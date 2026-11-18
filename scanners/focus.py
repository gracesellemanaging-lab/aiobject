"""Focus pipeline for mouse, remote, keyboard, chargers and cables.

1. COCO detections are already collected by the scanner at a low floor.
2. YOLO-World runs once with several prompts per priority object. Still images
   also get four overlapping tiles so small or thin objects are easier to see.
3. ``ai.focus_fusion`` merges the two sources, rejects look-alikes and keeps
   one label per object.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ai.extra_objects import EXTRA_OBJECTS, WORLD_CONFIDENCE_CAP
from ai.focus_fusion import AGREE_RELAX, fuse, shape_ok, threshold
from ai.focus_targets import (
    COCO_TO_KEY,
    FOCUS_COCO_FLOOR,
    FOCUS_PROMPTS,
    FOCUS_TARGETS,
    NEGATIVE_PROMPTS,
    PROMPT_TO_KEY,
)
from ai.registry import registry
from ai.utils import _iou, make_detection, normalized_box
from scanners.common import yolo_predict

WORLD_FLOOR = float(os.environ.get("WORLD_FLOOR", "0.05"))
WORLD_LIVE = os.environ.get("WORLD_LIVE", "1") != "0"
WORLD_TILING = os.environ.get("WORLD_TILING", "1") != "0"
WORLD_LIVE_IMGSZ = int(os.environ.get("WORLD_LIVE_IMGSZ", "640"))
WORLD_STILL_IMGSZ = int(os.environ.get("WORLD_STILL_IMGSZ", "640"))
TILE_FRACTION = 0.60
EDGE_PX = 3
REFERENCE_KEYS = {
    "charger_adapter": "charger-adapter",
    "charger_cable": "charger-cable",
    "charging_connector": "charging-connector",
}


def _tile_views(image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    height, width = image.shape[:2]
    if not WORLD_TILING or min(height, width) < 480:
        return []
    tile_width = int(width * TILE_FRACTION)
    tile_height = int(height * TILE_FRACTION)
    views: list[tuple[int, int, np.ndarray]] = []
    for offset_y in (0, height - tile_height):
        for offset_x in (0, width - tile_width):
            views.append((
                offset_x,
                offset_y,
                image[offset_y:offset_y + tile_height, offset_x:offset_x + tile_width],
            ))
    return views


def run_world(image: np.ndarray, *, live: bool) -> list[tuple[str, float, list[float]]]:
    """Run YOLO-World and return ``(prompt, confidence, [x1, y1, x2, y2] px)``."""
    if live and (not WORLD_LIVE or registry.world_state != "ready"):
        return []  # never block a live frame while the open-vocabulary model is loading
    model = registry.world()
    if model is None:
        return []

    names = model.names
    if not isinstance(names, dict):
        names = dict(enumerate(names))
    live_classes = None
    if live:
        live_classes = [index for index, prompt in names.items() if prompt in FOCUS_PROMPTS]
        if not live_classes:
            return []

    height, width = image.shape[:2]
    views: list[tuple[int, int, np.ndarray]] = [(0, 0, image)]
    if not live:
        views.extend(_tile_views(image))

    found: list[tuple[str, float, list[float]]] = []
    for offset_x, offset_y, view in views:
        view_height, view_width = view.shape[:2]
        try:
            with registry.world_lock:
                result = model.predict(
                    view,
                    imgsz=WORLD_LIVE_IMGSZ if live else WORLD_STILL_IMGSZ,
                    conf=WORLD_FLOOR,
                    iou=0.5,
                    classes=live_classes,
                    max_det=50 if live else 100,
                    verbose=False,
                    device="cpu",
                )[0]
        except Exception:  # noqa: BLE001 - extras must never break a scan
            continue

        for box in getattr(result, "boxes", None) or []:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            if view is not image:  # drop boxes cut by an inner tile border
                cut_x = (
                    (offset_x > 0 and x1 < EDGE_PX)
                    or (offset_x + view_width < width and x2 > view_width - EDGE_PX)
                )
                cut_y = (
                    (offset_y > 0 and y1 < EDGE_PX)
                    or (offset_y + view_height < height and y2 > view_height - EDGE_PX)
                )
                if cut_x or cut_y:
                    continue
            found.append((
                names.get(int(box.cls[0].item()), ""),
                float(box.conf[0].item()),
                [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
            ))
    return found


def focus_detections(
    image: np.ndarray,
    confidence: float,
    live: bool,
    *,
    coco_raw: list[dict[str, Any]] | None = None,
    world_raw: list[tuple[str, float, list[float]]] | None = None,
    coco_all: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fuse low-floor COCO hits with YOLO-World prompts into final detections."""
    height, width = image.shape[:2]

    if coco_raw is None:
        coco_raw = yolo_predict(
            image,
            min(confidence, FOCUS_COCO_FLOOR),
            live=live,
            class_names=set(COCO_TO_KEY),
            max_detections=30,
        )
        if coco_all is None:
            coco_all = coco_raw
    if world_raw is None:
        world_raw = run_world(image, live=live)

    coco_candidates: list[dict[str, Any]] = []
    for raw in coco_raw:
        key = COCO_TO_KEY.get(raw["raw"])
        if not key or not shape_ok(key, raw["bbox"], width, height):
            continue
        if raw["confidence"] >= AGREE_RELAX * threshold(key, confidence):
            coco_candidates.append({
                "key": key,
                "conf": raw["confidence"],
                "bbox": raw["bbox"],
            })

    world_candidates: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for prompt, score, (x1, y1, x2, y2) in world_raw or []:
        bbox = normalized_box(x1, y1, x2, y2, width, height)
        if prompt in NEGATIVE_PROMPTS:
            negatives.extend(
                {"key": target_key, "conf": score, "bbox": bbox}
                for target_key in NEGATIVE_PROMPTS[prompt]
            )
            continue
        key = PROMPT_TO_KEY.get(prompt)
        if not key or not shape_ok(key, bbox, width, height):
            continue
        if score >= AGREE_RELAX * threshold(key, confidence):
            world_candidates.append({"key": key, "conf": score, "bbox": bbox})

    # A phone is the most common look-alike for a remote control or a mouse;
    # a TV can be mistaken for a remote control.
    for raw in coco_all or []:
        if raw.get("raw") in {"cell phone", "smartphone", "tablet"}:
            for target_key in ("remote", "mouse", "charger_adapter"):
                negatives.append({
                    "key": target_key,
                    "conf": raw["confidence"],
                    "bbox": raw["bbox"],
                })
        elif raw.get("raw") == "tv":
            negatives.append({
                "key": "remote",
                "conf": raw["confidence"],
                "bbox": raw["bbox"],
            })

    detections: list[dict[str, Any]] = []
    for item in fuse(coco_candidates, world_candidates, negatives, confidence):
        target = FOCUS_TARGETS[item["key"]]
        source_names = " + ".join(
            {"coco": "YOLO11s", "world": "YOLO-World"}[source]
            for source in item["sources"]
        )
        detections.append(make_detection(
            label=target.label,
            category=target.category,
            confidence=item["conf"],
            bbox=item["bbox"],
            description=target.description,
            reference_key=REFERENCE_KEYS.get(item["key"], target.coco or item["key"]),
            source=f"{source_names} · focus object",
            details={"focus_key": item["key"], "sources": item["sources"]},
        ))
    return detections


def closest_guesses(world_raw: list[tuple[str, float, list[float]]], limit: int = 3) -> list[str]:
    """Best open-vocabulary guesses, shown when every candidate is too weak."""
    best: dict[str, float] = {}
    for prompt, score, _box in world_raw:
        if prompt and prompt not in NEGATIVE_PROMPTS:
            best[prompt] = max(best.get(prompt, 0.0), score)
    top = sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [f"{name} {round(score * 100)}%" for name, score in top if score >= 0.04]


def extra_object_detections(
    image: np.ndarray,
    confidence: float,
    world_raw: list[tuple[str, float, list[float]]],
) -> list[dict[str, Any]]:
    """Build detections for the non-priority open-vocabulary objects.

    Focus prompts (chargers, cables, mouse, remote, keyboard) are handled by
    ``focus_detections`` so they are skipped here.
    """
    height, width = image.shape[:2]
    floor = max(0.12, min(confidence, WORLD_CONFIDENCE_CAP))
    detections: list[dict[str, Any]] = []
    for prompt, score, (x1, y1, x2, y2) in world_raw:
        entry = EXTRA_OBJECTS.get(prompt)
        if not entry or score < floor:
            continue
        # Tiny specks are the open-vocabulary model's main false positive.
        if (x2 - x1) * (y2 - y1) < 0.003 * width * height:
            continue
        label, category, description = entry
        detections.append(make_detection(
            label=label,
            category=category,
            confidence=score,
            bbox=normalized_box(x1, y1, x2, y2, width, height),
            description=description,
            reference_key=label.lower(),
            source="YOLO-World · open vocabulary",
            details={"prompt": prompt},
        ))
    return detections
