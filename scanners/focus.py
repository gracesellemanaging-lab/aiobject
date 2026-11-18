"""Accurate detection of the priority objects: mouse, remote, keyboard, ballpen, adapter, connector.

Pipeline
1. YOLO11 (COCO) at a LOW floor for mouse / remote / keyboard, so weak-but-real hits survive.
2. YOLO-World (open vocabulary) with several prompts per object, plus look-alike prompts.
   On still images it also scans 4 overlapping tiles, which is what makes thin or small things
   (ballpen, connector) findable.
3. ai.focus_fusion merges both: agreement boosts the score, look-alikes and size outliers are dropped.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ai.focus_fusion import area_ok, fuse, nms, threshold
from ai.focus_targets import COCO_TO_KEY, FOCUS_TARGETS, NEGATIVE_PROMPTS, PROMPT_TO_KEY
from ai.registry import registry
from ai.utils import make_detection, normalized_box
from scanners.common import yolo_predict

WORLD_FLOOR = 0.04
TILING = os.environ.get("FOCUS_TILING", "1") != "0"
LIVE_WORLD = os.environ.get("FOCUS_LIVE_WORLD", "1") != "0"
_EDGE_PX = 3


def run_world(image: np.ndarray, *, live: bool) -> list[tuple[str, float, list[float]]]:
    """Run YOLO-World once (plus tiles on stills). Returns ``(prompt, conf, [x1,y1,x2,y2] px)``."""
    if live and (not LIVE_WORLD or registry.world_state != "ready"):
        return []  # never block a live frame while the open-vocabulary model is still loading
    model = registry.world()
    if model is None:
        return []
    height, width = image.shape[:2]
    imgsz = 512 if live else 640
    views: list[tuple[int, int, np.ndarray]] = [(0, 0, image)]
    if TILING and not live and min(height, width) >= 480:
        tw, th = int(width * 0.6), int(height * 0.6)
        for oy in (0, height - th):
            for ox in (0, width - tw):
                views.append((ox, oy, image[oy:oy + th, ox:ox + tw]))

    found: list[tuple[str, float, list[float]]] = []
    for ox, oy, view in views:
        vh, vw = view.shape[:2]
        try:
            with registry.world_lock:
                result = model.predict(view, imgsz=imgsz, conf=WORLD_FLOOR, iou=0.5,
                                       verbose=False, device="cpu")[0]
        except Exception:  # noqa: BLE001 - extras must never break a scan
            continue
        names = result.names
        for box in getattr(result, "boxes", None) or []:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            if view is not image:  # drop boxes cut by an inner tile border (partial objects)
                cut_x = (ox > 0 and x1 < _EDGE_PX) or (ox + vw < width and x2 > vw - _EDGE_PX)
                cut_y = (oy > 0 and y1 < _EDGE_PX) or (oy + vh < height and y2 > vh - _EDGE_PX)
                if cut_x or cut_y:
                    continue
            found.append((names.get(int(box.cls[0].item()), ""), float(box.conf[0].item()),
                          [x1 + ox, y1 + oy, x2 + ox, y2 + oy]))
    return found


def focus_detections(
    image: np.ndarray,
    confidence: float,
    live: bool,
    world_raw: list[tuple[str, float, list[float]]] | None = None,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]

    # 1) COCO at a low floor (per-target thresholds are applied during fusion)
    floor = max(0.12, 0.6 * min(t.min_conf for t in FOCUS_TARGETS.values() if t.coco))
    coco = []
    for raw in yolo_predict(image, min(confidence, floor), live=live,
                            class_names=set(COCO_TO_KEY), max_detections=30):
        key = COCO_TO_KEY.get(raw["raw"])
        if key and area_ok(key, raw["bbox"]) and raw["confidence"] >= 0.6 * threshold(key, confidence):
            coco.append({"key": key, "conf": raw["confidence"], "bbox": raw["bbox"]})

    # 2) YOLO-World
    world: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    if world_raw is None:
        world_raw = run_world(image, live=live)
    for prompt, conf, (x1, y1, x2, y2) in world_raw or []:
        bbox = normalized_box(x1, y1, x2, y2, width, height)
        if prompt in NEGATIVE_PROMPTS:
            negatives.append({"key": NEGATIVE_PROMPTS[prompt], "conf": conf, "bbox": bbox})
            continue
        key = PROMPT_TO_KEY.get(prompt)
        if key and area_ok(key, bbox) and conf >= 0.6 * threshold(key, confidence):
            world.append({"key": key, "conf": conf, "bbox": bbox})

    # 3) fuse
    out = []
    for item in fuse(coco, nms(world), negatives, confidence):
        target = FOCUS_TARGETS[item["key"]]
        coco_name = target.coco or target.key
        out.append(make_detection(
            label=target.label, category=target.category, confidence=item["conf"],
            bbox=item["bbox"], description=target.description, reference_key=coco_name,
            source=" + ".join({"coco": "YOLO11s", "world": "YOLO-World"}[s] for s in item["sources"])
                   + " · focus objects",
            details={"focus_key": item["key"], "sources": item["sources"]},
        ))
    return out


def closest_guesses(world_raw: list[tuple[str, float, list[float]]], limit: int = 3) -> list[str]:
    """Best open-vocabulary guesses, shown to the user when nothing passes the thresholds."""
    best: dict[str, float] = {}
    for prompt, conf, _box in world_raw:
        if prompt:
            best[prompt] = max(best.get(prompt, 0.0), conf)
    top = sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [f"{name} {round(score * 100)}%" for name, score in top if score >= 0.04]
