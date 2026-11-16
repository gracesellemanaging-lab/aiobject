"""Shared helpers for the scanner modules."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ai.config import LIVE_IMAGE_SIZE, STILL_IMAGE_SIZE
from ai.description import scene_description, search_keywords
from ai.knowledge import SCANNER_META
from ai.references import reference_service
from ai.registry import registry
from ai.utils import image_size, normalized_box


PERSON_CONFIDENCE_CAP = 0.40


def yolo_predict_with_people(image: np.ndarray, confidence: float, **kwargs: Any) -> list[dict[str, Any]]:
    """Run YOLO at a lower floor for people, then apply the user's threshold to other objects.

    People are large, easy targets; a high slider value (e.g. 83%) otherwise hides them in
    live camera frames.
    """
    floor = min(confidence, PERSON_CONFIDENCE_CAP)
    raw = yolo_predict(image, floor, **kwargs)
    return [d for d in raw if d["raw"] == "person" or d["confidence"] >= confidence]


def yolo_predict(
    image: np.ndarray,
    confidence: float,
    *,
    live: bool = True,
    class_names: set[str] | None = None,
    max_detections: int = 40,
) -> list[dict[str, Any]]:
    model = registry.yolo()
    names = dict(model.names)
    class_ids = None
    if class_names is not None:
        class_ids = [index for index, name in names.items() if name in class_names]
        if not class_ids:
            return []

    height, width = image.shape[:2]
    imgsz = LIVE_IMAGE_SIZE if live else STILL_IMAGE_SIZE
    with registry.yolo_lock:
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=max(0.12, min(0.95, confidence)),
            iou=0.45,
            classes=class_ids,
            max_det=max_detections,
            verbose=False,
            device="cpu",
        )[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        raw = names.get(class_id, str(class_id))
        coords = box.xyxy[0].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = coords
        detections.append({
            "raw": raw,
            "class_id": class_id,
            "confidence": float(box.conf[0].item()),
            "bbox_pixels": [x1, y1, x2, y2],
            "bbox": normalized_box(x1, y1, x2, y2, width, height),
        })
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def attach_references(detections: list[dict[str, Any]], prime: bool = True) -> None:
    """Attach cached references and warm missing ones in the background."""
    priming: list[tuple[str, str]] = []
    seen: set[str] = set()
    for detection in detections:
        key = detection.get("reference_key")
        if not key:
            detection["reference"] = None
            continue
        cached = reference_service.cached(key)
        detection["reference"] = cached
        if not cached and prime and key not in seen:
            seen.add(key)
            priming.append((key, detection.get("label", key)))
    if priming:
        reference_service.prime(priming)


def group_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate identical labels so the UI can show real quantities."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for detection in detections:
        label = str(detection.get("label") or "Object")
        category = str(detection.get("category") or "Object")
        key = (label.lower(), category)
        confidence = float(detection.get("confidence") or 0.0)
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "label": label,
                "category": category,
                "count": 1,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 1),
            }
        else:
            group["count"] += 1
            if confidence > group["confidence"]:
                group["confidence"] = confidence
                group["confidence_percent"] = round(confidence * 100, 1)
    return sorted(groups.values(), key=lambda group: (-group["confidence"], group["label"]))


def annotate_quantities(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the number of same-label detections to every item."""
    counts: dict[tuple[str, str], int] = {}
    for detection in detections:
        key = (str(detection.get("label") or "").lower(), str(detection.get("category") or ""))
        counts[key] = counts.get(key, 0) + 1
    for detection in detections:
        key = (str(detection.get("label") or "").lower(), str(detection.get("category") or ""))
        detection["quantity"] = counts[key]
    return detections


def result_payload(
    mode: str,
    detections: list[dict[str, Any]],
    notes: list[str],
    started: float,
    image: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = SCANNER_META.get(mode, SCANNER_META["object"])
    annotate_quantities(detections)
    groups = group_detections(detections)
    dimensions = image_size(image)
    people = sum(1 for item in detections if item.get("category") == "People")
    expressions = sum(1 for item in detections if item.get("category") == "Expressions")
    fruits = sum(
        1 for item in detections
        if item.get("category") in {"Fruits", "Vegetables", "Food", "Food & Drink"}
    )
    objects = max(0, len(detections) - people - expressions - fruits)
    payload = {
        "mode": mode,
        "detections": detections,
        "notes": notes,
        "description": scene_description(detections, dimensions),
        "keywords": search_keywords(detections),
        "meta": {
            "title": meta["title"],
            "engine": meta["engine"],
            "capability_note": meta["note"],
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            "image": dimensions,
            "server_time": time.time(),
            "people": people,
            "objects": objects,
            "fruits": fruits,
            "expressions": expressions,
            "groups": groups,
        },
    }
    if mode == "fruit":
        payload["meta"]["fruit_summary"] = [
            {
                "label": group["label"],
                "quantity": group["count"],
                "estimated": True,
                "confidence_percent": group["confidence_percent"],
            }
            for group in groups
            if group["category"] in {"Fruits", "Vegetables", "Food", "Food & Drink"}
        ]
    if extra:
        payload["meta"].update(extra)
    return payload
