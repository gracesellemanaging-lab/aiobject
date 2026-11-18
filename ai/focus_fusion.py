"""Combine COCO and open-vocabulary boxes for the priority objects.

Pure Python on normalised ``[x, y, w, h]`` boxes, so it can be unit-tested
without loading a model.

Candidate dict: ``{"key": str, "conf": float, "bbox": [x, y, w, h]}``.
"""
from __future__ import annotations

from typing import Any

from .focus_targets import FOCUS_TARGETS
from .utils import _iou

AGREE_IOU = 0.40         # COCO box and World box count as the same object above this overlap
AGREE_BONUS = 0.08       # small confidence boost when both models agree
AGREE_RELAX = 0.60       # when both agree, each only needs 60% of the normal threshold
COCO_RELAX = 0.85        # a COCO-only hit is trusted slightly below its floor
SAME_KEY_NMS = 0.50
CROSS_KEY_NMS = 0.60
NEGATIVE_IOU = 0.50


def threshold(key: str, confidence: float) -> float:
    """The slider can make a priority object easier, never stricter than its floor."""
    return min(FOCUS_TARGETS[key].min_conf, confidence)


def shape_ok(key: str, bbox: list[float], width: float = 1.0, height: float = 1.0) -> bool:
    """Reject specks, frame-sized boxes and shapes that cannot be this object."""
    target = FOCUS_TARGETS[key]
    area = float(bbox[2]) * float(bbox[3])
    if not (target.min_area <= area <= target.max_area):
        return False
    box_width = max(1.0, float(bbox[2]) * float(width))
    box_height = max(1.0, float(bbox[3]) * float(height))
    ratio = max(box_width, box_height) / min(box_width, box_height)
    return target.min_ratio <= ratio <= target.max_ratio


def nms(items: list[dict[str, Any]], iou: float = SAME_KEY_NMS) -> list[dict[str, Any]]:
    """Keep the strongest box for the same label."""
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda detection: detection["conf"], reverse=True):
        if all(
            item["key"] != other["key"] or _iou(item["bbox"], other["bbox"]) < iou
            for other in kept
        ):
            kept.append(item)
    return kept


def fuse(
    coco: list[dict[str, Any]],
    world: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    confidence: float,
) -> list[dict[str, Any]]:
    """Return final focus detections with key, bbox, conf and sources."""
    world = nms(world)
    candidates: list[dict[str, Any]] = []
    used_world: set[int] = set()

    for coco_item in coco:
        best_index = None
        best_overlap = 0.0
        for index, world_item in enumerate(world):
            if index in used_world or world_item["key"] != coco_item["key"]:
                continue
            overlap = _iou(coco_item["bbox"], world_item["bbox"])
            if overlap >= AGREE_IOU and overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is None:
            candidates.append({**coco_item, "agree": False, "sources": ["coco"]})
            continue
        used_world.add(best_index)
        world_item = world[best_index]
        top = coco_item if coco_item["conf"] >= world_item["conf"] else world_item
        candidates.append({
            "key": coco_item["key"],
            "bbox": top["bbox"],
            "agree": True,
            "sources": ["coco", "world"],
            "conf": min(0.95, max(coco_item["conf"], world_item["conf"]) + AGREE_BONUS),
        })

    for index, world_item in enumerate(world):
        if index not in used_world:
            candidates.append({**world_item, "agree": False, "sources": ["world"]})

    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        target_threshold = threshold(candidate["key"], confidence)
        if candidate["sources"] == ["coco"]:
            target_threshold *= COCO_RELAX
        if candidate["agree"] or candidate["conf"] >= target_threshold:
            kept.append(candidate)

    # Drop targets that a look-alike explains better (phone vs remote, etc.).
    survivors: list[dict[str, Any]] = []
    for candidate in kept:
        beaten = any(
            negative["key"] == candidate["key"]
            and negative["conf"] > candidate["conf"]
            and _iou(negative["bbox"], candidate["bbox"]) >= NEGATIVE_IOU
            for negative in negatives
        )
        if not beaten:
            survivors.append(candidate)

    # One label per object: same-label duplicates and cross-label conflicts.
    final: list[dict[str, Any]] = []
    for candidate in sorted(survivors, key=lambda detection: detection["conf"], reverse=True):
        clash = False
        for other in final:
            limit = SAME_KEY_NMS if other["key"] == candidate["key"] else CROSS_KEY_NMS
            if _iou(candidate["bbox"], other["bbox"]) >= limit:
                clash = True
                break
        if not clash:
            final.append(candidate)
    return final
