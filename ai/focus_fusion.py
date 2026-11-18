"""Combine COCO (YOLO11) and open-vocabulary (YOLO-World) boxes for the focus objects.

Pure Python on normalised ``[x, y, w, h]`` boxes, so it is unit-testable without any model.

Candidate dict: ``{"key": str, "conf": float, "bbox": [x, y, w, h]}``.
"""
from __future__ import annotations

from typing import Any

from .focus_targets import FOCUS_TARGETS
from .utils import _iou

AGREE_IOU = 0.40        # COCO box and World box count as the same object above this overlap
AGREE_BONUS = 0.20      # confidence boost when both models agree
AGREE_RELAX = 0.60      # when both agree each only has to reach 60% of the normal threshold
SAME_KEY_NMS = 0.50
CROSS_KEY_NMS = 0.60
NEGATIVE_IOU = 0.50


def threshold(key: str, confidence: float) -> float:
    """The user's slider can only make a target MORE sensitive, never stricter than its own floor."""
    return min(FOCUS_TARGETS[key].min_conf, confidence)


def area_ok(key: str, bbox: list[float]) -> bool:
    target = FOCUS_TARGETS[key]
    return target.min_area <= bbox[2] * bbox[3] <= target.max_area


def nms(items: list[dict[str, Any]], iou: float = SAME_KEY_NMS) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda d: d["conf"], reverse=True):
        if all(item["key"] != other["key"] or _iou(item["bbox"], other["bbox"]) < iou for other in kept):
            kept.append(item)
    return kept


def fuse(
    coco: list[dict[str, Any]],
    world: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    confidence: float,
) -> list[dict[str, Any]]:
    """Return the final focus detections, each with ``conf``, ``bbox``, ``key`` and ``sources``."""
    world = nms(world)
    candidates: list[dict[str, Any]] = []
    used_world: set[int] = set()

    for c in coco:
        best_i, best_iou = None, 0.0
        for i, w in enumerate(world):
            if i in used_world or w["key"] != c["key"]:
                continue
            overlap = _iou(c["bbox"], w["bbox"])
            if overlap >= AGREE_IOU and overlap > best_iou:
                best_i, best_iou = i, overlap
        if best_i is None:
            candidates.append({**c, "agree": False, "sources": ["coco"]})
            continue
        used_world.add(best_i)
        w = world[best_i]
        top = c if c["conf"] >= w["conf"] else w
        candidates.append({
            "key": c["key"], "bbox": top["bbox"], "agree": True, "sources": ["coco", "world"],
            "conf": min(0.99, max(c["conf"], w["conf"]) + AGREE_BONUS),
        })
    for i, w in enumerate(world):
        if i not in used_world:
            candidates.append({**w, "agree": False, "sources": ["world"]})

    # 1) per-target threshold (relaxed when both models agree)
    kept = []
    for cand in candidates:
        thr = threshold(cand["key"], confidence)
        if cand["agree"] or cand["conf"] >= thr:
            kept.append(cand)

    # 2) drop targets that a look-alike explains better (phone / game controller vs remote)
    survivors = []
    for cand in kept:
        beaten = any(
            neg["key"] == cand["key"] and neg["conf"] > cand["conf"]
            and _iou(neg["bbox"], cand["bbox"]) >= NEGATIVE_IOU
            for neg in negatives
        )
        if not beaten:
            survivors.append(cand)

    # 3) one label per object: same-key duplicates and cross-key conflicts, strongest wins
    final: list[dict[str, Any]] = []
    for cand in sorted(survivors, key=lambda d: d["conf"], reverse=True):
        clash = False
        for other in final:
            limit = SAME_KEY_NMS if other["key"] == cand["key"] else CROSS_KEY_NMS
            if _iou(cand["bbox"], other["bbox"]) >= limit:
                clash = True
                break
        if not clash:
            final.append(cand)
    return final
