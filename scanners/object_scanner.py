"""Object scanner: everyday objects only (no people). COCO 80 minus person + extra open-vocabulary objects."""
from __future__ import annotations

import time

import numpy as np

from ai.extra_objects import EXTRA_OBJECTS, WORLD_CONFIDENCE_CAP
from ai.focus_targets import COCO_TO_KEY
from ai.knowledge import lookup
from ai.registry import registry
from ai.utils import _iou, dedupe_detections, make_detection, normalized_box
from scanners.common import attach_references, result_payload, yolo_predict
from scanners.focus import focus_detections, run_world


def _extra_objects(world_raw: list, image: np.ndarray, confidence: float) -> list[dict]:
    """Objects that are not in COCO (earphones, wallet...), from the shared YOLO-World pass."""
    height, width = image.shape[:2]
    conf = max(0.12, min(confidence, WORLD_CONFIDENCE_CAP))
    found = []
    for prompt, score, (x1, y1, x2, y2) in world_raw:
        entry = EXTRA_OBJECTS.get(prompt)  # focus prompts are handled by scanners.focus
        if not entry or score < conf:
            continue
        # Tiny specks are the open-vocabulary model's main false positive.
        if (x2 - x1) * (y2 - y1) < 0.003 * width * height:
            continue
        found.append({
            "label": entry[0], "category": entry[1], "description": entry[2], "prompt": prompt,
            "confidence": score, "bbox": normalized_box(x1, y1, x2, y2, width, height),
        })
    return found


class ObjectScanner:
    mode = "object"

    def scan(self, image: np.ndarray, confidence: float = 0.35, *, live: bool = True,
             max_detections: int = 25) -> dict:
        started = time.perf_counter()
        # mouse / remote / keyboard are handled by the focus pipeline (lower floor + YOLO-World fusion)
        coco_names = {name for name in registry.yolo_names.values()
                      if name != "person" and name not in COCO_TO_KEY}
        raw_detections = yolo_predict(image, confidence, live=live, class_names=coco_names,
                                      max_detections=max(6, max_detections * 2))

        detections = []
        for raw in raw_detections:
            info = lookup(raw["raw"])
            if info:
                label, category, description = info["label"], info["category"], info["description"]
            else:
                label = raw["raw"].replace("_", " ").title()
                category = "Object"
                description = "An object detected by the YOLO11s model."
            detections.append(make_detection(
                label=label,
                category=category,
                confidence=raw["confidence"],
                bbox=raw["bbox"],
                description=description,
                reference_key=raw["raw"],
                source="YOLO11s · COCO 80",
                details={"coco_class": raw["raw"], "class_id": raw["class_id"]},
            ))

        world_raw = run_world(image, live=live)
        detections.extend(focus_detections(image, confidence, live, world_raw=world_raw))

        if not live:
            for extra in _extra_objects(world_raw, image, confidence):
                # skip if a COCO detection already covers this spot
                if any(_iou(extra["bbox"], d["bbox"]) > 0.6 for d in detections):
                    continue
                detections.append(make_detection(
                    label=extra["label"],
                    category=extra["category"],
                    confidence=extra["confidence"],
                    bbox=extra["bbox"],
                    description=extra["description"],
                    reference_key=extra["label"].lower(),
                    source="YOLO-World · open vocabulary",
                    details={"prompt": extra["prompt"]},
                ))

        detections = dedupe_detections(detections)
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        detections = detections[:max_detections]

        notes: list[str] = []
        if not detections:
            notes.append(
                "No object detected. People are excluded here: use Person Detection or Smart Scan. "
                "Move closer and add light."
            )
        if not live and registry.world_state != "ready":
            notes.append("Extra objects (earphones, wallet...) are still loading or unavailable.")

        attach_references(detections)
        return result_payload("object", detections, notes, started, image, {
            "supported_vocabulary": "COCO 80 (no person) + extra objects",
            "count": len(detections),
        })


object_scanner = ObjectScanner()
