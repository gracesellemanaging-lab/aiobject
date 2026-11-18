"""Object scanner: everyday objects only (no people). COCO 80 + priority objects."""
from __future__ import annotations

import time

import numpy as np

from ai.focus_targets import COCO_TO_KEY, FOCUS_COCO_FLOOR
from ai.knowledge import lookup
from ai.registry import registry
from ai.utils import _iou, dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict
from scanners.focus import (
    closest_guesses,
    extra_object_detections,
    focus_detections,
    run_world,
)


class ObjectScanner:
    mode = "object"

    def scan(self, image: np.ndarray, confidence: float = 0.35, *, live: bool = True,
             max_detections: int = 25) -> dict:
        started = time.perf_counter()
        coco_names = {name for name in registry.yolo_names.values() if name != "person"}
        raw_detections = yolo_predict(
            image,
            min(confidence, FOCUS_COCO_FLOOR),
            live=live,
            class_names=coco_names,
            max_detections=max(12, max_detections * 3),
        )

        world_started = time.perf_counter()
        world_raw = run_world(image, live=live)
        world_ms = round((time.perf_counter() - world_started) * 1000, 1)

        detections = []
        focus_raw = []
        for raw in raw_detections:
            if raw["raw"] in COCO_TO_KEY:
                focus_raw.append(raw)  # handled by the focus pipeline
                continue
            if raw["confidence"] < confidence:
                continue
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

        detections.extend(focus_detections(
            image,
            confidence,
            live,
            coco_raw=focus_raw,
            world_raw=world_raw,
            coco_all=raw_detections,
        ))

        if not live:
            for extra in extra_object_detections(image, confidence, world_raw):
                # skip if a COCO detection already covers this spot
                if any(_iou(extra["bbox"], detection["bbox"]) > 0.6 for detection in detections):
                    continue
                detections.append(extra)

        detections = dedupe_detections(detections)
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        detections = detections[:max_detections]

        notes: list[str] = []
        if not detections:
            notes.append(
                "No object detected. People are excluded here: use Person Detection or Smart Scan. "
                "Move closer, add light, or tap Scan Now for a full-resolution scan."
            )
        if registry.world_state in {"idle", "loading"}:
            notes.append(
                "Charger, cable and adapter detection is still loading. "
                "Mouse, remote and keyboard detection already works."
            )
        elif registry.world_state == "error":
            reason = str(registry.world_error or "YOLO-World unavailable")[:160]
            notes.append(
                "Charger, cable and adapter detection is unavailable: "
                f"{reason}. Mouse, remote and keyboard detection still works."
            )
        elif not live and not detections:
            guesses = closest_guesses(world_raw)
            if guesses:
                notes.append("Closest open-vocabulary guesses (too weak to show): " + ", ".join(guesses) + ".")

        attach_references(detections)
        return result_payload("object", detections, notes, started, image, {
            "supported_vocabulary": "COCO 80 (no person) + priority objects (mouse, remote, keyboard, charger, cable)",
            "count": len(detections),
            "world_ms": world_ms,
        })


object_scanner = ObjectScanner()
