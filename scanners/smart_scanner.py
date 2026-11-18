"""Combined scanner for the dashboard: people, objects and visible expressions."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai.attributes import detect_faces
from ai.expressions import analyze_expressions
from ai.focus_targets import COCO_TO_KEY, FOCUS_COCO_FLOOR
from ai.knowledge import GESTURE_INFO, lookup
from ai.registry import registry
from ai.utils import _iou, dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict_with_people
from scanners.focus import (
    extra_object_detections,
    focus_detections,
    run_world,
)
from scanners.person_scanner import person_detection


class SmartScanner:
    mode = "smart"

    def scan(self, image: np.ndarray, confidence: float = 0.32, *, live: bool = True,
             max_detections: int = 25) -> dict:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as pool:
            faces_future = pool.submit(detect_faces, image)
            world_future = pool.submit(run_world, image, live=live)
            raw_detections = yolo_predict_with_people(
                image,
                confidence,
                live=live,
                max_detections=max(25, max_detections * 3),
                object_floor=FOCUS_COCO_FLOOR,
                keep_classes=set(COCO_TO_KEY) | {"cell phone", "tablet"},
            )
            world_raw = world_future.result()
            faces = faces_future.result() if raw_detections else []
        used_faces: set[int] = set()

        detections = []
        focus_raw = []
        for raw in raw_detections:
            if raw["raw"] in COCO_TO_KEY:
                focus_raw.append(raw)  # handled by the focus pipeline below
                continue
            if raw["raw"] == "person":
                detections.append(person_detection(
                    image, raw, faces, used_faces, source="YOLO11s · Smart Scan person"
                ))
                continue
            # Phones/tablets are kept at a low floor only to suppress remote/mouse
            # look-alikes; do not show them below the user's confidence slider.
            if raw["confidence"] < confidence:
                continue
            info = lookup(raw["raw"])
            if not info:
                continue
            detections.append(make_detection(
                label=info["label"],
                category=info["category"],
                confidence=raw["confidence"],
                bbox=raw["bbox"],
                description=info["description"],
                reference_key=raw["raw"],
                source="YOLO11s · COCO 80",
                details={"coco_class": raw["raw"]},
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
                if any(_iou(extra["bbox"], detection["bbox"]) > 0.6 for detection in detections):
                    continue
                detections.append(extra)

        expression_faces = []
        if faces:
            try:
                expression_faces = analyze_expressions(image)[:3]
            except Exception:
                expression_faces = []
        for face in expression_faces:
            info = GESTURE_INFO[face["expression_key"]]
            detections.append(make_detection(
                label=info["label"],
                category="Expressions",
                confidence=face["confidence"],
                bbox=face["bbox"],
                description=info["description"],
                reference_key=face["expression_key"],
                source="MediaPipe Face Landmarker · blendshapes",
                details={**face["scores"], "landmark_count": face["landmarks"]},
                note="Visible-expression estimate only. It does not identify emotions, mood or personality.",
            ))

        detections = dedupe_detections(detections)
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        detections = detections[:max_detections]

        notes: list[str] = []
        if not detections:
            notes.append(
                "Nothing from the supported vocabulary was confidently detected. "
                "Try better lighting, move closer, or tap Scan Now for a full-resolution scan."
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

        people = sum(1 for item in detections if item["category"] == "People")
        expressions = sum(1 for item in detections if item["category"] == "Expressions")
        objects = max(0, len(detections) - people - expressions)

        attach_references(detections)
        return result_payload("smart", detections, notes, started, image, {
            "count": len(detections),
            "people": people,
            "objects": objects,
            "expressions": expressions,
        })


smart_scanner = SmartScanner()
