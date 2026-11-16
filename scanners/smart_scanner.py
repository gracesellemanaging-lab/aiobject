"""Combined scanner for the dashboard: people, objects and visible expressions."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai.attributes import detect_faces
from ai.expressions import analyze_expressions
from ai.knowledge import GESTURE_INFO, lookup
from ai.utils import dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict_with_people
from scanners.person_scanner import person_detection


class SmartScanner:
    mode = "smart"

    def scan(self, image: np.ndarray, confidence: float = 0.32, *, live: bool = True,
             max_detections: int = 25) -> dict:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            faces_future = pool.submit(detect_faces, image)
            raw_detections = yolo_predict_with_people(image, confidence, live=live, max_detections=max_detections)
            faces = faces_future.result() if raw_detections else []
        used_faces: set[int] = set()

        detections = []
        for raw in raw_detections:
            if raw["raw"] == "person":
                detections.append(person_detection(
                    image, raw, faces, used_faces, source="YOLOv8n · Smart Scan person"
                ))
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
                source="YOLOv8n · COCO 80",
                details={"coco_class": raw["raw"]},
            ))

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
                "Try better lighting, a closer subject, or a dedicated scanner mode."
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
