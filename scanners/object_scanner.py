"""Universal COCO object scanner."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ai.attributes import detect_faces
from scanners.person_scanner import person_detection

from ai.knowledge import lookup
from ai.utils import dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict


class ObjectScanner:
    mode = "object"

    def scan(self, image: np.ndarray, confidence: float = 0.35, *, live: bool = True,
             max_detections: int = 25) -> dict:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            faces_future = pool.submit(detect_faces, image)
            raw_detections = yolo_predict(image, confidence, live=live, max_detections=max(6, max_detections * 2))
            faces = faces_future.result() if any(r["raw"] == "person" for r in raw_detections) else []
        used_faces: set[int] = set()

        detections = []
        for raw in raw_detections:
            if raw["raw"] == "person":
                detections.append(person_detection(
                    image, raw, faces, used_faces, source="YOLOv8n · Object Detection person"
                ))
                continue
            info = lookup(raw["raw"])
            if info:
                label = info["label"]
                category = info["category"]
                description = info["description"]
            else:
                label = raw["raw"].replace("_", " ").title()
                category = "Object"
                description = "An object detected by the YOLOv8n model."
            detections.append(make_detection(
                label=label,
                category=category,
                confidence=raw["confidence"],
                bbox=raw["bbox"],
                description=description,
                reference_key=raw["raw"],
                source="YOLOv8n · COCO 80",
                details={"coco_class": raw["raw"], "class_id": raw["class_id"]},
            ))

        detections = dedupe_detections(detections)
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        detections = detections[:max_detections]

        notes: list[str] = []
        if not detections:
            notes.append(
                "No object from the supported 80-class COCO vocabulary was detected. "
                "Move closer, add light, or scan an object in the camera view."
            )

        attach_references(detections)
        return result_payload("object", detections, notes, started, image, {
            "supported_vocabulary": "COCO 80 classes",
            "count": len(detections),
        })


object_scanner = ObjectScanner()
