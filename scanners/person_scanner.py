"""Person detection with clearly-labeled approximate age-range estimates."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ai.attributes import ATTRIBUTE_DISCLAIMER, detect_faces, estimate_attributes, face_inside_box
from ai.utils import dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict


def person_detection(image: np.ndarray, raw: dict, faces: list[dict], used_faces: set[int],
                     source: str = "YOLOv8n · COCO person") -> dict[str, Any]:
    box = raw["bbox"]
    candidate = None
    for face in faces:
        if id(face) in used_faces:
            continue
        if face_inside_box(face, box):
            candidate = face
            break
    attributes: dict[str, Any] = {
        "face_visible": candidate is not None,
        "age_range": None,
        "age_is_estimate": True,
        "disclaimer": ATTRIBUTE_DISCLAIMER,
    }
    details: dict[str, Any] = {"detector": "YOLOv8n", "coco_class": "person"}
    if candidate is not None:
        used_faces.add(id(candidate))
        estimate = estimate_attributes(image, candidate)
        if estimate.get("available"):
            attributes.update({
                "age_range": estimate["age_range"],
                "age_confidence": estimate["age_confidence"],
                "face_confidence": estimate["face_confidence"],
                "age_distribution": estimate.get("age_distribution", {}),
            })
            details["attribute_engine"] = "FairFace age head"
        else:
            attributes["reason"] = estimate.get("reason", "Age estimate unavailable")
    else:
        attributes["reason"] = "No clearly visible face inside the person box"

    note = (
        "Age is an uncertain AI prediction from visible facial features, not an exact age. "
        "No identity is recognized or stored, and gender is not inferred."
    )
    return make_detection(
        label="Person Detected",
        category="People",
        confidence=raw["confidence"],
        bbox=box,
        description=(
            "A human was localized. The system does not identify people by name, "
            "face, gender or personal identity."
        ),
        reference_key="person",
        source=source,
        attributes=attributes,
        details=details,
        note=note,
    )


class PersonScanner:
    mode = "person"

    def scan(self, image: np.ndarray, confidence: float = 0.35, *, live: bool = True,
             max_detections: int = 12) -> dict:
        started = time.perf_counter()
        raw_detections = yolo_predict(
            image, confidence, live=live, class_names={"person"}, max_detections=max_detections
        )
        faces = detect_faces(image) if raw_detections else []
        used_faces: set[int] = set()
        detections = [person_detection(image, raw, faces, used_faces) for raw in raw_detections]

        detections = dedupe_detections(detections)[:max_detections]
        detections.sort(key=lambda item: item["confidence"], reverse=True)

        notes: list[str] = []
        if not detections:
            notes.append("No person detected. Make sure a person is clearly visible in the camera view.")
        elif not any(item["attributes"].get("face_visible") for item in detections):
            notes.append("A person was detected, but no face was clearly visible, so the age prediction is unavailable.")
        notes.append("Age ranges are approximate AI predictions, not exact ages. Gender is not inferred.")

        attach_references(detections)
        return result_payload("person", detections, notes, started, image, {
            "count": len(detections),
            "face_count": len(faces),
            "attribute_engine": "FairFace (age head only)",
        })


person_scanner = PersonScanner()
