"""Fruit / vegetable scanner: YOLO localization + ImageNet food classification."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ai.classifier import CLASSIFIER_SOURCE, analyze_food_box, analyze_food_image, best_candidate
from ai.knowledge import FRUIT_SCAN_SUPPORTED, YOLO_FRUIT_CLASSES, lookup
from ai.utils import dedupe_detections, make_detection
from scanners.common import attach_references, result_payload, yolo_predict

UNSUPPORTED_FRUITS = ["Mango", "Grapes", "Watermelon", "Papaya", "Dragon Fruit"]
FOOD_CATEGORIES = {"Fruits", "Vegetables", "Food", "Food & Drink"}


class FruitScanner:
    mode = "fruit"

    def scan(self, image: np.ndarray, confidence: float = 0.3, *, live: bool = True,
             max_detections: int = 12) -> dict:
        started = time.perf_counter()
        raw_detections = yolo_predict(
            image,
            max(0.22, confidence),
            live=live,
            class_names=YOLO_FRUIT_CLASSES,
            max_detections=max(6, max_detections * 2),
        )

        detections: list[dict[str, Any]] = []
        targets = raw_detections[:max_detections]
        for raw in targets:
            base_info = lookup(raw["raw"])
            candidates: list[dict] = []
            try:
                candidates = analyze_food_box(image, raw["bbox"])
            except Exception:
                candidates = []
            candidate = best_candidate(candidates, minimum=0.22)

            chosen = base_info
            chosen_key = raw["raw"]
            chosen_confidence = raw["confidence"]
            source = "YOLO11s · COCO"
            note = None
            if candidate and candidate.get("label"):
                same_label = base_info and candidate["label"].lower() == base_info["label"].lower()
                confident_override = candidate["probability"] >= 0.55
                if candidate["category"] in FOOD_CATEGORIES and (same_label or confident_override):
                    chosen = candidate
                    chosen_key = candidate["raw"]
                    chosen_confidence = candidate["probability"]
                    source = f"{CLASSIFIER_SOURCE} · refined crop"
                    if not same_label:
                        note = "The classifier refined the YOLO label using the localized crop."
                if chosen_confidence < raw["confidence"] and same_label:
                    chosen_confidence = raw["confidence"]

            if not chosen:
                continue
            details = {
                "yolo_class": raw["raw"],
                "yolo_confidence": round(raw["confidence"], 4),
                "classifier": [
                    {"name": item["name"], "probability": item["probability"]}
                    for item in candidates[:3]
                ],
            }
            detections.append(make_detection(
                label=chosen["label"],
                category=chosen["category"],
                confidence=chosen_confidence,
                bbox=raw["bbox"],
                description=chosen["description"],
                reference_key=chosen_key,
                source=source,
                details=details,
                note=note,
            ))

        # If no fruit was localized, try the whole frame for close-up shots.
        whole_note = None
        if not detections:
            whole_candidates: list[dict] = []
            try:
                whole_candidates = analyze_food_image(image)
            except Exception:
                whole_candidates = []
            candidate = best_candidate(whole_candidates, minimum=0.4)
            if candidate and candidate["category"] in FOOD_CATEGORIES:
                detections.append(make_detection(
                    label=candidate["label"],
                    category=candidate["category"],
                    confidence=candidate["probability"],
                    bbox=None,
                    description=candidate["description"],
                    reference_key=candidate["raw"],
                    source=f"{CLASSIFIER_SOURCE} · whole image",
                    details={
                        "classifier": [
                            {"name": item["name"], "probability": item["probability"]}
                            for item in whole_candidates[:3]
                        ],
                    },
                    note="No individual fruit was localized; this label comes from whole-image classification.",
                    localized=False,
                ))
                whole_note = "No bounding box was produced because the fruit was recognized from the whole frame."

        detections = dedupe_detections(detections)[:max_detections]
        detections.sort(key=lambda item: item["confidence"], reverse=True)

        notes: list[str] = []
        if whole_note:
            notes.append(whole_note)
        if not detections:
            notes.append(
                "No supported fruit or vegetable was recognized. Supported items include apple, banana, "
                "orange, lemon, pineapple, strawberry, pomegranate, fig, cucumber, bell pepper and broccoli."
            )
            notes.append(
                "Mango, grapes, watermelon, papaya and dragon fruit are not supported in this build, "
                "so the app will not guess them."
            )

        attach_references(detections)
        return result_payload("fruit", detections, notes, started, image, {
            "count": len(detections),
            "supported": FRUIT_SCAN_SUPPORTED,
            "unsupported": UNSUPPORTED_FRUITS,
        })


fruit_scanner = FruitScanner()
