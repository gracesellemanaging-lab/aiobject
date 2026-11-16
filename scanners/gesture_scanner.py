"""Gesture, posture and visible-expression scanner."""
from __future__ import annotations

import time

import numpy as np

from ai.gestures import gesture_engine
from scanners.common import attach_references, result_payload


class GestureScanner:
    mode = "gesture"

    def scan(self, image: np.ndarray, confidence: float = 0.35, *, live: bool = True,
             session_id: str | None = None) -> dict:
        started = time.perf_counter()
        analysis = gesture_engine.analyze(image, session_id=session_id, include_expressions=True)
        detections = analysis["detections"]
        notes = list(analysis["notes"])
        if not live and session_id is None:
            notes.append("Use Start Camera for waving, clapping and walking, which are tracked across live frames.")

        attach_references(detections)
        return result_payload("gesture", detections, notes, started, image, {
            "count": len(detections),
            "hands": analysis["hand_count"],
            "temporal_tracking": bool(session_id),
        })


gesture_scanner = GestureScanner()
