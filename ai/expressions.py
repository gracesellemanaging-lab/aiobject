"""Basic visible facial-expression estimation using MediaPipe blendshapes."""
from __future__ import annotations

import cv2
import numpy as np

from .registry import registry
from .utils import bbox_from_landmarks


def analyze_expressions(image: np.ndarray) -> list[dict]:
    """Return visible-expression estimates for every detected face."""
    try:
        _, _, landmarker = registry.mediapipe()
        if landmarker is None:
            return []
        import mediapipe as mp

        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with registry.mediapipe_lock:
            result = landmarker.detect(mp_image)
    except Exception:
        return []

    faces: list[dict] = []
    for index, landmarks in enumerate(result.face_landmarks or []):
        blendshapes = {}
        if index < len(result.face_blendshapes or []):
            blendshapes = {item.category_name: float(item.score) for item in result.face_blendshapes[index]}

        smile = (blendshapes.get("mouthSmileLeft", 0.0) + blendshapes.get("mouthSmileRight", 0.0)) / 2.0
        jaw_open = blendshapes.get("jawOpen", 0.0)
        brow_inner = blendshapes.get("browInnerUp", 0.0)
        brow_outer = (blendshapes.get("browOuterUpLeft", 0.0) + blendshapes.get("browOuterUpRight", 0.0)) / 2.0
        frown = (blendshapes.get("mouthFrownLeft", 0.0) + blendshapes.get("mouthFrownRight", 0.0)) / 2.0
        brow_down = (blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)) / 2.0
        surprise = min(1.0, jaw_open * 0.75 + brow_inner * 0.5 + brow_outer * 0.15)
        sadness = min(1.0, frown * 0.65 + brow_down * 0.45)

        if smile >= 0.36 and smile >= surprise:
            key, confidence = "smile", smile
        elif sadness >= 0.34 and smile < 0.3 and sadness >= surprise:
            key, confidence = "sad", sadness
        elif surprise >= 0.42 and brow_inner >= 0.18 and jaw_open >= 0.22:
            key, confidence = "surprised", surprise
        else:
            key, confidence = "neutral", max(0.35, 1.0 - max(smile, surprise, sadness))

        points = [(point.x, point.y) for point in landmarks]
        faces.append({
            "expression_key": key,
            "confidence": round(float(min(0.97, max(0.05, confidence))), 3),
            "bbox": bbox_from_landmarks(points, padding=0.04),
            "scores": {
                "smile": round(smile, 3),
                "jaw_open": round(jaw_open, 3),
                "brow_inner_up": round(brow_inner, 3),
                "frown": round(frown, 3),
                "brow_down": round(brow_down, 3),
                "surprise": round(surprise, 3),
                "sadness": round(sadness, 3),
            },
            "landmarks": len(landmarks),
        })
    return faces
