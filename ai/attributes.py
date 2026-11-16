"""Face localization and clearly-labeled age-range estimates."""
from __future__ import annotations

import cv2
import numpy as np

from .registry import registry
from .utils import clamp_box, normalized_box

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["man", "woman"]  # FairFace gender head order: Male, Female
GENDER_MIN_CONFIDENCE = 0.6
ATTRIBUTE_DISCLAIMER = (
    "Uncertain age-range and man/woman estimates from visible facial features. This is not "
    "identity recognition and must not be treated as an exact age or a statement about who someone is."
)


def detect_faces(image: np.ndarray, score_threshold: float = 0.45) -> list[dict]:
    """Detect faces with OpenCV's SSD face detector."""
    height, width = image.shape[:2]
    try:
        net = registry.faces()
        with registry.face_lock:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0), False, False)
            net.setInput(blob)
            raw = net.forward()
    except Exception:
        return []

    faces: list[dict] = []
    for index in range(raw.shape[2]):
        score = float(raw[0, 0, index, 2])
        if score < score_threshold:
            continue
        x1 = float(raw[0, 0, index, 3]) * width
        y1 = float(raw[0, 0, index, 4]) * height
        x2 = float(raw[0, 0, index, 5]) * width
        y2 = float(raw[0, 0, index, 6]) * height
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)
        if (x2 - x1) < 18 or (y2 - y1) < 18:
            continue
        faces.append({
            "pixels": [x1, y1, x2, y2],
            "bbox": normalized_box(x1, y1, x2, y2, width, height),
            "score": round(score, 4),
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
        })
    faces.sort(key=lambda item: item["score"], reverse=True)
    return faces[:12]


def _softmax(values: np.ndarray) -> np.ndarray:
    exp = np.exp(values - np.max(values))
    return exp / np.sum(exp)


def estimate_attributes(image: np.ndarray, face: dict) -> dict:
    """Predict an approximate age range and man/woman estimate for a detected face.

    The age and gender heads of FairFace are used. The race head is ignored and
    identity is never inferred.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = face["pixels"]
    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.25)
    x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"available": False, "reason": "Face crop was empty"}

    try:
        session = registry.fairface()
        with registry.fairface_lock:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = (rgb - mean) / std
            blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
            names = [item.name for item in session.get_outputs()]
            outputs = dict(zip(names, session.run(None, {session.get_inputs()[0].name: blob})))
    except Exception as exc:  # noqa: BLE001 - attributes are optional
        return {"available": False, "reason": f"Age model unavailable: {exc}"}

    age_logits = np.asarray(outputs["age_output"][0], dtype=np.float64)
    age_probs = _softmax(age_logits)
    age_index = int(np.argmax(age_probs))
    age_confidence = float(age_probs[age_index])

    gender = None
    gender_confidence = 0.0
    if "gender_output" in outputs:
        gender_probs = _softmax(np.asarray(outputs["gender_output"][0], dtype=np.float64))
        gender_index = int(np.argmax(gender_probs))
        gender_confidence = float(gender_probs[gender_index])
        if gender_confidence >= GENDER_MIN_CONFIDENCE:
            gender = GENDER_LABELS[gender_index]

    return {
        "available": True,
        "gender": gender,
        "gender_confidence": round(gender_confidence, 3),
        "age_range": AGE_LABELS[age_index],
        "age_confidence": round(age_confidence, 3),
        "face_confidence": face["score"],
        "face_bbox": face["bbox"],
        "age_distribution": {
            label: round(float(prob), 3)
            for label, prob in zip(AGE_LABELS, age_probs)
            if prob >= 0.08
        },
        "disclaimer": ATTRIBUTE_DISCLAIMER,
    }


def face_inside_box(face: dict, box: list[float], margin: float = 0.04) -> bool:
    """Whether a face centre lies inside a normalized person box (with margin)."""
    x, y, w, h = box
    fx, fy, fw, fh = face["bbox"]
    fcx, fcy = fx + fw / 2, fy + fh / 2
    return (x - margin) <= fcx <= (x + w + margin) and (y - margin) <= fcy <= (y + h + margin)
