"""MobileNetV3 ImageNet classifier used to refine / extend fruit recognition."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .knowledge import IMAGENET_FOOD_INFO
from .registry import registry
from .utils import crop_with_padding

CLASSIFIER_SOURCE = "MobileNetV3 · ImageNet"


def _top_predictions(image: np.ndarray, top_k: int = 8) -> list[dict]:
    import torch

    model, weights, transform = registry.classifier()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0)
    with registry.classifier_lock:
        with torch.inference_mode():
            logits = model(tensor)[0]
            probabilities = torch.softmax(logits, dim=0)
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    categories = weights.meta["categories"]
    return [
        {
            "raw": categories[int(index)].lower(),
            "name": categories[int(index)],
            "probability": round(float(value), 4),
        }
        for value, index in zip(values.tolist(), indices.tolist())
    ]


def analyze_food_box(image: np.ndarray, box: list[float] | None) -> list[dict]:
    """Classify one localized crop and return mapped food candidates."""
    crop = crop_with_padding(image, box, padding=0.04) if box else image
    if crop.size == 0:
        return []
    predictions = _top_predictions(crop)
    mapped = []
    for prediction in predictions:
        info = IMAGENET_FOOD_INFO.get(prediction["raw"])
        if info:
            mapped.append({**prediction, **info})
    return mapped


def analyze_food_image(image: np.ndarray) -> list[dict]:
    """Classify a whole frame when no object was localized."""
    predictions = _top_predictions(image)
    mapped = []
    for prediction in predictions:
        info = IMAGENET_FOOD_INFO.get(prediction["raw"])
        if info:
            mapped.append({**prediction, **info})
    return mapped


def best_candidate(candidates: list[dict], minimum: float = 0.16) -> dict | None:
    for candidate in candidates:
        if candidate["probability"] >= minimum:
            return candidate
    return None
