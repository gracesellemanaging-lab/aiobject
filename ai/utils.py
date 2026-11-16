"""Shared image, geometry and response helpers."""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import SNAPSHOT_DIR

CATEGORY_COLORS = {
    "People": "#3ee9ff",
    "Electronics": "#4d9bff",
    "Furniture": "#a78bfa",
    "Vehicles": "#f7b955",
    "Animals": "#5be7a9",
    "Fruits": "#ffb347",
    "Vegetables": "#8bd450",
    "Food": "#ff9366",
    "Food & Drink": "#ff9366",
    "Gestures": "#ffd166",
    "Pose": "#b287ff",
    "Expressions": "#ff7ad9",
    "Sports": "#59d2fe",
    "Accessories": "#f78fb3",
    "Clothing": "#f78fb3",
    "Home": "#74d3c6",
    "Appliances": "#64b5f6",
    "Stationery": "#c6a7ff",
    "Urban": "#9db2cc",
    "Toys": "#f9a8d4",
    "Personal Care": "#7dd3fc",
    "Object": "#879bb8",
    "Unknown": "#9db2cc",
}


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, "#3ee9ff")


def decode_image(data: bytes, max_side: int = 1600) -> np.ndarray:
    """Decode an uploaded/captured image into a BGR array, honoring EXIF rotation."""
    if not data:
        raise ValueError("Empty image data")
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("The image could not be decoded") from exc

    if max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def normalized_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[float]:
    x = max(0.0, min(1.0, x1 / width))
    y = max(0.0, min(1.0, y1 / height))
    w = max(0.0, min(1.0 - x, (x2 - x1) / width))
    h = max(0.0, min(1.0 - y, (y2 - y1) / height))
    return [round(x, 5), round(y, 5), round(w, 5), round(h, 5)]


def crop_with_padding(image: np.ndarray, box: Iterable[float] | None, padding: float = 0.12) -> np.ndarray:
    height, width = image.shape[:2]
    if box is None:
        return image
    x, y, w, h = list(box)[:4]
    x1 = int(x * width)
    y1 = int(y * height)
    x2 = int((x + w) * width)
    y2 = int((y + h) * height)
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1, y1, x2, y2 = clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)
    return image[y1:y2, x1:x2].copy()


def save_snapshot(image: np.ndarray, box: Iterable[float] | None, prefix: str = "scan") -> str | None:
    """Save a small JPEG crop for the history view. Returns a filename."""
    try:
        if box is None:
            crop = image.copy()
        else:
            crop = crop_with_padding(image, box, padding=0.18)
        h, w = crop.shape[:2]
        if w < 4 or h < 4:
            return None
        if max(w, h) > 420:
            ratio = 420.0 / max(w, h)
            crop = cv2.resize(crop, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)
        filename = f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
        target = SNAPSHOT_DIR / filename
        cv2.imwrite(str(target), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        return filename
    except Exception:  # noqa: BLE001 - a missing thumbnail must never break a scan
        return None


def make_detection(
    *,
    label: str,
    category: str,
    confidence: float,
    bbox: list[float] | None,
    description: str,
    reference_key: str,
    source: str,
    attributes: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    note: str | None = None,
    localized: bool = True,
) -> dict[str, Any]:
    confidence = float(max(0.0, min(1.0, confidence)))
    return {
        "id": uuid.uuid4().hex[:12],
        "label": label,
        "category": category,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "bbox": bbox,
        "localized": localized and bbox is not None,
        "description": description,
        "reference_key": reference_key,
        "color": category_color(category),
        "source": source,
        "attributes": attributes or {},
        "details": details or {},
        "note": note,
        "timestamp": time.time(),
    }


def dedupe_detections(detections: list[dict[str, Any]], min_iou: float = 0.55) -> list[dict[str, Any]]:
    """Keep the strongest detection when two boxes overlap heavily and have the same label."""
    kept: list[dict[str, Any]] = []
    for item in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        duplicate = False
        for other in kept:
            if item["label"] != other["label"]:
                continue
            if _iou(item.get("bbox"), other.get("bbox")) >= min_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def _iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_from_landmarks(points: Iterable[tuple[float, float]], padding: float = 0.03) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    return [
        round(max(0.0, x1 - w * padding), 5),
        round(max(0.0, y1 - h * padding), 5),
        round(min(1.0, w * (1 + padding * 2)), 5),
        round(min(1.0, h * (1 + padding * 2)), 5),
    ]


def image_size(image: np.ndarray) -> dict[str, int]:
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}
