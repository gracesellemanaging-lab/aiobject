"""Plain-language scene descriptions generated from real model outputs.

The descriptions are deterministic summaries of the detections that the models
produced for a specific image.  Nothing here is a canned or hard-coded result:
every number, label and expression is read from the live scan payload.
"""
from __future__ import annotations

from typing import Any

CATEGORY_ORDER = (
    "People",
    "Expressions",
    "Gestures",
    "Pose",
    "Fruits",
    "Vegetables",
    "Food",
    "Food & Drink",
    "Animals",
    "Vehicles",
    "Electronics",
    "Furniture",
    "Appliances",
    "Accessories",
    "Clothing",
    "Home",
    "Stationery",
    "Sports",
    "Urban",
    "Toys",
    "Personal Care",
    "Object",
)


def _join(items: list[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _plural(label: str, count: int) -> str:
    if count == 1:
        return label
    lowered = label.lower()
    if lowered.endswith(("s", "sh", "ch", "x", "z")):
        return f"{label}es"
    if lowered.endswith("y") and len(label) > 1 and label[-2].lower() not in "aeiou":
        return f"{label[:-1]}ies"
    return f"{label}s"


def _unique_groups(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for detection in detections:
        label = str(detection.get("label") or "Object")
        category = str(detection.get("category") or "Object")
        key = (label.lower(), category)
        group = groups.get(key)
        confidence = float(detection.get("confidence") or 0.0)
        if group is None:
            groups[key] = {
                "label": label,
                "category": category,
                "count": 1,
                "confidence": confidence,
            }
        else:
            group["count"] += 1
            group["confidence"] = max(group["confidence"], confidence)
    return sorted(
        groups.values(),
        key=lambda group: (CATEGORY_ORDER.index(group["category"]) if group["category"] in CATEGORY_ORDER else 99, -group["confidence"]),
    )


def _position_phrase(bbox: Any) -> str:
    try:
        x, y, w, h = [float(value) for value in list(bbox)[:4]]
    except (TypeError, ValueError):
        return ""
    horizontal = "left" if x + w / 2 < 0.38 else ("right" if x + w / 2 > 0.62 else "center")
    vertical = "top" if y + h / 2 < 0.36 else ("bottom" if y + h / 2 > 0.64 else "middle")
    if horizontal == "center" and vertical == "middle":
        return "in the centre"
    if vertical == "middle":
        return f"on the {horizontal}"
    if horizontal == "center":
        return f"in the {vertical}"
    return f"in the {vertical}-{horizontal}"


def scene_description(detections: list[dict[str, Any]], image_size: dict[str, int] | None = None) -> str:
    """Build a short, factual caption from the detections of one scan."""
    if not detections:
        return (
            "The AI models did not find any supported person, object or fruit in this image. "
            "The frame may be too dark, too small, or contain items outside the model vocabulary."
        )

    groups = _unique_groups(detections)
    people = [group for group in groups if group["category"] == "People"]
    expressions = [group for group in groups if group["category"] == "Expressions"]
    fruits = [group for group in groups if group["category"] in {"Fruits", "Vegetables"}]
    others = [
        group for group in groups
        if group["category"] not in {"People", "Expressions", "Fruits", "Vegetables"}
    ]

    sentences: list[str] = []
    total_people = sum(group["count"] for group in people)
    total_objects = sum(group["count"] for group in others)
    total_fruits = sum(group["count"] for group in fruits)

    if total_people:
        gendered = [d for d in detections if d.get("category") == "People" and d.get("label") in {"Man", "Woman"}]
        if total_people == 1 and gendered:
            subject = f"a {gendered[0]['label'].lower()}"
        else:
            men = sum(1 for d in gendered if d["label"] == "Man")
            women = sum(1 for d in gendered if d["label"] == "Woman")
            subject = f"{total_people} {'person' if total_people == 1 else 'people'}"
            if men or women:
                subject += f" ({men} man, {women} woman)" if total_people == 1 else f" ({men} men, {women} women)"
        sentences.append(
            f"The image shows {subject}"
            + (" with a clearly visible face." if any(
                (detection.get("attributes") or {}).get("face_visible") for detection in detections
                if detection.get("category") == "People"
            ) else ".")
        )

    if expressions:
        parts = [
            f"{group['label'].replace(' Detected', '').replace(' Expression', '').lower()} "
            f"({round(group['confidence'] * 100)}% confidence)"
            for group in expressions
        ]
        sentences.append(f"Visible facial expressions include {_join(parts)}.")

    if fruits:
        parts = [
            f"{group['count']} × {group['label']}" if group["count"] > 1 else group["label"]
            for group in fruits
        ]
        sentences.append(
            f"Fruit and vegetable analysis identified {_join(parts).lower() if len(parts) == 1 else _join(parts)}."
        )

    if others:
        parts = []
        for group in others[:6]:
            parts.append(
                f"{group['count']} × {group['label']}" if group["count"] > 1 else group["label"]
            )
        sentences.append(
            f"Detected objects include {_join(parts)}."
            + (f" {max(0, len(others) - 6)} more object type(s) were also found." if len(others) > 6 else "")
        )

    if total_people == 0 and total_objects == 0 and total_fruits == 0:
        sentences.append("No supported category was detected in this frame.")

    strongest = max(detections, key=lambda item: float(item.get("confidence") or 0.0))
    strongest_position = _position_phrase(strongest.get("bbox"))
    best_text = (
        f"Strongest detection: {strongest.get('label')} at "
        f"{round(float(strongest.get('confidence') or 0.0) * 100, 1)}% confidence"
    )
    if strongest_position:
        best_text += f" {strongest_position}"
    sentences.append(best_text + ".")

    if image_size:
        width = image_size.get("width")
        height = image_size.get("height")
        if width and height:
            orientation = "landscape" if width > height else ("portrait" if height > width else "square")
            sentences.append(f"The source image is {width} × {height} pixels ({orientation}).")

    return " ".join(sentences)


def search_keywords(detections: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Derive Wikimedia search phrases from the strongest detections."""
    from .knowledge import reference_info

    ordered = sorted(
        detections,
        key=lambda item: (
            0 if item.get("category") == "Expressions" else
            1 if item.get("category") in {"Gestures", "Pose"} else
            2 if item.get("category") == "People" else
            3 if item.get("category") in {"Fruits", "Vegetables"} else 4,
            -float(item.get("confidence") or 0.0),
        ),
    )
    keywords: list[str] = []
    seen: set[str] = set()
    for detection in ordered:
        key = str(detection.get("reference_key") or detection.get("label") or "").strip()
        label = str(detection.get("label") or "").strip()
        query = str(detection.get("reference_query") or "").strip()
        if not query and key:
            try:
                query = reference_info(key, label).get("query", "")
            except Exception:  # noqa: BLE001 - keyword generation is best effort
                query = ""
        query = query or label
        if not query:
            continue
        lowered = query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(query)
        if len(keywords) >= limit:
            break
    return keywords
