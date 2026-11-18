"""Everyday objects that are NOT in the 80 COCO classes, detected with YOLO-World (open vocabulary).

Each text prompt maps to a display label. Several prompts can share one label.
"""
from __future__ import annotations

# prompt -> (label, category, description)
EXTRA_OBJECTS: dict[str, tuple[str, str, str]] = {
    "earphones": ("Earphones", "Electronics", "Small audio devices worn in or near the ears, wired or wireless (earbuds / earpods)."),
    "earbuds": ("Earphones", "Electronics", "Small audio devices worn in or near the ears, wired or wireless (earbuds / earpods)."),
    "wireless earbuds": ("Earphones", "Electronics", "Small audio devices worn in or near the ears, wired or wireless (earbuds / earpods)."),
    "earbuds charging case": ("Earbuds Case", "Electronics", "A small case that stores and charges wireless earbuds."),
    "headphones": ("Headphones", "Electronics", "Over-ear or on-ear audio headset."),
    "wallet": ("Wallet", "Accessories", "A small flat case for carrying cash and cards."),
    "pen": ("Pen", "Stationery", "A writing instrument that uses ink."),
    "pencil": ("Pencil", "Stationery", "A writing instrument with a graphite core."),
    "key": ("Key", "Accessories", "A metal key used to open a lock."),
    "power bank": ("Power Bank", "Electronics", "A portable battery for charging devices."),
    "eyeglasses": ("Eyeglasses", "Accessories", "Glasses worn to correct vision."),
    "wrist watch": ("Wrist Watch", "Accessories", "A watch worn on the wrist."),
    "calculator": ("Calculator", "Electronics", "A handheld device for arithmetic."),
    "lighter": ("Lighter", "Objects", "A small device that makes a flame."),
    "id card": ("ID Card", "Objects", "A plastic or paper identification card."),
    "padlock": ("Padlock", "Objects", "A portable lock with a shackle."),
}

from .focus_targets import FOCUS_PROMPTS  # noqa: E402

# Chargers and cables moved to focus_targets with better prompts and shape rules.
PROMPTS = list(dict.fromkeys([*EXTRA_OBJECTS, *FOCUS_PROMPTS]))
WORLD_CONFIDENCE_CAP = 0.30
