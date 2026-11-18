"""Priority everyday objects with dedicated open-vocabulary prompts.

The base YOLO model only knows the 80 COCO classes. Chargers and cables are not
COCO classes, so they are detected with YOLO-World using several wordings for
each object. Mouse, remote and keyboard are in COCO *and* in the open
vocabulary set; the fusion layer combines both sources.

Tuning guide
------------
* ``min_conf`` - minimum score for a single-model detection. YOLO-World scores
  tend to run lower than COCO scores.
* ``min_area`` / ``max_area`` - box area as a fraction of the frame. Rejects
  specks and boxes that cover almost the whole frame.
* ``min_ratio`` / ``max_ratio`` - long side / short side of the box in pixels.
  Charger cables are long and thin; adapters are blockier.
* When both models agree on the same box the bar is relaxed and the score gets
  a small boost (see ``ai.focus_fusion``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusTarget:
    key: str
    label: str
    category: str
    description: str
    prompts: tuple[str, ...]
    coco: str | None
    min_conf: float
    min_area: float
    max_area: float
    min_ratio: float
    max_ratio: float


FOCUS_TARGETS: dict[str, FocusTarget] = {
    target.key: target
    for target in (
        FocusTarget(
            "mouse", "Computer Mouse", "Electronics",
            "A hand-held pointing device used with a computer.",
            ("computer mouse", "wireless mouse", "usb optical mouse", "mouse"),
            coco="mouse", min_conf=0.18, min_area=0.0008, max_area=0.30,
            min_ratio=1.0, max_ratio=3.5,
        ),
        FocusTarget(
            "remote", "Remote Control", "Electronics",
            "A wireless hand-held device for controlling a TV, air conditioner or other appliance.",
            ("tv remote control", "remote controller", "air conditioner remote", "remote control"),
            coco="remote", min_conf=0.18, min_area=0.0008, max_area=0.45,
            min_ratio=1.2, max_ratio=6.0,
        ),
        FocusTarget(
            "keyboard", "Keyboard", "Electronics",
            "A panel of keys used to input text and commands into a computer.",
            ("computer keyboard", "wireless keyboard", "laptop keyboard", "keyboard"),
            coco="keyboard", min_conf=0.20, min_area=0.004, max_area=0.80,
            min_ratio=1.2, max_ratio=8.0,
        ),
        FocusTarget(
            "charger_adapter", "Charger Adapter", "Electronics",
            "A power adapter (adaptor) charger block that plugs into a wall outlet to charge a device.",
            (
                "charger", "phone charger", "charger adapter", "charger adaptor",
                "power adapter", "wall charger", "usb charger", "laptop charger",
                "ac adapter", "charger plug", "power brick", "charging brick",
                "usb power adapter",
            ),
            coco=None, min_conf=0.16, min_area=0.0008, max_area=0.40,
            min_ratio=0.6, max_ratio=3.0,
        ),
        FocusTarget(
            "charger_cable", "Charger Cable", "Electronics",
            "A charging or data cable used to connect a device to a charger or computer.",
            (
                "charger cable", "charging cable", "usb cable", "usb-c cable",
                "type-c cable", "lightning cable", "phone cable", "data cable",
                "charging cord", "usb cord",
            ),
            coco=None, min_conf=0.14, min_area=0.0004, max_area=0.60,
            min_ratio=1.0, max_ratio=20.0,
        ),
        FocusTarget(
            "charging_connector", "Charging Connector", "Electronics",
            "The plug end of a charging cable (USB, USB-C, Lightning or barrel plug).",
            (
                "usb connector", "usb plug", "usb-c plug", "type-c connector",
                "lightning connector", "charging connector", "barrel plug",
            ),
            coco=None, min_conf=0.12, min_area=0.0003, max_area=0.25,
            min_ratio=1.0, max_ratio=6.0,
        ),
    )
}

# Look-alikes. If one of these scores higher on the same box than the target,
# the target detection is dropped (for example a phone read as a remote).
NEGATIVE_PROMPTS: dict[str, tuple[str, ...]] = {
    "smartphone": ("remote", "mouse", "charger_adapter"),
    "cell phone": ("remote", "mouse", "charger_adapter"),
    "mobile phone": ("remote", "mouse", "charger_adapter"),
    "tablet": ("remote", "mouse", "charger_adapter"),
    "game controller": ("remote",),
    "calculator": ("remote",),
    "earphones": ("charger_cable",),
}

PROMPT_TO_KEY: dict[str, str] = {
    prompt: target.key for target in FOCUS_TARGETS.values() for prompt in target.prompts
}
COCO_TO_KEY: dict[str, str] = {
    target.coco: target.key for target in FOCUS_TARGETS.values() if target.coco is not None
}
FOCUS_COCO_CLASSES = frozenset(COCO_TO_KEY)
FOCUS_PROMPTS: list[str] = list(dict.fromkeys([*PROMPT_TO_KEY, *NEGATIVE_PROMPTS]))
FOCUS_COCO_FLOOR = 0.15
