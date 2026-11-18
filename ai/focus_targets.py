"""Priority objects that must be detected accurately: mouse, remote, keyboard, ballpen, adapter, connector.

Pure data (no model imports) so it can be shared by the registry, the scanners and the tests.

Tuning guide
------------
* ``min_conf``  - minimum score for a single-model detection. YOLO-World scores run lower than
                  COCO scores, so thin / small things (pen, connector) use a lower value.
* ``min_area`` / ``max_area`` - box size as a fraction of the frame. Rejects specks and boxes
                  that swallow the whole frame (typical false positives).
* Both models agreeing on the same box lowers the bar and raises the score (see focus_fusion).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusTarget:
    key: str
    label: str
    category: str
    description: str
    prompts: tuple[str, ...]      # YOLO-World text prompts (several wordings = better recall)
    coco: str | None              # matching YOLO11 COCO class, if any
    min_conf: float
    min_area: float
    max_area: float


FOCUS_TARGETS: dict[str, FocusTarget] = {
    t.key: t
    for t in (
        FocusTarget(
            "mouse", "Computer Mouse", "Electronics",
            "A hand-held pointing device used with a computer.",
            ("computer mouse", "wireless mouse", "usb optical mouse"),
            coco="mouse", min_conf=0.25, min_area=0.0008, max_area=0.30,
        ),
        FocusTarget(
            "remote", "Remote Control", "Electronics",
            "A wireless hand-held device for controlling a TV, aircon or other appliance.",
            ("tv remote control", "remote controller", "air conditioner remote"),
            coco="remote", min_conf=0.25, min_area=0.0008, max_area=0.35,
        ),
        FocusTarget(
            "keyboard", "Keyboard", "Electronics",
            "A panel of keys used to input text and commands into a computer.",
            ("computer keyboard", "wireless keyboard"),
            coco="keyboard", min_conf=0.28, min_area=0.004, max_area=0.80,
        ),
        FocusTarget(
            "ballpen", "Ballpen", "Stationery",
            "A ballpoint pen: a slim writing instrument with a retractable or capped ink tip.",
            ("ballpoint pen", "ballpen", "pen"),
            coco=None, min_conf=0.10, min_area=0.0003, max_area=0.12,
        ),
        FocusTarget(
            "adapter", "Adapter Charger", "Electronics",
            "A power adapter / charger block that plugs into a wall outlet to charge a device.",
            ("phone charger adapter", "usb wall charger", "power adapter", "laptop charger"),
            coco=None, min_conf=0.12, min_area=0.001, max_area=0.30,
        ),
        FocusTarget(
            "connector", "Charging Connector", "Electronics",
            "The plug end of a charging or data cable (USB, Type-C, Lightning, barrel plug).",
            ("usb cable connector", "usb type-c plug", "charging cable plug"),
            coco=None, min_conf=0.10, min_area=0.0002, max_area=0.08,
        ),
    )
}

# Look-alikes. YOLO-World also looks for these; if one of them scores higher on the same box
# than the target it points at, the target detection is dropped (e.g. a phone read as a remote).
NEGATIVE_PROMPTS: dict[str, str] = {
    "smartphone": "remote",
    "game controller": "remote",
}

PROMPT_TO_KEY: dict[str, str] = {
    prompt: target.key for target in FOCUS_TARGETS.values() for prompt in target.prompts
}
COCO_TO_KEY: dict[str, str] = {t.coco: t.key for t in FOCUS_TARGETS.values() if t.coco}
FOCUS_PROMPTS: list[str] = list(PROMPT_TO_KEY) + list(NEGATIVE_PROMPTS)
