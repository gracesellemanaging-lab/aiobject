"""Scanner implementations exposed to the Flask API."""
from __future__ import annotations

from .fruit_scanner import fruit_scanner
from .gesture_scanner import gesture_scanner
from .object_scanner import object_scanner
from .person_scanner import person_scanner
from .smart_scanner import smart_scanner

SCANNERS = {
    "smart": smart_scanner,
    "person": person_scanner,
    "object": object_scanner,
    "fruit": fruit_scanner,
    "gesture": gesture_scanner,
}

SCAN_MODES = tuple(SCANNERS)

__all__ = ["SCANNERS", "SCAN_MODES"]
