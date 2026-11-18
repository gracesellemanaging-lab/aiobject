"""Landmark-based gesture, posture and visible-expression analysis."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from .expressions import analyze_expressions
from .knowledge import GESTURE_INFO
from .registry import registry
from .utils import bbox_from_landmarks, make_detection

# MediaPipe pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# MediaPipe hand landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


class GestureTracker:
    """Small in-memory temporal buffer per browser scanning session."""

    def __init__(self, maxlen: int = 18, ttl: float = 45.0) -> None:
        self.maxlen = maxlen
        self.ttl = ttl
        self._sessions: dict[str, deque] = {}
        self._lock = threading.Lock()

    def update(self, session_id: str, features: dict) -> list[dict]:
        now = time.time()
        with self._lock:
            stale = [key for key, frames in self._sessions.items() if not frames or now - frames[-1]["t"] > self.ttl]
            for key in stale:
                self._sessions.pop(key, None)
            history = self._sessions.setdefault(session_id, deque(maxlen=self.maxlen))
            history.append({"t": now, **features})
            frames = list(history)
            if len(self._sessions) > 40:
                oldest = sorted(self._sessions.items(), key=lambda item: item[1][-1]["t"] if item[1] else 0)
                for key, _ in oldest[:-30]:
                    self._sessions.pop(key, None)
        return _temporal_gestures(frames)


class GestureEngine:
    def __init__(self) -> None:
        self.tracker = GestureTracker()

    def analyze(self, image: np.ndarray, session_id: str | None = None, include_expressions: bool = True,
                include_note_callback=None) -> dict[str, Any]:
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose, hands, _ = registry.mediapipe()
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_landmarks = None
        hand_landmarks = []
        hand_handedness = []
        with registry.mediapipe_lock:
            try:
                pose_result = pose.detect(mp_image)
                pose_landmarks = (pose_result.pose_landmarks or [None])[0]
            except Exception:
                pose_landmarks = None
            try:
                hand_result = hands.detect(mp_image)
                hand_landmarks = list(hand_result.hand_landmarks or [])
                hand_handedness = list(hand_result.handedness or [])
            except Exception:
                hand_landmarks, hand_handedness = [], []

        detections: list[dict] = []
        notes: list[str] = []
        pose_points = _pose_points(pose_landmarks)
        pose_bbox = _pose_bbox(pose_points)
        features: dict[str, Any] = {
            "pose_visible": bool(pose_points),
            "hands": [],
            "raised_wrists": {},
            "wrist_distance": None,
            "ankle_separation": None,
            "posture": None,
        }

        # ----------------------------------------------------------- posture
        posture = _posture(pose_points)
        if posture:
            features["posture"] = posture["key"]
            info = GESTURE_INFO[posture["key"]]
            detections.append(make_detection(
                label=info["label"],
                category=info["category"],
                confidence=posture["confidence"],
                bbox=pose_bbox,
                description=info["description"],
                reference_key=posture["key"],
                source="MediaPipe Pose · landmark geometry",
                details={"heuristic": True, "knee_angles": posture.get("knee_angles")},
                note="Posture is inferred from visible 2D body landmarks.",
            ))
        elif pose_points:
            notes.append("Lower body is not clearly visible, so sitting / standing could not be classified.")

        # --------------------------------------------------------- hands
        thumb_gestures: list[dict] = []
        for index, hand in enumerate(hand_landmarks[:2]):
            handedness = "Hand"
            if index < len(hand_handedness):
                entry = hand_handedness[index]
                category = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                label = getattr(category, "category_name", None) or getattr(category, "label", None)
                if label:
                    handedness = label
            normalized = [(point.x, point.y) for point in _hand_points(hand)]
            hand_info = _hand_geometry(normalized)
            hand_info["handedness"] = handedness
            hand_info["wrist"] = normalized[WRIST]
            hand_info["bbox"] = bbox_from_landmarks(normalized, padding=0.06)
            features["hands"].append({
                "handedness": handedness,
                "wrist": normalized[WRIST],
                "gesture": hand_info["gesture"],
            })

            if hand_info["gesture"] in ("thumbs-up", "thumbs-down", "pointing"):
                thumb_gestures.append(hand_info)
                info = GESTURE_INFO[hand_info["gesture"]]
                detections.append(make_detection(
                    label=info["label"],
                    category="Gestures",
                    confidence=hand_info["confidence"],
                    bbox=hand_info["bbox"],
                    description=info["description"],
                    reference_key=hand_info["gesture"],
                    source="MediaPipe Hands · landmark geometry",
                    details={
                        "heuristic": True,
                        "handedness": handedness,
                        "extended_fingers": hand_info["extended"],
                        "thumb_direction": hand_info["thumb_direction"],
                    },
                    note="Hand shapes are estimated from 2D landmarks and can be affected by hand angle or occlusion.",
                ))

        # --------------------------------------------------- raised wrists
        raised = _raised_hands(pose_points)
        features["raised_wrists"] = raised
        if raised and len(raised) >= 1:
            both = "left" in raised and "right" in raised
            key = "raising-both-hands" if both else "raising-one-hand"
            if not thumb_gestures:
                info = GESTURE_INFO[key]
                wrist_points = [raised[side] for side in raised]
                detections.append(make_detection(
                    label=info["label"],
                    category="Gestures",
                    confidence=0.62 if not both else 0.7,
                    bbox=bbox_from_landmarks(wrist_points, padding=0.14),
                    description=info["description"],
                    reference_key=key,
                    source="MediaPipe Pose · landmark geometry",
                    details={"heuristic": True, "sides": list(raised.keys())},
                    note="A raised hand is inferred when a visible wrist is above its shoulder.",
                ))

        # ------------------------------------------------- temporal state
        temporal: list[dict] = []
        if pose_points:
            shoulder_ys = [pose_points[i][1] for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if pose_points.get(i)]
            hip_ys = [pose_points[i][1] for i in (LEFT_HIP, RIGHT_HIP) if pose_points.get(i)]
            torso = {
                "shoulder": float(np.mean(shoulder_ys)) if shoulder_ys else None,
                "hip": float(np.mean(hip_ys)) if hip_ys else None,
            }
            features["torso"] = torso
            ankles = [pose_points[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if pose_points.get(i)]
            if len(ankles) == 2:
                features["ankle_separation"] = abs(ankles[0][0] - ankles[1][0])
                features["ankle_vertical"] = abs(ankles[0][1] - ankles[1][1])
        if len(hand_landmarks) >= 2:
            wrists = [_hand_points(hand)[WRIST] for hand in hand_landmarks[:2]]
            features["wrist_distance"] = math.dist((wrists[0].x, wrists[0].y), (wrists[1].x, wrists[1].y))

        if session_id:
            temporal = self.tracker.update(session_id, features)
            for item in temporal:
                info = GESTURE_INFO[item["key"]]
                detections.append(make_detection(
                    label=info["label"],
                    category=info["category"],
                    confidence=item["confidence"],
                    bbox=item.get("bbox") or pose_bbox,
                    description=info["description"],
                    reference_key=item["key"],
                    source="Temporal landmark analysis",
                    details={"heuristic": True, "frames": item.get("frames", 0)},
                    note="Detected from movement across multiple live frames.",
                ))
        else:
            notes.append("Waving, clapping and walking need continuous live scanning.")

        # ------------------------------------------------------ expressions
        if include_expressions:
            try:
                expressions = analyze_expressions(image)
            except Exception:
                expressions = []
            for face in expressions[:3]:
                key = face["expression_key"]
                info = GESTURE_INFO[key]
                detections.append(make_detection(
                    label=info["label"],
                    category="Expressions",
                    confidence=face["confidence"],
                    bbox=face["bbox"],
                    description=info["description"],
                    reference_key=key,
                    source="MediaPipe Face Landmarker · blendshapes",
                    details={**face["scores"], "landmark_count": face["landmarks"]},
                    note="Visible-expression estimate only. It does not identify emotions, mood or personality.",
                ))

        detections = _dedupe(detections)
        if not detections:
            if pose_points or hand_landmarks:
                notes.append("A person was found, but no supported gesture, posture or expression was confidently detected.")
            else:
                notes.append("No person pose detected. Stand in frame with your upper body and hands visible.")

        return {
            "detections": detections,
            "notes": notes,
            "pose_bbox": pose_bbox,
            "hand_count": len(hand_landmarks),
            "features": features,
        }


def _hand_points(hand) -> list:
    return list(getattr(hand, "landmark", hand))


def _pose_points(pose_landmarks) -> dict[int, tuple[float, float, float]]:
    if not pose_landmarks:
        return {}
    points = {}
    for index, landmark in enumerate(getattr(pose_landmarks, "landmark", pose_landmarks)):
        points[index] = (float(landmark.x), float(landmark.y), float(getattr(landmark, "visibility", 0.0)))
    return points


def _pose_bbox(points: dict[int, tuple[float, float, float]]) -> list[float] | None:
    if not points:
        return None
    visible = [(x, y) for x, y, visibility in points.values() if visibility >= 0.3]
    if len(visible) < 5:
        return None
    return bbox_from_landmarks(visible, padding=0.035)


def _visible(points: dict[int, tuple[float, float, float]], index: int, threshold: float = 0.4):
    point = points.get(index)
    if point is None or point[2] < threshold:
        return None
    return point[:2]


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*ab) * math.hypot(*cb)
    if norm == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / norm))
    return math.degrees(math.acos(cosine))


def _posture(points: dict[int, tuple[float, float, float]]) -> dict | None:
    hips = [p for p in (_visible(points, LEFT_HIP), _visible(points, RIGHT_HIP)) if p]
    knees = [p for p in (_visible(points, LEFT_KNEE), _visible(points, RIGHT_KNEE)) if p]
    ankles = [p for p in (_visible(points, LEFT_ANKLE), _visible(points, RIGHT_ANKLE)) if p]
    if not hips or not knees or not ankles:
        return None

    angles = []
    for hip_index, knee_index, ankle_index in ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)):
        hip = _visible(points, hip_index)
        knee = _visible(points, knee_index)
        ankle = _visible(points, ankle_index)
        if hip and knee and ankle:
            angles.append(_angle(hip, knee, ankle))
    if not angles:
        return None

    average = sum(angles) / len(angles)
    hip_y = float(np.mean([p[1] for p in hips]))
    knee_y = float(np.mean([p[1] for p in knees]))
    ankle_y = float(np.mean([p[1] for p in ankles]))
    vertical_span = ankle_y - hip_y
    if average >= 150 and vertical_span > 0.22 and hip_y < knee_y < ankle_y:
        confidence = min(0.94, 0.62 + (average - 150) / 100 + min(0.12, vertical_span * 0.2))
        return {"key": "standing", "confidence": round(confidence, 3), "knee_angles": [round(a, 1) for a in angles]}
    if average <= 138 or (knee_y - hip_y < 0.035):
        confidence = min(0.9, 0.58 + (145 - average) / 120)
        return {"key": "sitting", "confidence": round(confidence, 3), "knee_angles": [round(a, 1) for a in angles]}
    return None


def _hand_geometry(points: list[tuple[float, float]]) -> dict[str, Any]:
    wrist = points[WRIST]
    palm_size = math.dist(wrist, points[MIDDLE_MCP]) or 1e-6
    finger_angles = {
        "thumb": _angle(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP]),
        "index": _angle(points[INDEX_MCP], points[INDEX_PIP], points[INDEX_TIP]),
        "middle": _angle(points[MIDDLE_MCP], points[MIDDLE_PIP], points[MIDDLE_TIP]),
        "ring": _angle(points[RING_MCP], points[RING_PIP], points[RING_TIP]),
        "pinky": _angle(points[PINKY_MCP], points[PINKY_PIP], points[PINKY_TIP]),
    }
    extended = {name: angle >= (148 if name != "thumb" else 140) for name, angle in finger_angles.items()}
    other_folded = sum(1 for name in ("index", "middle", "ring", "pinky") if not extended[name]) >= 3
    index_extended = extended["index"] and finger_angles["index"] >= 155

    gesture = None
    confidence = 0.0
    thumb_direction = None
    if extended["thumb"] and not extended["index"] and other_folded:
        thumb_tip_y = points[THUMB_TIP][1]
        wrist_y = wrist[1]
        thumb_size = math.dist(wrist, points[THUMB_MCP]) or 1e-6
        if thumb_tip_y < wrist_y - thumb_size * 0.45:
            gesture, thumb_direction = "thumbs-up", "up"
        elif thumb_tip_y > wrist_y + thumb_size * 0.45:
            gesture, thumb_direction = "thumbs-down", "down"
        if gesture:
            confidence = min(0.86, 0.58 + (finger_angles["thumb"] - 140) / 120 + min(0.1, palm_size * 0.5))
    if gesture is None and index_extended and other_folded:
        gesture = "pointing"
        confidence = min(0.86, 0.62 + (finger_angles["index"] - 155) / 150)

    return {
        "gesture": gesture,
        "confidence": round(confidence, 3),
        "extended": [name for name, value in extended.items() if value],
        "finger_angles": {name: round(angle, 1) for name, angle in finger_angles.items()},
        "thumb_direction": thumb_direction,
        "palm_size": round(palm_size, 4),
    }


def _raised_hands(points: dict[int, tuple[float, float, float]]) -> dict[str, tuple[float, float]]:
    raised: dict[str, tuple[float, float]] = {}
    for side, wrist_index, shoulder_index in (
        ("left", LEFT_WRIST, LEFT_SHOULDER),
        ("right", RIGHT_WRIST, RIGHT_SHOULDER),
    ):
        wrist = _visible(points, wrist_index, threshold=0.35)
        shoulder = _visible(points, shoulder_index, threshold=0.35)
        if wrist and shoulder and wrist[1] < shoulder[1] - 0.015:
            raised[side] = wrist
    return raised


def _temporal_gestures(frames: list[dict]) -> list[dict]:
    if len(frames) < 4:
        return []
    recent = frames[-12:]
    results: list[dict] = []

    # ------------------------------------------------------------- waving
    for side in ("left", "right"):
        xs = [frame["raised_wrists"][side][0] for frame in recent if side in frame.get("raised_wrists", {})]
        if len(xs) >= 5:
            spread = max(xs) - min(xs)
            direction_changes = 0
            previous_direction = 0
            for a, b in zip(xs, xs[1:]):
                direction = 1 if b - a > 0.004 else (-1 if b - a < -0.004 else 0)
                if direction and previous_direction and direction != previous_direction:
                    direction_changes += 1
                if direction:
                    previous_direction = direction
            if spread > 0.045 and direction_changes >= 1:
                bbox = bbox_from_landmarks([frame["raised_wrists"][side] for frame in recent if side in frame.get("raised_wrists", {})], padding=0.18)
                results.append({
                    "key": "waving",
                    "confidence": round(min(0.92, 0.63 + spread * 1.6), 3),
                    "bbox": bbox,
                    "frames": len(xs),
                })
                break

    # ----------------------------------------------------------- clapping
    distances = [frame["wrist_distance"] for frame in recent if frame.get("wrist_distance") is not None]
    if len(distances) >= 4 and min(distances) < 0.105 and max(distances) > 0.145:
        results.append({
            "key": "clapping",
            "confidence": round(min(0.9, 0.66 + (max(distances) - min(distances)) * 1.2), 3),
            "bbox": None,
            "frames": len(distances),
        })

    # ------------------------------------------------------------ walking
    separations = [frame["ankle_separation"] for frame in recent if frame.get("ankle_separation") is not None]
    vertical = [frame.get("ankle_vertical", 0.0) for frame in recent if frame.get("ankle_separation") is not None]
    standing = sum(1 for frame in recent if frame.get("posture") == "standing")
    if len(separations) >= 5 and standing >= 3:
        spread = max(separations) - min(separations)
        alternation = max(vertical) - min(vertical) if vertical else 0
        if spread > 0.035 and alternation > 0.018:
            results.append({
                "key": "walking",
                "confidence": round(min(0.86, 0.6 + spread * 2.0), 3),
                "bbox": None,
                "frames": len(separations),
            })

    return results


def _dedupe(detections: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in detections:
        key = (item["reference_key"], item["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


gesture_engine = GestureEngine()
