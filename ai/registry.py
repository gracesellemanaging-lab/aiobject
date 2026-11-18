"""Thread-safe, lazily loaded registry for every AI model used by the app."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    FAIRFACE_MODEL,
    FACE_DETECTOR_MODEL,
    FACE_DETECTOR_PROTO,
    FACE_LANDMARKER_MODEL,
    HAND_LANDMARKER_MODEL,
    POSE_LANDMARKER_MODEL,
    YOLO_MODEL,
    WORLD_MODEL,
)
from .downloads import ensure_models

logger = logging.getLogger(__name__)

_MODEL_KEYS = ("yolo", "faces", "fairface", "mediapipe", "classifier")
_MODEL_LABELS = {
    "yolo": "YOLOv8n object detector",
    "faces": "OpenCV face detector",
    "fairface": "FairFace age estimator",
    "mediapipe": "MediaPipe pose / hands / face",
    "classifier": "MobileNetV3 food classifier",
}


class ModelRegistry:
    def __init__(self) -> None:
        self._init_lock = threading.RLock()
        self.yolo_lock = threading.RLock()
        self.face_lock = threading.RLock()
        self.fairface_lock = threading.RLock()
        self.mediapipe_lock = threading.RLock()
        self.classifier_lock = threading.RLock()

        self._yolo: Any = None
        self._face_net: Any = None
        self._fairface: Any = None
        self._pose: Any = None
        self._hands: Any = None
        self._face_landmarker: Any = None
        self._classifier: Any = None
        self._classifier_weights: Any = None
        self._classifier_transform: Any = None
        self._world: Any = None
        self.world_lock = threading.RLock()
        self.world_state = "idle"
        self.world_error: str | None = None

        self._state = {key: "idle" for key in _MODEL_KEYS}
        self._errors: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._started_at = time.time()

    # ------------------------------------------------------------------ boot
    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._load_all, name="model-loader", daemon=True)
        self._thread.start()

    def _load_all(self) -> None:
        ensure_models()
        for key, loader in (
            ("faces", self.faces),
            ("yolo", self.yolo),
            ("fairface", self.fairface),
            ("mediapipe", self.mediapipe),
            ("classifier", self.classifier),
        ):
            try:
                loader()
            except Exception as exc:  # noqa: BLE001 - reported through health
                logger.exception("Failed to initialize %s", key)
                self._state[key] = "error"
                self._errors[key] = str(exc)
        self.world()  # optional extra-object detector; never blocks the core models

    def _ready(self, key: str) -> None:
        self._state[key] = "ready"

    def _fail(self, key: str, exc: Exception) -> None:
        self._state[key] = "error"
        self._errors[key] = str(exc)
        logger.exception("Model %s failed", key)

    # ----------------------------------------------------------------- yolo
    def yolo(self):
        if self._yolo is not None:
            return self._yolo
        with self._init_lock:
            if self._yolo is None:
                from ultralytics import YOLO

                if self._state["yolo"] != "ready":
                    self._state["yolo"] = "loading"
                model = YOLO(str(YOLO_MODEL))
                warm = np.zeros((320, 320, 3), dtype=np.uint8)
                try:
                    model.predict(warm, imgsz=320, conf=0.9, verbose=False, device="cpu")
                except Exception:  # warmup is best-effort
                    logger.debug("YOLO warmup failed", exc_info=True)
                self._yolo = model
                self._ready("yolo")
        return self._yolo

    @property
    def yolo_names(self) -> dict[int, str]:
        model = self.yolo()
        return dict(model.names)

    # ---------------------------------------------------------------- faces
    def faces(self):
        if self._face_net is not None:
            return self._face_net
        with self._init_lock:
            if self._face_net is None:
                if self._state["faces"] != "ready":
                    self._state["faces"] = "loading"
                net = cv2.dnn.readNetFromTensorflow(str(FACE_DETECTOR_MODEL), str(FACE_DETECTOR_PROTO))
                self._face_net = net
                self._ready("faces")
        return self._face_net

    # ------------------------------------------------------------- fairface
    def fairface(self):
        if self._fairface is not None:
            return self._fairface
        with self._init_lock:
            if self._fairface is None:
                if self._state["fairface"] != "ready":
                    self._state["fairface"] = "loading"
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = max(2, min(4, os.cpu_count() or 2))
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._fairface = ort.InferenceSession(
                    str(FAIRFACE_MODEL),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._ready("fairface")
        return self._fairface

    # ------------------------------------------------------------- mediapipe
    def mediapipe(self):
        if self._pose is not None and self._hands is not None and self._face_landmarker is not None:
            return self._pose, self._hands, self._face_landmarker
        with self._init_lock:
            if self._pose is None:
                import mediapipe as mp  # noqa: F401 - registers the tasks runtime
                from mediapipe.tasks.python import BaseOptions, vision

                logging.getLogger("absl").setLevel(logging.ERROR)
                self._state["mediapipe"] = "loading"
                self._pose = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_buffer=POSE_LANDMARKER_MODEL.read_bytes()),
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.35,
                        min_pose_presence_confidence=0.35,
                    )
                )
                self._hands = vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_buffer=HAND_LANDMARKER_MODEL.read_bytes()),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.35,
                        min_hand_presence_confidence=0.35,
                    )
                )
            if self._face_landmarker is None:
                try:
                    import mediapipe as mp  # noqa: F401
                    from mediapipe.tasks.python import BaseOptions, vision

                    self._face_landmarker = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=BaseOptions(model_asset_buffer=FACE_LANDMARKER_MODEL.read_bytes()),
                            running_mode=vision.RunningMode.IMAGE,
                            num_faces=4,
                            min_face_detection_confidence=0.25,
                            min_face_presence_confidence=0.25,
                            min_tracking_confidence=0.25,
                            output_face_blendshapes=True,
                            output_facial_transformation_matrixes=False,
                        )
                    )
                except Exception as exc:  # expressions unavailable, gestures can still run
                    logger.warning("Face landmarker unavailable: %s", exc)
                    self._face_landmarker = None
            self._ready("mediapipe")
        return self._pose, self._hands, self._face_landmarker

    # ------------------------------------------------------------ classifier
    def classifier(self):
        if self._classifier is not None:
            return self._classifier, self._classifier_weights, self._classifier_transform
        with self._init_lock:
            if self._classifier is None:
                if self._state["classifier"] != "ready":
                    self._state["classifier"] = "loading"
                import torch
                from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

                torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
                weights = MobileNet_V3_Small_Weights.DEFAULT
                ckpt_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
                # A Git-LFS pointer (a few bytes) instead of the real .pth makes loading fail: remove it
                # so torchvision downloads the genuine weights.
                for bad in ckpt_dir.glob("*.pth"):
                    if bad.stat().st_size < 1_000_000:
                        bad.unlink(missing_ok=True)
                try:
                    model = mobilenet_v3_small(weights=weights).eval()
                except Exception:  # corrupt download: delete and retry once
                    for bad in ckpt_dir.glob("*.pth"):
                        bad.unlink(missing_ok=True)
                    model = mobilenet_v3_small(weights=weights).eval()
                self._classifier = model
                self._classifier_weights = weights
                self._classifier_transform = weights.transforms()
                self._ready("classifier")
        return self._classifier, self._classifier_weights, self._classifier_transform

    # ------------------------------------------------------- open vocabulary
    def world(self):
        """YOLO-World for objects outside COCO. Optional: returns None if unavailable."""
        if self._world is not None:
            return self._world
        if self.world_state == "error":
            return None
        with self._init_lock:
            if self._world is None and self.world_state != "error":
                try:
                    from ultralytics import YOLOWorld

                    from .extra_objects import PROMPTS

                    self.world_state = "loading"
                    model = YOLOWorld(str(WORLD_MODEL))
                    model.set_classes(PROMPTS)
                    self._world = model
                    self.world_state = "ready"
                except Exception as exc:  # noqa: BLE001 - extras are optional
                    self.world_state = "error"
                    self.world_error = str(exc)
                    logger.exception("YOLO-World unavailable")
        return self._world

    # ----------------------------------------------------------------- health
    def status(self) -> dict[str, Any]:
        models = []
        for key in _MODEL_KEYS:
            state = self._state[key]
            models.append({
                "key": key,
                "name": _MODEL_LABELS[key],
                "state": state,
                "error": self._errors.get(key),
            })
        ready = all(model["state"] == "ready" for model in models)
        return {
            "ready": ready,
            "models": models,
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "device": "CPU",
            "extra_objects": {"state": self.world_state, "error": self.world_error},
        }

    def close(self) -> None:
        for handle in (self._pose, self._hands, self._face_landmarker):
            try:
                if handle is not None:
                    handle.close()
            except Exception:  # noqa: BLE001 - shutdown best effort
                pass


registry = ModelRegistry()
