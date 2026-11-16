"""Central configuration and filesystem paths for AI Object Visualizer."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
DB_PATH = DATA_DIR / "history.db"
STATIC_DIR = BASE_DIR / "static"
REFERENCE_DIR = STATIC_DIR / "refs"
SIMILAR_DIR = STATIC_DIR / "similar"

for _directory in (MODELS_DIR, DATA_DIR, SNAPSHOT_DIR, REFERENCE_DIR, SIMILAR_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
FACE_DETECTOR_PROTO = MODELS_DIR / "opencv_face_detector.pbtxt"
FACE_DETECTOR_MODEL = MODELS_DIR / "opencv_face_detector_uint8.pb"
FAIRFACE_MODEL = MODELS_DIR / "fairface.onnx"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"

MODEL_URLS = {
    YOLO_MODEL.name: "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    FACE_DETECTOR_PROTO.name: "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    FACE_DETECTOR_MODEL.name: "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    FAIRFACE_MODEL.name: "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx",
    FACE_LANDMARKER_MODEL.name: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

DEFAULT_CONFIDENCE = 0.35
LIVE_IMAGE_SIZE = 512
STILL_IMAGE_SIZE = 640
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
REFERENCE_TIMEOUT = 12

os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
