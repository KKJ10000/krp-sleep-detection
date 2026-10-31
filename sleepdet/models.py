"""Downloads the two small OpenCV Zoo models on first run (YuNet face detector, SFace face embedder)."""
import os
import urllib.request

MODELS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

MODEL_DIR = os.environ.get("SLEEPDET_MODEL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"[sleepdet] downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths
