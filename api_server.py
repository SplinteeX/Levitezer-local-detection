"""Simple FastAPI server to receive snapshot images and run YOLO detection.

Endpoints:
  POST /detect  - multipart/form-data with file field `image`; returns JSON with detections and path to annotated image

Behavior:
  - Loads model once on startup (uses env var YOLO_WEIGHTS or default 'yolov11.pt')
  - Reads `data.yaml` for class names when available
  - Saves annotated images to `outputs/api_annotated/`

Run with:
  uvicorn api_server:app --host 0.0.0.0 --port 8000

Note: requires packages in requirements.txt: fastapi, uvicorn, python-multipart
"""
from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import yaml
import socket


def load_data_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf8") as f:
        return yaml.safe_load(f)


APP_BASE = Path(__file__).resolve().parent
OUT_DIR = APP_BASE / "outputs" / "api_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# configuration
# Prefer an explicit env var, otherwise prefer a found local weight like yolo11m.pt, then yolov11.pt
env_weights = os.environ.get("YOLO_WEIGHTS")
default_yolo11m = APP_BASE / "yolo11m.pt"
default_yolov11 = APP_BASE / "yolov11.pt"
if env_weights:
    WEIGHTS = env_weights
elif default_yolo11.exists():
    WEIGHTS = str(default_yolo11)
elif default_yolov11.exists():
    WEIGHTS = str(default_yolov11)
else:
    # fallback to the default path (may not exist) so the loader can try hub names
    WEIGHTS = str(default_yolov11)

IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
CONF = float(os.environ.get("YOLO_CONF", "0.25"))
DEVICE = os.environ.get("YOLO_DEVICE", None)

# load class names from data.yaml when available
DATA_YAML = APP_BASE / "data.yaml"
DATA = load_data_yaml(DATA_YAML)
NAMES = DATA.get("names") if DATA else None


class Detection(BaseModel):
    cls: int
    name: str | None
    conf: float
    xyxy: List[float]


class DetectResponse(BaseModel):
    detections: List[Detection]
    annotated_path: str | None


app = FastAPI(title="YOLO Snapshot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def load_model_on_startup():
    global MODEL, NAMES
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package is required. Install with: pip install ultralytics") from e

    print(f"Loading YOLO model from: {WEIGHTS}")
    MODEL = YOLO(WEIGHTS)
    # refresh names if the model has .model.names or data.yaml provided
    if not NAMES:
        try:
            # ultralytics models may store names in MODEL.model.names
            mdl = getattr(MODEL, "model", None)
            if mdl is not None and hasattr(mdl, "names"):
                NAMES = list(getattr(mdl, "names"))
        except Exception:
            NAMES = None
    # Print a helpful LAN-accessible URL so you can test from other machines on the same network
    def _get_local_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't have to be reachable; used to determine the outbound interface
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    local_ip = _get_local_ip()
    print(f"API server startup: model loaded. If your firewall allows inbound port 8000, you can reach the API at http://{local_ip}:8000/detect")


def results_to_detections(res) -> List[Dict[str, Any]]:
    boxes = getattr(res, "boxes", None)
    detections: List[Dict[str, Any]] = []
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else [1.0] * len(xyxy)
    cls_inds = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else [0] * len(xyxy)

    for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, cls_inds):
        name = NAMES[cls] if NAMES and cls < len(NAMES) else None
        detections.append({
            "cls": int(cls),
            "name": name,
            "conf": float(c),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })

    return detections


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...), save_annotated: bool = True, return_image: bool = False):
    """Accept an uploaded image (multipart/form-data) and run detection.

    Returns JSON with detections and annotated image path.
    If return_image=True, returns base64 string in annotated_path (not a file path).
    """
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    data = await image.read()
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        raise HTTPException(status_code=500, detail="Pillow and numpy are required on the server")

    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    img_np = np.array(img)  # RGB

    # run inference
    try:
        try:
            results = MODEL(img_np, imgsz=IMGSZ, conf=CONF, device=DEVICE)
        except TypeError:
            results = MODEL(img_np, size=IMGSZ, conf=CONF)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    if not results:
        return {"detections": [], "annotated_path": None}

    res = results[0]
    detections = results_to_detections(res)

    annotated_path = None
    if save_annotated:
        try:
            annotated = res.plot()
            import cv2
            ts = int(time.time() * 1000)
            filename = f"api_{ts}.jpg"
            out_file = OUT_DIR / filename
            cv2.imwrite(str(out_file), annotated[:, :, ::-1])
            annotated_path = str(out_file.relative_to(APP_BASE))
            if return_image:
                # return base64-encoded image instead of path
                import base64

                with open(out_file, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                annotated_path = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            # fail gracefully but include detections
            annotated_path = None

    return {"detections": detections, "annotated_path": annotated_path}
