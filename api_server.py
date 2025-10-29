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

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Form

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
# Weight selection (no environment variables). Prefer repo root best.pt, then
# yolo11m.pt, then yolov11.pt. Fallback to yolov11.pt so Ultralytics can try hub names.
default_yolo11m = APP_BASE / "yolo11m.pt"
default_yolov11 = APP_BASE / "yolov11.pt"
root_best = APP_BASE / "best.pt"
if root_best.exists():
    WEIGHTS = str(root_best)
elif default_yolo11m.exists():
    WEIGHTS = str(default_yolo11m)
elif default_yolov11.exists():
    WEIGHTS = str(default_yolov11)
else:
    # fallback to the default path (may not exist) so the loader can try hub names
    WEIGHTS = str(default_yolov11)

IMGSZ = 640
CONF = 0.25
DEVICE = None

# load class names from data.yaml when available
DATA_YAML = APP_BASE / "data.yaml"
DATA = load_data_yaml(DATA_YAML)
NAMES = DATA.get("names") if DATA else None
MODEL_NAMES = None


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
    # capture model's internal names separately so we can map indices -> model label
    try:
        mdl = getattr(MODEL, "model", None)
        if mdl is not None and hasattr(mdl, "names"):
            global MODEL_NAMES
            MODEL_NAMES = list(getattr(mdl, "names"))
    except Exception:
        MODEL_NAMES = None
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
        model_name = None
        try:
            if MODEL_NAMES and cls < len(MODEL_NAMES):
                model_name = MODEL_NAMES[cls]
        except Exception:
            model_name = None
        detections.append({
            "cls": int(cls),
            "name": name,
            "model_name": model_name,
            "conf": float(c),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        })

    return detections


@app.post("/detect")
async def detect(request: Request):
    """Unified handler for multipart, base64, or JSON inputs (no File() param)."""
    import base64, json
    from PIL import Image
    import numpy as np
    from io import BytesIO

    # Read body once (safe now that FastAPI didn't parse it)
    content_type = (request.headers.get("content-type") or "").lower()
    raw_body = await request.body()
    data = None
    is_roboflow_style = False

    # ----- Multipart upload -----
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("image")
        if not upload:
            raise HTTPException(status_code=400, detail="Missing 'image' file in multipart form")
        data = await upload.read()

    # ----- JSON -----
    elif content_type.startswith("application/json"):
        is_roboflow_style = True
        try:
            body = json.loads(raw_body.decode("utf8", errors="ignore"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        img_b64 = body.get("image") or body.get("image_base64")
        image_url = body.get("image_url") or body.get("url")
        if img_b64:
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            data = base64.b64decode(img_b64)
        elif image_url:
            from urllib.request import urlopen
            with urlopen(image_url) as resp:
                data = resp.read()
        else:
            raise HTTPException(status_code=422, detail="JSON must include 'image' or 'image_url'")

    # ----- Base64-only body -----
    elif (
        "form-urlencoded" in content_type
        or content_type.startswith("text")
        or content_type.startswith("application/x-www-form-urlencoded")
        or content_type == ""
    ):
        is_roboflow_style = True
        text = raw_body.decode("utf8", errors="replace").strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[1]
        data = base64.b64decode(text)

    else:
        raise HTTPException(status_code=422, detail="Unsupported content-type")

    # ---------------- YOLO Inference ----------------
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    img_np = np.array(img)
    try:
        try:
            results = MODEL(img_np, imgsz=IMGSZ, conf=CONF, device=DEVICE)
        except TypeError:
            results = MODEL(img_np, size=IMGSZ, conf=CONF)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    if not results:
        if is_roboflow_style:
            return {"predictions": [], "annotated": None}
        return {"detections": [], "annotated_path": None}

    res = results[0]
    detections = results_to_detections(res)

    # Filter detections to dataset classes (use data.yaml names when available)
    allowed_names = set(DATA.get("names", [])) if DATA else set()
    if allowed_names:
        filtered_detections = [
            d for d in detections if (d.get("name") in allowed_names) or (d.get("model_name") in allowed_names)
        ]
    else:
        filtered_detections = detections

    # Annotate and save (draw only filtered detections)
    annotated_path = None
    try:
        import cv2, time

        # start from original image (RGB) and draw boxes
        annotated = np.array(img).copy()
        for d in filtered_detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d.get("xyxy", [0, 0, 0, 0])]
            label = d.get("name") or d.get("model_name") or str(d.get("cls"))
            conf = d.get("conf", 0.0)
            # draw rectangle (RGB), will convert to BGR for cv2.imwrite
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{label} {conf:.2f}"
            # put text background
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(annotated, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        ts = int(time.time() * 1000)
        filename = f"api_{ts}.jpg"
        out_file = OUT_DIR / filename
        cv2.imwrite(str(out_file), annotated[:, :, ::-1])
        annotated_path = str(out_file.relative_to(APP_BASE))
    except Exception:
        annotated_path = None

    # ----- Roboflow-style response -----
    if is_roboflow_style:
        width, height = img.size
        predictions = []
        for d in filtered_detections:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = x2 - x1, y2 - y1
            predictions.append({
                "x": cx / width, "y": cy / height,
                "width": w / width, "height": h / height,
                "confidence": d["conf"],
                "class": d.get("name") or str(d["cls"]),
                "bbox": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            })
        return {"predictions": predictions, "annotated": annotated_path}

    # ----- Default response -----
    return {"detections": filtered_detections, "annotated_path": annotated_path}