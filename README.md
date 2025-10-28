# Levitezer - Local Drone Detection (YOLO)

This repository contains a small local object-detection toolkit adapted for drone detection using a YOLO model and the Ultralytics API.

It provides:
- `api_server.py` — a FastAPI server that accepts snapshot image uploads (POST /detect) and returns detections + annotated images.
- `requirements.txt` — Python dependencies to install.
- A `.gitignore` that excludes model weights, virtualenvs and outputs.

This README explains how to set up a development environment on Windows (PowerShell), add model weights, run the API, and test it from another machine on the same LAN.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r .\\requirements.txt
```

3. Provide model weights (one of these):

- Copy your local weights file into the repository root and name it `yolo11m.pt` (preferred):

- You can find model versions here: https://docs.ultralytics.com/models/yolo11/#usage-examples

```powershell
Copy-Item 'C:\\path\\to\\yolo11m.pt' -Destination .\\yolo11m.pt
```

- Or set the `YOLO_WEIGHTS` environment variable to the absolute path of your weights (session-only):

```powershell
$env:YOLO_WEIGHTS = 'C:\\full\\path\\to\\yolo11m.pt'
```

If you don't have a local weight file, you can point `YOLO_WEIGHTS` to a hub model (for testing), e.g. `yolov8n.pt` and Ultralytics will download it.

4. Start the API server (binds to all interfaces so other LAN machines can connect):

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

On startup the server will print the LAN-accessible URL (e.g., http://192.168.1.42:8000/detect).

## Test the API from another machine on the same LAN

1. Make sure the host machine firewall allows inbound connections to port 8000 (run as Administrator on the host):

```powershell
New-NetFirewallRule -DisplayName "YOLO API (8000)" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

2. On a client machine (same network), POST an image using curl or Postman:

```bash
curl -F "image=@/path/to/snapshot.jpg" http://<HOST_LAN_IP>:8000/detect
```

The response will be JSON with detections and an `annotated_path` (relative path under the project `outputs/` folder) or a data URL if `return_image=true` was requested.

## CLI: local detection and camera mode

Run `run_yolov11.py` for local testing or live camera input.

Examples:

Process a folder of images and save annotated images and YOLO-format txts:

```powershell
python .\run_yolov11.py --save-annotated --save-txt
```

Run webcam live detection (camera index 0):

```powershell
python .\run_yolov11.py --camera 0 --save-annotated --display
```

Watch a snapshot directory (process new files as they appear):

```powershell
python .\run_yolov11.py --snapshot-dir .\snapshots --save-annotated
```

## Environment variables

- `YOLO_WEIGHTS` — path or hub name for model weights. If not set, the server prefers `yolo11m.pt` then `yolov11.pt` in the repo root.
- `YOLO_IMGSZ` — inference image size (default 640)
- `YOLO_CONF` — confidence threshold (default 0.25)
- `YOLO_DEVICE` — device for inference (e.g., `cpu` or `0` for GPU). If omitted the Ultralytics library will try to auto-detect.

## Notes and best practices

- Do NOT commit model weights to your Git repository. `.gitignore` already excludes `*.pt` and `outputs/`.
- Running multiple Uvicorn workers will load the model once per worker. For large models prefer a single worker or a dedicated model server.
- For production or internet exposure, add authentication and use HTTPS / a reverse proxy.

## Troubleshooting

- If the server fails at startup with FileNotFoundError referencing a `.pt` file, confirm the path exists or set `YOLO_WEIGHTS` correctly.
- If inference is slow on CPU, try a smaller model (e.g., `yolov8n.pt`) for testing or run on a machine with GPU and proper CUDA drivers.

## Next steps / optional improvements

- Add an API key header to `api_server.py` to avoid open access on LAN.
- Add a small Windows PowerShell script to start the server and open the firewall automatically (requires admin).
- Add a Dockerfile for containerized deployment.

If you want any of those, tell me which and I'll add them.
