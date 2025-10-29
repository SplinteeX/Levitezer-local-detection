"""Train script for Levitezer-local-detection

Usage examples:
  # basic training (uses YOLO_WEIGHTS env or repo weights)
  python train_yolov11.py --data data.yaml --epochs 50

  # pretune stage then full train
  python train_yolov11.py --data data.yaml --pretune-epochs 5 --epochs 50

This script uses the Ultralytics YOLO API (ultralytics package). It supports a
"pretune" stage that runs a short fine-tune on the provided base weights and
then resumes full training from the pre-tuned weights.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import yaml
import time

try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics is required. Install it with: pip install ultralytics")
    raise


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"data yaml not found: {path}")
    with path.open("r", encoding="utf8") as f:
        return yaml.safe_load(f)


def find_weights(arg_weights: str | None) -> str:
    # prefer explicit arg, then YOLO_WEIGHTS env, then local yolo11m.pt, then yolov11.pt
    if arg_weights:
        return arg_weights
    env = os.environ.get("YOLO_WEIGHTS")
    if env:
        return env
    base = Path(__file__).resolve().parent
    w1 = base / "yolo11m.pt"
    w2 = base / "yolov11.pt"
    if w1.exists():
        return str(w1)
    if w2.exists():
        return str(w2)
    # fallback to a small hub model for quick tests
    return "yolov8n.pt"


def find_best_from_run(project: str, name: str):
    # Ultralytics default run path: runs/train/{name}/weights/best.pt
    base = Path("runs") / "train" / name / "weights"
    candidate = base / "best.pt"
    if candidate.exists():
        return str(candidate)
    last = base / "last.pt"
    if last.exists():
        return str(last)
    return None


def train_stage(weights: str, data: str, epochs: int, imgsz: int, batch: int, lr: float, device: str, project: str, name: str, patience: int | None = None):
    print(f"Training: weights={weights}, data={data}, epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}, device={device}")
    model = YOLO(weights)
    # build kwargs for train
    kwargs = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
    )
    if patience is not None:
        kwargs["patience"] = patience

    # run training (this will block until finished)
    model.train(**kwargs)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train YOLO on the project's dataset with optional pretune stage")
    p.add_argument("--data", default="data.yaml", help="path to data.yaml")
    p.add_argument("--weights", default=None, help="initial weights (file or hub name)")
    p.add_argument("--epochs", type=int, default=50, help="main training epochs")
    p.add_argument("--pretune-epochs", type=int, default=0, help="run a short pretune stage first (epochs)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--pretune-lr", type=float, default=0.01)
    p.add_argument("--device", default="", help="device for training, e.g. '0' or 'cpu'. Defaults to auto-detect by ultralytics")
    p.add_argument("--project", default="runs/train", help="where to save runs")
    p.add_argument("--name", default=None, help="run name (defaults to timestamp)")
    p.add_argument("--patience", type=int, default=None, help="early stopping patience (optional)")
    p.add_argument("--no-pretune-save", action="store_true", help="don't use pretune weights for the main run (if pretune used)")
    args = p.parse_args(argv)

    data_yaml = Path(args.data).resolve()
    data = load_yaml(data_yaml)
    print(f"Loaded data.yaml: {data_yaml}")

    base_name = args.name or f"levitezer_{int(time.time())}"
    project = args.project

    init_weights = find_weights(args.weights)

    # PRETUNE
    pretune_weights = None
    if args.pretune_epochs and args.pretune_epochs > 0:
        pretune_name = base_name + "_pretune"
        print(f"Starting pretune stage: epochs={args.pretune_epochs}, name={pretune_name}")
        train_stage(
            weights=init_weights,
            data=str(data_yaml),
            epochs=args.pretune_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr=args.pretune_lr,
            device=args.device,
            project=project,
            name=pretune_name,
            patience=args.patience,
        )
        found = find_best_from_run(project, pretune_name)
        if found:
            pretune_weights = found
            print(f"Pretune produced weights: {pretune_weights}")
        else:
            print("Pretune completed but no best.pt/last.pt found; will continue with initial weights")

    # MAIN TRAIN
    main_name = base_name + "_train"
    main_weights = init_weights
    if pretune_weights and not args.no_pretune_save:
        main_weights = pretune_weights

    print(f"Starting main training: epochs={args.epochs}, name={main_name}, weights={main_weights}")
    train_stage(
        weights=main_weights,
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        project=project,
        name=main_name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
