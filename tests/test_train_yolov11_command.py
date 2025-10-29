import os
from pathlib import Path

import train_yolov11


def test_train_command_invokes_yolo_with_given_weights_and_hyperparams(monkeypatch, tmp_path):
    """Ensure running train_yolov11 with specific CLI args instantiates YOLO with
    the provided weights and calls train() with the expected hyperparameters.
    This mirrors the command:
      python .\train_yolov11.py --data data.yaml --weights yolov8n.pt --epochs 3 --imgsz 416 --batch 8
    """

    instantiate_calls = []
    train_calls = []


    class DummyYOLO:
        def __init__(self, weights):
            instantiate_calls.append(str(weights))

        def train(self, **kwargs):
            train_calls.append(kwargs)

    monkeypatch.setattr(train_yolov11, "YOLO", DummyYOLO)

    # Run in a temp directory to avoid creating runs/ in the repo
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        argv = [
            "--data",
            "data.yaml",
            "--weights",
            "yolov8n.pt",
            "--epochs",
            "3",
            "--imgsz",
            "416",
            "--batch",
            "8",
        ]
        train_yolov11.main(argv)
    finally:
        os.chdir(cwd)

    # Verify a YOLO model was instantiated with the provided weights
    assert len(instantiate_calls) >= 1
    assert instantiate_calls[0].endswith("yolov8n.pt")

    # Verify train was called once with the expected hyperparameters
    assert len(train_calls) == 1
    tc = train_calls[0]
    assert tc.get("epochs") == 3
    assert tc.get("imgsz") == 416
    assert tc.get("batch") == 8
