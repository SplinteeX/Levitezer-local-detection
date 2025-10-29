import os
import types
import tempfile
from pathlib import Path

import train_yolov11


def test_large_training_flow(monkeypatch, tmp_path):
    """Simulate a large pretune+main training run by mocking YOLO and run-finder.

    This test ensures train_yolov11 invokes a pretune stage (when requested)
    and then uses the pretune-produced weights for the main training run.
    It uses very small epoch counts and a Dummy YOLO so it runs quickly.
    """

    instantiate_calls = []
    train_calls = []


    class DummyYOLO:
        def __init__(self, weights):
            # record the exact weights string used to instantiate the model
            instantiate_calls.append(str(weights))

        def train(self, **kwargs):
            # record the training kwargs and return quickly
            train_calls.append(kwargs)

    # Monkeypatch the YOLO class used by the script
    monkeypatch.setattr(train_yolov11, "YOLO", DummyYOLO)

    # Pretend the pretune run created a best.pt; return a known path
    fake_best = tmp_path / "pretune_run" / "weights" / "best.pt"
    fake_best.parent.mkdir(parents=True, exist_ok=True)
    fake_best.write_text("fake")

    def fake_find_best(project, name):
        # always return our fake best path for the pretune run
        return str(fake_best)

    monkeypatch.setattr(train_yolov11, "find_best_from_run", fake_find_best)

    # Run the main entry with a short pretune and main epochs
    argv = [
        "--data",
        "data.yaml",
        "--weights",
        "yolov8n.pt",
        "--pretune-epochs",
        "2",
        "--epochs",
        "3",
        "--imgsz",
        "416",
        "--batch",
        "1",
    ]

    # Ensure we run in tmp to avoid writing to repo runs/
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        train_yolov11.main(argv)
    finally:
        os.chdir(cwd)

    # Two instantiations: pretune model (initial weights), then main model (pretune best)
    assert len(instantiate_calls) >= 2
    assert instantiate_calls[0].endswith("yolov8n.pt")
    assert instantiate_calls[1] == str(fake_best)

    # Two train calls (pretune + main)
    assert len(train_calls) == 2

    # Names used for runs should contain _pretune then _train in order
    pretune_name = train_calls[0].get("name", "")
    main_name = train_calls[1].get("name", "")
    assert "_pretune" in pretune_name
    assert "_train" in main_name
