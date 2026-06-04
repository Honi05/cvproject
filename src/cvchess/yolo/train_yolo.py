from __future__ import annotations
from pathlib import Path

from cvchess.config import CACHE_DIR, OUTPUTS_DIR
from cvchess.yolo.prepare_yolo import write_yolo_split, write_data_yaml
from cvchess.logging_utils import get_logger

log = get_logger("train_yolo")


def prepare(out_dir: Path | None = None,
            train_manifest: Path | None = None,
            test_manifest: Path | None = None) -> Path:
    out_dir = out_dir or (OUTPUTS_DIR / "yolo_dataset")
    train_manifest = train_manifest or (CACHE_DIR / "train_manifest.json")
    test_manifest = test_manifest or (CACHE_DIR / "test_manifest.json")
    write_yolo_split(train_manifest, out_dir, "train")
    write_yolo_split(test_manifest, out_dir, "test")
    return write_data_yaml(out_dir)


def train(data_yaml: Path, model_name: str = "yolov8n.pt",
          epochs: int = 30, imgsz: int = 400, project: str = "chess-cnn-vs-yolo"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                          project=str(OUTPUTS_DIR / "yolo_runs"), name="baseline",
                          exist_ok=True)
    log.info("YOLO training complete: %s", results.save_dir)
    return model, results
