from __future__ import annotations
import json
import shutil
from pathlib import Path

import yaml

from cvchess.config import IMG_SIZE, YOLO_CLASSES


def to_yolo_label_lines(boxes, img_size: int = IMG_SIZE) -> list[str]:
    """Convert [(cls,x1,y1,x2,y2)] pixel boxes to YOLO txt lines (normalized)."""
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_size
        yc = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def write_yolo_split(manifest_path, out_dir: Path, split: str) -> None:
    records = json.loads(Path(manifest_path).read_text())
    img_dir = Path(out_dir) / "images" / split
    lbl_dir = Path(out_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        src = Path(rec["path"])
        shutil.copy(src, img_dir / src.name)
        lines = to_yolo_label_lines(rec["boxes"])
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))


def write_data_yaml(out_dir: Path) -> Path:
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/test",
        "names": {i: n for i, n in enumerate(YOLO_CLASSES)},
    }
    yaml_path = Path(out_dir) / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path
