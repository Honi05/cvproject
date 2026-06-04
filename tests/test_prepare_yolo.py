import json
from pathlib import Path
import numpy as np
from PIL import Image
from cvchess.yolo.prepare_yolo import to_yolo_label_lines, write_yolo_split


def test_to_yolo_label_lines_normalized():
    boxes = [(5, 200, 350, 250, 400)]
    lines = to_yolo_label_lines(boxes, img_size=400)
    parts = lines[0].split()
    assert parts[0] == "5"
    xc, yc, w, h = map(float, parts[1:])
    assert abs(xc - 225/400) < 1e-6
    assert abs(w - 50/400) < 1e-6
    assert 0 <= xc <= 1 and 0 <= yc <= 1


def test_write_yolo_split_creates_files(tmp_path):
    fen = "8-8-8-8-8-8-8-4K3"
    img = Image.fromarray(np.full((400, 400, 3), 127, dtype=np.uint8))
    p = tmp_path / f"{fen}.png"
    img.save(p)
    from cvchess.data.build_dataset import board_record
    rec = board_record(p)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps([rec]))
    out = tmp_path / "yolo"
    write_yolo_split(m, out, "train")
    assert (out / "images" / "train" / f"{fen}.png").exists()
    assert (out / "labels" / "train" / f"{fen}.txt").exists()
