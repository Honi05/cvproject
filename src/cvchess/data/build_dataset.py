from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cvchess.config import CELL, GRID, CNN_TO_YOLO
from cvchess.data.fen import (
    filename_to_fen, fen_to_label_grid, label_grid_to_cnn_indices,
)
from cvchess.logging_utils import get_logger

log = get_logger("build_dataset")


def slice_cells(img: np.ndarray) -> np.ndarray:
    """Slice a 400x400x3 board into (64, 50, 50, 3) in row-major order."""
    cells = np.empty((GRID * GRID, CELL, CELL, img.shape[2]), dtype=img.dtype)
    k = 0
    for r in range(GRID):
        for c in range(GRID):
            cells[k] = img[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
            k += 1
    return cells


def derive_yolo_boxes(idx: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """For each occupied cell, a (yolo_cls, x1, y1, x2, y2) cell-box."""
    boxes = []
    for r in range(GRID):
        for c in range(GRID):
            cnn = int(idx[r, c])
            if cnn == 0:
                continue
            boxes.append((CNN_TO_YOLO[cnn],
                          c*CELL, r*CELL, c*CELL+CELL, r*CELL+CELL))
    return boxes


def board_record(path: Path) -> dict:
    """Build a manifest record for one board image (no pixel copies)."""
    fen = filename_to_fen(Path(path).name)
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)
    flat = idx.reshape(-1).tolist()
    boxes = derive_yolo_boxes(idx)
    return {
        "path": str(path),
        "fen": fen,
        "cell_labels": flat,
        "boxes": boxes,
        "n_pieces": int((idx != 0).sum()),
    }


def build_manifest(split_dir: Path, out_json: Path, limit: int | None = None) -> dict:
    """Scan a split dir of *.png, write a manifest JSON, return summary."""
    split_dir = Path(split_dir)
    exts = ("*.png", "*.jpeg", "*.jpg")
    imgs = sorted(p for ext in exts for p in split_dir.glob(ext))
    if limit:
        imgs = imgs[:limit]
    records = [board_record(p) for p in imgs]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records))
    log.info("Wrote %d records to %s", len(records), out_json)
    return {"count": len(records), "path": str(out_json)}


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def build_crop_cache(manifest_path, cells_path, labels_path) -> dict:
    """Precompute a uint8 memmap of all 64 cells per board + an int64 label
    array, so training reads 50x50 crops without re-decoding images.
    cells shape: (n_boards, 64, 50, 50, 3); labels shape: (n_boards, 64).
    Board order matches the manifest record order exactly."""
    from pathlib import Path as _Path
    records = json.loads(_Path(manifest_path).read_text())
    n = len(records)
    cells_path, labels_path = _Path(cells_path), _Path(labels_path)
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    cells = np.lib.format.open_memmap(
        cells_path, mode="w+", dtype=np.uint8,
        shape=(n, GRID * GRID, CELL, CELL, 3))
    labels = np.zeros((n, GRID * GRID), dtype=np.int64)
    for i, rec in enumerate(records):
        img = load_image(rec["path"])
        cells[i] = slice_cells(img)
        labels[i] = np.asarray(rec["cell_labels"], dtype=np.int64)
        if (i + 1) % 2000 == 0:
            log.info("crop cache: %d/%d boards", i + 1, n)
    cells.flush()
    del cells
    np.save(labels_path, labels)
    log.info("Wrote crop cache: %s (%d boards) + %s", cells_path, n, labels_path)
    return {"cells": str(cells_path), "labels": str(labels_path), "n": n}
