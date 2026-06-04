from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cvchess.config import CELL, GRID
from cvchess.data.build_dataset import slice_cells, load_image


class CellDataset(Dataset):
    """Yields (cell_tensor[3,50,50] float[0,1], cnn_label) for every cell.
    Optionally caps empty cells relative to occupied for class balance."""

    def __init__(self, manifest_path, max_empty_ratio: float | None = None):
        self.records = json.loads(Path(manifest_path).read_text())
        self.index: list[tuple[int, int]] = []
        for ri, rec in enumerate(self.records):
            labels = rec["cell_labels"]
            piece_cells = [(ri, ci) for ci, l in enumerate(labels) if l != 0]
            empty_cells = [(ri, ci) for ci, l in enumerate(labels) if l == 0]
            if max_empty_ratio is not None:
                cap = max(1, int(len(piece_cells) * max_empty_ratio))
                empty_cells = empty_cells[:cap]
            self.index.extend(piece_cells + empty_cells)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ri, ci = self.index[i]
        rec = self.records[ri]
        img = load_image(rec["path"])
        cells = slice_cells(img)
        cell = cells[ci].astype(np.float32) / 255.0
        tensor = torch.from_numpy(cell).permute(2, 0, 1).contiguous()
        label = int(rec["cell_labels"][ci])
        return tensor, label
