from __future__ import annotations
import os
import numpy as np
from cvchess.config import FEN_TO_CNN


def filename_to_fen(filename: str) -> str:
    """Strip directory + extension, keep the dash-separated FEN board."""
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def fen_to_label_grid(fen: str) -> list[list[str]]:
    """Convert a dash-separated FEN board to an 8x8 grid of FEN chars
    ('' for empty). Row 0 = rank 8 (top), col 0 = file a (left)."""
    rows = fen.split("-")
    if len(rows) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(rows)}: {fen}")
    grid: list[list[str]] = []
    for r in rows:
        cells: list[str] = []
        for ch in r:
            if ch.isdigit():
                cells.extend([""] * int(ch))
            else:
                cells.append(ch)
        if len(cells) != 8:
            raise ValueError(f"Rank '{r}' expands to {len(cells)} cells, need 8")
        grid.append(cells)
    return grid


def label_grid_to_cnn_indices(grid: list[list[str]]) -> np.ndarray:
    """Map an 8x8 char grid to an 8x8 int array of CNN class indices."""
    idx = np.zeros((8, 8), dtype=np.int64)
    for r in range(8):
        for c in range(8):
            ch = grid[r][c]
            idx[r, c] = FEN_TO_CNN[ch] if ch else 0
    return idx
