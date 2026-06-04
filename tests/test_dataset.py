import json
import numpy as np
from PIL import Image
import torch
from cvchess.data.dataset import CellDataset


def _make_manifest(tmp_path):
    fen = "8-8-8-8-8-8-8-4K3"
    img = Image.fromarray(np.full((400, 400, 3), 127, dtype=np.uint8))
    p = tmp_path / f"{fen}.png"
    img.save(p)
    from cvchess.data.fen import fen_to_label_grid, label_grid_to_cnn_indices
    idx = label_grid_to_cnn_indices(fen_to_label_grid(fen)).reshape(-1).tolist()
    rec = {"path": str(p), "fen": fen, "cell_labels": idx, "boxes": [], "n_pieces": 1}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps([rec]))
    return mpath


def test_dataset_len_is_64_per_board(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m)
    assert len(ds) == 64


def test_dataset_item_shape_and_label(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m)
    found = [ds[i] for i in range(len(ds)) if int(ds[i][1]) == 6]
    assert len(found) == 1
    img, label = found[0]
    assert img.shape == (3, 50, 50)
    assert img.dtype == torch.float32
    assert int(label) == 6


def test_balanced_indices_reduce_empty(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m, max_empty_ratio=1.0)
    labels = [int(ds[i][1]) for i in range(len(ds))]
    n_empty = sum(1 for l in labels if l == 0)
    n_piece = sum(1 for l in labels if l != 0)
    assert n_empty <= max(1, n_piece)
