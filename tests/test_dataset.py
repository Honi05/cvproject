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


def test_dataset_uses_crop_cache(tmp_path):
    import numpy as np
    from PIL import Image
    import json
    from cvchess.data.build_dataset import build_crop_cache, board_record
    fen = "8-8-8-8-8-8-8-4K3"
    arr = np.full((400, 400, 3), 50, dtype=np.uint8)
    # mark e1 cell (row7,col4) so we can tell crops are real
    arr[7*50:8*50, 4*50:5*50] = 200
    p = tmp_path / f"{fen}.png"
    Image.fromarray(arr).save(p)
    rec = board_record(p)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps([rec]))
    cells_p = tmp_path / "cells.npy"
    labels_p = tmp_path / "labels.npy"
    info = build_crop_cache(m, cells_p, labels_p)
    assert info["n"] == 1
    ds = CellDataset(m, cells_path=cells_p, labels_path=labels_p)
    # find the wK (label 6) item; its crop should be the bright one
    items = [ds[i] for i in range(len(ds)) if int(ds[i][1]) == 6]
    assert len(items) == 1
    img, label = items[0]
    assert img.shape == (3, 50, 50)
    assert label == 6
    assert img.max().item() > 0.7  # bright cell preserved through cache
