import numpy as np
from PIL import Image
from cvchess.data.build_dataset import (
    slice_cells, derive_yolo_boxes, board_record,
)
from cvchess.data.fen import fen_to_label_grid, label_grid_to_cnn_indices


def test_slice_cells_shapes():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cells = slice_cells(img)
    assert cells.shape == (64, 50, 50, 3)


def test_derive_boxes_only_for_occupied():
    fen = "8-8-8-8-8-8-8-4K3"
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)
    boxes = derive_yolo_boxes(idx)
    assert len(boxes) == 1
    cls, x1, y1, x2, y2 = boxes[0]
    assert cls == 5
    assert (x1, y1, x2, y2) == (4*50, 7*50, 4*50+50, 7*50+50)


def test_board_record_roundtrip(synthetic_board):
    path, fname, fen = synthetic_board
    rec = board_record(path)
    assert rec["fen"] == fen
    assert len(rec["cell_labels"]) == 64
    assert rec["n_pieces"] == sum(1 for v in rec["cell_labels"] if v != 0)


def test_build_manifest_finds_jpeg_and_png(tmp_path):
    from PIL import Image
    import numpy as np
    import json
    from cvchess.data.build_dataset import build_manifest
    fen = "8-8-8-8-8-8-8-4K3"
    arr = np.full((400, 400, 3), 127, dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / f"{fen}.jpeg")
    out = tmp_path / "m.json"
    summary = build_manifest(tmp_path, out)
    assert summary["count"] == 1
    records = json.loads(out.read_text())
    assert records[0]["fen"] == fen
