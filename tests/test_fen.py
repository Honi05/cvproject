import numpy as np
from cvchess.data.fen import (
    filename_to_fen, fen_to_label_grid, label_grid_to_cnn_indices,
)


def test_filename_to_fen_strips_extension_and_keeps_dashes():
    assert filename_to_fen("1B1b3R-2q5-8-8-8-8-8-8.png") == "1B1b3R-2q5-8-8-8-8-8-8"


def test_fen_to_label_grid_starting_position():
    fen = "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    grid = fen_to_label_grid(fen)
    assert grid[0][0] == "r"
    assert grid[0][4] == "k"
    assert grid[7][4] == "K"
    assert grid[3][3] == ""
    assert grid[1][0] == "p"


def test_label_grid_to_cnn_indices():
    fen = "8-8-8-8-8-8-8-4K3"
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)
    assert idx.shape == (8, 8)
    assert idx[7][4] == 6
    assert idx[0][0] == 0
    assert (idx == 0).sum() == 63


def test_each_rank_sums_to_eight():
    fen = "1B1b3R-2q5-8-8-8-8-8-8"
    grid = fen_to_label_grid(fen)
    assert all(len(row) == 8 for row in grid)
