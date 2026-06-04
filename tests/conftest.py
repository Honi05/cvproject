import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def synthetic_board(tmp_path):
    """Create a 400x400 board image whose filename encodes a known FEN.
    Each occupied cell gets a distinct solid color so slicing is testable.
    Returns (path, fen_filename, expected_label_grid)."""
    fen = "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    img = Image.new("RGB", (400, 400), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    # checkerboard background
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                draw.rectangle([c*50, r*50, c*50+50, r*50+50], fill=(120, 120, 120))
    path = tmp_path / f"{fen}.png"
    img.save(path)
    return path, f"{fen}.png", fen
