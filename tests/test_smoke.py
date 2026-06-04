import json
import numpy as np
import pytest
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from cvchess.data.build_dataset import build_manifest
from cvchess.data.dataset import CellDataset
from cvchess.models.cnn import ChessCNN
from cvchess.eval.compare import evaluate_cnn, measure_latency_ms, compute_profile


def _make_boards(d, n=6):
    d.mkdir(parents=True, exist_ok=True)
    fens = ["8-8-8-8-8-8-8-4K3", "4k3-8-8-8-8-8-8-4K3",
            "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"]
    for i in range(n):
        fen = fens[i % len(fens)]
        img = Image.new("RGB", (400, 400), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2:
                    draw.rectangle([c*50, r*50, c*50+50, r*50+50], fill=(90, 90, 90))
        img.save(d / f"{fen}.png")


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    train_dir, test_dir = tmp_path / "train", tmp_path / "test"
    _make_boards(train_dir, 6)
    _make_boards(test_dir, 3)
    train_m = tmp_path / "train.json"
    test_m = tmp_path / "test.json"
    build_manifest(train_dir, train_m)
    build_manifest(test_dir, test_m)

    ds = CellDataset(train_m, max_empty_ratio=1.0)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = ChessCNN(num_classes=13)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for x, y in loader:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    test_ds = CellDataset(test_m, max_empty_ratio=1.0)
    test_loader = DataLoader(test_ds, batch_size=16)
    acc = evaluate_cnn(model, test_loader, "cpu")
    assert 0.0 <= acc <= 1.0
    assert measure_latency_ms(model, device="cpu", iters=3, warmup=1) > 0
    assert compute_profile(model)["params"] > 0
