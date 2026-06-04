import torch
from cvchess.eval.compare import (
    accuracy_from_preds, measure_latency_ms, compute_profile,
)
from cvchess.models.cnn import ChessCNN


def test_accuracy_from_preds():
    preds = torch.tensor([1, 2, 3, 0])
    labels = torch.tensor([1, 2, 9, 0])
    assert abs(accuracy_from_preds(preds, labels) - 0.75) < 1e-6


def test_measure_latency_positive():
    model = ChessCNN(num_classes=13).eval()
    ms = measure_latency_ms(model, input_shape=(1, 3, 50, 50), device="cpu", iters=3, warmup=1)
    assert ms > 0


def test_compute_profile_keys():
    model = ChessCNN(num_classes=13)
    prof = compute_profile(model, input_shape=(1, 3, 50, 50))
    assert "params" in prof and prof["params"] > 0
    assert "flops" in prof
