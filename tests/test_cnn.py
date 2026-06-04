import torch
from cvchess.models.cnn import ChessCNN, count_params


def test_forward_shape():
    model = ChessCNN(num_classes=13)
    x = torch.rand(8, 3, 50, 50)
    out = model(x)
    assert out.shape == (8, 13)


def test_param_count_positive():
    model = ChessCNN(num_classes=13)
    assert count_params(model) > 0


def test_configurable_depth_width():
    small = ChessCNN(num_classes=13, channels=(16, 32), fc_dim=64)
    big = ChessCNN(num_classes=13, channels=(32, 64, 128), fc_dim=256)
    assert count_params(big) > count_params(small)


def test_dropout_in_eval_is_deterministic():
    model = ChessCNN(num_classes=13, dropout=0.5).eval()
    x = torch.rand(2, 3, 50, 50)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.allclose(a, b)
