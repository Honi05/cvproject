import torch
from cvchess.data.augment import build_augmentation


def test_augmentation_preserves_shape_and_range():
    aug = build_augmentation(strength=0.5)
    x = torch.rand(4, 3, 50, 50)
    y = aug(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 - 1e-4
    assert y.max() <= 1.0 + 1e-4


def test_augmentation_is_differentiable():
    aug = build_augmentation(strength=0.5)
    x = torch.rand(2, 3, 50, 50, requires_grad=True)
    y = aug(x).sum()
    y.backward()
    assert x.grad is not None


def test_strength_zero_is_near_identity():
    aug = build_augmentation(strength=0.0)
    x = torch.rand(2, 3, 50, 50)
    y = aug(x)
    assert torch.allclose(x, y, atol=1e-3)
