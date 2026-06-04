from __future__ import annotations
import torch
import torch.nn as nn
import kornia.augmentation as K


class _Identity(nn.Module):
    """True identity module. Used for strength==0.0 because some kornia
    versions still alter the tensor (e.g. RandomGaussianBlur / RandomAffine)
    even when p=0.0, which breaks the 'near identity' contract."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ClampedAug(nn.Module):
    """Wraps a kornia AugmentationSequential and clamps the result to [0,1].
    Gaussian noise/jitter can push values slightly out of range; clamp keeps
    the pipeline output valid while remaining differentiable."""

    def __init__(self, aug: nn.Module):
        super().__init__()
        self.aug = aug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x).clamp(0.0, 1.0)


def build_augmentation(strength: float = 0.5) -> nn.Module:
    """Differentiable GPU augmentation pipeline for 50x50 piece cells.
    strength in [0,1] scales magnitude; 0 => near identity (true identity)."""
    s = max(0.0, min(1.0, strength))
    if s == 0.0:
        # Return a real identity so strength=0 is exactly the input. Relying on
        # p=0.0 is not enough across kornia versions.
        return _Identity()
    p = 0.5
    aug = K.AugmentationSequential(
        K.RandomAffine(degrees=8.0 * s, translate=(0.05 * s, 0.05 * s),
                       scale=(1.0 - 0.1 * s, 1.0 + 0.1 * s), p=p),
        K.ColorJitter(brightness=0.2 * s, contrast=0.2 * s,
                      saturation=0.2 * s, hue=0.05 * s, p=p),
        K.RandomGaussianNoise(mean=0.0, std=0.03 * s, p=p),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1 + 1.0 * s), p=p),
        same_on_batch=False,
    )
    return _ClampedAug(aug)


class GpuAugmenter(nn.Module):
    """Wraps the pipeline; moves to device and clamps output to [0,1]."""
    def __init__(self, strength: float = 0.5, device: str = "cuda"):
        super().__init__()
        self.aug = build_augmentation(strength).to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x.to(self.device)).clamp_(0.0, 1.0)
