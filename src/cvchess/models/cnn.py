from __future__ import annotations
import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    """Configurable from-scratch CNN for 50x50 RGB piece cells."""

    def __init__(self, num_classes: int = 13,
                 channels: tuple[int, ...] = (32, 64, 128),
                 fc_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_c = 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_c, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_from_trial(trial, num_classes: int = 13) -> "ChessCNN":
    """Construct a ChessCNN from an Optuna trial's suggested hyperparams."""
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    base = trial.suggest_categorical("base_channels", [16, 32, 64])
    channels = tuple(base * (2 ** i) for i in range(n_blocks))
    fc_dim = trial.suggest_categorical("fc_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return ChessCNN(num_classes, channels, fc_dim, dropout)
