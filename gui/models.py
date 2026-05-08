"""1D-CNN architectures for the GUI inference heads.

A classifier (object / size) and a regressor (planar location), both
sharing a three-block 1D-CNN backbone over the flattened spatial
channels of the 3x3 SPAD histogram.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


__all__ = [
    "Conv1DClassifier",
    "Conv1DRegressor",
]


class _Conv1DBackbone(nn.Module):
    """Three Conv1D blocks (Conv-BN-GELU) over the flattened 9 spatial channels."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.LazyConv1d(hidden,         kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv1d   (hidden, hidden,    kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d   (hidden, hidden//2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden // 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b, h, w, t = x.shape
        x = x.reshape(b, h * w, t)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        return x


class Conv1DClassifier(_Conv1DBackbone):
    """1D-CNN classifier with an MLP head."""

    def __init__(self, num_classes: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__(hidden=hidden)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class Conv1DRegressor(_Conv1DBackbone):
    """1D-CNN regressor for planar (x, y) location."""

    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__(hidden=hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_linear = nn.Linear(hidden // 4, 2)
        # Bundled state dicts include `out_tanh.*` keys; kept for `strict=True` load.
        self.out_tanh = nn.Linear(hidden // 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).squeeze(-1)
        return self.out_linear(self.mlp(x))
