"""Architectures for the main benchmark table.

Class names are preserved so that bundled state dicts load with
`strict=True`. Each family has a paired classifier (`*Net`) and regressor
(`*Reg`) sharing the same backbone:

    SmallMLP             / SmallMLPReg              -- 4-layer MLP
    MediumConv1DNet      / MediumConv1DNetReg       -- 1D CNN
    DeepConv3DNet        / MediumConv3DOverCRegNet  -- 3-block 3D CNN
    MediumConv3DOverCNet (classification only)      -- single-block 3D CNN
    TransformerNet       / TransformerNetReg        -- cls-token Transformer

`MediumConv3DOverCNet` and `MediumConv3DOverCRegNet` are different
architectures despite the name overlap; both are kept verbatim because the
saved state-dict keys depend on the class hierarchy. Lazy modules
(`LazyLinear`, `LazyConv1d`) require a dummy forward pass before
`load_state_dict(strict=True)` — see `loaders.materialize_lazy_layers`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "SmallMLP",
    "SmallMLPReg",
    "MediumConv1DNet",
    "MediumConv1DNetReg",
    "MediumConv3DOverCNet",
    "MediumConv3DOverCRegNet",
    "DeepConv3DNet",
    "TransformerNet",
    "TransformerNetReg",
]


# Input adapters: histograms arrive as `(B, n, n, T)` (n=3 spatial grid,
# T=128 time bins). Each model needs the data in a different layout.

def _as_matrix(x: torch.Tensor) -> torch.Tensor:
    """`(B, n, n, T)` -> `(B, n*n, T)`. For 1D CNN: spatial pixels become channels."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    b, h, w, t = x.shape
    return x.reshape(b, h * w, t)


def _as_token_sequence(x: torch.Tensor) -> torch.Tensor:
    """`(B, n, n, T)` -> `(B, T, n*n)`. For Transformer: one token per time bin."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    b, h, w, t = x.shape
    return x.permute(0, 3, 1, 2).reshape(b, t, h * w)


def _as_volume(x: torch.Tensor) -> torch.Tensor:
    """`(B, n, n, T)` -> `(B, 1, n, n, T)`. For 3D CNN: single-channel volume."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.unsqueeze(1)


# ---- MLP --------------------------------------------------------------------

class _SmallMLPBase(nn.Module):
    """Four fully-connected layers with GELU + dropout."""

    def __init__(self, num_outputs: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, hidden // 4)
        self.fc4 = nn.Linear(hidden // 4, num_outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
        x = self.dropout(F.gelu(self.fc3(x)))
        return self.fc4(x)


class SmallMLP(_SmallMLPBase):
    """MLP classifier."""
    def __init__(self, num_classes: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__(num_outputs=num_classes, hidden=hidden, dropout=dropout)


class SmallMLPReg(_SmallMLPBase):
    """MLP regressor (two-output head: planar `(x, y)`)."""
    def __init__(self, hidden: int = 512, dropout: float = 0.1):
        super().__init__(num_outputs=2, hidden=hidden, dropout=dropout)


# ---- 1D CNN -----------------------------------------------------------------

class _MediumConv1DBase(nn.Module):
    """Three-block 1D CNN backbone (Conv-BN-GELU x3) over time."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.LazyConv1d(hidden,         kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv1d   (hidden, hidden,    kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d   (hidden, hidden//2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden // 2)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_matrix(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        return x


class MediumConv1DNet(_MediumConv1DBase):
    """1D CNN classifier."""

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
        return self.head(self._features(x))


class MediumConv1DNetReg(_MediumConv1DBase):
    """1D CNN regressor (two-output head)."""

    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__(hidden=hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_linear = nn.Linear(hidden // 4, 2)
        # Unused at inference; bundled state dicts include `out_tanh.*` keys
        # that `strict=True` requires.
        self.out_tanh = nn.Linear(hidden // 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self._features(x)).squeeze(-1)
        return self.out_linear(self.mlp(x))


# ---- 3D CNN -----------------------------------------------------------------

class MediumConv3DOverCNet(nn.Module):
    """Single-block 3D CNN classifier with kernel `(5, 5, 16)` and an MLP head."""

    def __init__(self, num_classes: int, hidden: int = 32, dropout: float = 0.1,
                 stride_hw: int = 2, stride_t: int = 4):
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, hidden,
            kernel_size=(5, 5, 16),
            stride=(stride_hw, stride_hw, stride_t),
            padding=(2, 2, 8),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(hidden)
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_volume(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(F.gelu(self.fc2(x)))
        x = self.dropout(F.gelu(self.fc3(x)))
        return self.fc4(x)


class _DeepConv3DBase(nn.Module):
    """Three-block 3D CNN backbone (Conv-BN-GELU x3) with kernel `(3, 3, 10)`."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(1,      hidden,    kernel_size=(3, 3, 10), padding="same", bias=False)
        self.conv2 = nn.Conv3d(hidden, hidden,    kernel_size=(3, 3, 10), padding="same", bias=False)
        self.conv3 = nn.Conv3d(hidden, hidden//2, kernel_size=(3, 3, 10), padding="same", bias=False)
        self.bn1 = nn.BatchNorm3d(hidden)
        self.bn2 = nn.BatchNorm3d(hidden)
        self.bn3 = nn.BatchNorm3d(hidden // 2)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_volume(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        return x


def _pooled_3d_head(in_features: int, num_outputs: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(in_features, in_features // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_features // 2, num_outputs),
    )


class DeepConv3DNet(_DeepConv3DBase):
    """Three-block 3D CNN classifier."""
    def __init__(self, num_outputs: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__(hidden=hidden)
        self.head = _pooled_3d_head(hidden // 2, num_outputs, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x))


class MediumConv3DOverCRegNet(_DeepConv3DBase):
    """Three-block 3D CNN regressor (two-output head; same backbone as `DeepConv3DNet`)."""
    def __init__(self, hidden: int = 128, dropout: float = 0.1):
        super().__init__(hidden=hidden)
        self.head = _pooled_3d_head(hidden // 2, 2, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x))


# ---- Transformer ------------------------------------------------------------

class _TransformerBase(nn.Module):
    """Cls-token Transformer encoder (one token per time bin, n*n features per token).

    Subclasses differ only in where `input_norm` runs relative to the CLS
    prepend; both orderings are intentional and match the bundled checkpoints.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1, seq_len: int = 128):
        super().__init__()
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.dropout_p = dropout

        self.token_embed  = nn.LazyLinear(d_model)
        self.pos_encoding = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_drop   = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_token_sequence(x)
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.size(1)}.")
        x = self.token_embed(x) * math.sqrt(self.d_model)
        return x + self.pos_encoding[:, : x.size(1)]

    def _prepend_cls(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(x.size(0), 1, -1)
        return torch.cat([cls, x], dim=1)


class TransformerNet(_TransformerBase):
    """Transformer classifier; CLS is prepended *before* `input_norm`."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        d, p = self.d_model, self.dropout_p
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(d // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepend_cls(self._embed(x))
        x = self.pos_drop(self.input_norm(x))
        x = self.transformer_encoder(x)
        return self.head(self.final_norm(x)[:, 0])


class TransformerNetReg(_TransformerBase):
    """Transformer regressor; `input_norm` runs *before* CLS is prepended (two-output head)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d, p = self.d_model, self.dropout_p
        self.head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(p),
            nn.Linear(d // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_drop(self.input_norm(self._embed(x)))
        x = self.transformer_encoder(self._prepend_cls(x))
        return self.head(self.final_norm(x)[:, 0])
