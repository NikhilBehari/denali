"""Classification and regression metrics.

Both functions accept either `torch.Tensor` or `numpy.ndarray`. Classification
inputs are raw logits; the function takes argmax internally.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_classification_metrics(
    logits:      torch.Tensor | np.ndarray,
    targets:     torch.Tensor | np.ndarray,
    *,
    num_classes: int | None = None,
    top_k:       int        = 5,
) -> dict[str, Any]:
    """Top-1 / Top-k accuracy plus macro-averaged precision, recall, and F1."""
    logits  = _as_numpy(logits)
    targets = _as_numpy(targets)
    if num_classes is None:
        num_classes = int(logits.shape[1])

    preds = logits.argmax(axis=1)
    top1  = float((preds == targets).mean())

    if num_classes >= top_k:
        topk_preds = np.argpartition(-logits, kth=top_k - 1, axis=1)[:, :top_k]
        topk = float((topk_preds == targets[:, None]).any(axis=1).mean())
    else:
        topk = 1.0

    return {
        "top1_acc":        top1,
        "top5_acc":        topk,
        "macro_f1":        float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(targets, preds, average="macro", zero_division=0)),
    }


def compute_regression_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets:     torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Overall and per-axis MSE / MAE for 2D `(x, y)` regression."""
    predictions = _as_numpy(predictions, dtype=np.float32)
    targets     = _as_numpy(targets,     dtype=np.float32)
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}.")
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"Expected shape [N, 2]; got {predictions.shape}.")

    diff = predictions - targets
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    return {
        "mse":   mse,
        "rmse":  float(np.sqrt(mse)),
        "mae":   mae,
        "x_mse": float(np.mean(diff[:, 0] ** 2)),
        "y_mse": float(np.mean(diff[:, 1] ** 2)),
        "x_mae": float(np.mean(np.abs(diff[:, 0]))),
        "y_mae": float(np.mean(np.abs(diff[:, 1]))),
    }


def _as_numpy(t, dtype=None) -> np.ndarray:
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
]
