"""Evaluate a loaded cell on the canonical 70/30 test split.

Mirrors the split and preprocessing used during training: random for
regression, stratified-on-label for classification, and the cell's own
`log_transform` / `normalize` setting.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder

from .cells import Cell
from .data import (
    ClassificationArrayDataset,
    RegressionArrayDataset,
    collate_with_meta,
    random_split,
    stratified_split,
)
from .metrics import compute_classification_metrics, compute_regression_metrics


__all__ = [
    "evaluate",
    "format_metrics",
]


def evaluate(
    cell:       Cell,
    model:      torch.nn.Module,
    query:      Mapping[str, list],
    *,
    device:     str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, float]:
    """Run ``model`` over ``cell``'s canonical test split and return its metrics.

    Returns:
        Regression entries: `{mse, rmse, mae, x_mse, y_mse, x_mae, y_mae}`.
        Classification entries: `{top1_acc, top5_acc, macro_f1, macro_precision, macro_recall}`.
    """
    if cell.task == "regression":
        return _evaluate_regression(cell, model, query, device, batch_size)
    return _evaluate_classification(cell, model, query, device, batch_size)


def format_metrics(cell: Cell, metrics: Mapping[str, float]) -> str:
    """Format ``metrics`` for the cell's task as a single line."""
    m = metrics
    if cell.task == "regression":
        return f"RMSE {m['rmse']:.4f} / MAE {m['mae']:.4f}"
    if cell.task == "object_classification":
        return f"Top1 {m['top1_acc']:.4f} / Top5 {m['top5_acc']:.4f} / F1 {m['macro_f1']:.4f}"
    return f"P {m['macro_precision']:.4f} / R {m['macro_recall']:.4f} / Acc {m['top1_acc']:.4f}"


def _evaluate_regression(cell, model, query, device, batch_size) -> dict[str, float]:
    split = random_split(
        len(query["spad_histograms"]),
        test_size=cell.config.test_size,
        random_state=cell.config.random_state,
    )
    dataset = RegressionArrayDataset(
        query["spad_histograms"], query["locations"],
        meta={"sizes": query["sizes"]},
        log_transform=cell.config.log_transform,
        normalize=cell.config.normalize,
    )
    preds, targets = _run_loader(model, _make_loader(dataset, split.eval_indices, batch_size), device)
    return {k: float(v) for k, v in compute_regression_metrics(preds, targets).items()}


def _evaluate_classification(cell, model, query, device, batch_size) -> dict[str, float]:
    encoder = LabelEncoder().fit(query[cell.y_key])
    y_idx   = encoder.transform(query[cell.y_key]).astype(np.int64)
    split   = stratified_split(
        query[cell.y_key],
        test_size=cell.config.test_size,
        random_state=cell.config.random_state,
    )
    dataset = ClassificationArrayDataset(
        query["spad_histograms"], y_idx,
        meta={"sizes": query["sizes"]},
        log_transform=cell.config.log_transform,
        normalize=cell.config.normalize,
    )
    preds, targets = _run_loader(model, _make_loader(dataset, split.eval_indices, batch_size), device)
    raw = compute_classification_metrics(
        preds, targets,
        num_classes=cell.n_outputs,
        top_k=min(5, cell.n_outputs),
    )
    return {k: float(v) for k, v in raw.items()}


def _make_loader(dataset, indices, batch_size):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_meta,
    )


def _run_loader(model, loader, device):
    preds, targets = [], []
    with torch.inference_mode():
        for x, y, _meta in loader:
            preds.append(model(x.to(device)).detach().cpu())
            targets.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)
