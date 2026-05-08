"""From-scratch training loops for classification and regression cells.

Adam (`betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`), CE loss for
classification, summed MSE for regression, 70/30 random split with
`random_state=42`. Saves the best-by-metric state dict alongside its
`test_metrics`, `epoch`, and run hyperparameters.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import LabelEncoder

from .data import (
    ClassificationArrayDataset,
    RegressionArrayDataset,
    collate_with_meta,
    random_split,
    stratified_split,
)
from .loaders import materialize_lazy_layers
from .metrics import compute_classification_metrics, compute_regression_metrics


def set_random_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs (CPU + CUDA).

    With ``deterministic=True`` (default) cuDNN is pinned to deterministic
    algorithms and its kernel auto-tuner is disabled, so retraining the same
    cell twice on the same hardware lands at the same trajectory. Costs a
    small amount of throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


@dataclass
class TrainResult:
    best_epoch:          int
    best_metric:         float
    test_metrics:        dict[str, float]
    normalization_mean:  float
    normalization_std:   float
    train_indices:       list[int]
    eval_indices:        list[int]


def train_classification_experiment(
    *,
    query:         Mapping[str, list],
    model_factory: Callable[[], nn.Module],
    y_key:         str,
    num_classes:   int,
    num_epochs:    int,
    batch_size:    int,
    lr:            float,
    log_transform: bool   = True,
    normalize:     bool   = True,
    test_size:     float  = 0.3,
    random_state:  int    = 42,
    device:        str | torch.device = "cuda",
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    """Train a classifier and save the best-Top1 epoch as a checkpoint."""
    set_random_seed(random_state)

    encoder   = LabelEncoder().fit(query[y_key])
    y_idx     = encoder.transform(query[y_key]).astype(np.int64)
    split     = stratified_split(query[y_key], test_size=test_size, random_state=random_state)
    full_ds   = ClassificationArrayDataset(
        query["spad_histograms"], y_idx,
        meta={"sizes": query["sizes"]},
        log_transform=log_transform, normalize=normalize,
    )
    train_loader = _make_loader(full_ds, split.train_indices, batch_size, shuffle=True)
    test_loader  = _make_loader(full_ds, split.eval_indices,  batch_size, shuffle=False)

    model = model_factory().to(device)
    materialize_lazy_layers(model, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    crit  = nn.CrossEntropyLoss()

    # Snapshot the epoch with the highest test Top-1; at low lr it tracks
    # more stably epoch-to-epoch than F1.
    best_top1, best_state, best_epoch, best_metrics = -1.0, None, 0, {}
    for epoch in range(1, num_epochs + 1):
        _train_one_epoch(model, train_loader, optim, crit, device)
        metrics = _evaluate_classification(model, test_loader, device,
                                            num_classes=num_classes,
                                            top_k=min(5, num_classes))
        if metrics["top1_acc"] > best_top1:
            best_top1    = metrics["top1_acc"]
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch
            best_metrics = metrics
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path,
                                 model_state_dict=best_state,
                                 epoch=best_epoch,
                                 test_metrics=best_metrics,
                                 classes=list(encoder.classes_),
                                 config=dict(lr=lr, num_epochs=num_epochs,
                                             batch_size=batch_size, random_state=random_state,
                                             log_transform=log_transform, normalize=normalize))

    return TrainResult(
        best_epoch          = best_epoch,
        best_metric         = best_top1,
        test_metrics        = best_metrics,
        normalization_mean  = full_ds.normalization_stats.mean,
        normalization_std   = full_ds.normalization_stats.std,
        train_indices       = split.train_indices,
        eval_indices        = split.eval_indices,
    )


def train_regression_experiment(
    *,
    query:         Mapping[str, list],
    model_factory: Callable[[], nn.Module],
    num_epochs:    int,
    batch_size:    int,
    lr:            float,
    log_transform: bool   = True,
    normalize:     bool   = True,
    test_size:     float  = 0.3,
    random_state:  int    = 42,
    device:        str | torch.device = "cuda",
    checkpoint_path: str | Path | None = None,
) -> TrainResult:
    """Train a regressor and save the best-MAE epoch as a checkpoint."""
    set_random_seed(random_state)

    split   = random_split(len(query["spad_histograms"]),
                            test_size=test_size, random_state=random_state)
    full_ds = RegressionArrayDataset(
        query["spad_histograms"], query["locations"],
        meta={"sizes": query["sizes"]},
        log_transform=log_transform, normalize=normalize,
    )
    train_loader = _make_loader(full_ds, split.train_indices, batch_size, shuffle=True)
    test_loader  = _make_loader(full_ds, split.eval_indices,  batch_size, shuffle=False)

    model = model_factory().to(device)
    materialize_lazy_layers(model, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    crit  = nn.MSELoss(reduction="sum")

    best_mae, best_state, best_epoch, best_metrics = float("inf"), None, 0, {}
    for epoch in range(1, num_epochs + 1):
        _train_one_epoch(model, train_loader, optim, crit, device)
        metrics = _evaluate_regression(model, test_loader, device)
        if metrics["mae"] < best_mae:
            best_mae     = metrics["mae"]
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch
            best_metrics = metrics
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path,
                                 model_state_dict=best_state,
                                 epoch=best_epoch,
                                 test_metrics=best_metrics,
                                 config=dict(lr=lr, num_epochs=num_epochs,
                                             batch_size=batch_size, random_state=random_state,
                                             log_transform=log_transform, normalize=normalize))

    return TrainResult(
        best_epoch          = best_epoch,
        best_metric         = best_mae,
        test_metrics        = best_metrics,
        normalization_mean  = full_ds.normalization_stats.mean,
        normalization_std   = full_ds.normalization_stats.std,
        train_indices       = split.train_indices,
        eval_indices        = split.eval_indices,
    )


def _make_loader(dataset, indices, batch_size, *, shuffle: bool):
    return DataLoader(
        Subset(dataset, indices),
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_with_meta,
    )


def _train_one_epoch(model, loader, optim, crit, device) -> None:
    model.train()
    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        optim.step()


def _evaluate_classification(model, loader, device, *, num_classes: int, top_k: int):
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for x, y, _meta in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    return compute_classification_metrics(
        torch.cat(preds), torch.cat(targets),
        num_classes=num_classes, top_k=top_k,
    )


def _evaluate_regression(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for x, y, _meta in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    return compute_regression_metrics(torch.cat(preds), torch.cat(targets))


def _save_checkpoint(path: str | Path, **payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


__all__ = [
    "TrainResult",
    "set_random_seed",
    "train_classification_experiment",
    "train_regression_experiment",
]
