"""Cell-aware wrapper around `training.train_*_experiment`.

`train_cell` is the public entry point: pass a :class:`Cell`, get back the
path of a freshly written checkpoint that subsequently loads cleanly with
`load_cell`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .cells import Cell
from .training import train_classification_experiment, train_regression_experiment


__all__ = [
    "train_cell",
]


def train_cell(
    cell:      Cell,
    query:     Mapping[str, list],
    *,
    device:    str | torch.device | None = None,
    overwrite: bool = False,
) -> Path:
    """Train ``cell`` from scratch with its registered config and write `cell.ckpt`.

    Refuses to overwrite an existing checkpoint unless ``overwrite=True``.
    """
    if cell.ckpt.exists() and not overwrite:
        raise FileExistsError(
            f"{cell.ckpt} already exists. Pass overwrite=True (or remove the file) to retrain."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = cell.config
    cell.ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] {cell.short()}  device={device}  -> {cell.ckpt}")

    common = dict(
        query=query,
        model_factory=cell.arch_factory,
        num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        log_transform=cfg.log_transform, normalize=cfg.normalize,
        test_size=cfg.test_size, random_state=cfg.random_state,
        device=device, checkpoint_path=cell.ckpt,
    )
    if cell.task == "regression":
        train_regression_experiment(**common)
    else:
        train_classification_experiment(
            **common, y_key=cell.y_key, num_classes=cell.n_outputs,
        )

    print(f"[train] {cell.name} done  -> {cell.ckpt}")
    return cell.ckpt
