"""Cell-aware checkpoint and dataset loading.

`load_cell` instantiates a cell's architecture, materializes its `LazyConv1d`
/ `LazyLinear` parameters with a dummy forward pass, then loads the bundled
`.pth` with `strict=True`. `build_query` resolves the saved 3x3 dataset and
the AprilTag-derived gantry positions, and returns the same query result the
training scripts use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .cells import Cell, PACKAGE_DIR
from .data import (
    BASE_OBJECTS,
    DataSet,
    load_capture_number_to_location_mapping,
)


__all__ = [
    "DEFAULT_TAG_POSITIONS",
    "DEFAULT_SAVED_DATASET",
    "SAVED_DATASET_ENV",
    "DUMMY_INPUT_SHAPE",
    "load_cell",
    "build_query",
    "materialize_lazy_layers",
]


# `tag1_positions.json` lives in the shared repo-level `assets/` folder;
# `saved_dataset/` sits one level up at the benchmark/ root and can also be
# located via env var or kwarg.
DEFAULT_TAG_POSITIONS = (PACKAGE_DIR / ".." / ".." / "assets" / "tag1_positions.json").resolve()
DEFAULT_SAVED_DATASET = (PACKAGE_DIR / ".." / "saved_dataset").resolve()
SAVED_DATASET_ENV     = "DENALI_SAVED_DATASET"

# Lazy modules (`LazyLinear`, `LazyConv1d`) materialize their parameters on
# the first forward pass; this happens before `load_state_dict(strict=True)`
# can accept the saved keys. Shape is `(B, n, n, T)` with n=3 and T=128 bins.
DUMMY_INPUT_SHAPE = (1, 3, 3, 128)


def load_cell(cell: Cell, *, device: str | torch.device = "cpu") -> nn.Module:
    """Build ``cell``'s architecture, load its checkpoint, return ``model.eval()``.

    Raises ``FileNotFoundError`` if the checkpoint is missing (with a hint to
    rerun the training script) and ``RuntimeError`` from
    ``load_state_dict(strict=True)`` on any key mismatch.
    """
    if not cell.ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint missing for entry '{cell.name}': {cell.ckpt}\n"
            f"  Regenerate it with:\n"
            f"      python -m main_table.scripts.train_cell --cell {cell.name}"
        )
    model = cell.arch_factory()
    materialize_lazy_layers(model, device="cpu")
    payload = torch.load(cell.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(_extract_state_dict(payload), strict=True)
    model.to(device).eval()
    return model


def materialize_lazy_layers(model: nn.Module, *, device: str | torch.device) -> None:
    """Run a dummy forward pass so `LazyLinear` / `LazyConv1d` materialize."""
    sample = torch.zeros(DUMMY_INPUT_SHAPE, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample)
    if was_training:
        model.train()


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def build_query(
    *,
    saved_dataset_dir: Path | None = None,
    tag_positions:     Path | None = None,
) -> dict[str, list]:
    """Load the saved 3x3 dataset and produce the canonical query result.

    Returns the 3x3 captures (default-excluded objects removed,
    `average_spad_histograms=False`) used by every cell's training and
    evaluation, so the `random_state=42` 70/30 split is reproducible.

    Resolution order for `saved_dataset_dir`: kwarg > `$DENALI_SAVED_DATASET`
    env var > `main_table/../saved_dataset/`.
    """
    saved_dataset_dir = (
        saved_dataset_dir
        or _path_from_env(SAVED_DATASET_ENV)
        or DEFAULT_SAVED_DATASET
    )
    tag_positions = tag_positions or DEFAULT_TAG_POSITIONS

    if not saved_dataset_dir.exists():
        raise FileNotFoundError(
            f"Saved dataset directory not found: {saved_dataset_dir}\n"
            f"  Set ${SAVED_DATASET_ENV} or pass `saved_dataset_dir=...` to point at "
            f"the joblib + .npy artifacts produced by `DataSet.save_to_directory()`."
        )
    if not tag_positions.exists():
        raise FileNotFoundError(f"Tag positions JSON not found: {tag_positions}")

    dataset          = DataSet.load_from_directory(saved_dataset_dir)
    location_mapping = load_capture_number_to_location_mapping(tag_positions)
    return dataset.query_data(
        capture_resolutions=["3x3"],
        average_spad_histograms=False,
        location_mapping=location_mapping,
        included_objects=BASE_OBJECTS,
    )


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else None
