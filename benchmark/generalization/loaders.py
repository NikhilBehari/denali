"""Checkpoint and held-out dataset loading for the supplement experiments.

Each generalization checkpoint carries its own training-set normalization
statistics (`train_mean`, `train_std`); evaluation must apply log1p +
z-score with those exact statistics so that the test-time inputs match the
distribution the model trained on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn

from .experiments import Experiment, PACKAGE_DIR


__all__ = [
    "DEFAULT_HELDOUT_ROOT",
    "HELDOUT_ROOT_ENV",
    "DEFAULT_TAG_POSITIONS",
    "DUMMY_INPUT_SHAPE",
    "HeldOutDataset",
    "load_held_out_dataset",
    "load_experiment_model",
]


# The held-out captures are read from the shared `saved_dataset/` built by
# `main_table.scripts.build_dataset`. To point at a different parent
# directory, pass `held_out_root=` or set ``$DENALI_HELDOUT_DATASETS``.
DEFAULT_HELDOUT_ROOT = PACKAGE_DIR.parent
HELDOUT_ROOT_ENV     = "DENALI_HELDOUT_DATASETS"

# AprilTag positions live in the shared repo-level `assets/` folder. They
# provide (x, y) per capture number for the regression task; the held-out
# joblib datasets do not store locations directly.
DEFAULT_TAG_POSITIONS = (PACKAGE_DIR / ".." / ".." / "assets" / "tag1_positions.json").resolve()

# Lazy modules need a single forward pass before `strict=True` load.
DUMMY_INPUT_SHAPE = (1, 3, 3, 128)


@dataclass(frozen=True)
class HeldOutDataset:
    """Slice of the held-out joblib dataset matching one experiment's filter."""
    spad_histograms: list[np.ndarray]
    objects:         list[str]
    sizes:           list[str]
    locations:       list[tuple[float, float]] | None  # populated when available

    def __len__(self) -> int:
        return len(self.spad_histograms)


def load_held_out_dataset(
    experiment:        Experiment,
    *,
    held_out_root:     Path | None = None,
    capture_resolution: str        = "3x3",
) -> HeldOutDataset:
    """Read the joblib dataset for ``experiment`` and return its held-out slice.

    Filter applied:
        capture_resolution == ``capture_resolution`` (default 3x3)
        object IN ``experiment.held_out_objects``
        size == ``experiment.size_filter`` if set
    """
    root = held_out_root or _resolve_root()
    folder = root / experiment.saved_dataset_subdir
    if not folder.exists():
        raise FileNotFoundError(
            f"Saved dataset not found: {folder}\n"
            f"  Build it with `python -m main_table.scripts.build_dataset "
            f"--data-dir ../denali-data/data --output-dir saved_dataset` or set "
            f"${HELDOUT_ROOT_ENV} to point at the parent directory."
        )

    objects = joblib.load(folder / "objects.joblib")
    sizes   = joblib.load(folder / "sizes.joblib")
    res     = joblib.load(folder / "capture_resolutions.joblib")
    capture_numbers = joblib.load(folder / "capture_numbers.joblib")
    spad_dir = folder / "spad_histograms"

    held_out = set(experiment.held_out_objects)
    keep_idx = [
        i for i in range(len(objects))
        if objects[i] in held_out
        and res[i] == capture_resolution
        and (experiment.size_filter is None or sizes[i] == experiment.size_filter)
    ]

    spad_files = sorted(spad_dir.glob("*.npy"))
    histograms_per_capture = [np.load(spad_files[i]) for i in keep_idx]
    # The joblib dataset stores three repeats per capture; flatten them out
    # so that each repeat becomes its own evaluation sample.
    spad_histograms: list[np.ndarray] = []
    objects_out:     list[str] = []
    sizes_out:       list[str] = []
    capture_nums:    list[int] = []
    for stack, i in zip(histograms_per_capture, keep_idx):
        for h in stack:
            spad_histograms.append(np.asarray(h, dtype=np.float32))
            objects_out.append(objects[i])
            sizes_out.append(sizes[i])
            capture_nums.append(int(capture_numbers[i]))

    locations: list[tuple[float, float]] | None = None
    if experiment.task == "regression":
        location_mapping = _load_tag_positions(DEFAULT_TAG_POSITIONS)
        locations = [location_mapping[n] for n in capture_nums]

    return HeldOutDataset(
        spad_histograms=spad_histograms,
        objects=objects_out,
        sizes=sizes_out,
        locations=locations,
    )


def load_experiment_model(
    experiment: Experiment, *, device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the architecture, load the checkpoint, return ``(model, payload)``.

    The payload dict carries `train_mean`, `train_std`, `classes`, and the
    bundled validation metrics from when the checkpoint was produced.
    """
    if not experiment.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {experiment.ckpt}")

    payload = torch.load(experiment.ckpt, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    model = _build_arch_for(experiment)
    with torch.no_grad():
        model(torch.zeros(DUMMY_INPUT_SHAPE))    # materialize lazy layers
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, payload


def _build_arch_for(experiment: Experiment) -> nn.Module:
    # Local import so that `experiments.py` stays import-light.
    from .models import MediumConv1DNet, MediumConv1DNetReg
    if experiment.task == "regression":
        return MediumConv1DNetReg(hidden=128, dropout=0.1)
    if experiment.task == "object_classification":
        return MediumConv1DNet(num_classes=30, hidden=128, dropout=0.1)
    return MediumConv1DNet(num_classes=2, hidden=128, dropout=0.1)


def _resolve_root() -> Path:
    raw = os.environ.get(HELDOUT_ROOT_ENV)
    return Path(raw).expanduser().resolve() if raw else DEFAULT_HELDOUT_ROOT


def _load_tag_positions(path: Path) -> dict[int, tuple[float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, tuple[float, float]] = {}
    for k, v in raw.items():
        if not k.startswith("frame"):
            continue
        n = int(k.split("_", 1)[0].removeprefix("frame"))
        out[n] = (float(v[0]), float(v[1]))
    return out
