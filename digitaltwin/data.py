"""Loaders for the captured tracking RGB and SPAD histogram.

Reads from the canonical layout under ``data_dir``:

    <data_dir>/<object>_<size>_3x3_lighton_NLOSdata/
        tracking_rgb/<NNN>.png
        spad_histogram/<NNN>.npy        (shape: 3 x 3 x 3 x 128 = repeats x H x W x bins)

The 3x3 SPAD cube is averaged over the three repeated captures and an
optional no-object background (``NOOBJECT_<size>_3x3_lighton_NLOSdata``)
is subtracted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


__all__ = ["load_real_tracking_rgb", "load_real_spad_histogram"]


def _capture_folder(data_dir: Path, obj: str, size: str) -> Path:
    return data_dir / f"{obj}_{size}_3x3_lighton_NLOSdata"


def load_real_tracking_rgb(data_dir: Path, obj: str, size: str, location: int) -> np.ndarray:
    """Return the tracking-RGB frame for ``(obj, size, location)``."""
    rgb_path = _capture_folder(data_dir, obj, size) / "tracking_rgb" / f"{int(location):03d}.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"tracking RGB not found: {rgb_path}")
    return plt.imread(rgb_path)


def load_real_spad_histogram(
    data_dir:        Path,
    obj:             str,
    size:            str,
    location:        int,
    *,
    crop_start_bin:  int  = 70,
    subtract_bg:     bool = True,
    bg_size:         str  = "8inch",
) -> np.ndarray:
    """Return the centre-pixel SPAD histogram (length-128 vector).

    Averages the three repeats, optionally subtracts the matching no-object
    background, clips to >=0, and zeroes the first ``crop_start_bin`` bins
    (the direct return that the digital twin does not model).
    """
    obj_path = _capture_folder(data_dir, obj, size) / "spad_histogram" / f"{int(location):03d}.npy"
    if not obj_path.exists():
        raise FileNotFoundError(f"SPAD histogram not found: {obj_path}")
    obj_hist = np.load(obj_path)
    if obj_hist.ndim != 4 or obj_hist.shape[-1] != 128:
        raise ValueError(f"unexpected histogram shape {obj_hist.shape} at {obj_path}")
    obj_mean = obj_hist.mean(axis=0).astype(np.float32)

    if subtract_bg:
        bg_path = _capture_folder(data_dir, "NOOBJECT", bg_size) / "spad_histogram" / f"{int(location):03d}.npy"
        if not bg_path.exists():
            raise FileNotFoundError(f"background histogram not found: {bg_path}")
        bg_mean = np.load(bg_path).mean(axis=0).astype(np.float32)
        if obj_mean.shape != bg_mean.shape:
            raise ValueError(f"object/background shape mismatch: {obj_mean.shape} vs {bg_mean.shape}")
        center = obj_mean - bg_mean
    else:
        center = obj_mean

    center = np.clip(center, 0.0, None)
    center[..., :crop_start_bin] = 0
    return center[1, 1, :].astype(np.float32)
