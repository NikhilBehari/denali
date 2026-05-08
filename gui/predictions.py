"""Inference for the three GUI heads.

    object     30-class softmax over the trained object set
    size       2-class softmax over [4inch, 8inch]
    location   2D (x, y) in world meters (same coords as `tag1_positions.json`)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from .models import Conv1DClassifier, Conv1DRegressor


__all__ = [
    "Prediction",
    "predict",
    "OBJECT_CLASSES",
    "SIZE_CLASSES",
]


PACKAGE_DIR     = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"
DEVICE          = torch.device("cpu")

# Class orderings expected by the trained classifiers.
OBJECT_CLASSES: list[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "I", "O", "S", "U", "V",
    "circle", "downarrow", "halfcircle", "hexagon", "minus",
    "plus", "square", "triangle", "uparrow", "widerectangle",
]
SIZE_CLASSES: list[str] = ["4inch", "8inch"]


@dataclass(frozen=True)
class Prediction:
    gt_xy:          tuple[float, float]
    pred_xy:        tuple[float, float]
    err_m:          float
    object_classes: list[str]
    object_probs:   np.ndarray   # (30,)
    object_gt_idx:  int          # -1 if GT is not in the trained set (e.g. NOOBJECT)
    size_classes:   list[str]
    size_probs:     np.ndarray   # (2,)
    size_gt_idx:    int          # -1 if GT is not in the trained set


_MODELS:      dict[str, torch.nn.Module] = {}
_MODELS_LOCK: Lock = Lock()
_CACHE:       dict[tuple[str, int], Prediction] = {}


def _load(model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state   = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    with torch.no_grad():
        model(torch.zeros(1, 3, 3, 128))   # materialize lazy layers
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


def _models() -> dict[str, torch.nn.Module]:
    if not _MODELS:
        with _MODELS_LOCK:
            if not _MODELS:
                _MODELS["object"]     = _load(Conv1DClassifier(num_classes=30), CHECKPOINTS_DIR / "object.pth")
                _MODELS["size"]       = _load(Conv1DClassifier(num_classes=2),  CHECKPOINTS_DIR / "size.pth")
                _MODELS["regression"] = _load(Conv1DRegressor(),                CHECKPOINTS_DIR / "regression.pth")
    return _MODELS


def predict(
    obj:       str,
    size:      str,
    *,
    hist:      np.ndarray,             # (3, 3, 128) float32
    gt_xy:     tuple[float, float],
    cache_key: tuple[str, int] | None = None,
) -> Prediction:
    """Run the three GUI heads on ``hist`` and return predictions + GT context."""
    if cache_key is not None and cache_key in _CACHE:
        return _CACHE[cache_key]

    models = _models()
    x = torch.from_numpy(hist[None].astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        obj_probs  = torch.softmax(models["object"](x), dim=-1)[0].cpu().numpy()
        size_probs = torch.softmax(models["size"](x),   dim=-1)[0].cpu().numpy()
        pred_xy    = models["regression"](x)[0].cpu().numpy()

    pred_xy_t = (float(pred_xy[0]), float(pred_xy[1]))
    err_m     = float(np.hypot(pred_xy_t[0] - gt_xy[0], pred_xy_t[1] - gt_xy[1]))

    p = Prediction(
        gt_xy=gt_xy,
        pred_xy=pred_xy_t,
        err_m=err_m,
        object_classes=OBJECT_CLASSES,
        object_probs=obj_probs,
        object_gt_idx=OBJECT_CLASSES.index(obj) if obj in OBJECT_CLASSES else -1,
        size_classes=SIZE_CLASSES,
        size_probs=size_probs,
        size_gt_idx=SIZE_CLASSES.index(size) if size in SIZE_CLASSES else -1,
    )
    if cache_key is not None:
        _CACHE[cache_key] = p
    return p
