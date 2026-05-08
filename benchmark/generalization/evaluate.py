"""End-to-end evaluation for one supplement generalization experiment.

Pipeline:
    1. Load the experiment's checkpoint and its saved `train_mean / train_std`.
    2. Read the held-out joblib dataset, restricted to the experiment's
       held-out objects (and optional size filter).
    3. Apply `log1p` + z-score using the checkpoint's training-set statistics.
    4. Run inference and compute task-appropriate metrics on the held-out slice.

For object classification, held-out object names are mapped to base
class labels via `experiment.base_class_mapping` (e.g. the italic-A
variant maps to base class `A`).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from .experiments import Experiment, SplitExperiment
from .loaders import HeldOutDataset, load_experiment_model, load_held_out_dataset
from .splits import load_split_model, load_split_test_set


__all__ = [
    "BASE_OBJECT_CLASSES",
    "BASE_SIZE_CLASSES",
    "evaluate",
    "evaluate_split",
    "format_metrics",
]


# 30 base object classes the classifiers were trained on, in the order
# `LabelEncoder().fit(...)` would produce after sorting alphabetically.
BASE_OBJECT_CLASSES: list[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "I", "O", "S", "U", "V",
    "circle", "downarrow", "halfcircle", "hexagon", "minus",
    "plus", "square", "triangle", "uparrow", "widerectangle",
]
BASE_SIZE_CLASSES: list[str] = ["4inch", "8inch"]


def evaluate(
    experiment: Experiment,
    *,
    device:     str | torch.device = "cpu",
    batch_size: int = 256,
    held_out_root=None,
) -> dict[str, float]:
    """Run one held-out experiment and return its task metrics."""
    model, payload = load_experiment_model(experiment, device=device)
    dataset = load_held_out_dataset(experiment, held_out_root=held_out_root)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No held-out samples found for experiment '{experiment.name}'. "
            f"Check the saved dataset path and held_out_objects filter."
        )

    train_mean = float(payload["train_mean"])
    train_std  = float(payload["train_std"])
    X = _preprocess(dataset.spad_histograms, train_mean, train_std)
    preds = _run_inference(model, X, device, batch_size)

    if experiment.task == "regression":
        return _regression_metrics(preds, dataset.locations)
    if experiment.task == "object_classification":
        return _object_classification_metrics(preds, dataset, experiment)
    return _size_classification_metrics(preds, dataset)


def format_metrics(experiment, m: Mapping[str, float]) -> str:
    """One-line summary, formatted by task. Accepts Experiment or SplitExperiment."""
    if experiment.task == "regression":
        return f"RMSE {m['rmse']:.4f} / MAE {m['mae']:.4f} / MSE {m['mse']:.6f}"
    if experiment.task == "object_classification":
        return f"Top-1 {m['top1_acc']:.4f} / Top-5 {m['top5_acc']:.4f} / F1 {m['macro_f1']:.4f}"
    return f"P {m['macro_precision']:.4f} / R {m['macro_recall']:.4f} / Acc {m['top1_acc']:.4f}"


def evaluate_split(
    experiment: SplitExperiment,
    query:      Mapping[str, list],
    *,
    device:     str | torch.device = "cpu",
    batch_size: int = 256,
) -> dict[str, float]:
    """Run one Table-`identity_splits` entry and return its task metrics.

    Loads the bundled split checkpoint, reads its test indices from
    ``split_info`` (or recomputes the random-grouped split), applies
    `log1p + z-score` preprocessing, and computes task metrics.
    """
    model, _ = load_split_model(experiment, device=device)
    test_set = load_split_test_set(experiment, dict(query))

    test_idx = test_set.test_indices
    if not test_idx:
        raise RuntimeError(f"No test samples for {experiment.name}.")

    histograms = [np.asarray(query["spad_histograms"][i], dtype=np.float32) for i in test_idx]
    X = _preprocess(histograms, test_set.train_mean, test_set.train_std)
    preds = _run_inference(model, X, device, batch_size)

    if experiment.task == "regression":
        targets = [tuple(map(float, query["locations"][i])) for i in test_idx]
        return _regression_metrics(preds, targets)

    if experiment.task == "object_classification":
        targets = [str(query["objects"][i]) for i in test_idx]
        return _classification_metrics(preds, targets, BASE_OBJECT_CLASSES, num_classes=30)

    targets = [str(query["sizes"][i]) for i in test_idx]
    return _classification_metrics(preds, targets, BASE_SIZE_CLASSES, num_classes=2)


def _classification_metrics(
    logits: np.ndarray, target_labels: list[str], class_list: list[str], *, num_classes: int,
) -> dict[str, float]:
    encoder = LabelEncoder().fit(class_list)
    targets = encoder.transform(target_labels).astype(np.int64)

    preds = logits.argmax(axis=1)
    top1  = float((preds == targets).mean())

    if num_classes >= 5:
        topk_preds = np.argpartition(-logits, kth=4, axis=1)[:, :5]
        top5 = float((topk_preds == targets[:, None]).any(axis=1).mean())
    else:
        top5 = 1.0

    return {
        "top1_acc":        top1,
        "top5_acc":        top5,
        "macro_f1":        float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(targets, preds, average="macro", zero_division=0)),
    }


def _preprocess(histograms, train_mean: float, train_std: float) -> np.ndarray:
    X = np.stack(histograms, axis=0).astype(np.float32)
    X = np.log1p(np.clip(X, 0.0, None))
    X = (X - train_mean) / max(train_std, 1e-6)
    return X.astype(np.float32, copy=False)


def _run_inference(model, X: np.ndarray, device, batch_size: int) -> np.ndarray:
    out_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            out_chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _regression_metrics(preds: np.ndarray, locations) -> dict[str, float]:
    targets = np.asarray(locations, dtype=np.float32)
    diff = preds.astype(np.float32) - targets
    mse = float(np.mean(diff ** 2))
    return {
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "mae":  float(np.mean(np.abs(diff))),
    }


def _object_classification_metrics(
    logits: np.ndarray, dataset: HeldOutDataset, experiment: Experiment,
) -> dict[str, float]:
    encoder = LabelEncoder().fit(BASE_OBJECT_CLASSES)
    base_targets = np.array([
        encoder.transform([experiment.base_class_mapping[o]])[0]
        for o in dataset.objects
    ], dtype=np.int64)

    preds = logits.argmax(axis=1)
    top1  = float((preds == base_targets).mean())

    top5_preds = np.argpartition(-logits, kth=4, axis=1)[:, :5]
    top5 = float((top5_preds == base_targets[:, None]).any(axis=1).mean())

    macro_f1        = float(f1_score(base_targets, preds, average="macro", zero_division=0))
    macro_precision = float(precision_score(base_targets, preds, average="macro", zero_division=0))
    macro_recall    = float(recall_score(base_targets, preds, average="macro", zero_division=0))

    return {
        "top1_acc":        top1,
        "top5_acc":        top5,
        "macro_f1":        macro_f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
    }


def _size_classification_metrics(
    logits: np.ndarray, dataset: HeldOutDataset,
) -> dict[str, float]:
    encoder = LabelEncoder().fit(BASE_SIZE_CLASSES)
    targets = encoder.transform(dataset.sizes).astype(np.int64)

    preds = logits.argmax(axis=1)
    top1  = float((preds == targets).mean())

    return {
        "top1_acc":        top1,
        "macro_f1":        float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(targets, preds, average="macro", zero_division=0)),
    }
