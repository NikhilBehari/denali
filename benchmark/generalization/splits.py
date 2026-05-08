"""Split-strategy harness for the supplement `identity_splits` entries.

Each split checkpoint stores a `split_info` dict that fully determines the
test slice: train/test locations for `by_location`, train/test objects for
`by_object`, train/test sizes for `by_size`. For `random_grouped` we
recompute the deterministic group-shuffle partition with `random_state=42`.

The base 3x3 dataset (the same one the main table uses) is loaded via
`main_table.build_query`; this module adds the splitting and preprocessing
harness on top.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .experiments import PACKAGE_DIR, SplitExperiment


__all__ = [
    "PREPROCESSING_MANIFEST_PATH",
    "SplitTestSet",
    "load_preprocessing_manifest",
    "load_split_test_set",
    "load_split_model",
]


PREPROCESSING_MANIFEST_PATH = PACKAGE_DIR / "split_checkpoints" / "preprocessing_manifest.json"


@dataclass(frozen=True)
class SplitTestSet:
    """Test indices into the base query plus the resolved per-cell preprocessing."""
    test_indices: list[int]
    train_mean:   float
    train_std:    float


def load_preprocessing_manifest() -> dict[str, Any]:
    with open(PREPROCESSING_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_split_test_set(experiment: SplitExperiment, query: dict[str, list]) -> SplitTestSet:
    """Resolve the test indices for one split experiment.

    For `by_*` strategies, the test slice comes straight from the
    checkpoint's `split_info`. For `random_grouped`, it's recomputed
    deterministically with `np.random.RandomState(experiment.random_state)`.
    """
    payload = torch.load(experiment.ckpt, map_location="cpu", weights_only=False)
    split_info = payload.get("split_info", {})

    if experiment.split_strategy == "random_grouped":
        test_indices = _split_random_grouped(query, experiment.test_size, experiment.random_state)
    elif experiment.split_strategy == "by_location":
        test_indices = _filter_by_location(query, split_info["test_locations"])
    elif experiment.split_strategy == "by_object":
        test_indices = _filter_by_object(query, split_info["test_objects"])
    elif experiment.split_strategy == "by_size":
        test_indices = _filter_by_size(query, split_info["test_sizes"])
    else:
        raise ValueError(f"Unknown split strategy: {experiment.split_strategy}")

    manifest_entry = load_preprocessing_manifest()[experiment.ckpt.name]
    stats = manifest_entry["normalization"]
    return SplitTestSet(
        test_indices=test_indices,
        train_mean=float(stats["mean"]),
        train_std=float(stats["std"]),
    )


def load_split_model(experiment: SplitExperiment, *, device="cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Build the architecture, load the checkpoint, return (model, payload)."""
    from .models import MediumConv1DNet, MediumConv1DNetReg

    payload = torch.load(experiment.ckpt, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"]

    if experiment.task == "regression":
        model = MediumConv1DNetReg(hidden=128, dropout=0.1)
    elif experiment.task == "object_classification":
        model = MediumConv1DNet(num_classes=30, hidden=128, dropout=0.1)
    else:
        model = MediumConv1DNet(num_classes=2, hidden=128, dropout=0.1)

    with torch.no_grad():
        model(torch.zeros(1, 3, 3, 128))
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, payload


# Split implementations.

def _group_id_for(query: dict[str, list], i: int) -> tuple:
    """Group key shared by all 3 repeats of one capture."""
    return (
        query["names"][i],
        query["capture_numbers"][i],
        query["objects"][i],
        query["sizes"][i],
        query["capture_resolutions"][i],
        query["lightings"][i],
    )


def _build_int_groups(query: dict[str, list]) -> dict[int, list[int]]:
    """Group sample indices by capture key, keyed by integer group id assigned in
    the order each key is first seen. The integer ordering is what makes the
    `random_grouped` split deterministic across runs."""
    int_groups: dict[int, list[int]] = defaultdict(list)
    key_to_id: dict[tuple, int] = {}
    next_id = 0
    for i in range(len(query["spad_histograms"])):
        k = _group_id_for(query, i)
        if k not in key_to_id:
            key_to_id[k] = next_id
            next_id += 1
        int_groups[key_to_id[k]].append(i)
    return dict(int_groups)


def _split_random_grouped(query, test_size: float, random_state: int) -> list[int]:
    groups = _build_int_groups(query)
    keys = sorted(groups.keys())               # int order = insertion order
    n_test = int(len(keys) * test_size)
    shuffled = list(keys)
    np.random.RandomState(random_state).shuffle(shuffled)
    test_keys = set(shuffled[:n_test])
    test_idx: list[int] = []
    for k in sorted(groups.keys()):
        if k in test_keys:
            test_idx.extend(groups[k])
    return test_idx


def _filter_by_location(query, test_locations) -> list[int]:
    target = {tuple(map(float, loc)) for loc in test_locations}
    return [
        i for i in range(len(query["spad_histograms"]))
        if tuple(map(float, query["locations"][i])) in target
    ]


def _filter_by_object(query, test_objects) -> list[int]:
    target = set(test_objects)
    return [i for i in range(len(query["spad_histograms"])) if query["objects"][i] in target]


def _filter_by_size(query, test_sizes) -> list[int]:
    target = set(test_sizes)
    return [i for i in range(len(query["spad_histograms"])) if query["sizes"][i] in target]
