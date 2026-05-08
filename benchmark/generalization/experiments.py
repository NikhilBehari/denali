"""Registry of the 15 supplement generalization experiments (Sec. 7).

Two families:

1. **Held-out object variants** (supplement Figs `non_rr` and `italics`, 5
   entries). Train on the base dataset; evaluate on unseen object variants
   loaded from a separate joblib dataset. One :class:`Experiment` per entry.

2. **Train/test split variants** (supplement Table `identity_splits`, 10
   entries). Train the 1D CNN under different stratifications of the same
   base dataset. Each :class:`SplitExperiment` reads the test slice
   directly from the bundled checkpoint's `split_info`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


__all__ = [
    "EXPERIMENTS",
    "Experiment",
    "SPLIT_EXPERIMENTS",
    "SplitExperiment",
    "Task",
    "SplitStrategy",
    "PACKAGE_DIR",
    "CHECKPOINTS_DIR",
    "SPLIT_CKPTS_DIR",
    "experiments_in_table_order",
    "split_experiments_in_table_order",
]


PACKAGE_DIR     = Path(__file__).resolve().parent
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"
SPLIT_CKPTS_DIR = PACKAGE_DIR / "split_checkpoints"

Task           = Literal["regression", "object_classification", "size_classification"]
SplitStrategy  = Literal["random_grouped", "by_location", "by_object", "by_size"]


# Held-out object-variant experiments.

@dataclass(frozen=True)
class Experiment:
    """One held-out object-variant entry (italics, cardstock, or white)."""
    name:                 str
    task:                 Task
    ckpt:                 Path
    saved_dataset_subdir: str
    held_out_objects:     tuple[str, ...]
    base_class_mapping:   dict[str, str] = field(default_factory=dict)
    size_filter:          str | None = None


EXPERIMENTS: dict[str, Experiment] = {
    "italics_loc": Experiment(
        name="italics_loc",
        task="regression",
        ckpt=CHECKPOINTS_DIR / "italics_loc.pth",
        saved_dataset_subdir="saved_dataset",
        held_out_objects=(
            "generalization1italic",
            "generalizationAitalic",
            "generalizationcircleitalic",
        ),
        size_filter="8inch",
    ),
    "italics_obj": Experiment(
        name="italics_obj",
        task="object_classification",
        ckpt=CHECKPOINTS_DIR / "italics_obj.pth",
        saved_dataset_subdir="saved_dataset",
        held_out_objects=(
            "generalization1italic",
            "generalizationAitalic",
            "generalizationcircleitalic",
        ),
        base_class_mapping={
            "generalization1italic":      "1",
            "generalizationAitalic":      "A",
            "generalizationcircleitalic": "circle",
        },
        size_filter="8inch",
    ),
    "nonrr_loc": Experiment(
        name="nonrr_loc",
        task="regression",
        ckpt=CHECKPOINTS_DIR / "nonrr_loc.pth",
        saved_dataset_subdir="saved_dataset",
        held_out_objects=("generalization1cardstock", "generalization1white"),
    ),
    "nonrr_obj": Experiment(
        name="nonrr_obj",
        task="object_classification",
        ckpt=CHECKPOINTS_DIR / "nonrr_obj.pth",
        saved_dataset_subdir="saved_dataset",
        held_out_objects=("generalization1cardstock", "generalization1white"),
        base_class_mapping={
            "generalization1cardstock": "1",
            "generalization1white":     "1",
        },
    ),
    "nonrr_size": Experiment(
        name="nonrr_size",
        task="size_classification",
        ckpt=CHECKPOINTS_DIR / "nonrr_size.pth",
        saved_dataset_subdir="saved_dataset",
        held_out_objects=("generalization1cardstock", "generalization1white"),
    ),
}


def experiments_in_table_order() -> list[Experiment]:
    """Return the 5 held-out experiments in supplement-figure order."""
    return list(EXPERIMENTS.values())


# Split-variant experiments.

@dataclass(frozen=True)
class SplitExperiment:
    """One `identity_splits` entry: the 1D CNN trained under a specific split strategy."""
    name:           str
    task:           Task
    ckpt:           Path
    split_strategy: SplitStrategy

    # `random_grouped` is recomputed deterministically from these; `by_*`
    # strategies read the test slice from the checkpoint's `split_info`.
    test_size:      float = 0.3
    random_state:   int   = 42


SPLIT_EXPERIMENTS: dict[str, SplitExperiment] = {

    # Location regression.
    "splits_loc_random_grouped": SplitExperiment(
        name="splits_loc_random_grouped", task="regression",
        ckpt=SPLIT_CKPTS_DIR / "splits_loc_random_grouped.pth",
        split_strategy="random_grouped",
    ),
    "splits_loc_by_location": SplitExperiment(
        name="splits_loc_by_location", task="regression",
        ckpt=SPLIT_CKPTS_DIR / "splits_loc_by_location.pth",
        split_strategy="by_location",
    ),
    "splits_loc_by_object": SplitExperiment(
        name="splits_loc_by_object", task="regression",
        ckpt=SPLIT_CKPTS_DIR / "splits_loc_by_object.pth",
        split_strategy="by_object",
    ),
    "splits_loc_by_size": SplitExperiment(
        name="splits_loc_by_size", task="regression",
        ckpt=SPLIT_CKPTS_DIR / "splits_loc_by_size.pth",
        split_strategy="by_size",
    ),

    # Object classification (no `by_object` entry: classifying objects unseen at training is ill-posed).
    "splits_obj_random_grouped": SplitExperiment(
        name="splits_obj_random_grouped", task="object_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_obj_random_grouped.pth",
        split_strategy="random_grouped",
    ),
    "splits_obj_by_location": SplitExperiment(
        name="splits_obj_by_location", task="object_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_obj_by_location.pth",
        split_strategy="by_location",
    ),
    "splits_obj_by_size": SplitExperiment(
        name="splits_obj_by_size", task="object_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_obj_by_size.pth",
        split_strategy="by_size",
    ),

    # Size classification (no `by_size` entry: classifying sizes unseen at training is ill-posed).
    "splits_size_random_grouped": SplitExperiment(
        name="splits_size_random_grouped", task="size_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_size_random_grouped.pth",
        split_strategy="random_grouped",
    ),
    "splits_size_by_location": SplitExperiment(
        name="splits_size_by_location", task="size_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_size_by_location.pth",
        split_strategy="by_location",
    ),
    "splits_size_by_object": SplitExperiment(
        name="splits_size_by_object", task="size_classification",
        ckpt=SPLIT_CKPTS_DIR / "splits_size_by_object.pth",
        split_strategy="by_object",
    ),
}


def split_experiments_in_table_order() -> list[SplitExperiment]:
    """Return the 10 split entries in the order they appear in Table `identity_splits`."""
    order = [
        "splits_loc_random_grouped", "splits_loc_by_location", "splits_loc_by_object", "splits_loc_by_size",
        "splits_obj_random_grouped", "splits_obj_by_location", "splits_obj_by_size",
        "splits_size_random_grouped", "splits_size_by_location", "splits_size_by_object",
    ]
    return [SPLIT_EXPERIMENTS[k] for k in order]
