"""DENALI supplement generalization experiments (Sec. 7).

Two families:

- **Held-out object variants** (supplement Figs `non_rr` and `italics`):
  train on the base 30-object dataset, evaluate on unseen object variants
  (italic, cardstock, white). 5 entries in :data:`EXPERIMENTS`.

- **Train/test split variants** (supplement Table `identity_splits`):
  train the 1D CNN under different stratifications of the same base
  dataset. 10 entries in :data:`SPLIT_EXPERIMENTS`.

Public API
----------
    EXPERIMENTS, SPLIT_EXPERIMENTS    registries
    Experiment, SplitExperiment       dataclasses
    evaluate                          run a held-out experiment, return metrics
    evaluate_split                    run a split experiment, return metrics
    format_metrics                    pretty-print a metrics dict

Typical usage
-------------
    >>> from generalization import EXPERIMENTS, evaluate, format_metrics
    >>> exp = EXPERIMENTS["italics_obj"]
    >>> print(format_metrics(exp, evaluate(exp, device="cuda")))
"""
from .experiments import (
    EXPERIMENTS,
    Experiment,
    SPLIT_EXPERIMENTS,
    SplitExperiment,
    experiments_in_table_order,
    split_experiments_in_table_order,
)
from .evaluate import evaluate, evaluate_split, format_metrics

__all__ = [
    "EXPERIMENTS",
    "Experiment",
    "experiments_in_table_order",
    "SPLIT_EXPERIMENTS",
    "SplitExperiment",
    "split_experiments_in_table_order",
    "evaluate",
    "evaluate_split",
    "format_metrics",
]
