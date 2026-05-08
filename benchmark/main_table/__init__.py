"""DENALI main benchmark table (Sec. 4): 12 entries across 3 tasks x 4 architectures.

Each entry is described by a single :class:`Cell` record in the :data:`CELLS`
registry. Loading, evaluation, and training all flow through that registry,
so the checkpoint, the canonical 70/30 split, and the training config stay
tied together.

Public API
----------
    CELLS               registry of all 12 entries
    Cell, TrainConfig   dataclasses describing one entry
    load_cell           load an entry's architecture + checkpoint
    build_query         load the saved 3x3 dataset
    evaluate            run a loaded entry on the canonical 70/30 split
    format_metrics      pretty-print evaluation metrics for an entry
    train_cell          retrain an entry from scratch using its TrainConfig

Typical usage
-------------
    >>> from main_table import CELLS, build_query, load_cell, evaluate, format_metrics
    >>> query = build_query()
    >>> cell  = CELLS["obj_1dcnn"]
    >>> model = load_cell(cell, device="cuda")
    >>> print(format_metrics(cell, evaluate(cell, model, query, device="cuda")))
"""
from .cells import CELLS, Cell, TrainConfig, cells_in_table_order
from .loaders import load_cell, build_query
from .evaluate import evaluate, format_metrics
from .train import train_cell

__all__ = [
    "CELLS",
    "Cell",
    "TrainConfig",
    "cells_in_table_order",
    "load_cell",
    "build_query",
    "evaluate",
    "format_metrics",
    "train_cell",
]
