#!/usr/bin/env python3
"""Retrain one Table 2 cell from scratch using its registered hyperparameters.

The best checkpoint is written to ``main_table/checkpoints/<cell>.pth``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import torch

from main_table import CELLS, build_query, train_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", required=True, choices=sorted(CELLS),
                   help="Which Table 2 cell to retrain.")
    p.add_argument("--device", default="auto",
                   help="'auto' (default), 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the existing checkpoint if it's already on disk.")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    query  = build_query()
    train_cell(CELLS[args.cell], query, device=device, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
