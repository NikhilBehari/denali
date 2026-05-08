#!/usr/bin/env python3
"""Evaluate the held-out object-variant and train/test split experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import torch

from generalization import (
    evaluate,
    evaluate_split,
    experiments_in_table_order,
    format_metrics,
    split_experiments_in_table_order,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=("all", "held_out", "splits"), default="all",
                   help="Which family of experiments to run.")
    p.add_argument("--device", default="auto",
                   help="'auto' (default), 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def main() -> int:
    args   = parse_args()
    device = resolve_device(args.device)
    print(f"device: {device}", flush=True)
    failed = 0

    if args.only in ("all", "held_out"):
        print("\n--- Held-out generalization (italic / non-RR) ---", flush=True)
        print(f"{'experiment':<14}  metrics", flush=True)
        print("-" * 80, flush=True)
        for exp in experiments_in_table_order():
            try:
                m = evaluate(exp, device=device, batch_size=args.batch_size)
            except Exception as e:                                   # noqa: BLE001
                print(f"{exp.name:<14}  FAILED - {type(e).__name__}: {e!s:.80s}", flush=True)
                failed += 1
                continue
            print(f"{exp.name:<14}  {format_metrics(exp, m)}", flush=True)

    if args.only in ("all", "splits"):
        # Splits use the base 3x3 dataset queried via main_table.
        from main_table import build_query
        print("\n--- Train/test split analysis (Table identity_splits) ---", flush=True)
        query = build_query()
        print(f"queried 3x3 samples: {len(query['spad_histograms'])}", flush=True)
        print(f"{'experiment':<28}  metrics", flush=True)
        print("-" * 90, flush=True)
        for exp in split_experiments_in_table_order():
            try:
                m = evaluate_split(exp, query, device=device, batch_size=args.batch_size)
            except Exception as e:                                   # noqa: BLE001
                print(f"{exp.name:<28}  FAILED - {type(e).__name__}: {e!s:.80s}", flush=True)
                failed += 1
                continue
            print(f"{exp.name:<28}  {format_metrics(exp, m)}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
