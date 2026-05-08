#!/usr/bin/env python3
"""Evaluate every Table 2 cell's checkpoint on the canonical 70/30 test split.

Exit code 0 on a clean run; 1 if any checkpoint is missing or fails to load.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import torch

from main_table import (
    CELLS,
    build_query,
    cells_in_table_order,
    evaluate,
    format_metrics,
    load_cell,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", nargs="*", default=None,
                   help="Optional list of cell names to evaluate; defaults to all 12.")
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
    cells  = [CELLS[c] for c in args.cells] if args.cells else cells_in_table_order()

    query = build_query()
    print(f"device: {device}   |   queried 3x3 samples: {len(query['spad_histograms'])}\n", flush=True)
    print(f"{'cell':<18}  metrics", flush=True)
    print("-" * 80, flush=True)

    failed = 0
    for cell in cells:
        try:
            model   = load_cell(cell, device=device)
            metrics = evaluate(cell, model, query, device=device, batch_size=args.batch_size)
        except FileNotFoundError:
            print(f"{cell.name:<18}  CHECKPOINT MISSING - {cell.ckpt.name}", flush=True)
            failed += 1
            continue
        except Exception as e:                                       # noqa: BLE001
            print(f"{cell.name:<18}  LOAD/EVAL FAILED - {type(e).__name__}: {e!s:.80s}", flush=True)
            failed += 1
            continue
        print(f"{cell.name:<18}  {format_metrics(cell, metrics)}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
