#!/usr/bin/env python3
"""Build the joblib + .npy bundle from the raw NLOS captures.

The bundle is the on-disk format `main_table` and `generalization` both
read from. It includes every capture under ``--data-dir`` (base objects
and held-out variants alike); each consumer applies its own object
filter at query time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from main_table.data import DataSet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Path to the raw captures folder (contains *_NLOSdata subdirs).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Destination for the joblib + .npy bundle.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        print(f"[error] --data-dir does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    print(f"[build] reading raw captures from {args.data_dir}", flush=True)
    ds = DataSet.from_raw_captures(args.data_dir)
    print(f"[build] read {len(ds)} captures across {len(set(ds.objects))} unique objects", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] writing joblib + .npy bundle to {args.output_dir}", flush=True)
    ds.save_to_directory(args.output_dir)
    print("[build] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
