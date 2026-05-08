#!/usr/bin/env python3
"""Print the registered cells with architectures and training hyperparameters."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from main_table import cells_in_table_order


def main() -> int:
    cells = cells_in_table_order()
    width = max(len(c.name) for c in cells)
    for cell in cells:
        cfg = cell.config
        present = "[x]" if cell.ckpt.exists() else "[ ]"
        preproc = "log+znorm" if cfg.log_transform and cfg.normalize else "raw"
        print(f"  {present} {cell.name:<{width}}  {cell.arch_name:<42}  "
              f"lr={cfg.lr:<8} ep={cfg.num_epochs:<3} bs={cfg.batch_size:<3} {preproc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
