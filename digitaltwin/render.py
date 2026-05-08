"""Command-line entry point for the digital-twin renderer.

Run from inside ``denali_public/``:

    python -m digitaltwin --object 7 --size 8inch --location 19
    python -m digitaltwin --object A --size 4inch --location 50 --output-dir custom/

For a given (object, size, location) this:

    1. loads the captured tracking-RGB frame and the centre-pixel SPAD
       histogram from ``denali_public/denali-data/data/``;
    2. re-renders the scene in Mitsuba, returning the simulated tracking
       RGB and a transient-NLOS capture cube;
    3. subtracts the no-object background transient and writes the five
       PNGs and two ``.npy`` traces to
       ``<output-dir>/<object>_<size>_<location>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .data  import load_real_spad_histogram, load_real_tracking_rgb
from .plot  import save_outputs
from .scene import render_scene_with_transients


__all__ = ["main"]


PACKAGE_DIR  = Path(__file__).resolve().parent
ASSETS_DIR   = (PACKAGE_DIR / ".." / "assets").resolve()
POSES_PATH   = ASSETS_DIR / "poses.json"
OBJECTS_DIR  = ASSETS_DIR / "object_files" / "objects" / "objs_shifted"
DEFAULT_DATA_DIR   = (PACKAGE_DIR / ".." / "denali-data" / "data").resolve()
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"


def _obj_path(obj: str, size: str) -> Path:
    """Resolve `<obj>_<size_code>.obj` (`8inch` -> `8in`)."""
    return OBJECTS_DIR / f"{obj}_{size.replace('inch', 'in')}.obj"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m digitaltwin",
        description="Render a real + simulated capture pair (tracking RGB and LiDAR histogram).",
    )
    p.add_argument("--object",   required=True,
                   help="Object name (e.g. 7, A, circle, widerectangle).")
    p.add_argument("--size",     choices=("4inch", "8inch"), default="8inch",
                   help="Object size (default: 8inch).")
    p.add_argument("--location", type=int, required=True,
                   help="Gantry location index (0-99).")
    p.add_argument("--data-dir",   type=Path, default=None,
                   help="Path to the captured NLOS data (default: ../denali-data/data).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write outputs (default: digitaltwin/outputs/).")
    p.add_argument("--block-size", type=int, default=8,
                   help="Side length of the centre block extracted from the "
                        "transient cube for the simulated trace (default: 8).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    obj_file = _obj_path(args.object, args.size)
    if not obj_file.exists():
        print(f"[error] object mesh not found: {obj_file}", file=sys.stderr)
        return 1

    data_dir = (args.data_dir or DEFAULT_DATA_DIR).resolve()
    if not data_dir.is_dir():
        print(f"[error] data directory not found: {data_dir}", file=sys.stderr)
        return 1

    out_dir = (args.output_dir or DEFAULT_OUTPUT_DIR) / \
              f"{args.object}_{args.size}_{int(args.location):03d}"

    print(f"[render] object={args.object} size={args.size} location={args.location:03d}", flush=True)
    print(f"[render] data dir : {data_dir}", flush=True)
    print(f"[render] output   : {out_dir}", flush=True)

    real_rgb  = load_real_tracking_rgb(data_dir, args.object, args.size, args.location)
    real_hist = load_real_spad_histogram(data_dir, args.object, args.size, args.location)

    print("[render] rendering simulated scene (Mitsuba) ...", flush=True)
    sim = render_scene_with_transients(obj_file, args.location, POSES_PATH)
    transient = np.clip(
        (sim["transient_full_obj"] - sim["transient_full_bg"]).astype(np.float32),
        0.0, None,
    )

    print("[render] writing outputs ...", flush=True)
    save_outputs(
        out_dir,
        real_rgb=real_rgb,
        sim_rgb=sim["rgb_image"],
        real_hist=real_hist,
        transient_full=transient,
        block_size=args.block_size,
    )
    print(f"[render] done -> {out_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
