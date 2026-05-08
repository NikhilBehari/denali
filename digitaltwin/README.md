# digitaltwin

Renders the calibrated DENALI capture scene in Mitsuba 3 and pairs it
with the captured tracking RGB and LiDAR histogram for any
`(object, size, location)` in the dataset.

## Install

```bash
pip install -r requirements.txt
```

Requires `mitsuba >= 3.5` with the `cuda_ad_rgb` variant (a CUDA GPU is
recommended). [`mitransient`](https://github.com/diegoroyo/mitransient)
provides the NLOS capture-meter sensor used for the transient pass.

On hosts without a PTX-capable GPU, fall back to the LLVM (CPU) variant:

```bash
export DENALI_MITSUBA_VARIANT=llvm_ad_rgb
export DRJIT_LIBLLVM_PATH=/path/to/libLLVM-14.so   # only if drjit cannot auto-locate it
```

## Run

From inside `denali_public/`:

```bash
python -m digitaltwin --object 7 --size 8inch --location 19
python -m digitaltwin --object A --size 4inch --location 50 --output-dir custom/
```

The data directory defaults to `../denali-data/data/` (the layout that
the public dataset archive expands into); override with `--data-dir`.
Outputs default to
`digitaltwin/outputs/<object>_<size>_<location>/`; override with
`--output-dir`.

Each render produces five PNGs and two `.npy` histogram traces:

| File             | Contents                                                       |
|------------------|----------------------------------------------------------------|
| `combined.png`   | three-panel: real RGB · simulated RGB · histogram overlay      |
| `real_rgb.png`   | captured tracking-camera frame                                 |
| `sim_rgb.png`    | Mitsuba-rendered tracking-camera view                          |
| `real_hist.png`  | SPAD histogram (3x3 centre pixel, background-subtracted)       |
| `sim_hist.png`   | simulated NLOS histogram (centre-block trace)                  |
| `real_hist.npy`  | values plotted in `real_hist.png`                              |
| `sim_hist.npy`   | values plotted in `sim_hist.png`                               |

## Layout

```
digitaltwin/
├── README.md
├── requirements.txt
├── __init__.py / __main__.py
├── render.py        CLI entry point
├── scene.py         Mitsuba scene builder and transient renderer
├── data.py          loaders for the captured RGB and SPAD histogram
├── plot.py          figure and trace writers
├── calibration.py   tracking and co-located camera intrinsics/extrinsics
└── outputs/         render destination (auto-created on first run)
```

The renderer reads the per-tag world poses (`poses.json`) and the object
meshes (`object_files/`) from the shared [`assets/`](../assets/) folder
at the repo root.
