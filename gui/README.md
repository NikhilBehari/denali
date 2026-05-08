# DENALI GUI

A web app showing real NLOS tracking on the captured 3x3 SPAD data: scene
RGB with projected predictions, the live SPAD histograms, and the three
inference heads (object, size, 2D location).

The 1D-CNN checkpoints in [`checkpoints/`](checkpoints/) drive the three
inference heads. They are trained on the full 3x3 base dataset (no
held-out split), so the GUI is intended for visualization rather than
generalization measurement.

## Run

```bash
pip install -r requirements.txt
python -m gui                                      # http://127.0.0.1:8050
python -m gui --port 8080                          # custom port
python -m gui --host 0.0.0.0 --port 8080           # bind on all interfaces
python -m gui --data-dir /alt/path/to/data         # override data directory
```

Run from inside `denali_public/`. By default the app reads the captures
from `denali_public/denali-data/data/`; override with `--data-dir`.

## Layout

```
gui/
├── README.md
├── requirements.txt
├── app.py                Dash app + Flask route for the scene image
├── predictions.py        wraps the three checkpoints behind one `predict()`
├── calibration.py        tracking-camera intrinsics + extrinsics
├── models.py             1D-CNN classifier + regressor architectures
└── checkpoints/
    ├── object.pth        30-class object classifier
    ├── size.pth          2-class size classifier
    └── regression.pth    2D location regressor
```

The scene index (`manifest.json`) is read from the shared
[`assets/`](../assets/) folder at the repo root.
