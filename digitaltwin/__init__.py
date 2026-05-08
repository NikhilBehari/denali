"""Digital-twin renderer for DENALI captures.

Reproduces the calibrated capture setup in Mitsuba 3 and emits a
side-by-side view of the captured tracking-camera frame, the simulated
tracking-camera render, and the real and simulated LiDAR histograms for
any ``(object, size, location)`` in the dataset.

Public API:

    render_scene_with_transients   build the scene; render RGB and transients
    load_real_tracking_rgb         load the captured tracking-camera frame
    load_real_spad_histogram       load the captured 3x3 SPAD histogram
    save_outputs                   write the five PNGs and two `.npy` traces
    main                           CLI entry (``python -m digitaltwin --help``)
"""
from .data   import load_real_spad_histogram, load_real_tracking_rgb
from .plot   import save_outputs
from .render import main
from .scene  import render_scene_with_transients


__all__ = [
    "load_real_spad_histogram",
    "load_real_tracking_rgb",
    "main",
    "render_scene_with_transients",
    "save_outputs",
]
