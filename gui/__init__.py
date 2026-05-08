"""DENALI Dash GUI for non-line-of-sight tracking on 3x3 SPAD captures.

A browser-based viewer that runs the three 1D-CNN inference heads
(object, size, planar location) on each capture's SPAD histogram and
overlays the predictions on the tracking-camera RGB image. Markers are
projected from world meters to pixels with the calibration in
:mod:`gui.calibration`.
"""
from .predictions import predict, Prediction, OBJECT_CLASSES, SIZE_CLASSES
from .calibration import project_world_to_pixel, world_to_norm_xy
from .app import main

__all__ = [
    "predict",
    "Prediction",
    "OBJECT_CLASSES",
    "SIZE_CLASSES",
    "project_world_to_pixel",
    "world_to_norm_xy",
    "main",
]
