"""Tracking-camera intrinsics & extrinsics for the GUI overlay.

Provides `project_world_to_pixel(xyz_world) -> (px, py)` mapping a 3D
world-space point (meters) into the 1280x720 tracking-RGB image.
Reprojection RMS on the calibration set was 0.13 px.
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "IMG_W",
    "IMG_H",
    "K",
    "R",
    "RVEC",
    "TVEC",
    "project_world_to_pixel",
    "world_to_norm_xy",
]


# Tracking camera (RealSense D-series, 1280x720)
_FX, _FY = 923.57, 923.94
_CX, _CY = 641.69, 365.96

K = np.array(
    [[_FX, 0.0, _CX],
     [0.0, _FY, _CY],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# Rotation / translation: world -> camera frame.
RVEC = np.array(
    [2.455429880042772, 0.01494441359033604, 0.013295024135541745],
    dtype=np.float64,
)
TVEC = np.array(
    [-0.7449249475839963, 0.3983587029616176, 1.1458284076684249],
    dtype=np.float64,
)

R = np.array(
    [[0.9998823028765109,   0.007364662021945467, 0.01345890588677102],
     [0.014224765565310535, -0.77366878320271,    -0.6334305565271144],
     [0.005747733377462908,  0.6335474533536857,  -0.7736824852031233]],
    dtype=np.float64,
)

IMG_W, IMG_H = 1280, 720


def project_world_to_pixel(xyz_world: np.ndarray) -> tuple[float, float] | None:
    """Project a single 3D world point ``[x, y, z]`` (meters) into the tracking
    RGB pixel coordinates. Returns ``(px, py)`` or ``None`` if the point is
    behind the camera."""
    p = R @ np.asarray(xyz_world, dtype=np.float64) + TVEC
    if p[2] <= 1e-6:
        return None
    px = _FX * p[0] / p[2] + _CX
    py = _FY * p[1] / p[2] + _CY
    return float(px), float(py)


def world_to_norm_xy(xyz_world: np.ndarray) -> tuple[float, float] | None:
    """Same as `project_world_to_pixel` but normalised to ``[0, 1]`` image coords."""
    pix = project_world_to_pixel(xyz_world)
    if pix is None:
        return None
    return pix[0] / IMG_W, pix[1] / IMG_H
