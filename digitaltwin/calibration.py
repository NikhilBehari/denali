"""Calibrated intrinsics and extrinsics for the tracking and co-located cameras.

Both cameras are calibrated against the AprilTag rig in the capture setup.
``rotation_matrix`` and ``position`` give each camera's pose in the same
world frame as ``poses.json``. Calibration reprojection RMS is below 0.2 px
on the calibration set.
"""
from __future__ import annotations


__all__ = ["TRACKING_CAMERA", "COLOCATED_CAMERA"]


TRACKING_CAMERA = {
    "rvec": [2.455429880042772, 0.01494441359033604, 0.013295024135541745],
    "rotation_matrix": [
        [0.9998823028765109,   0.007364662021945467, 0.01345890588677102],
        [0.014224765565310535, -0.77366878320271,    -0.6334305565271144],
        [0.005747733377462908,  0.6335474533536857,  -0.7736824852031233],
    ],
    "tvec":     [-0.7449249475839963, 0.3983587029616176, 1.1458284076684249],
    "position": [ 0.7325847967163193, -0.4122528561894433, 1.1488658197378823],
    "intrinsics": {"fx": 923.57, "fy": 923.94, "cx": 641.69, "cy": 365.96},
    "res": [1280, 720],
}


COLOCATED_CAMERA = {
    "rvec": [1.6210261895847597, -0.34607420166357206, 0.395278428354992],
    "rotation_matrix": [
        [0.8923183116801053,   -0.4487731425355246,  0.048689805700537686],
        [0.011046951680332567, -0.08612041839810997, -0.996223488175975  ],
        [0.4512715318950873,    0.88948633495615,    -0.07188925112516481],
    ],
    "tvec":     [-0.2838756345833701, 0.3936884484228392, -0.46559250766961424],
    "position": [ 0.45906701388666116, 0.32064702650552074, 0.37255243212751327],
    "intrinsics": {"fx": 605.79, "fy": 605.37, "cx": 426.69, "cy": 247.06},
    "res": [848, 480],
}
