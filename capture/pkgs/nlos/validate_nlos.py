"""Pre-capture validation for the DENALI NLOS rig.

`precapture_validate()` grabs a single frame from the tracking RealSense,
posts it to a running AprilTag detection server, and returns
``(detections, is_dark)`` so the capture script can verify the gantry-side
AprilTags are visible and the room lighting matches the requested mode.

The AprilTag server is the FastAPI app shipped at
``capture/pkgs/april_tag/scripts/apriltag_server.py``. It listens on port
8000 by default; set ``$APRILTAG_SERVER`` to point at a different host
(default: ``http://127.0.0.1:8000``).
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
import pyrealsense2 as rs
import requests
from PIL import Image


__all__ = ["precapture_validate"]


SERVER_URL         = os.environ.get("APRILTAG_SERVER", "http://127.0.0.1:8000")
TRACKING_RS_SERIAL = os.environ.get("DENALI_TRACKING_SERIAL", "")


def _capture_realsense_image(serial: str, width: int = 1280, height: int = 720) -> Image.Image:
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        for _ in range(30):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame from RealSense.")
        bgr = np.asanyarray(color_frame.get_data())
        return Image.fromarray(bgr[:, :, ::-1])
    finally:
        pipeline.stop()


def _post_to_apriltag_server(img: Image.Image, server_url: str) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    response = requests.post(f"{server_url}/process/", files={"file": buf})
    response.raise_for_status()
    return response.json()


def _process_detections(data: dict, *, show_overlay: bool = False) -> list[dict]:
    detections = data.get("detections", [])
    for det in detections:
        print("Found Tag ID", det["tag_id"])
    overlay_b64 = data.get("overlay_image")
    if overlay_b64 and show_overlay:
        Image.open(io.BytesIO(base64.b64decode(overlay_b64))).show()
    return detections


def _is_image_dark(img: Image.Image, *, threshold: float = 30.0) -> bool:
    return float(np.array(img.convert("L")).mean()) < threshold


def precapture_validate() -> tuple[list[dict], bool]:
    """Return (AprilTag detections, image-is-dark) for the tracking RealSense frame."""
    if not TRACKING_RS_SERIAL:
        raise RuntimeError(
            "Tracking RealSense serial is not set. Set $DENALI_TRACKING_SERIAL "
            "before running pre-capture validation."
        )
    img  = _capture_realsense_image(TRACKING_RS_SERIAL)
    data = _post_to_apriltag_server(img, SERVER_URL)
    return _process_detections(data, show_overlay=False), _is_image_dark(img)
