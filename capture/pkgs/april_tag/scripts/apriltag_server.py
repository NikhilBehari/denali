"""FastAPI AprilTag detection server used by the DENALI capture validator.

Run as ``python apriltag_server.py`` (default port 8000). The capture
script ``capture_spad_nlos.py`` POSTs a tracking-camera frame to
``/process/`` to verify which AprilTags are visible before each session.

Camera intrinsics and tag side length are read from env vars; the
defaults below are the DENALI-rig values (1280x720 RealSense tracking
camera, 6 cm AprilTags). Override with ``APRILTAG_CAMERA_FX/FY/CX/CY``
and ``APRILTAG_TAG_SIZE_M`` to use the server with a different rig.
"""
import base64
import os

import apriltag
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


_DEFAULT_CAMERA_PARAMS = (
    float(os.environ.get("APRILTAG_CAMERA_FX", "923.57")),
    float(os.environ.get("APRILTAG_CAMERA_FY", "923.94")),
    float(os.environ.get("APRILTAG_CAMERA_CX", "641.69")),
    float(os.environ.get("APRILTAG_CAMERA_CY", "365.36")),
)
_DEFAULT_TAG_SIZE = float(os.environ.get("APRILTAG_TAG_SIZE_M", "0.06"))


def apriltag_server(image_bytes=None,
                    camera_params=_DEFAULT_CAMERA_PARAMS,
                    tag_size=_DEFAULT_TAG_SIZE):
    if image_bytes is None:
        raise ValueError("No image bytes provided.")

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes.")

    parser = apriltag.ArgumentParser(description="Detect AprilTags from images.")
    apriltag.add_arguments(parser)
    # Quad-contour detection is more robust under shadow.
    parser.set_defaults(quad_contours=True)
    options = parser.parse_args([])

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    detections, overlay = apriltag.detect_tags(
        img,
        detector,
        camera_params=camera_params,
        tag_size=tag_size,
        vizualization=3,
        verbose=1,
        annotation=True
    )

    return detections, overlay


app = FastAPI()


def serialize_detection(det):
    if hasattr(det, "__dict__"):
        data = det.__dict__
    elif isinstance(det, dict):
        data = det
    else:
        try:
            data = {
                "tag_family": getattr(det[0], "tag_family", None),
                "tag_id": getattr(det[0], "tag_id", None),
                "hamming": getattr(det[0], "hamming", None),
                "goodness": getattr(det[0], "goodness", None),
                "decision_margin": getattr(det[0], "decision_margin", None),
                "homography": det[0].homography.tolist() if hasattr(det[0], "homography") else None,
                "center": det[0].center.tolist() if hasattr(det[0], "center") else None,
                "corners": det[0].corners.tolist() if hasattr(det[0], "corners") else None,
                "pose": det[1].tolist() if len(det) > 1 and isinstance(det[1], np.ndarray) else None,
                "init_error": float(det[2]) if len(det) > 2 else None,
                "final_error": float(det[3]) if len(det) > 3 else None
            }
        except Exception:
            data = {"raw": str(det)}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()
        if isinstance(v, bytes):
            data[k] = v.decode()
    return data


@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections_raw, overlay = apriltag_server(image_bytes)

    # group detections [detection, pose, e0, e1]
    grouped = [detections_raw[i:i+4] for i in range(0, len(detections_raw), 4)]
    serialized = [serialize_detection(det) for det in grouped]

    _, png_bytes = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(png_bytes.tobytes()).decode()

    return JSONResponse({
        "detections": serialized,
        "overlay_image": overlay_b64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
