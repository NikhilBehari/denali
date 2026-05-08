#!/usr/bin/env python3
"""DENALI NLOS capture: SPAD histograms + dual-RealSense RGB/depth, snake-scanned over a 2D gantry.

Run from inside ``capture/``:

    python capture_spad_nlos.py object=<obj>_<size>_<grid>_<light>

where ``<obj>`` is the object name, ``<size>`` is ``4inch`` or ``8inch``,
``<grid>`` is ``3x3`` or ``8x8``, and ``<light>`` is ``lighton`` or
``lightoff``. Pass ``object=test`` to do a dry run with no light/tag checks.

Per gantry position the script:

    1. moves the snake-controller to the next (x, y);
    2. accumulates ``SAMPLES_PER`` SPAD histograms via the TMF8828;
    3. pulls one aligned depth+color frame from each RealSense;
    4. appends the combined record to the run's ``.pkl`` log.

Pre-capture validation (an AprilTag check + room-light check) ensures the
rig is in the expected state before any data lands on disk. The AprilTag
detection runs against the FastAPI server in
``capture/pkgs/april_tag/scripts/apriltag_server.py``.
"""
import os
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"]   = "false"

import re
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import serial.tools.list_ports

from pkgs.nlos.validate_nlos import precapture_validate

from cc_hardware.drivers.spads import SPADSensor, SPADDataType
from cc_hardware.drivers.spads.spad_wrappers import SPADMergeWrapperConfig
from cc_hardware.drivers.spads.tmf8828 import TMF8828Config, SPADID, RangeMode
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    SnakeStepperController,
    SnakeStepperControllerConfig,
    SnakeControllerAxisConfig,
)
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    TelemetrixStepperMotorSystem,
    TelemetrixStepperMotorSystemConfig,
    SingleDrive1AxisGantryXConfig,
    SingleDrive1AxisGantryYConfig,
    StepperMotorSystemAxis,
)
from cc_hardware.tools.dashboard       import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils                 import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers   import PklHandler
from cc_hardware.utils.manager         import Manager


# ---- Capture configuration -------------------------------------------------

X_SAMPLES         = 10
Y_SAMPLES         = 10
GANTRY_RANGE_X    = (0, 32)
GANTRY_RANGE_Y    = (0, 32)
SAMPLES_PER       = 3
RANGE_MODE        = RangeMode.SHORT

# Set the serial numbers of your two RealSense cameras (find them via
# ``rs-enumerate-devices`` or the Intel RealSense Viewer). The colocated
# camera shares the SPAD's optical axis; the tracking camera observes the
# scene head-on and is the source the AprilTag validator reads from.
COLOCATED_RS_SERIAL = os.environ.get("DENALI_COLOCATED_SERIAL", "")
TRACKING_RS_SERIAL  = os.environ.get("DENALI_TRACKING_SERIAL",  "")
COLOCATED_RES       = (848,  480)   # (width, height)
TRACKING_RES        = (1280, 720)
REALSENSE_FPS       = 30

# AprilTags that must be visible in light-on captures (gantry tag 1 may be
# occluded by the mounted object; that is allowed to fail).
REQUIRED_TAG_IDS = {0, 1, 5, 6, 7, 10, 12, 13, 14, 15}
TAG_1_OPTIONAL   = True

OBJECT_NAME_FORMAT = "{object}_{4inch|8inch}_{3x3|8x8}_{lighton|lightoff}"
OBJECT_NAME_RE     = re.compile(
    r"^(test|[a-zA-Z0-9]+_(4inch|8inch)_(3x3|8x8)_(lighton|lightoff))$"
)

NOW    = datetime.now()
LOGDIR = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")


# ---- Object-name parsing (sets OBJECT_NAME, SPAD_ID, LIGHT_MODE) -----------

OBJECT_NAME = next(
    (a.split("=", 1)[1].strip() for a in sys.argv[1:] if a.startswith("object=")),
    None,
)
if not OBJECT_NAME:
    print(f"\033[91mError: missing 'object=' argument.\nExpected: {OBJECT_NAME_FORMAT}\033[0m")
    sys.exit(1)

_match = OBJECT_NAME_RE.match(OBJECT_NAME)
if not _match:
    print(f"\033[91mInvalid object name. Expected: {OBJECT_NAME_FORMAT}\033[0m")
    sys.exit(1)

if OBJECT_NAME == "test":
    SPAD_ID    = SPADID.ID6
    LIGHT_MODE = None
else:
    SPAD_ID      = SPADID.ID6 if _match.group(3) == "3x3" else SPADID.ID15
    OBJECT_NAME += "_short" if RANGE_MODE == RangeMode.SHORT else "_long"
    LIGHT_MODE   = _match.group(4)

if not COLOCATED_RS_SERIAL or not TRACKING_RS_SERIAL:
    print(
        "\033[91mError: RealSense serials are not set.\n"
        "  Either edit COLOCATED_RS_SERIAL / TRACKING_RS_SERIAL at the top of\n"
        "  this file, or set $DENALI_COLOCATED_SERIAL and $DENALI_TRACKING_SERIAL.\033[0m"
    )
    sys.exit(1)


# ---- Serial-port discovery -------------------------------------------------

def find_ports() -> tuple[str, str]:
    """Locate the SPAD (Arduino Uno) and gantry (USB-Serial) ports by description."""
    ports = {p.description: p.device for p in serial.tools.list_ports.comports()}
    spad_port = next((dev for desc, dev in ports.items() if "ttyACM0"        in desc), None)
    gantry    = next((dev for desc, dev in ports.items() if "USB2.0-Serial"  in desc), None)
    if not spad_port or not gantry:
        raise RuntimeError("Could not find both SPAD (Arduino Uno) and Gantry (USB-Serial) ports.")
    return spad_port, gantry


SPAD_PORT, GANTRY_PORT = find_ports()


# ---- Setup, loop, cleanup --------------------------------------------------

def _start_realsense(serial: str, width: int, height: int) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16,  REALSENSE_FPS)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, REALSENSE_FPS)
    pipeline.start(config)
    return pipeline, rs.align(rs.stream.color)


def _check_lighting_and_tags(detections: list[dict], dark: bool) -> None:
    if LIGHT_MODE == "lightoff":
        if not dark:
            print("\033[91mError: room is not dark enough for light-off mode\033[0m")
            sys.exit(1)
        print("\033[92mRoom is sufficiently dark for light-off mode\033[0m")
        return
    if dark:
        print("\033[91mError: room is too dark for light-on mode\033[0m")
        sys.exit(1)
    detected = {d["tag_id"] for d in detections}
    missing  = REQUIRED_TAG_IDS - detected
    if not missing:
        print("\033[92mAll required AprilTags visible and lights correctly on\033[0m")
    elif TAG_1_OPTIONAL and missing == {1}:
        print("\033[93mWarning: tag 1 not detected, continuing anyway\033[0m")
    else:
        print(f"\033[91mError: missing required AprilTags: {sorted(missing)}\033[0m")
        sys.exit(1)


def setup(
    manager:    Manager,
    *,
    dashboard:  SPADDashboardConfig,
    logdir:     Path,
    object:     str,
    spad_id:    SPADID,
    range_mode: RangeMode,
    gantry:     StepperMotorSystem,
) -> None:
    logdir.mkdir(parents=True, exist_ok=True)

    sensor_config = SPADMergeWrapperConfig.create(
        wrapped=TMF8828Config.create(spad_id=spad_id, range_mode=range_mode, port=SPAD_PORT),
        data_type="HISTOGRAM",
    )
    spad = SPADSensor.create_from_config(sensor_config)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize SPAD sensor")
        return
    manager.add(spad=spad)

    dashboard = dashboard.create_from_registry(config=dashboard, sensor=spad)
    dashboard.setup()
    manager.add(dashboard=dashboard)
    win    = manager.components["dashboard"].win
    screen = win.screen().geometry()
    win.move(screen.width() - win.width() - 10, 10)
    win.show()

    controller = SnakeStepperController(SnakeStepperControllerConfig(
        axes={
            "x": SnakeControllerAxisConfig(range=GANTRY_RANGE_X, samples=X_SAMPLES),
            "y": SnakeControllerAxisConfig(range=GANTRY_RANGE_Y, samples=Y_SAMPLES),
        },
    ))
    manager.add(gantry=gantry, gantry_controller=controller)

    output_pkl = logdir / f"{object}_NLOSdata.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    print("\033[92m\nConfirming tag visibility + light conditions\033[0m")
    detections, dark = precapture_validate()
    _check_lighting_and_tags(detections, dark)

    spad_pipeline,     spad_align     = _start_realsense(COLOCATED_RS_SERIAL, *COLOCATED_RES)
    tracking_pipeline, tracking_align = _start_realsense(TRACKING_RS_SERIAL,  *TRACKING_RES)
    manager.add(
        spad_realsense_pipeline=spad_pipeline,
        spad_realsense_align=spad_align,
        tracking_realsense_pipeline=tracking_pipeline,
        tracking_realsense_align=tracking_align,
    )

    metadata = {
        "object":              object,
        "start_time":          NOW.isoformat(),
        "samples_per_capture": SAMPLES_PER,
        "x_samples":           X_SAMPLES,
        "y_samples":           Y_SAMPLES,
        "light_mode":          LIGHT_MODE,
        "spad_res":             str(spad_id.name),
        "spad_range":           str(range_mode.name),
        "capture_mode":         "sequential",
        "realsense_configs": {
            "colocated_realsense": {
                "serial_number": COLOCATED_RS_SERIAL,
                "color_width":   COLOCATED_RES[0], "color_height": COLOCATED_RES[1],
                "depth_width":   COLOCATED_RES[0], "depth_height": COLOCATED_RES[1],
                "fps":           REALSENSE_FPS,
            },
            "tracking_realsense": {
                "serial_number": TRACKING_RS_SERIAL,
                "color_width":   TRACKING_RES[0],  "color_height": TRACKING_RES[1],
                "depth_width":   TRACKING_RES[0],  "depth_height": TRACKING_RES[1],
                "fps":           REALSENSE_FPS,
            },
        },
    }
    pkl_writer.append({"metadata": metadata})


def loop(
    iter:      int,
    manager:   Manager,
    spad:      SPADSensor,
    dashboard: SPADDashboard,
    writer:    PklHandler,
    **kwargs,
) -> bool:
    gantry            = manager.components["gantry"]
    gantry_controller = manager.components["gantry_controller"]
    pos = gantry_controller.get_position(iter, verbose=False)
    if pos is None:
        print("\n=== exiting loop ===")
        return False

    print(f"\n\033[1;32m=== capturing iter {iter+1}/{X_SAMPLES * Y_SAMPLES} ===\033[0m")
    gantry.move_to(pos["x"], pos["y"])
    time.sleep(0.5)

    spad.accumulate()                                                # reset
    data_list    = spad.accumulate(num_samples=SAMPLES_PER)
    stacked_hist = np.stack([d[SPADDataType.HISTOGRAM] for d in data_list])
    dashboard.update(iter, data=data_list[0])
    record: dict = {"iter": iter, "histogram": stacked_hist}

    for prefix in ("spad", "tracking"):
        pipeline = manager.components[f"{prefix}_realsense_pipeline"]
        align    = manager.components[f"{prefix}_realsense_align"]
        aligned  = align.process(pipeline.wait_for_frames())
        record[f"{prefix}_realsense_data"] = {
            "aligned_depth_image": np.asanyarray(aligned.get_depth_frame().get_data()),
            "aligned_rgb_image":   np.asanyarray(aligned.get_color_frame().get_data()),
        }

    writer.append(record)
    get_logger().info(f"Captured and recorded iter {iter}")
    return True


def cleanup(*, gantry: StepperMotorSystem, manager: Manager, **kwargs) -> None:
    gantry.move_to(0, 0)
    gantry.close()
    manager.components["spad_realsense_pipeline"].stop()
    manager.components["tracking_realsense_pipeline"].stop()


@register_cli
def spad_capture(
    dashboard:  SPADDashboardConfig,
    object:     str       = OBJECT_NAME,
    spad_id:    SPADID    = SPAD_ID,
    range_mode: RangeMode = RANGE_MODE,
    logdir:     Path      = LOGDIR,
):
    gantry_config = TelemetrixStepperMotorSystemConfig(
        axes={
            StepperMotorSystemAxis.X: [SingleDrive1AxisGantryXConfig()],
            StepperMotorSystemAxis.Y: [SingleDrive1AxisGantryYConfig()],
        },
        port=GANTRY_PORT,
    )
    gantry = TelemetrixStepperMotorSystem(config=gantry_config)
    with Manager() as manager:
        manager.run(
            setup=partial(setup,
                          dashboard=dashboard, logdir=logdir, object=object,
                          spad_id=spad_id, range_mode=range_mode, gantry=gantry),
            loop=loop,
            cleanup=partial(cleanup, gantry=gantry, manager=manager),
        )
    print(f"\n\033[1;32mAll done. Data saved to {(logdir / f'{object}_NLOSdata.pkl').resolve()}\033[0m\n")


if __name__ == "__main__":
    run_cli(spad_capture)
