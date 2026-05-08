# DENALI capture

Hardware-side code for the DENALI capture rig: a 2D snake-scanned gantry
with a TMF8828 SPAD (LiDAR) and two Intel RealSense cameras — one
co-located with the SPAD and one observing the scene head-on.

## Pipeline

Per session, `capture_spad_nlos.py`:

1. **Validates the rig.** Pulls one frame from the tracking RealSense and
   POSTs it to a local AprilTag detection server
   ([`pkgs/april_tag/scripts/apriltag_server.py`](pkgs/april_tag/scripts/apriltag_server.py)),
   then asserts the required tag IDs are visible (light-on mode) or that
   the room is dark enough (light-off mode).
2. **Initializes hardware.** Brings up the TMF8828 SPAD over the Arduino
   Uno serial port, the Telemetrix gantry over its USB-Serial port, and
   both RealSense pipelines (depth + color, aligned to the color stream).
3. **Snake-scans the gantry.** For each of the 100 (10×10) gantry
   positions, accumulates `SAMPLES_PER` SPAD histograms, pulls one
   aligned RGB+depth frame from each RealSense, and appends the combined
   record to an incremental `<object>_NLOSdata.pkl` log.

## Hardware

| Component                      | Notes                                          |
|--------------------------------|------------------------------------------------|
| ams TMF8828 SPAD               | flashed with [`pkgs/drivers/cc_hardware/drivers/data/tmf8828/tmf8828.ino`](pkgs/drivers/cc_hardware/drivers/data/tmf8828/tmf8828.ino) on an Arduino Uno (default port: `ttyACM0`) |
| 2D linear gantry               | driven by Telemetrix-firmware steppers (default port: USB-Serial) |
| 2× Intel RealSense (D-series)  | one co-located with the SPAD, one tracking the scene |
| AprilTags (tag36h11, 6 cm)     | placed on the relay wall and the gantry frame  |

## Setup

```bash
# Install the cc_hardware sub-packages (drivers / tools / utils) and capture deps.
cd capture/
pip install -e .

# Build and install the AprilTag C library + Python bindings.
cd pkgs/april_tag/
./install.sh
cd -
```

You'll also need:
- `pyrealsense2` (Intel RealSense SDK Python bindings).
- `arduino-cli` if you want to (re)flash the SPAD firmware.

## Configure

The capture script reads the following environment variables (or you can
edit the corresponding constants at the top of
[`capture_spad_nlos.py`](capture_spad_nlos.py)):

| Variable                       | Purpose                                                 |
|--------------------------------|---------------------------------------------------------|
| `DENALI_COLOCATED_SERIAL`      | RealSense serial of the SPAD-side camera                |
| `DENALI_TRACKING_SERIAL`       | RealSense serial of the head-on tracking camera         |
| `APRILTAG_SERVER`              | URL of the AprilTag server (default `http://127.0.0.1:8000`) |
| `APRILTAG_CAMERA_FX/FY/CX/CY`  | tracking-camera intrinsics for the AprilTag server     |
| `APRILTAG_TAG_SIZE_M`          | physical AprilTag side length in metres (default `0.06`) |

Find your RealSense serials with `rs-enumerate-devices` or the Intel
RealSense Viewer.

## Run

In one terminal, start the AprilTag server:

```bash
cd pkgs/april_tag/scripts/
python apriltag_server.py        # listens on 0.0.0.0:8000
```

In another, run a capture (object name encodes object × size × grid × light):

```bash
cd capture/
python capture_spad_nlos.py object=circle_8inch_3x3_lighton
python capture_spad_nlos.py object=test                # dry run, no validation
```

Captures land in `logs/<YYYY-MM-DD>/<HH-MM-SS>/<object>_NLOSdata.pkl`.

## Layout

```
capture/
├── README.md
├── pyproject.toml / poetry.lock
├── docker/                docker-compose recipes for dev / Jetson environments
├── docker-compose.yml
├── capture_spad_nlos.py   capture entry point
└── pkgs/
    ├── nlos/validate_nlos.py     pre-capture AprilTag + light check
    ├── april_tag/                AprilTag C library + Python bindings + FastAPI server
    ├── drivers/cc_hardware/drivers/
    │   ├── spads/                TMF8828 + spad wrappers + base SPAD interface
    │   ├── stepper_motors/       Telemetrix gantry drivers
    │   └── data/tmf8828/         Arduino firmware for the TMF8828 + Uno
    ├── tools/cc_hardware/tools/dashboard/   pyqtgraph SPAD live dashboard
    └── utils/cc_hardware/utils/             logger, manager, registry, file handlers
```
