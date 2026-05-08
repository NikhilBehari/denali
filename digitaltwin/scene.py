"""Mitsuba scene construction and rendering for the digital twin.

Two passes per render:

    * tracking-camera RGB through the calibrated perspective sensor, with
      the full visible scene (tags, mounts, plates, hidden object, tables,
      relay wall);
    * NLOS transient through the co-located camera, run twice (with and
      without the hidden object) so the caller can subtract the no-object
      background.

Set ``DENALI_MITSUBA_VARIANT`` to switch the Mitsuba variant
(default: ``cuda_ad_rgb``; use ``llvm_ad_rgb`` on hosts without a
PTX-capable GPU).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import mitsuba as mi

mi.set_variant(os.environ.get("DENALI_MITSUBA_VARIANT", "cuda_ad_rgb"))

import mitransient as mitr  # noqa: E402

from .calibration import COLOCATED_CAMERA, TRACKING_CAMERA  # noqa: E402


__all__ = ["render_scene_with_transients"]


PACKAGE_DIR  = Path(__file__).resolve().parent
ASSETS_DIR   = (PACKAGE_DIR / ".." / "assets").resolve()
OBJECT_FILES = ASSETS_DIR / "object_files"
TAG_OBJ      = OBJECT_FILES / "tag_plane.obj"
TAG_DIR_BIG  = OBJECT_FILES / "tags" / "tags_1cm_border"
TAG_DIR_WALL = OBJECT_FILES / "tags" / "tags_0.4cm_border"

# 128 bins × ~2.34 cm OPL ≈ 1.5 m short-range LiDAR window.
DEFAULT_BIN_WIDTH_OPL = 3.0 / 128
DEFAULT_START_OPL     = -1.05

OBJECT_REFL_VISUAL    = (0.4, 0.4, 0.4)
OBJECT_REFL_TRANSIENT = (0.5, 0.5, 0.5)
PARTS_REFL            = (0.8, 0.8, 0.8)
TABLE_REFL            = (0.6, 0.5, 0.4)


# ---- helpers ---------------------------------------------------------------

def _to_mi(matrix: np.ndarray) -> mi.ScalarTransform4f:
    return mi.ScalarTransform4f(matrix.astype(np.float32))


def _yup_to_zup() -> np.ndarray:
    a = -np.pi / 2
    return np.array(
        [[1, 0,         0,         0],
         [0, np.cos(a), -np.sin(a), 0],
         [0, np.sin(a),  np.cos(a), 0],
         [0, 0,         0,         1]],
        dtype=np.float32,
    )


def _diffuse(value=None, tex_file: str | None = None) -> dict:
    if tex_file is not None:
        return {"type": "diffuse", "reflectance": {"type": "bitmap", "filename": tex_file}}
    if isinstance(value, (list, tuple, np.ndarray)):
        return {"type": "diffuse",
                "reflectance": {"type": "rgb", "value": [float(v) for v in value]}}
    if value is None:
        return {"type": "diffuse"}
    return {"type": "diffuse", "reflectance": float(value)}


def _obj_shape(filename: str | Path, *, matrix: np.ndarray | None = None,
               bsdf_value=None, tex_file: str | None = None) -> dict:
    s = {"type": "obj", "filename": str(filename),
         "bsdf": _diffuse(bsdf_value, tex_file)}
    if matrix is not None:
        s["to_world"] = _to_mi(matrix)
    return s


def _mat_from_tag(tag: dict) -> np.ndarray:
    R = np.array(tag["rotation_matrix"], dtype=np.float32)
    c = np.array(tag["center"],          dtype=np.float32).reshape(3, 1)
    return np.vstack([np.hstack([R, c]), np.array([0, 0, 0, 1], dtype=np.float32)])


def _camera(cam: dict) -> dict:
    width, height = cam["res"]
    fov_deg = float(np.degrees(2 * np.arctan(width / (2 * cam["intrinsics"]["fx"]))))
    R_wc = np.array(cam["rotation_matrix"], dtype=np.float32).T
    flip = np.diag([-1, -1, 1]).astype(np.float32)
    to_world = np.eye(4, dtype=np.float32)
    to_world[:3, :3] = R_wc @ flip
    to_world[:3, 3]  = cam["position"]
    return {
        "type": "perspective",
        "film": {"type": "hdrfilm", "width": int(width), "height": int(height)},
        "fov": fov_deg,
        "to_world": _to_mi(to_world),
    }


def _projector(cam_dict: dict, *, irradiance: float = 100.0, fov: float = 0.2) -> dict:
    return {"type": "projector", "irradiance": irradiance, "fov": fov,
            "to_world": cam_dict["to_world"]}


# ---- shapes ----------------------------------------------------------------

def _tag(tag_id: str, pose: np.ndarray, tex_dir: Path) -> dict:
    scale = 0.068 / 0.08 if 5 <= int(tag_id) <= 7 else 1.0
    S = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
    aligned = pose @ _yup_to_zup() @ S
    tex = str(tex_dir / f"tag36_11_{int(tag_id):05d}.png")
    return _obj_shape(TAG_OBJ, matrix=aligned, tex_file=tex)


def _table(tag_pose: np.ndarray, *, tag_id: str,
           size_xy=(0.78, 1.524),
           x_offset: float = 0.0, y_offset: float = 0.0,
           z_offset: float = 0.001) -> dict:
    x_dir = tag_pose[:3, 0]; y_dir = tag_pose[:3, 1]
    z_dir = -np.cross(x_dir, y_dir)
    pos = tag_pose[:3, 3] - z_offset * z_dir
    if   tag_id == "14": pos += 0.5 * size_xy[0] * x_dir - 0.5 * size_xy[1] * y_dir
    elif tag_id == "15": pos -= 0.5 * size_xy[0] * x_dir + 0.5 * size_xy[1] * y_dir
    pos += x_offset * x_dir + y_offset * y_dir
    R = np.eye(4, dtype=np.float32); R[:3, 0], R[:3, 1], R[:3, 2] = x_dir, y_dir, z_dir
    S = np.diag([size_xy[0] / 2, size_xy[1] / 2, 1, 1]).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
    return {"type": "rectangle", "to_world": _to_mi(T @ R @ S),
            "bsdf": _diffuse(TABLE_REFL)}


def _mount_assembly(tag_id: str, pose: np.ndarray, *,
                    plate_file: Path, mount_file: Path,
                    mount_offset: tuple[float, float, float],
                    main_object_file: Path | None = None) -> dict[str, dict]:
    rotated = pose @ _yup_to_zup()
    mount = rotated @ np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, mount_offset[1]],
         [0, 0, 1, mount_offset[2]],
         [0, 0, 0, 1]], dtype=np.float32,
    )
    tex = str(TAG_DIR_BIG / f"tag36_11_{int(tag_id):05d}.png")
    out = {
        f"mount_{tag_id}_tag":   _obj_shape(TAG_OBJ,    matrix=rotated, tex_file=tex),
        f"mount_{tag_id}_plate": _obj_shape(plate_file, matrix=rotated, bsdf_value=PARTS_REFL),
        f"mount_{tag_id}_mount": _obj_shape(mount_file, matrix=mount,   bsdf_value=PARTS_REFL),
    }
    if main_object_file is not None:
        out[f"mount_{tag_id}_object"] = _obj_shape(
            main_object_file, matrix=mount, bsdf_value=OBJECT_REFL_VISUAL,
        )
    return out


def _backdrop() -> dict:
    S = np.diag([5.0, 5.0, 1.0, 1.0]).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, 3] = (0.762, 0.4, -1.0)
    return {"type": "rectangle", "to_world": _to_mi(T @ S),
            "bsdf": _diffuse((0.0, 0.0, 0.0))}


def _visual_relay(tag_data: dict, *, width_x: float = 0.93,
                  height_z: float = 0.6096, z_offset: float = 0.001) -> dict:
    """Plain visual relay-wall rectangle (no NLOS sensor)."""
    tag5 = _mat_from_tag(tag_data["5"])
    x_dir = tag5[:3, 0]; y_dir = tag5[:3, 1]
    z_dir = -np.cross(x_dir, y_dir)
    pos = (
        tag5[:3, 3]
        - z_offset * z_dir
        + (width_x  / 2 - 0.068 / 2)         * x_dir
        + (height_z / 2 - 0.068 / 2 - 0.186) * y_dir
    )
    R = np.eye(4, dtype=np.float32); R[:3, 0], R[:3, 1], R[:3, 2] = x_dir, y_dir, z_dir
    S = np.diag([width_x / 2, height_z / 2, 1, 1]).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
    return {"type": "rectangle", "to_world": _to_mi(T @ R @ S),
            "bsdf": _diffuse(PARTS_REFL)}


def _nlos_relay(cam: dict, tag_data: dict, *,
                film_size: int = 64, temporal_bins: int = 128,
                bin_width_opl: float = DEFAULT_BIN_WIDTH_OPL,
                start_opl: float = DEFAULT_START_OPL,
                sample_count: int = 25_000) -> dict:
    """Relay-wall rectangle hosting an `nlos_capture_meter`."""
    width_x, height_z = 0.93, 0.6096
    tag5  = _mat_from_tag(tag_data["5"])
    x_dir = tag5[:3, 0] / np.linalg.norm(tag5[:3, 0])
    y_dir = tag5[:3, 1] / np.linalg.norm(tag5[:3, 1])
    center = (
        tag5[:3, 3]
        + (width_x  / 2 - 0.068 / 2)         * x_dir
        + (height_z / 2 - 0.068 / 2 - 0.186) * y_dir
    )
    R_face = np.array([[1, 0,  0, 0],
                       [0, 0, -1, 0],
                       [0, 1,  0, 0],
                       [0, 0,  0, 1]], dtype=np.float32)
    n_base, x_base = R_face[:3, 2], R_face[:3, 0]
    v = x_dir - np.dot(x_dir, n_base) * n_base
    if np.linalg.norm(v) < 1e-8:
        R_inplane = np.eye(4, dtype=np.float32)
    else:
        v = v / np.linalg.norm(v)
        c = float(np.clip(np.dot(x_base, v), -1, 1))
        s = float(np.dot(n_base, np.cross(x_base, v)))
        K = np.array([[0, -n_base[2],  n_base[1]],
                      [n_base[2],  0, -n_base[0]],
                      [-n_base[1], n_base[0], 0]], dtype=np.float32)
        R3 = c * np.eye(3) + (1 - c) * np.outer(n_base, n_base) + s * K
        R_inplane = np.eye(4, dtype=np.float32); R_inplane[:3, :3] = R3
    S = np.diag([width_x / 2, height_z / 2, 1, 1]).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, 3] = center
    return {
        "type": "rectangle",
        "to_world": _to_mi(T @ (R_inplane @ R_face) @ S),
        "bsdf": {"type": "diffuse", "reflectance": 1.0},
        "nlos_sensor": {
            "type": "nlos_capture_meter",
            "sampler": {"type": "independent", "sample_count": int(sample_count)},
            "account_first_and_last_bounces": False,
            "sensor_origin": mi.ScalarPoint3f(cam["position"]),
            "transient_film": {
                "type": "transient_hdr_film",
                "width":  int(film_size),
                "height": int(film_size),
                "temporal_bins": int(temporal_bins),
                "bin_width_opl": float(bin_width_opl),
                "start_opl":     float(start_opl),
                "rfilter": {"type": "box"},
            },
        },
    }


# ---- scene assembly --------------------------------------------------------

def _visual_scene(tag_data: dict, tag1_pose: dict | None,
                  obj_file: Path, cam_dict: dict) -> dict:
    scene = {
        "type": "scene",
        "integrator":       {"type": "path"},
        "sensor":           cam_dict,
        "emitter_constant": {"type": "constant",
                             "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}},
        "backdrop":   _backdrop(),
        "table_14":   _table(_mat_from_tag(tag_data["14"]), tag_id="14",
                             x_offset=-0.04, y_offset=0.548),
        "table_15":   _table(_mat_from_tag(tag_data["15"]), tag_id="15",
                             x_offset= 0.04, y_offset=0.548),
        "relay_wall": _visual_relay(tag_data),
    }
    for tag_id, tag_info in tag_data.items():
        tex_dir = TAG_DIR_WALL if 5 <= int(tag_id) <= 7 else TAG_DIR_BIG
        scene[f"tag_{tag_id}"] = _tag(tag_id, _mat_from_tag(tag_info), tex_dir)
    if "0" in tag_data:
        scene.update(_mount_assembly(
            "0", _mat_from_tag(tag_data["0"]),
            plate_file=OBJECT_FILES / "tmf_mount_plate.obj",
            mount_file=OBJECT_FILES / "tmf_mount.obj",
            mount_offset=(0, -0.005, -0.0325),
        ))
    if tag1_pose is not None:
        scene.update(_mount_assembly(
            "1", _mat_from_tag(tag1_pose),
            plate_file=OBJECT_FILES / "object_mount_plate.obj",
            mount_file=OBJECT_FILES / "object_mount.obj",
            mount_offset=(0, -0.005, -0.04),
            main_object_file=obj_file,
        ))
    return scene


def _transient_geometry(tag_data: dict, tag1_pose: dict | None,
                        obj_file: Path) -> dict:
    geo: dict = {
        "table_14": _table(_mat_from_tag(tag_data["14"]), tag_id="14",
                           x_offset=-0.04, y_offset=0.548),
        "table_15": _table(_mat_from_tag(tag_data["15"]), tag_id="15",
                           x_offset= 0.04, y_offset=0.548),
    }
    if tag1_pose is not None:
        rotated = _mat_from_tag(tag1_pose) @ _yup_to_zup()
        mount = rotated @ np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, -0.005],
             [0, 0, 1, -0.04],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        geo["mounted_obj"] = _obj_shape(obj_file, matrix=mount,
                                        bsdf_value=OBJECT_REFL_TRANSIENT)
    return geo


def _render_transient(geo: dict, tag_data: dict, *,
                      film_size: int = 64,
                      bin_width_opl: float = DEFAULT_BIN_WIDTH_OPL,
                      start_opl:     float = DEFAULT_START_OPL) -> np.ndarray:
    emitter = mi.load_dict(_projector(_camera(COLOCATED_CAMERA)))
    relay   = mi.load_dict(_nlos_relay(
        COLOCATED_CAMERA, tag_data, film_size=film_size,
        bin_width_opl=bin_width_opl, start_opl=start_opl,
    ))
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {
            "type": "transient_nlos_path",
            "nlos_laser_sampling": True,
            "nlos_hidden_geometry_sampling": True,
            "nlos_hidden_geometry_sampling_do_rroulette": False,
            "temporal_filter": "box",
        },
        "emitter":    emitter,
        "relay_wall": relay,
        **geo,
    })
    mitr.nlos.focus_emitter_at_relay_wall_pixel(
        mi.Point2f(film_size / 2, film_size / 2), relay, emitter,
    )
    _, transient = mi.render(scene)
    return np.array(transient)


# ---- public API ------------------------------------------------------------

def render_scene_with_transients(
    obj_file:      str | Path,
    location:      int | str,
    poses_path:    str | Path,
    *,
    bin_width_opl: float = DEFAULT_BIN_WIDTH_OPL,
    start_opl:     float = DEFAULT_START_OPL,
) -> dict:
    """Render the tracking RGB and the with/without-object NLOS transients.

    Args:
        obj_file:    path to the hidden-object ``.obj``.
        location:    gantry index 0..99 (key into ``poses.json``).
        poses_path:  path to the bundled ``poses.json``.
        bin_width_opl, start_opl: temporal discretization of the NLOS film.

    Returns a dict with ``rgb_image`` (HxWx3 uint8) and the two transient
    cubes ``transient_full_obj`` / ``transient_full_bg``.
    """
    with open(poses_path) as f:
        poses = json.load(f)
    tag_data  = poses["tracking_avgposes"]
    tag1_pose = poses.get("tracking_tag1_index_avgposes", {}).get(str(location))
    obj_path  = Path(obj_file)

    visual = _visual_scene(tag_data, tag1_pose, obj_path, _camera(TRACKING_CAMERA))
    img    = mi.render(mi.load_dict(visual), spp=64)
    rgb    = (np.clip(img.numpy(), 0.0, 1.0) * 255).astype(np.uint8)

    transient_obj = _render_transient(
        _transient_geometry(tag_data, tag1_pose, obj_path), tag_data,
        bin_width_opl=bin_width_opl, start_opl=start_opl,
    )
    transient_bg = _render_transient(
        _transient_geometry(tag_data, None, obj_path), tag_data,
        bin_width_opl=bin_width_opl, start_opl=start_opl,
    )
    return {
        "rgb_image":          rgb,
        "transient_full_obj": transient_obj,
        "transient_full_bg":  transient_bg,
    }
