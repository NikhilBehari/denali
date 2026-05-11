"""Dash web app for the DENALI live-inference GUI.

Reads the 3x3 SPAD captures and tracking-camera RGB frames from
``denali_public/denali-data/data/``, runs the 1D-CNN inference heads
in ``gui/checkpoints/``, and projects the predictions onto the scene
image using the camera calibration in ``calibration.py``.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dash import Dash, Input, Output, State, dcc, html, ctx
from flask import abort, send_file
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .predictions import predict
from .calibration import project_world_to_pixel


# --------------------------------------------------------------------------- #
# Paths / manifest
# --------------------------------------------------------------------------- #

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR  = (PACKAGE_DIR / ".." / "assets").resolve()
MANIFEST    = ASSETS_DIR / "manifest.json"

with open(MANIFEST) as f:
    MAN = json.load(f)

OBJECTS: list[str] = MAN["objects"]
SIZES: list[str] = MAN["sizes"]
LIGHTS: list[str] = MAN["lights"]
SCENE_INDEX: dict[tuple[str, str, str], dict] = {
    (s["object"], s["size"], s["light"]): s for s in MAN["scenes"]
}
SCENE_BY_ID: dict[str, dict] = {s["id"]: s for s in MAN["scenes"]}

# Tag1 ground-truth positions ship inside the manifest (frame index -> [x, y, z]).
TAG1_POSITIONS: dict[str, list[float]] = MAN["tag1_positions"]

# Resolved at startup by `main()` -> `<denali_public>/denali-data/data/` (or --data-dir).
DATA_ROOT: Path = Path()


def _data_folder(scene: dict) -> Path:
    """Translate a manifest scene record to the on-disk capture folder.

    Manifest IDs look like ``<obj>_<size>_<light>``; the data folders bundle
    the ``_3x3_..._NLOSdata`` suffix used by `denali-data/data/`.
    """
    return DATA_ROOT / f"{scene['object']}_{scene['size']}_3x3_{scene['light']}_NLOSdata"


def gt_xyz_for(frame: int) -> tuple[float, float, float]:
    return tuple(TAG1_POSITIONS[f"frame{frame}_tag1"])


def gt_xy_for(frame: int) -> tuple[float, float]:
    p = TAG1_POSITIONS[f"frame{frame}_tag1"]
    return float(p[0]), float(p[1])


# Pre-build the full 100-point world grid (for the bottom panel).
_GRID_XY = np.array(
    [TAG1_POSITIONS[f"frame{i}_tag1"][:2] for i in range(100)], dtype=float
)


def pretty_obj(o: str) -> str:
    if o == "NOOBJECT":
        return "(no object)"
    if o == "NOOBJECTMOUNT":
        return "(no object — no mount)"
    return o


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
# Two named palettes drive both the CSS (via custom properties at runtime) and
# the Plotly figures (via the palette() helper below). Add new themes by
# extending PALETTES; everything else picks them up automatically.

DEFAULT_THEME = "light"

PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg":           "#0e1116",
        "panel":        "#161a22",
        "panel_alt":    "#1e242e",
        "accent":       "#5cc8ff",
        "accent_dim":   "#3a7fa6",
        "gt":           "#6cf28b",
        "pred":         "#ff6b6b",
        "text":         "#e6edf3",
        "text_dim":     "#8b95a5",
        "grid":         "#2a313c",
        "fill_accent":  "rgba(92, 200, 255, 0.45)",
        "info_bg":      "rgba(92, 200, 255, 0.06)",
        "info_border":  "rgba(92, 200, 255, 0.30)",
        "legend_bg":    "rgba(30, 36, 46, 0.85)",
        "scene_img_bg": "#000000",
        "shadow":       "0 1px 0 rgba(0,0,0,0.30)",
    },
    "light": {
        # Off-white workspace, white panels, deep blue accent, semantic green/red.
        "bg":           "#f6f7fb",
        "panel":        "#ffffff",
        "panel_alt":    "#f1f3f7",
        "accent":       "#2563eb",
        "accent_dim":   "#93c5fd",
        "gt":           "#16a34a",
        "pred":         "#dc2626",
        "text":         "#0f172a",
        "text_dim":     "#64748b",
        "grid":         "#e2e8f0",
        "fill_accent":  "rgba(37, 99, 235, 0.18)",
        "info_bg":      "rgba(37, 99, 235, 0.06)",
        "info_border":  "rgba(37, 99, 235, 0.25)",
        "legend_bg":    "rgba(255, 255, 255, 0.92)",
        "scene_img_bg": "#e2e8f0",
        "shadow":       "0 1px 2px rgba(15, 23, 42, 0.06), 0 1px 0 rgba(15, 23, 42, 0.04)",
    },
}


def palette(theme: str | None) -> dict[str, str]:
    """Resolve a palette dict, falling back to DEFAULT_THEME."""
    return PALETTES.get(theme or DEFAULT_THEME, PALETTES[DEFAULT_THEME])


def plot_layout_base(theme: str | None) -> dict:
    pal = palette(theme)
    return dict(
        paper_bgcolor=pal["panel"],
        plot_bgcolor=pal["panel"],
        font=dict(
            color=pal["text_dim"],
            family="ui-monospace, SF Mono, Menlo, monospace",
            size=11,
        ),
        margin=dict(l=44, r=14, t=20, b=36),
    )


# --------------------------------------------------------------------------- #
# Histogram caching
# --------------------------------------------------------------------------- #

_HIST_CACHE: dict[tuple[str, int], np.ndarray] = {}


def load_histogram(scene_id: str, frame: int) -> np.ndarray:
    """Load 3x3x128 averaged histogram (averaged over capture samples)."""
    key = (scene_id, frame)
    if key in _HIST_CACHE:
        return _HIST_CACHE[key]
    path = _data_folder(SCENE_BY_ID[scene_id]) / "spad_histogram" / f"{frame:03d}.npy"
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr.mean(axis=0)
    _HIST_CACHE[key] = arr
    return arr


def scene_id_for(obj: str, size: str, light: str) -> str | None:
    s = SCENE_INDEX.get((obj, size, light))
    return s["id"] if s is not None else None


# --------------------------------------------------------------------------- #
# Image serving
# --------------------------------------------------------------------------- #

_GT_RGB = (108, 242, 139)
_PRED_RGB = (255, 107, 107)


def _draw_marker(draw: ImageDraw.ImageDraw, cx: int, cy: int, color, *,
                 ring_r: int, dot_r: int, fill_alpha: int = 60,
                 outline_w: int = 3) -> None:
    draw.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=color + (230,), width=outline_w,
    )
    fr = max(1, int(ring_r * 0.7))
    draw.ellipse(
        [cx - fr, cy - fr, cx + fr, cy + fr],
        fill=color + (fill_alpha,),
    )
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=color + (255,),
    )


def overlay_markers(
    img_path: Path,
    *,
    gt_pixel: tuple[float, float] | None,
    pred_pixel: tuple[float, float] | None,
    target_w: int = 960,
) -> bytes:
    """Render a downsampled tracking-RGB JPEG with GT (green) and prediction
    (red) markers drawn at the *projected* pixel locations.

    `gt_pixel` and `pred_pixel` are in the original 1280x720 image space; this
    function rescales them to match the downsampled output.
    """
    im = Image.open(img_path).convert("RGB")
    src_w = im.width
    src_h = im.height

    if im.width > target_w:
        ratio = target_w / im.width
        im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
    sx = im.width / src_w
    sy = im.height / src_h

    draw = ImageDraw.Draw(im, "RGBA")

    if gt_pixel is not None:
        gx = int(gt_pixel[0] * sx)
        gy = int(gt_pixel[1] * sy)
        # Crosshairs only on the GT (less visual noise)
        draw.line([(gx, 0), (gx, im.height)], fill=_GT_RGB + (110,), width=1)
        draw.line([(0, gy), (im.width, gy)], fill=_GT_RGB + (110,), width=1)
        _draw_marker(draw, gx, gy, _GT_RGB, ring_r=26, dot_r=6)

    if pred_pixel is not None:
        px = int(pred_pixel[0] * sx)
        py = int(pred_pixel[1] * sy)
        _draw_marker(draw, px, py, _PRED_RGB, ring_r=20, dot_r=5,
                     fill_alpha=40, outline_w=2)
        # Connector between GT and pred
        if gt_pixel is not None:
            draw.line(
                [(int(gt_pixel[0] * sx), int(gt_pixel[1] * sy)), (px, py)],
                fill=_PRED_RGB + (180,), width=2,
            )

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Plotly figure builders
# --------------------------------------------------------------------------- #

def fig_histograms(hist: np.ndarray, y_max: float, theme: str | None = None) -> go.Figure:
    """3x3 grid of bin histograms sharing a y-axis limit."""
    pal = palette(theme)
    n_bins = hist.shape[-1]
    x = np.arange(n_bins)

    fig = make_subplots(
        rows=3, cols=3,
        shared_yaxes=True, shared_xaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.04,
    )

    for r in range(3):
        for c in range(3):
            y = hist[r, c]
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines",
                    fill="tozeroy",
                    line=dict(color=pal["accent"], width=1.0),
                    fillcolor=pal["fill_accent"],
                    hovertemplate=(
                        f"cell ({r},{c})<br>bin %{{x}}<br>counts %{{y:,}}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )
            # cell label
            fig.add_annotation(
                xref=f"x{(r * 3 + c + 1) if (r * 3 + c + 1) > 1 else ''} domain",
                yref=f"y{(r * 3 + c + 1) if (r * 3 + c + 1) > 1 else ''} domain",
                x=0.04, y=0.92, text=f"({r},{c})",
                showarrow=False, font=dict(color=pal["text_dim"], size=10),
            )
    fig.update_xaxes(
        showgrid=False, zeroline=False, showline=False,
        tickcolor=pal["grid"], tickfont=dict(color=pal["text_dim"], size=9),
        range=[0, n_bins - 1],
    )
    fig.update_yaxes(
        gridcolor=pal["grid"], zeroline=False, showline=False,
        tickcolor=pal["grid"], tickfont=dict(color=pal["text_dim"], size=9),
        range=[0, y_max],
    )
    layout = {**plot_layout_base(theme),
              "height": 270,
              "margin": dict(l=42, r=10, t=10, b=20)}
    fig.update_layout(**layout)
    return fig


def fig_location(pred, theme: str | None = None) -> go.Figure:
    pal = palette(theme)
    fig = go.Figure()

    # 100 GT capture grid points (in world meters) as a faded background.
    fig.add_trace(go.Scatter(
        x=_GRID_XY[:, 0], y=_GRID_XY[:, 1],
        mode="markers",
        marker=dict(color=pal["grid"], size=6, line=dict(width=0)),
        name="capture grid",
        hoverinfo="skip",
    ))
    # GT (open green ring)
    fig.add_trace(go.Scatter(
        x=[pred.gt_xy[0]], y=[pred.gt_xy[1]],
        mode="markers",
        marker=dict(symbol="circle-open", color=pal["gt"], size=24,
                    line=dict(width=2.6)),
        name="ground truth",
    ))
    # Predicted (red ×)
    fig.add_trace(go.Scatter(
        x=[pred.pred_xy[0]], y=[pred.pred_xy[1]],
        mode="markers",
        marker=dict(symbol="x-thin", color=pal["pred"], size=18, line=dict(width=3)),
        name="prediction",
    ))
    # Connector
    fig.add_trace(go.Scatter(
        x=[pred.gt_xy[0], pred.pred_xy[0]],
        y=[pred.gt_xy[1], pred.pred_xy[1]],
        mode="lines",
        line=dict(color=pal["pred"], dash="dot", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))

    # Bound the axes to the grid extent + a margin.
    pad = 0.06
    x_lo, x_hi = float(_GRID_XY[:, 0].min()) - pad, float(_GRID_XY[:, 0].max()) + pad
    y_lo, y_hi = float(_GRID_XY[:, 1].min()) - pad, float(_GRID_XY[:, 1].max()) + pad

    fig.update_xaxes(
        range=[x_lo, x_hi], gridcolor=pal["grid"], zeroline=False,
        title=dict(text="x  (world m)", font=dict(color=pal["text_dim"], size=11)),
        tickfont=dict(color=pal["text_dim"], size=9),
    )
    fig.update_yaxes(
        # World-y points up in the scene image, so y_lo at the bottom keeps
        # the 2D layout aligned with the gantry's visual motion.
        range=[y_lo, y_hi],
        gridcolor=pal["grid"], zeroline=False,
        title=dict(text="y  (world m)", font=dict(color=pal["text_dim"], size=11)),
        tickfont=dict(color=pal["text_dim"], size=9),
        scaleanchor="x", scaleratio=1,
    )

    fig.update_layout(
        **plot_layout_base(theme),
        height=320,
        showlegend=True,
        legend=dict(
            x=1, y=0, xanchor="right", yanchor="bottom",
            bgcolor=pal["legend_bg"], bordercolor=pal["grid"], borderwidth=1,
            font=dict(color=pal["text"], size=10),
        ),
        annotations=[
            dict(
                xref="paper", yref="paper", x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                text=f"err  <b>{pred.err_m * 100:0.1f}</b>  cm",
                showarrow=False, font=dict(color=pal["text"], size=12),
                bgcolor=pal["panel_alt"], bordercolor=pal["grid"], borderwidth=1,
                borderpad=6,
            ),
        ],
    )
    return fig


def fig_topk_classes(pred, k: int = 5, theme: str | None = None) -> go.Figure:
    pal = palette(theme)
    order = np.argsort(pred.object_probs)[::-1][:k]
    labels = [pretty_obj(pred.object_classes[i]) for i in order]
    probs = [float(pred.object_probs[i]) for i in order]
    colors = [pal["gt"] if i == pred.object_gt_idx else pal["accent_dim"] for i in order]

    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.2f}" for p in probs],
        textposition="outside",
        textfont=dict(color=pal["text_dim"], size=11),
        hovertemplate="%{y}<br>p=%{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(
        range=[0, max(0.65, max(probs) * 1.18)],
        gridcolor=pal["grid"], zeroline=False,
        title=dict(text="probability", font=dict(color=pal["text_dim"], size=11)),
        tickfont=dict(color=pal["text_dim"], size=9),
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False, zeroline=False,
        tickfont=dict(color=pal["text"], size=11),
    )
    # GT badge
    if pred.object_gt_idx in order:
        badge = "GT in top-5  ✓"
        col = pal["gt"]
    elif pred.object_gt_idx >= 0:
        rank = int((pred.object_probs > pred.object_probs[pred.object_gt_idx]).sum() + 1)
        badge = f"GT @ rank {rank}"
        col = pal["pred"]
    else:
        badge = ""
        col = pal["text_dim"]
    layout = {**plot_layout_base(theme),
              "height": 320,
              "margin": dict(l=88, r=24, t=20, b=36),
              "annotations": [
                  dict(
                      xref="paper", yref="paper", x=0.99, y=1.04,
                      xanchor="right", yanchor="bottom",
                      text=f"<b>{badge}</b>",
                      showarrow=False, font=dict(color=col, size=11),
                  ),
              ]}
    fig.update_layout(**layout)
    return fig


def fig_size(pred, theme: str | None = None) -> go.Figure:
    pal = palette(theme)
    labels = [s.replace("inch", '"') for s in pred.size_classes]
    colors = [pal["gt"] if i == pred.size_gt_idx else pal["accent_dim"]
              for i in range(len(pred.size_classes))]

    fig = go.Figure(go.Bar(
        x=labels, y=pred.size_probs,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.2f}" for p in pred.size_probs],
        textposition="outside",
        textfont=dict(color=pal["text_dim"], size=11),
        width=0.55,
        hovertemplate="%{x}<br>p=%{y:.3f}<extra></extra>",
    ))
    fig.update_xaxes(showgrid=False, zeroline=False,
                     tickfont=dict(color=pal["text"], size=12))
    fig.update_yaxes(range=[0, 1.1], gridcolor=pal["grid"], zeroline=False,
                     title=dict(text="probability", font=dict(color=pal["text_dim"], size=11)),
                     tickfont=dict(color=pal["text_dim"], size=9))
    fig.update_layout(**plot_layout_base(theme), height=320)
    return fig


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = Dash(
    __name__,
    title="DENALI — NLOS Tracking GUI",
    update_title=None,
    suppress_callback_exceptions=True,
)

def _vars_block(theme_name: str) -> str:
    pal = PALETTES[theme_name]
    return f""":root[data-theme="{theme_name}"] {{
    --bg:            {pal["bg"]};
    --panel:         {pal["panel"]};
    --panel-alt:     {pal["panel_alt"]};
    --accent:        {pal["accent"]};
    --accent-dim:    {pal["accent_dim"]};
    --gt:            {pal["gt"]};
    --pred:          {pal["pred"]};
    --text:          {pal["text"]};
    --text-dim:      {pal["text_dim"]};
    --grid:          {pal["grid"]};
    --info-bg:       {pal["info_bg"]};
    --info-border:   {pal["info_border"]};
    --scene-img-bg:  {pal["scene_img_bg"]};
    --shadow:        {pal["shadow"]};
}}"""


# Inline base CSS — keeps the GUI self-contained (no external CSS needed). All
# colors come from CSS custom properties on :root[data-theme=...] so a single
# attribute swap re-themes the whole UI without re-rendering Dash components.
INLINE_CSS = "\n".join(_vars_block(name) for name in PALETTES) + """

html, body {
    background: var(--bg);
    color: var(--text);
    margin: 0;
    font-family: ui-sans-serif, -apple-system, "SF Pro", Inter, "Segoe UI", system-ui;
    -webkit-font-smoothing: antialiased;
    transition: background-color 0.18s ease, color 0.18s ease;
}
.container {
    padding: 18px 22px 24px;
    max-width: 1480px;
    margin: 0 auto;
}
.header {
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 14px;
}
.h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.02em; }
.subtitle { color: var(--text-dim); font-size: 13px; }
.header-right {
    margin-left: auto;
    display: flex; align-items: center; gap: 12px;
}
.scene-id {
    color: var(--accent);
    font-family: ui-monospace, SF Mono, Menlo, monospace;
    font-size: 12px;
}
/* Theme toggle */
.theme-toggle {
    background: var(--panel);
    color: var(--text-dim);
    border: 1px solid var(--grid);
    border-radius: 999px;
    padding: 4px 4px;
    display: inline-flex; align-items: center; gap: 2px;
    cursor: pointer;
    font-size: 12px;
    box-shadow: var(--shadow);
    transition: all 0.15s ease;
}
.theme-toggle:hover { border-color: var(--accent-dim); }
.theme-toggle .opt {
    padding: 4px 10px;
    border-radius: 999px;
    color: var(--text-dim);
    transition: all 0.15s ease;
    line-height: 1;
}
.theme-toggle .opt.active {
    background: var(--accent);
    color: #ffffff;
}
:root[data-theme="dark"] .theme-toggle .opt.active { color: #0e1116; }
.row { display: flex; gap: 14px; }
.col { display: flex; flex-direction: column; gap: 14px; }
.card {
    background: var(--panel);
    border: 1px solid var(--grid);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
    transition: background-color 0.18s ease, border-color 0.18s ease;
}
.card-title {
    color: var(--text-dim);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin: 0 0 10px 0;
}
.scene-img {
    width: 100%; height: auto; display: block; border-radius: 6px;
    background: var(--scene-img-bg);
}
.legend { display: flex; gap: 18px; align-items: center; padding-top: 8px; color: var(--text-dim); font-size: 12px; }
.legend .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.frame-tag { margin-left: auto; color: var(--accent); font-family: ui-monospace, monospace; font-size: 12px; }
.controls { display: grid; grid-template-columns: 70px 1fr; row-gap: 10px; column-gap: 10px; align-items: center; }
.controls label { color: var(--text-dim); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
.btn-row { display: flex; gap: 6px; }
.btn {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--grid);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.12s ease;
}
.btn:hover:not(:disabled) { border-color: var(--accent-dim); color: var(--accent); }
.btn.active { background: var(--accent-dim); border-color: var(--accent); color: var(--text); }
:root[data-theme="light"] .btn.active { background: var(--accent); color: #ffffff; border-color: var(--accent); }
.btn:disabled { opacity: 0.35; cursor: not-allowed; }
.frame-nav { display: flex; align-items: center; gap: 8px; }
.frame-nav .btn { padding: 4px 10px; min-width: 32px; }
.frame-nav .frame-label { font-family: ui-monospace, monospace; font-size: 12px; color: var(--accent); min-width: 80px; }
.num-input {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--grid);
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 13px;
    font-family: ui-monospace, SF Mono, Menlo, monospace;
    width: 86px;
    text-align: right;
    -moz-appearance: textfield;
}
.num-input:focus { outline: none; border-color: var(--accent); }
.num-input::-webkit-inner-spin-button, .num-input::-webkit-outer-spin-button {
    opacity: 1; height: 22px;
}
.frame-nav .num-input { width: 64px; }
.frame-nav .frame-suffix { color: var(--text-dim); font-family: ui-monospace, monospace; font-size: 12px; }
.ymax-row { display: flex; align-items: center; gap: 10px; padding-top: 6px; }
.ymax-row label { color: var(--text-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
.ymax-row .hint { color: var(--text-dim); font-size: 11px; margin-left: auto; }
.info {
    background: var(--info-bg);
    border: 1px solid var(--info-border);
    border-radius: 6px;
    padding: 6px 10px;
    color: var(--accent);
    font-size: 11px;
    margin-bottom: 12px;
    letter-spacing: 0.02em;
}
/* Style the dropdown */
.Select-control, .Select-menu-outer, .Select-value, .Select-input,
.dash-dropdown .Select-control {
    background-color: var(--panel-alt) !important;
    color: var(--text) !important;
    border-color: var(--grid) !important;
}
.Select-value-label, .Select-placeholder { color: var(--text) !important; }
.Select-menu-outer { border-color: var(--grid) !important; }
.Select-option { background-color: var(--panel-alt) !important; color: var(--text) !important; }
.Select-option.is-focused { background-color: var(--accent-dim) !important; color: var(--text) !important; }
/* Slider */
.rc-slider-track { background-color: var(--accent-dim) !important; }
.rc-slider-rail { background-color: var(--grid) !important; }
.rc-slider-handle { border-color: var(--accent) !important; background: var(--accent) !important; }
.rc-slider-dot { display: none; }
.rc-slider-mark-text { color: var(--text-dim) !important; font-size: 10px !important; }
"""

app.index_string = f"""
<!DOCTYPE html>
<html data-theme="{DEFAULT_THEME}">
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{INLINE_CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </footer>
</body>
</html>
"""


def _btn(label, btn_id, active=False, disabled=False):
    cls = "btn active" if active else "btn"
    return html.Button(label, id=btn_id, n_clicks=0, className=cls, disabled=disabled)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

initial_state = dict(obj="8", size="8inch", light="lighton", frame=50, y_max=8000)


def _theme_toggle(theme: str) -> html.Div:
    """A simple two-segment pill toggle for light/dark."""
    light_cls = "opt active" if theme == "light" else "opt"
    dark_cls = "opt active" if theme == "dark" else "opt"
    return html.Div(
        id="theme-toggle",
        className="theme-toggle",
        n_clicks=0,
        title="Toggle light / dark theme",
        children=[
            html.Span("Light", className=light_cls, id="theme-toggle-light"),
            html.Span("Dark",  className=dark_cls,  id="theme-toggle-dark"),
        ],
    )


def serve_layout():
    return html.Div(className="container", children=[
        # ---- Stores ---- #
        dcc.Store(id="state-store", data=initial_state),
        dcc.Store(id="theme-store", data=DEFAULT_THEME),

        # ---- Header ---- #
        html.Div(className="header", children=[
            html.Div("DENALI", className="h1"),
            html.Div("Non-line-of-sight tracking · 3×3 SPAD GUI",
                     className="subtitle"),
            html.Div(className="header-right", children=[
                html.Div(id="scene-id", className="scene-id"),
                _theme_toggle(DEFAULT_THEME),
            ]),
        ]),
        html.Div(
            "Predictions are run live by the 1D-CNN heads in "
            "gui/checkpoints/ (object · size · 2D location). Markers on the "
            "scene image are projected from world meters with the "
            "tracking-camera calibration.",
            className="info",
        ),

        # ---- TOP ROW ---- #
        html.Div(className="row", style={"alignItems": "stretch"}, children=[

            # LEFT: scene image
            html.Div(className="card", style={"flex": "5", "minWidth": "0"}, children=[
                html.Div("Scene · Tracking camera", className="card-title"),
                html.Img(id="tracking-img", className="scene-img", src=""),
                html.Div(className="legend", children=[
                    html.Span([html.Span(className="dot",
                                         style={"background": "var(--gt)"}),
                               "Ground truth"]),
                    html.Span([html.Span(className="dot",
                                         style={"background": "var(--pred)"}),
                               "Prediction"]),
                    html.Div(id="frame-tag", className="frame-tag"),
                ]),
            ]),

            # RIGHT: controls + histograms
            html.Div(className="col", style={"flex": "4", "minWidth": "0"}, children=[
                html.Div(className="card", children=[
                    html.Div("Scene selector", className="card-title"),
                    html.Div(className="controls", children=[
                        html.Label("Object"),
                        dcc.Dropdown(
                            id="obj-dd",
                            options=[
                                {"label": pretty_obj(o), "value": o} for o in OBJECTS
                            ],
                            value=initial_state["obj"],
                            clearable=False,
                            searchable=True,
                            style={"width": "100%"},
                        ),
                        html.Label("Size"),
                        html.Div(className="btn-row", children=[
                            _btn('4"', "size-4", active=False),
                            _btn('8"', "size-8", active=True),
                        ]),
                        html.Label("Light"),
                        html.Div(className="btn-row", children=[
                            _btn("OFF", "light-off", active=False),
                            _btn("ON", "light-on", active=True),
                        ]),
                        html.Label("Position"),
                        html.Div(className="frame-nav", children=[
                            html.Button("◀", id="prev-btn", n_clicks=0, className="btn"),
                            dcc.Input(
                                id="frame-input",
                                type="number", min=0, max=99, step=1,
                                value=initial_state["frame"],
                                className="num-input",
                                debounce=False,
                            ),
                            html.Span("/ 99", className="frame-suffix"),
                            html.Button("▶", id="next-btn", n_clicks=0, className="btn"),
                        ]),
                    ]),
                ]),

                html.Div(className="card", style={"flex": 1}, children=[
                    html.Div("3×3 SPAD histograms", className="card-title"),
                    dcc.Graph(
                        id="hist-fig",
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "270px"},
                    ),
                    html.Div(className="ymax-row", children=[
                        html.Label("Y max"),
                        dcc.Input(
                            id="ymax-input",
                            type="number", min=100, max=200000, step=100,
                            value=initial_state["y_max"],
                            className="num-input",
                            debounce=False,
                        ),
                        html.Span("counts (shared across all 9 cells)",
                                 className="hint"),
                    ]),
                ]),
            ]),
        ]),

        # ---- BOTTOM: predictions ---- #
        html.Div(className="row", style={"marginTop": "14px"}, children=[
            html.Div(className="card", style={"flex": "4"}, children=[
                html.Div("Predicted location · 10×10 capture grid",
                         className="card-title"),
                dcc.Graph(id="loc-fig", config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),
            html.Div(className="card", style={"flex": "4"}, children=[
                html.Div("Predicted object · top-5", className="card-title"),
                dcc.Graph(id="cls-fig", config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),
            html.Div(className="card", style={"flex": "2"}, children=[
                html.Div("Predicted size", className="card-title"),
                dcc.Graph(id="size-fig", config={"displayModeBar": False},
                          style={"height": "320px"}),
            ]),
        ]),
    ])


app.layout = serve_layout


# --------------------------------------------------------------------------- #
# Flask route serving the tracking RGB image (with overlay drawn on the fly)
# --------------------------------------------------------------------------- #

@app.server.route("/scene/<scene_id>/<int:frame>.jpg")
def serve_scene_image(scene_id: str, frame: int):
    scene = SCENE_BY_ID.get(scene_id)
    if scene is None or not (0 <= frame < 100):
        abort(404)
    rgb = _data_folder(scene) / "tracking_rgb" / f"{frame:03d}.png"
    if not rgb.exists():
        abort(404)

    # Project GT (and predicted) world points into pixels using the
    # tracking-camera calibration.
    gt_xyz   = gt_xyz_for(frame)
    gt_pixel = project_world_to_pixel(gt_xyz)

    pred_pixel = None
    try:
        hist = load_histogram(scene_id, frame)
        pr = predict(
            scene["object"], scene["size"],
            hist=hist, gt_xy=(gt_xyz[0], gt_xyz[1]),
            cache_key=(scene_id, frame),
        )
        # Use GT z as a stand-in (the model only predicts xy in world meters;
        # tag1 sits on a fixed plane so this projection is accurate enough).
        pred_pixel = project_world_to_pixel(
            np.array([pr.pred_xy[0], pr.pred_xy[1], gt_xyz[2]])
        )
    except Exception:                                                # noqa: BLE001
        pred_pixel = None

    data = overlay_markers(rgb, gt_pixel=gt_pixel, pred_pixel=pred_pixel)
    return send_file(io.BytesIO(data), mimetype="image/jpeg")


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

# Single state-management callback. Inputs from every control fold into
# state-store; downstream callbacks read state-store.

@app.callback(
    Output("state-store", "data"),
    Output("size-4", "className"),
    Output("size-8", "className"),
    Output("light-off", "className"),
    Output("light-on", "className"),
    Output("size-4", "disabled"),
    Output("size-8", "disabled"),
    Output("light-off", "disabled"),
    Output("light-on", "disabled"),
    Output("frame-input", "value"),
    Input("obj-dd", "value"),
    Input("size-4", "n_clicks"),
    Input("size-8", "n_clicks"),
    Input("light-off", "n_clicks"),
    Input("light-on", "n_clicks"),
    Input("prev-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    Input("frame-input", "value"),
    Input("ymax-input", "value"),
    State("state-store", "data"),
    prevent_initial_call=False,
)
def update_state(obj_val, _s4, _s8, _loff, _lon, _prev, _nxt, frame_val, ymax_val, state):
    state = dict(state) if state else dict(initial_state)
    trig = ctx.triggered_id

    if trig == "obj-dd":
        state["obj"] = obj_val
    elif trig == "size-4":
        state["size"] = "4inch"
    elif trig == "size-8":
        state["size"] = "8inch"
    elif trig == "light-off":
        state["light"] = "lightoff"
    elif trig == "light-on":
        state["light"] = "lighton"
    elif trig == "prev-btn":
        state["frame"] = (state["frame"] - 1) % 100
    elif trig == "next-btn":
        state["frame"] = (state["frame"] + 1) % 100
    elif trig == "frame-input":
        if frame_val is not None:
            try:
                state["frame"] = int(frame_val) % 100
            except (TypeError, ValueError):
                pass
    elif trig == "ymax-input":
        if ymax_val is not None:
            try:
                state["y_max"] = max(1, int(ymax_val))
            except (TypeError, ValueError):
                pass

    # If the current (obj, size, light) doesn't have data, snap to a valid one.
    if (state["obj"], state["size"], state["light"]) not in SCENE_INDEX:
        for s in SIZES:
            if (state["obj"], s, state["light"]) in SCENE_INDEX:
                state["size"] = s
                break
        else:
            for l in LIGHTS:
                if (state["obj"], state["size"], l) in SCENE_INDEX:
                    state["light"] = l
                    break

    # Active-class for buttons
    s4 = "btn active" if state["size"] == "4inch" else "btn"
    s8 = "btn active" if state["size"] == "8inch" else "btn"
    loff = "btn active" if state["light"] == "lightoff" else "btn"
    lon = "btn active" if state["light"] == "lighton" else "btn"

    # Disabled when no data
    s4_dis = (state["obj"], "4inch", state["light"]) not in SCENE_INDEX
    s8_dis = (state["obj"], "8inch", state["light"]) not in SCENE_INDEX
    loff_dis = (state["obj"], state["size"], "lightoff") not in SCENE_INDEX
    lon_dis = (state["obj"], state["size"], "lighton") not in SCENE_INDEX

    return state, s4, s8, loff, lon, s4_dis, s8_dis, loff_dis, lon_dis, state["frame"]


@app.callback(
    Output("tracking-img", "src"),
    Output("scene-id", "children"),
    Output("frame-tag", "children"),
    Output("hist-fig", "figure"),
    Output("loc-fig", "figure"),
    Output("cls-fig", "figure"),
    Output("size-fig", "figure"),
    Input("state-store", "data"),
    Input("theme-store", "data"),
)
def render_all(state, theme):
    obj = state["obj"]
    size = state["size"]
    light = state["light"]
    frame = state["frame"]
    y_max = state["y_max"]

    sid = scene_id_for(obj, size, light)
    if sid is None:
        empty = go.Figure(layout=plot_layout_base(theme))
        return "", "· no data for this combination", "", empty, empty, empty, empty

    img_src = f"/scene/{sid}/{frame}.jpg"
    scene_id_text = f"· scene = {sid}"

    hist  = load_histogram(sid, frame)
    gt_xy = gt_xy_for(frame)
    pred  = predict(obj, size, hist=hist, gt_xy=gt_xy, cache_key=(sid, frame))

    frame_tag_text = (
        f"frame {frame:03d}/099   ·   "
        f"err = {pred.err_m * 100:.1f} cm"
    )

    return (
        img_src, scene_id_text, frame_tag_text,
        fig_histograms(hist, y_max, theme=theme),
        fig_location(pred, theme=theme),
        fig_topk_classes(pred, theme=theme),
        fig_size(pred, theme=theme),
    )


# --- Theme toggle ---------------------------------------------------------- #

@app.callback(
    Output("theme-store", "data"),
    Output("theme-toggle-light", "className"),
    Output("theme-toggle-dark", "className"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=False,
)
def toggle_theme(n_clicks, current):
    """Flip between light and dark on each toggle click."""
    current = current or DEFAULT_THEME
    if not n_clicks:                                    # first paint, no flip
        new_theme = current
    else:
        new_theme = "dark" if current == "light" else "light"
    light_cls = "opt active" if new_theme == "light" else "opt"
    dark_cls = "opt active" if new_theme == "dark" else "opt"
    return new_theme, light_cls, dark_cls


# Clientside: push the chosen theme onto <html data-theme> so the CSS
# variables under :root[data-theme=...] take effect immediately, without
# needing any Dash component to re-render.
app.clientside_callback(
    """
    function(theme) {
        if (theme) {
            document.documentElement.setAttribute('data-theme', theme);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-store", "data", allow_duplicate=True),
    Input("theme-store", "data"),
    prevent_initial_call="initial_duplicate",
)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def _default_data_dir() -> Path:
    """Resolve the default data directory (`<repo>/denali-data/data/`)."""
    return (PACKAGE_DIR / ".." / "denali-data" / "data").resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m gui",
        description="Run the DENALI NLOS-tracking web GUI.",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Interface to bind on (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8050,
                   help="Port to listen on (default: 8050).")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Path to the raw NLOS captures (default: ../denali-data/data).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global DATA_ROOT
    args = parse_args(argv)
    DATA_ROOT = (args.data_dir or _default_data_dir()).resolve()
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"data directory not found: {DATA_ROOT}\n"
            f"  pass --data-dir <path> or extract denali-data/ at the repo root."
        )
    print(f"\n  DENALI GUI running at http://{args.host}:{args.port}")
    print(f"  serving captures from {DATA_ROOT}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
