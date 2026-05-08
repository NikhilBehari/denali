"""Figure outputs for the digital-twin renderer.

Each render writes five PNGs and the two histogram traces as ``.npy``:

    combined.png   3-panel: real RGB | sim RGB | overlaid histograms
    real_rgb.png   real tracking-camera frame
    sim_rgb.png    simulated (Mitsuba) tracking-camera render
    real_hist.png  real SPAD histogram (3x3 centre, bg-subtracted)
    sim_hist.png   simulated NLOS histogram (centre-block trace)
    real_hist.npy  values plotted in `real_hist.png`
    sim_hist.npy   values plotted in `sim_hist.png`
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


__all__ = ["save_outputs"]


_HIST_TITLE = "LiDAR Histogram: Real vs Sim.\n(with background subtraction)"


def _sim_block_trace(transient_full: np.ndarray, *, block_size: int,
                     center_ij: tuple[int, int]) -> np.ndarray:
    """Mean over a centre block of the transient cube, summed across RGB."""
    H, W, _, _ = transient_full.shape
    half = block_size // 2
    si = max(0, min(center_ij[0] - half, H - block_size))
    sj = max(0, min(center_ij[1] - half, W - block_size))
    block = transient_full[si:si + block_size, sj:sj + block_size]
    return block.mean(axis=(0, 1)).sum(axis=-1)


def _draw_histograms(ax_sim, sim_hist: np.ndarray, real_hist: np.ndarray,
                     *, scale_sim: float) -> None:
    L = min(len(sim_hist), len(real_hist))
    ax_sim.bar(np.arange(len(sim_hist)), sim_hist * scale_sim, alpha=0.6, label="Simulated")
    ax_sim.set_xlabel("Time index")
    ax_sim.set_ylabel("Radiance (SIM)")
    ax_real = ax_sim.twinx()
    ax_real.plot(np.arange(L), real_hist[:L].astype(np.float32),
                 linewidth=2.0, alpha=0.6, color="red", label="Real (3x3 center)")
    ax_real.set_ylabel("Counts (SPAD)")
    ax_sim.set_ylim(0,  max(1e-9, ax_sim.get_ylim()[1]))
    ax_real.set_ylim(0, max(1e-9, ax_real.get_ylim()[1]))
    h1, l1 = ax_sim.get_legend_handles_labels()
    h2, l2 = ax_real.get_legend_handles_labels()
    ax_sim.legend(h1 + h2, l1 + l2, loc="upper left")
    ax_sim.tick_params(axis="both",  labelsize=8)
    ax_real.tick_params(axis="both", labelsize=8)


def _save_combined(out_dir: Path, real_rgb: np.ndarray, sim_rgb: np.ndarray,
                   sim_hist: np.ndarray, real_hist: np.ndarray, scale_sim: float) -> None:
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.03, width_ratios=[1, 1, 0.7])
    ax_real, ax_sim, ax_hist = (fig.add_subplot(gs[0, i]) for i in range(3))
    aspect = real_rgb.shape[0] / real_rgb.shape[1]
    for ax, img, title in ((ax_real, real_rgb, "Real tracking RGB"),
                           (ax_sim,  sim_rgb,  "Simulated RGB")):
        ax.set_box_aspect(aspect)
        ax.imshow(img); ax.set_title(title); ax.axis("off")
    ax_hist.set_box_aspect(aspect * 1.4)
    _draw_histograms(ax_hist, sim_hist, real_hist, scale_sim=scale_sim)
    ax_hist.set_title(_HIST_TITLE)
    pos = ax_hist.get_position()
    ax_hist.set_position([pos.x0 + 0.035, pos.y0, pos.width, pos.height])
    fig.savefig(out_dir / "combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_image(out_dir: Path, name: str, image: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image); ax.axis("off"); ax.set_title(title)
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_hist(out_dir: Path, name: str, hist: np.ndarray, *,
               kind: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "real":
        ax.plot(np.arange(len(hist)), hist, linewidth=2.0, color="red")
    else:
        ax.bar(np.arange(len(hist)), hist, alpha=0.7)
    ax.set_xlabel("Time index"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis="both", labelsize=9)
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    out_dir:        Path,
    *,
    real_rgb:       np.ndarray,
    sim_rgb:        np.ndarray,
    real_hist:      np.ndarray,
    transient_full: np.ndarray,
    block_size:     int = 8,
    center_ij:      tuple[int, int] = (30, 4),
    scale_sim:      float = 1.0,
) -> None:
    """Write the five PNGs and two histogram ``.npy`` files into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_hist = _sim_block_trace(transient_full, block_size=block_size, center_ij=center_ij)
    sim_scaled = sim_hist * scale_sim

    _save_combined(out_dir, real_rgb, sim_rgb, sim_hist, real_hist, scale_sim)
    _save_image(out_dir, "real_rgb.png", real_rgb, "Real tracking RGB")
    _save_image(out_dir, "sim_rgb.png",  sim_rgb,  "Simulated tracking RGB")
    _save_hist(out_dir, "real_hist.png", real_hist,  kind="real",
               ylabel="Counts (SPAD)",   title="Real SPAD histogram (3x3 centre)")
    _save_hist(out_dir, "sim_hist.png",  sim_scaled, kind="sim",
               ylabel="Radiance (SIM)",  title="Simulated NLOS histogram")
    np.save(out_dir / "real_hist.npy", real_hist.astype(np.float32))
    np.save(out_dir / "sim_hist.npy",  sim_scaled.astype(np.float32))
