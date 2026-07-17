#!/usr/bin/env python3
"""Render compact, high-resolution RADs transfer panels.

The diagnostic inference figure intentionally contains every intermediate
view. This renderer keeps only the panels needed to judge transfer quality and
draws invalid depth pixels as black. The separate mask sheet uses white only
for invalid pixels and labels that convention explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.rads_map_checkpoint_infer import (
    build_sample,
    shift_range_cube,
)


def show_magnitude(ax: plt.Axes, value: np.ndarray, title: str) -> None:
    vmax = float(np.percentile(value, 99.7))
    ax.imshow(
        value,
        cmap="magma",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        interpolation="nearest",
    )
    ax.set_title(title)


def show_probability(ax: plt.Axes, value: np.ndarray, title: str) -> None:
    image = ax.imshow(
        value,
        cmap="inferno",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def show_depth(
    ax: plt.Axes,
    value: np.ndarray,
    invalid: np.ndarray,
    title: str,
) -> None:
    depth = np.ma.array(value, mask=invalid.astype(bool) | ~np.isfinite(value))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    image = ax.imshow(
        depth,
        cmap=cmap,
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=63.0,
        interpolation="nearest",
    )
    ax.set_facecolor("black")
    ax.set_title(title + " | black=invalid")
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="range voxel")


def set_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def render_frame(
    *,
    frame: str,
    rads_root: Path,
    predictions_root: Path,
    output_root: Path,
    dpi: int,
) -> None:
    rel = Path(frame)
    cube_path = rads_root / rel.parent / f"{rel.name}.npy"
    pred_path = predictions_root / (
        f"rads_{rel.parent.name}_{rel.name}_crop_auto_predictions.npz"
    )
    cube = np.load(cube_path).astype(np.complex64, copy=False)
    with np.load(pred_path) as pred:
        crop_start = int(pred["crop_start"])
        gt_polar = pred["gt_polar"].astype(bool)
        bev_prob = pred["bev_polar_prob_maxe"].astype(np.float32)
        depth_m1 = pred["depth_logit_gt_m1"].astype(np.float32)
        invalid_m1 = pred["invalid_mask_logit_gt_m1"].astype(bool)
        invalid_0 = pred["invalid_mask_logit_gt_0"].astype(bool)

    shifted = shift_range_cube(cube, crop_start)
    _, raw_views, input_views, _ = build_sample(
        shifted,
        flip_azimuth=True,
        azimuth_flip_mode="index",
        target_azimuth_bins=8,
        a8_reducer="aperture_truncate",
        beam_sigma=18.0,
        aperture_start=0,
        range_smooth=0.0,
        az_smooth=0.0,
        spatial_smooth_mode="none",
        amp_scale=2.6245,
        amp_gamma=0.45,
        amp_clip=0.0,
        mag_keep_frac=1.0,
        doppler_keep_bins=11,
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    show_magnitude(axes[0, 0], raw_views["ra"], "Cropped raw RA")
    set_axes(axes[0, 0], "range bin", "azimuth bin")
    show_magnitude(axes[0, 1], raw_views["rd"], "Cropped raw RD")
    set_axes(axes[0, 1], "range bin", "Doppler bin")
    show_magnitude(axes[0, 2], input_views["ra"], "Model input RA (A8)")
    set_axes(axes[0, 2], "range bin", "physical aperture beam")

    axes[1, 0].imshow(
        gt_polar.astype(np.float32),
        cmap="gray",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1, 0].set_facecolor("black")
    axes[1, 0].set_title("Sparse RADs_gt occupancy (index space)")
    set_axes(axes[1, 0], "range cell", "azimuth cell")

    show_probability(axes[1, 1], bev_prob, "GRT BEV probability + RADs_gt")
    yy, xx = np.nonzero(gt_polar)
    axes[1, 1].scatter(
        xx, yy, s=12, facecolors="none", edgecolors="#00ffff", linewidths=0.8,
        label="RADs_gt",
    )
    axes[1, 1].legend(loc="upper right", framealpha=0.85)
    set_axes(axes[1, 1], "range cell", "azimuth cell")

    valid_fraction = 1.0 - float(invalid_m1.mean())
    show_depth(
        axes[1, 2],
        depth_m1,
        invalid_m1,
        f"GRT first-hit depth (logit>-1, valid={valid_fraction:.3f})",
    )
    set_axes(axes[1, 2], "horizontal output cell", "vertical output cell")

    fig.suptitle(
        f"RADs transfer with GRT RADs-like I/Q-1M best | "
        f"{rel.parent.name}/{rel.name} | whole-cube crop_start={crop_start}",
        fontsize=16,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    main_path = output_root / f"rads_{rel.parent.name}_{rel.name}_paper.png"
    fig.savefig(main_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mask, threshold in zip(axes, (invalid_m1, invalid_0), (-1, 0)):
        ax.imshow(
            mask.astype(np.float32), cmap="gray", origin="upper",
            aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(
            f"Invalid mask logit>{threshold} | white=invalid | "
            f"fraction={float(mask.mean()):.3f}")
        ax.set_xlabel("horizontal output cell")
        ax.set_ylabel("vertical output cell")
    mask_path = output_root / f"rads_{rel.parent.name}_{rel.name}_invalid_masks.png"
    fig.savefig(mask_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"SAVED {main_path} {mask_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rads-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    for frame in args.frames:
        render_frame(
            frame=frame,
            rads_root=args.rads_root,
            predictions_root=args.predictions_root,
            output_root=args.output_root,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
