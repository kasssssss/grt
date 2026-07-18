#!/usr/bin/env python3
"""Render matched high-resolution RADs panels before and after adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from deepradar import DeepRadar
from scripts.evaluate_rads_azimuth_adapter import load_adapter
from scripts.train_rads_azimuth_adapter import load_frame, model_sample


def first_hit_depth(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    occupied = logits > threshold
    valid = occupied.any(axis=-1)
    depth = np.argmax(occupied.astype(np.uint8), axis=-1).astype(np.float32) + 1.0
    depth[~valid] = np.nan
    return depth, float(valid.mean())


def dilated_f1(
    logits: np.ndarray, target: np.ndarray, threshold: float, radius: int
) -> float:
    tensor = torch.from_numpy(target[None, None].astype(np.float32))
    if radius:
        size = radius * 2 + 1
        tensor = F.max_pool2d(tensor, size, stride=1, padding=radius)
    truth = tensor.numpy()[0, 0] > 0.5
    pred = logits.max(axis=0) > threshold
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def show_magnitude(
    ax: plt.Axes,
    value: np.ndarray,
    title: str,
    vmax: float | None = None,
) -> None:
    if vmax is None:
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


def show_bev(
    ax: plt.Axes,
    probability: np.ndarray,
    target: np.ndarray,
    title: str,
) -> None:
    image = ax.imshow(
        probability,
        cmap="inferno",
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    yy, xx = np.nonzero(target)
    ax.scatter(
        xx,
        yy,
        s=18,
        facecolors="none",
        edgecolors="#00ffff",
        linewidths=0.9,
        label="RADs_gt",
    )
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def show_depth(ax: plt.Axes, depth: np.ndarray, title: str) -> None:
    masked = np.ma.array(depth, mask=~np.isfinite(depth))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    image = ax.imshow(
        masked,
        cmap=cmap,
        origin="upper",
        aspect="auto",
        vmin=1.0,
        vmax=64.0,
        interpolation="nearest",
    )
    ax.set_facecolor("black")
    ax.set_title(title + " | black=invalid")
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="range cell")


def label_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-mode", choices=("auto", "zero", "original"), default="auto")
    parser.add_argument("--bev-threshold", type=float, default=0.0)
    parser.add_argument("--depth-threshold", type=float, default=0.0)
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()

    device = torch.device("cuda")
    saved, adapter = load_adapter(args.adapter, device)
    phase_mode = saved.get("phase_mode", "zero") if args.phase_mode == "auto" else args.phase_mode
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint), hparams_file=str(args.hparams), map_location=device
    ).eval().to(device)
    entries = json.loads(args.manifest.read_text())["frames"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for name in args.frames:
            dar_np, target = load_frame(args.data_root, name, entries[name])
            dar = torch.from_numpy(dar_np).to(device)
            baseline = adapter.initial_forward(dar, azimuth_dim=1)
            adapted = adapter(dar, azimuth_dim=1)
            samples = torch.cat(
                (
                    model_sample(baseline, phase_mode),
                    model_sample(adapted, phase_mode),
                ),
                dim=0,
            )
            logits = model({"radar": samples})["map"].float().cpu().numpy()
            baseline_logit, adapted_logit = logits[0], logits[1]
            baseline_bev = torch.sigmoid(torch.from_numpy(baseline_logit)).numpy().max(axis=0)
            adapted_bev = torch.sigmoid(torch.from_numpy(adapted_logit)).numpy().max(axis=0)
            baseline_depth, baseline_valid = first_hit_depth(
                baseline_logit, args.depth_threshold)
            adapted_depth, adapted_valid = first_hit_depth(
                adapted_logit, args.depth_threshold)

            raw_ra = np.abs(dar_np).max(axis=0)
            baseline_input = model_sample(
                baseline, phase_mode)[0, :, :, 0, :, 0].amax(dim=0).cpu().numpy()
            adapted_input = model_sample(
                adapted, phase_mode)[0, :, :, 0, :, 0].amax(dim=0).cpu().numpy()
            input_vmax = float(np.percentile(
                np.concatenate((baseline_input.ravel(), adapted_input.ravel())),
                99.7,
            ))
            baseline_f1 = dilated_f1(
                baseline_logit, target, args.bev_threshold, 2)
            adapted_f1 = dilated_f1(
                adapted_logit, target, args.bev_threshold, 2)

            fig, axes = plt.subplots(
                2, 4, figsize=(24, 11), constrained_layout=True)
            show_magnitude(axes[0, 0], raw_ra, "Cropped raw RA (A256)")
            label_axes(axes[0, 0], "range bin", "azimuth bin")
            show_magnitude(
                axes[0, 1], baseline_input, "Physical start-4 A8 input", input_vmax)
            label_axes(axes[0, 1], "range bin", "A8 channel")
            show_magnitude(
                axes[0, 2], adapted_input,
                f"Learned A8 input (step {saved['step']})", input_vmax)
            label_axes(axes[0, 2], "range bin", "A8 channel")
            axes[0, 3].imshow(
                target.astype(np.float32),
                cmap="gray",
                origin="upper",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            axes[0, 3].set_facecolor("black")
            axes[0, 3].set_title("Sparse RADs_gt occupancy")
            label_axes(axes[0, 3], "range cell", "azimuth cell")

            show_bev(
                axes[1, 0], baseline_bev, target,
                f"Baseline BEV logit>{args.bev_threshold:g} "
                f"| tol2 F1={baseline_f1:.4f}")
            label_axes(axes[1, 0], "range cell", "azimuth cell")
            show_bev(
                axes[1, 1], adapted_bev, target,
                f"Adapter BEV logit>{args.bev_threshold:g} "
                f"| tol2 F1={adapted_f1:.4f}")
            label_axes(axes[1, 1], "range cell", "azimuth cell")
            show_depth(
                axes[1, 2], baseline_depth,
                f"Baseline first-hit logit>{args.depth_threshold:g} "
                f"| valid={baseline_valid:.3f}")
            label_axes(
                axes[1, 2], "azimuth output cell", "elevation output cell")
            show_depth(
                axes[1, 3], adapted_depth,
                f"Adapter first-hit logit>{args.depth_threshold:g} "
                f"| valid={adapted_valid:.3f}")
            label_axes(
                axes[1, 3], "azimuth output cell", "elevation output cell")
            fig.suptitle(
                f"Frozen GRT query-mixer on RADs | {name} | "
                "whole-cube crop | matched scales",
                fontsize=17,
            )
            path = args.output_root / (
                f"rads_{name.replace('/', '_')}_baseline_vs_adapter.png")
            fig.savefig(path, dpi=args.dpi, facecolor="white")
            plt.close(fig)
            print(f"SAVED {path}", flush=True)


if __name__ == "__main__":
    main()
