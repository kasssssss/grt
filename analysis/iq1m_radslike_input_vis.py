#!/usr/bin/env python3
"""Visualize the actual RADs-like I/Q-1M radar tensors used by GRT training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepradar import DeepRadar
from deepradar.config import load_config


def _display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.log1p(np.maximum(arr, 0.0))
    lo, hi = np.percentile(arr, [1.0, 99.7])
    if hi <= lo:
        return arr
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _imshow(ax: plt.Axes, arr: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    ax.imshow(_display(arr), cmap="magma", aspect="auto", origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _make_figure(sample: np.ndarray, out_path: Path, title: str) -> dict:
    if sample.ndim != 5 or sample.shape[-1] < 1:
        raise ValueError(f"Expected radar sample [D,A,E,R,C], got {sample.shape}")

    mag = np.asarray(sample[..., 0], dtype=np.float32)
    doppler, azimuth, elevation, ranges = mag.shape

    selected = np.linspace(0, doppler - 1, 16, dtype=np.int32)
    fig = plt.figure(figsize=(18, 9.8), constrained_layout=True)
    gs = fig.add_gridspec(4, 8, height_ratios=[1.0, 1.0, 1.25, 1.0])
    fig.suptitle(title, fontsize=12)

    for i, d_idx in enumerate(selected):
        row = i // 8
        col = i % 8
        ax = fig.add_subplot(gs[row, col])
        # D slice displayed as x=range, y=azimuth; the kept elevation axis is
        # maxed for robustness if a config later keeps more than one bin.
        d_view = mag[d_idx].max(axis=1)
        _imshow(ax, d_view, f"D{d_idx:02d}", "range", "azimuth")

    rd = mag.max(axis=(1, 2))
    ra = mag.max(axis=(0, 2))
    ad = mag.max(axis=(2, 3))
    doppler_energy = mag.max(axis=(1, 2, 3))

    ax = fig.add_subplot(gs[2, 0:2])
    _imshow(ax, rd, "Processed RD (x=range, y=doppler)", "range", "doppler")

    ax = fig.add_subplot(gs[2, 2:5])
    _imshow(ax, ra, "Processed RA (x=range, y=azimuth)", "range", "azimuth")

    ax = fig.add_subplot(gs[2, 5:8])
    _imshow(ax, ad, "Processed AD (x=azimuth, y=doppler)", "azimuth", "doppler")

    ax = fig.add_subplot(gs[3, 0:4])
    ax.plot(np.arange(doppler), doppler_energy, color="#1f77b4", lw=1.3)
    ax.set_title("Doppler max energy after RADs-like merge", fontsize=9)
    ax.set_xlabel("target Doppler bin, centered at 32; physical span [-90, 90]", fontsize=8)
    ax.set_ylabel("max magnitude", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[3, 4:8])
    flat = mag.reshape(doppler, -1)
    nonzero_ratio = (flat > 1e-6).mean(axis=1)
    ax.bar(np.arange(doppler), nonzero_ratio, color="#d62728", width=0.9)
    ax.set_title("Per-Doppler nonzero ratio", fontsize=9)
    ax.set_xlabel("target Doppler bin", fontsize=8)
    ax.set_ylabel("ratio", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return {
        "path": str(out_path),
        "shape": list(sample.shape),
        "selected_doppler_bins": selected.tolist(),
        "magnitude_min": float(np.min(mag)),
        "magnitude_mean": float(np.mean(mag)),
        "magnitude_max": float(np.max(mag)),
        "empty_doppler_bins_lte_1e-6": np.where(doppler_energy <= 1e-6)[0].astype(int).tolist(),
        "low_energy_doppler_bins_lte_1pct": np.where(
            doppler_energy <= max(float(np.max(doppler_energy)) * 0.01, 1e-6)
        )[0].astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--trace",
        default="outdoor/shadyside.west",
        help="Single I/Q-1M trace to visualize. Use an eval trace by default so loading is quick.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            "grt/grt",
            "grt/small",
            "obj/map",
            "data[indoor,outdoor,bike]",
            "splits/codex_iq1m_2gpu",
            "repr/rads_like",
        ],
    )
    args = parser.parse_args()

    cfg = load_config(*[str(Path("config") / item) for item in args.config])
    cfg["dataset"]["traces"] = [args.trace]
    model = DeepRadar(**cfg)
    data = model.get_dataset(str(args.data), n_workers=0)
    loader = data.eval_dataloader(args.trace, batch_size=max(args.samples, 2))
    samples = next(iter(loader))

    radar = samples.get("radar")
    if radar is None:
        raise KeyError(f"No radar key found in val samples: {sorted(samples.keys())}")

    if hasattr(radar, "detach"):
        radar = radar.detach().cpu().numpy()
    radar = np.asarray(radar)
    if radar.shape[0] < args.samples:
        raise ValueError(f"Requested {args.samples} samples, got {radar.shape[0]}")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for idx in range(args.samples):
        out_path = args.out / f"iq1m_radslike_sample_{idx:03d}.png"
        summaries.append(_make_figure(
            radar[idx],
            out_path,
            f"I/Q-1M after RADs-like preprocessing | sample={idx} | tensor={tuple(radar[idx].shape)}",
        ))

    summary_path = args.out / "iq1m_radslike_input_summary.json"
    summary_path.write_text(json.dumps({
        "data": str(args.data),
        "trace": args.trace,
        "configs": args.config,
        "samples": summaries,
    }, indent=2), encoding="utf-8")
    print(summary_path)
    for item in summaries:
        print(item["path"])


if __name__ == "__main__":
    main()
