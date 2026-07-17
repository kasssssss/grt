#!/usr/bin/env python3
"""Audit the complex A8/A256 contract on I/Q-1M and RADs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rads_map_checkpoint_infer import reduce_azimuth, to_dar


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq-cache-root", type=Path, required=True)
    parser.add_argument("--rads-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iq-samples", type=int, default=12)
    parser.add_argument(
        "--rads-frames", nargs="+",
        default=["100/000001.npy", "100/000050.npy", "101/000001.npy"],
    )
    return parser.parse_args()


def from_complex_phase(sample: np.ndarray) -> np.ndarray:
    magnitude = sample[..., 0].astype(np.float32)
    phase = sample[..., 1].astype(np.float32)
    return np.square(magnitude) * np.exp(1j * phase)


def expand_a8(value: np.ndarray, bins: int = 256) -> np.ndarray:
    aperture = np.fft.ifft(np.fft.ifftshift(value, axes=1), axis=1)
    padded = np.pad(aperture, ((0, 0), (0, bins - value.shape[1]), (0, 0)))
    return np.fft.fftshift(np.fft.fft(padded, axis=1), axes=1)


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm((actual - expected).reshape(-1))
        / max(np.linalg.norm(expected.reshape(-1)), EPS)
    )


def normalized_ra(value: np.ndarray) -> np.ndarray:
    image = np.sqrt(np.abs(value)).max(axis=0)
    lo, hi = np.percentile(image, [1.0, 99.7])
    return np.clip((image - lo) / max(float(hi - lo), EPS), 0.0, 1.0)


def load_iq_samples(root: Path, count: int) -> list[tuple[str, np.ndarray]]:
    shards = sorted(root.glob("traces/**/*.npy"))
    if not shards:
        raise FileNotFoundError(f"No cache shards found under {root}")
    indices = np.linspace(0, len(shards) - 1, min(count, len(shards)), dtype=int)
    samples = []
    for shard_index in indices:
        shard_path = shards[int(shard_index)]
        shard = np.load(shard_path, mmap_mode="r")
        sample_index = int(shard.shape[0] // 2)
        sample = np.asarray(shard[sample_index], dtype=np.float32)
        value = from_complex_phase(sample)[:, :, 0, :]
        samples.append((f"{shard_path.parent.name}:{sample_index}", value))
    return samples


def projection_metrics(value256: np.ndarray) -> dict[str, float | int]:
    aperture = np.fft.ifft(np.fft.ifftshift(value256, axes=1), axis=1)
    energy = np.sum(np.square(np.abs(aperture)), axis=(0, 2))
    total = max(float(energy.sum()), EPS)
    windows = np.convolve(energy, np.ones(8, dtype=np.float64), mode="valid")
    best_start = int(np.argmax(windows))
    projected8 = reduce_azimuth(
        value256, 8, mode="aperture_truncate", beam_sigma=18.0)
    sector8 = reduce_azimuth(
        value256, 8, mode="sector_coherent", beam_sigma=18.0)
    return {
        "first8_aperture_energy_fraction": float(energy[:8].sum() / total),
        "best8_aperture_energy_fraction": float(windows[best_start] / total),
        "best8_aperture_start": best_start,
        "aperture_projection_relative_l2": relative_l2(
            expand_a8(projected8), value256),
        "sector_projection_relative_l2": relative_l2(
            expand_a8(sector8), value256),
    }


def audit_iq(samples: list[tuple[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for name, native8 in samples:
        expanded = expand_a8(native8)
        recovered = reduce_azimuth(
            expanded, 8, mode="aperture_truncate", beam_sigma=18.0)
        sector = reduce_azimuth(
            expanded, 8, mode="sector_coherent", beam_sigma=18.0)
        metrics = projection_metrics(expanded)
        metrics.update({
            "sample": name,
            "a8_roundtrip_relative_l2": relative_l2(recovered, native8),
            "sector_a8_relative_l2": relative_l2(sector, native8),
        })
        rows.append(metrics)
    return rows


def audit_rads(
    root: Path, frames: list[str],
) -> tuple[list[dict[str, object]], list[tuple[str, np.ndarray]]]:
    rows = []
    values = []
    for frame in frames:
        cube = np.load(root / frame).astype(np.complex64, copy=False)
        value = to_dar(cube, flip_azimuth=True)
        metrics = projection_metrics(value)
        metrics["frame"] = frame
        rows.append(metrics)
        values.append((frame, value))
    return rows, values


def plot_examples(
    iq_samples: list[tuple[str, np.ndarray]],
    rads_samples: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    pairs = [("I/Q-1M", *iq_samples[0]), ("RADs", *rads_samples[0])]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
    for row, (domain, name, value) in enumerate(pairs):
        value256 = expand_a8(value) if value.shape[1] == 8 else value
        aperture8 = reduce_azimuth(
            value256, 8, mode="aperture_truncate", beam_sigma=18.0)
        sector8 = reduce_azimuth(
            value256, 8, mode="sector_coherent", beam_sigma=18.0)
        images = [
            ("native/expanded A256", normalized_ra(value256)),
            ("aperture A8 -> A256", normalized_ra(expand_a8(aperture8))),
            ("sector A8 -> A256", normalized_ra(expand_a8(sector8))),
        ]
        aperture = np.fft.ifft(np.fft.ifftshift(value256, axes=1), axis=1)
        energy = np.sum(np.square(np.abs(aperture)), axis=(0, 2))
        energy /= max(float(energy.sum()), EPS)
        for col, (title, image) in enumerate(images):
            axes[row, col].imshow(
                image, cmap="magma", aspect="auto", origin="lower")
            axes[row, col].set_title(f"{domain} {name}\n{title}")
            axes[row, col].set_xlabel("range")
            axes[row, col].set_ylabel("azimuth")
        axes[row, 3].plot(np.arange(256), energy)
        axes[row, 3].axvspan(
            -0.5, 7.5, color="tab:orange", alpha=0.25,
            label="training aperture 0:8")
        axes[row, 3].set_title(f"{domain} aperture energy")
        axes[row, 3].set_xlabel("aperture coefficient")
        axes[row, 3].set_ylabel("energy fraction")
        axes[row, 3].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    iq_samples = load_iq_samples(args.iq_cache_root, args.iq_samples)
    iq_rows = audit_iq(iq_samples)
    rads_rows, rads_samples = audit_rads(args.rads_root, args.rads_frames)
    report = {
        "contract": {
            "training_native_shape": "[D=64,A=8,E=1,R=256,C=2]",
            "model_shape": "[D=64,A=256,E=1,R=256,C=2]",
            "a256_definition": (
                "azimuth FFT of an 8-element aperture zero-padded to 256"),
        },
        "iq1m": iq_rows,
        "rads": rads_rows,
        "summary": {
            "iq_roundtrip_relative_l2": summarize(
                iq_rows, "a8_roundtrip_relative_l2"),
            "iq_sector_relative_l2": summarize(
                iq_rows, "sector_a8_relative_l2"),
            "rads_first8_energy_fraction": summarize(
                rads_rows, "first8_aperture_energy_fraction"),
            "rads_aperture_projection_relative_l2": summarize(
                rads_rows, "aperture_projection_relative_l2"),
            "rads_sector_projection_relative_l2": summarize(
                rads_rows, "sector_projection_relative_l2"),
        },
    }
    report_path = args.out_dir / "azimuth_contract.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_examples(
        iq_samples, rads_samples, args.out_dir / "azimuth_contract.png")
    print(json.dumps(report["summary"], indent=2))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
