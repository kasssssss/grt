#!/usr/bin/env python3
"""Audit model-input distribution gaps between RADs and RADs-like I/Q-1M."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


LOG_MAG_EDGES = np.linspace(-6.0, 3.0, 181, dtype=np.float64)
EPS = 1e-12


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("cache must be NAME=PATH")
    return name, Path(path)


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    total = profile.sum()
    return profile / total if total > 0 else np.zeros_like(profile)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize_profile(left)
    right = normalize_profile(right)
    middle = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    value = 0.5 * np.sum(left[left_mask] * np.log(left[left_mask] / middle[left_mask]))
    value += 0.5 * np.sum(right[right_mask] * np.log(right[right_mask] / middle[right_mask]))
    return float(value)


def profile_quantile(profile: np.ndarray, quantile: float) -> float:
    profile = normalize_profile(profile)
    if not np.any(profile):
        return float("nan")
    return float(np.searchsorted(np.cumsum(profile), quantile, side="left"))


def histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    histogram = normalize_profile(histogram)
    if not np.any(histogram):
        return float("nan")
    index = min(
        int(np.searchsorted(np.cumsum(histogram), quantile, side="left")),
        LOG_MAG_EDGES.size - 2,
    )
    return float(10.0 ** (0.5 * (LOG_MAG_EDGES[index] + LOG_MAG_EDGES[index + 1])))


@dataclass
class InputStats:
    doppler_power: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float64))
    azimuth_power: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    range_power: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float64))
    log_magnitude_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(LOG_MAG_EDGES.size - 1, dtype=np.float64))
    magnitude_sum: float = 0.0
    magnitude_max: float = 0.0
    nonzero: int = 0
    elements: int = 0
    phase_vector: complex = 0.0j
    phase_weight: float = 0.0
    samples: int = 0

    def update(self, sample: np.ndarray) -> None:
        if sample.ndim == 6:
            sample = sample[:, :, :, 0]
        if sample.ndim != 5 or sample.shape[-1] != 2:
            raise ValueError(f"expected [N,D,A,R,2], got {sample.shape}")
        magnitude = np.asarray(sample[..., 0], dtype=np.float64)
        phase = np.asarray(sample[..., 1], dtype=np.float64)
        power = np.square(np.square(magnitude))
        self.doppler_power += power.sum(axis=(0, 2, 3))
        self.azimuth_power += power.sum(axis=(0, 1, 3))
        self.range_power += power.sum(axis=(0, 1, 2))
        active = magnitude > 0
        active_magnitude = magnitude[active]
        if active_magnitude.size:
            self.log_magnitude_histogram += np.histogram(
                np.log10(active_magnitude.clip(min=10.0 ** LOG_MAG_EDGES[0])),
                bins=LOG_MAG_EDGES,
            )[0]
            self.magnitude_max = max(self.magnitude_max, float(active_magnitude.max()))
            self.phase_vector += np.sum(active_magnitude * np.exp(1j * phase[active]))
            self.phase_weight += float(active_magnitude.sum())
        self.magnitude_sum += float(magnitude.sum())
        self.nonzero += int(np.count_nonzero(active))
        self.elements += magnitude.size
        self.samples += sample.shape[0]

    def summary(self) -> dict[str, object]:
        doppler = normalize_profile(self.doppler_power)
        active_doppler = int(np.count_nonzero(doppler > doppler.max() * 1e-4))
        return {
            "samples": self.samples,
            "magnitude_mean": self.magnitude_sum / max(self.elements, 1),
            "magnitude_active_p50": histogram_quantile(self.log_magnitude_histogram, 0.5),
            "magnitude_active_p99": histogram_quantile(self.log_magnitude_histogram, 0.99),
            "magnitude_max": self.magnitude_max,
            "nonzero_fraction": self.nonzero / max(self.elements, 1),
            "phase_resultant": abs(self.phase_vector) / max(self.phase_weight, EPS),
            "doppler_active_bins": active_doppler,
            "range_power_q05": profile_quantile(self.range_power, 0.05),
            "range_power_q50": profile_quantile(self.range_power, 0.50),
            "range_power_q95": profile_quantile(self.range_power, 0.95),
            "profiles": {
                "doppler_power": doppler.tolist(),
                "azimuth_power": normalize_profile(self.azimuth_power).tolist(),
                "range_power": normalize_profile(self.range_power).tolist(),
                "log_magnitude": normalize_profile(self.log_magnitude_histogram).tolist(),
            },
        }


def expand_a8_complex(spectrum: np.ndarray, target_bins: int = 256) -> np.ndarray:
    aperture = np.fft.ifft(np.fft.ifftshift(spectrum, axes=1), axis=1)
    padded = np.zeros(
        (spectrum.shape[0], target_bins, spectrum.shape[2]), dtype=np.complex64)
    padded[:, :spectrum.shape[1]] = aperture
    return np.fft.fftshift(np.fft.fft(padded, axis=1), axes=1).astype(np.complex64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", action="append", type=parse_named_path, required=True)
    parser.add_argument("--iq-traces", nargs="+", default=["outdoor/baum", "outdoor/cmu.east"])
    parser.add_argument("--iq-offset", type=int, default=256)
    parser.add_argument("--iq-samples-per-trace", type=int, default=256)
    parser.add_argument("--rads-root", type=Path, required=True)
    parser.add_argument("--rads-samples", type=int, default=32)
    parser.add_argument("--crop-fraction", type=float, default=1.0)
    parser.add_argument("--amp-scale", type=float, default=7.0)
    parser.add_argument("--doppler-keep-bins", type=int, default=11)
    parser.add_argument("--range-smooth", type=float, default=1.4)
    parser.add_argument("--az-smooth", type=float, default=0.45)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from scripts.evaluate_doppler_cache_variants import load_range  # noqa: PLC0415
    from scripts.rads_map_checkpoint_infer import (  # noqa: PLC0415
        keep_center_doppler,
        prepare_modes,
        reduce_azimuth,
        shift_range_cube,
        to_dar,
    )

    accumulators: dict[str, InputStats] = {}
    for name, cache_root in args.cache:
        accumulator = InputStats()
        for trace in args.iq_traces:
            accumulator.update(load_range(
                cache_root,
                trace,
                args.iq_offset,
                args.iq_samples_per_trace,
            ))
        accumulators[name] = accumulator

    files = sorted(args.rads_root.rglob("*.npy"))
    if not files:
        raise ValueError(f"No RADs files found below {args.rads_root}")
    indices = np.unique(np.linspace(
        0, len(files) - 1, min(args.rads_samples, len(files)), dtype=np.int64))
    rads_stats = InputStats()
    crop_starts: list[int] = []
    reconstruction_errors: list[float] = []
    selected_files: list[str] = []
    for index in indices:
        path = files[int(index)]
        prepared = prepare_modes(
            path,
            ["crop_auto"],
            flip_azimuth=True,
            target_azimuth_bins=8,
            a8_reducer="aperture_truncate",
            beam_sigma=18.0,
            aperture_start=0,
            range_smooth=args.range_smooth,
            az_smooth=args.az_smooth,
            amp_scale=args.amp_scale,
            crop_fraction=args.crop_fraction,
            mag_keep_frac=1.0,
            doppler_keep_bins=args.doppler_keep_bins,
        )[0]
        rads_stats.update(prepared.sample[None])
        crop_starts.append(prepared.crop_start)
        selected_files.append(str(path.relative_to(args.rads_root)))

        direct = keep_center_doppler(
            to_dar(shift_range_cube(prepared.radar, 0), flip_azimuth=True),
            args.doppler_keep_bins,
        )
        projected = reduce_azimuth(direct, 8, "aperture_truncate", 18.0)
        reconstructed = expand_a8_complex(projected)
        reconstruction_errors.append(float(
            np.linalg.norm(reconstructed - direct) / max(np.linalg.norm(direct), EPS)))
    accumulators["rads_crop_a8"] = rads_stats

    summaries = {name: value.summary() for name, value in accumulators.items()}
    comparisons: dict[str, dict[str, float]] = {}
    names = list(accumulators)
    for left_index, left_name in enumerate(names):
        left = accumulators[left_name]
        for right_name in names[left_index + 1:]:
            right = accumulators[right_name]
            comparisons[f"{left_name}__vs__{right_name}"] = {
                "doppler_js": js_divergence(left.doppler_power, right.doppler_power),
                "azimuth_js": js_divergence(left.azimuth_power, right.azimuth_power),
                "range_js": js_divergence(left.range_power, right.range_power),
                "log_magnitude_js": js_divergence(
                    left.log_magnitude_histogram, right.log_magnitude_histogram),
            }

    result = {
        "iq_traces": args.iq_traces,
        "iq_offset": args.iq_offset,
        "iq_samples_per_trace": args.iq_samples_per_trace,
        "rads_files": selected_files,
        "crop_fraction": args.crop_fraction,
        "rads_crop_start": {
            "min": int(np.min(crop_starts)),
            "median": float(np.median(crop_starts)),
            "max": int(np.max(crop_starts)),
        },
        "rads_a256_to_a8_to_a256_relative_l2": {
            "mean": float(np.mean(reconstruction_errors)),
            "median": float(np.median(reconstruction_errors)),
            "max": float(np.max(reconstruction_errors)),
        },
        "summaries": summaries,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    concise = {
        "rads_crop_start": result["rads_crop_start"],
        "rads_reconstruction": result["rads_a256_to_a8_to_a256_relative_l2"],
        "summaries": {
            name: {key: value[key] for key in (
                "magnitude_mean",
                "magnitude_active_p50",
                "magnitude_active_p99",
                "nonzero_fraction",
                "phase_resultant",
                "doppler_active_bins",
                "range_power_q05",
                "range_power_q50",
                "range_power_q95",
            )}
            for name, value in summaries.items()
        },
        "comparisons": comparisons,
    }
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
