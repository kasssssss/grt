#!/usr/bin/env python3
"""Infer a trained DeepRadar map checkpoint on RADs cubes.

This script is intentionally separate from training code. It converts RADs
native complex cubes into the same single-elevation RADs-like tensor convention
used by the AutoDL I/Q-1M experiment: [D, A, E, R, C] with C=(sqrt(abs), phase).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EPS = 1e-12


@dataclass(frozen=True)
class Prepared:
    mode: str
    crop_start: int
    crop_source: str
    raw_crop_start: int
    gt_crop_start: int | None
    azimuth_bins: int
    radar: np.ndarray
    sample: np.ndarray
    raw_views: dict[str, np.ndarray]
    input_views: dict[str, np.ndarray]
    input_stats: dict[str, float | str]
    gt_polar: np.ndarray | None


def normalize_image(arr: np.ndarray, pct: tuple[float, float] = (1.0, 99.7)) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, pct)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def display_mag(arr: np.ndarray) -> np.ndarray:
    return normalize_image(np.log1p(np.maximum(arr, 0.0)))


def first_signal_index(cube_raz: np.ndarray) -> tuple[int, float]:
    mag = np.abs(cube_raz)
    profile = mag.mean(axis=(1, 2))
    med = float(np.median(profile))
    mad = float(np.median(np.abs(profile - med)))
    mx = float(np.max(profile))
    threshold = max(med + 6.0 * mad, mx * 0.08)
    hits = np.flatnonzero(profile >= threshold)
    if hits.size == 0:
        return 0, threshold
    return int(hits[0]), threshold


def first_gt_signal_index(gt_cube_raz: np.ndarray) -> int | None:
    """Return the first range bin marked occupied by sparse RADs_gt."""
    hits = np.flatnonzero(np.any(gt_cube_raz != 0, axis=(1, 2)))
    return int(hits[0]) if hits.size else None


def select_crop_start(
    cube_raz: np.ndarray,
    gt_cube_raz: np.ndarray | None,
    crop_source: str,
) -> tuple[int, str, int, int | None]:
    """Select a reproducible whole-cube crop, preferring GT when requested."""
    raw_start, _ = first_signal_index(cube_raz)
    gt_start = (
        first_gt_signal_index(gt_cube_raz)
        if gt_cube_raz is not None
        else None
    )
    if crop_source == "gt" and gt_start is not None:
        return gt_start, "gt", raw_start, gt_start
    if crop_source not in {"gt", "raw"}:
        raise ValueError(f"unknown crop_source {crop_source!r}")
    return raw_start, "raw", raw_start, gt_start


def shift_range_cube(cube_raz: np.ndarray, start: int) -> np.ndarray:
    out = np.zeros_like(cube_raz)
    if start <= 0:
        out[...] = cube_raz
    elif start < cube_raz.shape[0]:
        out[: cube_raz.shape[0] - start] = cube_raz[start:]
    return out


def project_gt_ar(
    gt_ar: np.ndarray,
    start: int,
    *,
    flip_azimuth: bool,
    azimuth_flip_mode: str,
) -> np.ndarray:
    """Crop and project sparse native [azimuth,range] GT to [128,64]."""
    if gt_ar.shape != (256, 256):
        raise ValueError(f"expected manifest GT [256,256], got {gt_ar.shape}")
    shifted = np.zeros(gt_ar.shape, dtype=bool)
    if start <= 0:
        shifted[...] = gt_ar
    elif start < gt_ar.shape[1]:
        shifted[:, :gt_ar.shape[1] - start] = gt_ar[:, start:]
    if flip_azimuth:
        shifted = shifted[::-1]
        if azimuth_flip_mode == "spectral":
            shifted = np.roll(shifted, 1, axis=0)
        elif azimuth_flip_mode != "index":
            raise ValueError(
                f"unknown azimuth_flip_mode {azimuth_flip_mode!r}")
    return shifted.reshape(128, 2, 64, 4).max(axis=(1, 3))


def to_dar(
    cube_raz: np.ndarray,
    flip_azimuth: bool = True,
    azimuth_flip_mode: str = "index",
) -> np.ndarray:
    """RADs [range, azimuth, doppler] -> [doppler, azimuth, range]."""
    dar = np.moveaxis(cube_raz, [2, 1, 0], [0, 1, 2]).astype(np.complex64, copy=False)
    if flip_azimuth:
        dar = dar[:, ::-1, :]
        if azimuth_flip_mode == "spectral":
            # On an even fftshifted grid, exact u -> -u reflection is a
            # reversed array rolled by one bin. Plain index reversal reflects
            # around the boundary between the two center bins instead.
            dar = np.roll(dar, 1, axis=1)
        elif azimuth_flip_mode != "index":
            raise ValueError(
                f"unknown azimuth_flip_mode {azimuth_flip_mode!r}")
    return dar


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(x / sigma)).astype(np.float32)
    kernel /= max(float(kernel.sum()), EPS)
    return kernel


def smooth_axis(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return arr.astype(np.float32, copy=False)
    moved = np.moveaxis(arr.astype(np.float32, copy=False), axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty_like(flat)
    pad = kernel.size // 2
    for i, row in enumerate(flat):
        out[i] = np.convolve(np.pad(row, (pad, pad), mode="edge"), kernel, mode="valid")
    return np.moveaxis(out.reshape(moved.shape), -1, axis)


def smooth_complex_power_peak_phase(
    arr: np.ndarray, sigma: float, axis: int,
) -> np.ndarray:
    """Smooth power while taking phase from the strongest local contributor."""
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return arr.astype(np.complex64, copy=False)
    moved = np.moveaxis(arr.astype(np.complex64, copy=False), axis, -1)
    pad = kernel.size // 2
    padded = np.pad(
        moved, [(0, 0)] * (moved.ndim - 1) + [(pad, pad)], mode="edge")
    smoothed_power = np.zeros(moved.shape, dtype=np.float32)
    best_score = np.full(moved.shape, -np.inf, dtype=np.float32)
    best_source = np.zeros(moved.shape, dtype=np.complex64)
    width = moved.shape[-1]
    for offset, weight in enumerate(kernel):
        source = padded[..., offset:offset + width]
        score = float(weight) * np.square(np.abs(source)).astype(np.float32)
        smoothed_power += score
        update = score > best_score
        best_score = np.where(update, score, best_score)
        best_source = np.where(update, source, best_source)
    smoothed = np.sqrt(smoothed_power) * np.exp(1j * np.angle(best_source))
    return np.moveaxis(smoothed.astype(np.complex64), -1, axis)


def soft_beam_weights(n_src: int, n_beams: int, sigma: float) -> np.ndarray:
    centers = np.linspace(0, n_src - 1, n_beams, dtype=np.float32)
    x = np.arange(n_src, dtype=np.float32)
    weights = np.exp(-0.5 * np.square((x[None, :] - centers[:, None]) / sigma))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    return weights.astype(np.float32)


def reduce_azimuth(
    dar: np.ndarray,
    target_bins: int,
    mode: str,
    beam_sigma: float,
    aperture_start: int = 0,
) -> np.ndarray:
    """Reduce pseudo-A256 RADs azimuth while retaining complex phase."""
    if dar.shape[1] == target_bins:
        return dar
    if target_bins != 8 or dar.shape[1] % target_bins != 0:
        raise ValueError(
            f"Unsupported azimuth reduction A={dar.shape[1]} -> A={target_bins}.")
    if mode == "aperture_truncate":
        # The RADs-like I/Q-1M A256 path zero-pads an eight-element aperture
        # before the azimuth FFT. Its matched projection is the inverse
        # operation, not averaging neighboring angular bins.
        aperture = np.fft.ifft(
            np.fft.ifftshift(dar, axes=1), axis=1)
        aperture8 = aperture[:, :target_bins, :]
        return np.fft.fftshift(
            np.fft.fft(aperture8, axis=1), axes=1
        ).astype(np.complex64)
    if mode == "aperture_window":
        aperture = np.fft.ifft(
            np.fft.ifftshift(dar, axes=1), axis=1)
        indices = (aperture_start + np.arange(target_bins)) % aperture.shape[1]
        aperture8 = aperture[:, indices, :]
        return np.fft.fftshift(
            np.fft.fft(aperture8, axis=1), axes=1
        ).astype(np.complex64)
    if mode == "gaussian_coherent":
        weights = soft_beam_weights(dar.shape[1], target_bins, beam_sigma)
        return np.einsum("ba,dar->dbr", weights, dar, optimize=True).astype(
            np.complex64)

    sectors = dar.reshape(
        dar.shape[0], target_bins, dar.shape[1] // target_bins, dar.shape[2])
    if mode == "sector_coherent":
        return sectors.mean(axis=2).astype(np.complex64)
    if mode == "sector_energy_circular":
        amplitude = np.abs(sectors).mean(axis=2)
        phase = np.angle(np.sum(sectors * np.abs(sectors), axis=2))
        return (amplitude * np.exp(1j * phase)).astype(np.complex64)
    if mode == "sector_max_energy":
        index = np.argmax(np.abs(sectors), axis=2)
        return np.take_along_axis(
            sectors, index[:, :, None, :], axis=2)[:, :, 0].astype(np.complex64)
    raise ValueError(f"Unknown A8 reducer {mode!r}.")


def keep_center_doppler(dar: np.ndarray, keep_bins: int) -> np.ndarray:
    """Keep the centered physical Doppler support used by RADs-like I/Q-1M."""
    keep_bins = int(keep_bins)
    if keep_bins <= 0 or keep_bins >= dar.shape[0]:
        return dar
    start = (dar.shape[0] - keep_bins) // 2
    out = np.zeros_like(dar)
    out[start:start + keep_bins] = dar[start:start + keep_bins]
    return out


def apply_mag_keep_frac(sample: np.ndarray, keep_frac: float) -> tuple[np.ndarray, float]:
    keep_frac = float(keep_frac)
    if keep_frac >= 1.0:
        return sample, 0.0
    if keep_frac <= 0.0:
        out = sample.copy()
        out[..., 0] = 0.0
        out[..., 1] = 0.0
        return out, float("inf")
    mag = sample[..., 0]
    threshold = float(np.quantile(mag.reshape(-1), 1.0 - keep_frac))
    out = sample.copy()
    mask = out[..., 0] >= threshold
    out[..., 0] = np.where(mask, out[..., 0], 0.0)
    out[..., 1] = np.where(mask, out[..., 1], 0.0)
    return out, threshold


def build_sample(
    cube_raz: np.ndarray,
    *,
    flip_azimuth: bool,
    azimuth_flip_mode: str,
    target_azimuth_bins: int,
    a8_reducer: str,
    beam_sigma: float,
    aperture_start: int,
    range_smooth: float,
    az_smooth: float,
    spatial_smooth_mode: str,
    amp_scale: float,
    amp_gamma: float,
    amp_clip: float,
    mag_keep_frac: float,
    doppler_keep_bins: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float | str],
]:
    dar = to_dar(
        cube_raz,
        flip_azimuth=flip_azimuth,
        azimuth_flip_mode=azimuth_flip_mode,
    )
    raw_mag = np.sqrt(np.abs(dar)).astype(np.float32)
    raw_views = {
        "ra": raw_mag.max(axis=0),      # [azimuth, range]
        "rd": raw_mag.max(axis=1),      # [doppler, range]
        "ad": raw_mag.max(axis=2),      # [doppler, azimuth]
    }
    dar = keep_center_doppler(dar, doppler_keep_bins)

    complex_model = reduce_azimuth(
        dar, target_azimuth_bins, a8_reducer, beam_sigma, aperture_start)
    if spatial_smooth_mode == "power_peak_phase":
        complex_model = smooth_complex_power_peak_phase(
            complex_model, range_smooth, axis=2)
        complex_model = smooth_complex_power_peak_phase(
            complex_model, az_smooth, axis=1)
    elif spatial_smooth_mode not in {"none", "legacy_amplitude"}:
        raise ValueError(
            f"unknown spatial_smooth_mode {spatial_smooth_mode!r}")
    amplitude = np.sqrt(np.abs(complex_model)).astype(np.float32)
    if spatial_smooth_mode == "legacy_amplitude":
        amplitude = smooth_axis(amplitude, range_smooth, axis=2)
        amplitude = smooth_axis(amplitude, az_smooth, axis=1)
    if amp_gamma <= 0.0:
        raise ValueError("amp_gamma must be positive.")
    amplitude = float(amp_scale) * np.power(amplitude, float(amp_gamma))
    if amp_clip > 0.0:
        amplitude = np.minimum(amplitude, float(amp_clip))
    phase = ((np.angle(complex_model) + np.pi) % (2 * np.pi) - np.pi).astype(
        np.float32)
    sample = np.stack([amplitude, phase], axis=-1)[:, :, None, :, :].astype(
        np.float32, copy=False)
    sample, mag_keep_threshold = apply_mag_keep_frac(sample, mag_keep_frac)
    amp_model = sample[:, :, 0, :, 0]
    phase_model = sample[:, :, 0, :, 1]
    input_views = {
        "ra": amp_model.max(axis=0),
        "rd": amp_model.max(axis=1),
        "ad": amp_model.max(axis=2),
    }
    stats = {
        "mag_keep_frac": float(mag_keep_frac),
        "doppler_keep_bins": float(doppler_keep_bins),
        "azimuth_bins": float(target_azimuth_bins),
        "amp_gamma": float(amp_gamma),
        "amp_clip": float(amp_clip),
        "spatial_smooth_mode": spatial_smooth_mode,
        "mag_keep_threshold": float(mag_keep_threshold),
        "mag_nonzero_frac": float(np.mean(amp_model > 0.0)),
        "mag_min": float(amp_model.min()),
        "mag_mean": float(amp_model.mean()),
        "mag_p99": float(np.percentile(amp_model, 99.0)),
        "mag_max": float(amp_model.max()),
        "phase_min": float(phase_model.min()),
        "phase_max": float(phase_model.max()),
    }
    return sample, raw_views, input_views, stats


def prepare_modes(
    path: Path,
    modes: list[str],
    *,
    flip_azimuth: bool,
    azimuth_flip_mode: str = "index",
    target_azimuth_bins: int,
    a8_reducer: str,
    beam_sigma: float,
    aperture_start: int,
    range_smooth: float,
    az_smooth: float,
    spatial_smooth_mode: str = "none",
    amp_scale: float = 1.0,
    amp_gamma: float = 1.0,
    amp_clip: float = 0.0,
    crop_fraction: float = 1.0,
    crop_source: str = "gt",
    mag_keep_frac: float = 1.0,
    doppler_keep_bins: int = 0,
    gt_path: Path | None = None,
    gt_crop_start: int | None = None,
    gt_ar_manifest: np.ndarray | None = None,
) -> list[Prepared]:
    cube = np.load(path)
    if cube.shape != (256, 256, 64):
        raise ValueError(f"expected RADs [range, azimuth, doppler] shape (256,256,64), got {cube.shape}")
    cube = cube.astype(np.complex64, copy=False)
    gt_cube = None
    if gt_path is not None and gt_path.exists():
        gt_cube = np.load(gt_path)
        if gt_cube.shape != cube.shape:
            raise ValueError(
                f"RADs_gt must match RADs shape {cube.shape}, got {gt_cube.shape}.")
    if not 0.0 <= crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in [0, 1].")
    if gt_cube is not None:
        auto_start, selected_crop_source, raw_start, gt_start = select_crop_start(
            cube, gt_cube, crop_source)
    else:
        raw_start, _ = first_signal_index(cube)
        gt_start = gt_crop_start
        if crop_source == "gt" and gt_start is not None:
            auto_start, selected_crop_source = gt_start, "gt_manifest"
        else:
            auto_start, selected_crop_source = raw_start, "raw"
    prepared: list[Prepared] = []
    for mode in modes:
        if mode == "nocrop":
            start = 0
        elif mode == "crop_auto":
            start = int(round(auto_start * crop_fraction))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        shifted = shift_range_cube(cube, start)
        sample, raw_views, input_views, stats = build_sample(
            shifted,
            flip_azimuth=flip_azimuth,
            azimuth_flip_mode=azimuth_flip_mode,
            target_azimuth_bins=target_azimuth_bins,
            a8_reducer=a8_reducer,
            beam_sigma=beam_sigma,
            aperture_start=aperture_start,
            range_smooth=range_smooth,
            az_smooth=az_smooth,
            spatial_smooth_mode=spatial_smooth_mode,
            amp_scale=amp_scale,
            amp_gamma=amp_gamma,
            amp_clip=amp_clip,
            mag_keep_frac=mag_keep_frac,
            doppler_keep_bins=doppler_keep_bins,
        )
        gt_polar = None
        if gt_cube is not None:
            shifted_gt = shift_range_cube(gt_cube, start)
            gt_ar = np.any(
                np.abs(to_dar(
                    shifted_gt,
                    flip_azimuth=flip_azimuth,
                    azimuth_flip_mode=azimuth_flip_mode,
                )) > 0,
                axis=0,
            )
            gt_polar = gt_ar.reshape(128, 2, 64, 4).max(axis=(1, 3))
        elif gt_ar_manifest is not None:
            gt_polar = project_gt_ar(
                gt_ar_manifest,
                start,
                flip_azimuth=flip_azimuth,
                azimuth_flip_mode=azimuth_flip_mode,
            )
        prepared.append(Prepared(
            mode,
            start,
            "none" if mode == "nocrop" else selected_crop_source,
            raw_start,
            gt_start,
            target_azimuth_bins,
            shifted,
            sample,
            raw_views,
            input_views,
            stats,
            gt_polar,
        ))
    return prepared


def first_hit_depth(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    occ = logits > threshold
    has = occ.any(axis=-1)
    depth = np.argmax(occ.astype(np.uint8), axis=-1).astype(np.float32)
    depth[~has] = np.nan
    return depth, float(has.mean())


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Sparse occupancy diagnostics used only for relative RADs comparisons."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"metric shape mismatch: pred={pred.shape}, target={target.shape}")
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_fraction": float(pred.mean()),
        "target_fraction": float(target.mean()),
    }


def polar_to_cartesian(az_range: np.ndarray, fov_deg: float = 90.0, n: int = 360) -> np.ndarray:
    az_count, range_count = az_range.shape
    angles = np.deg2rad(np.linspace(-fov_deg / 2, fov_deg / 2, az_count))
    ranges = np.arange(range_count, dtype=np.float32) + 0.5
    rr, aa = np.meshgrid(ranges, angles, indexing="xy")
    x = rr * np.sin(aa)
    y = rr * np.cos(aa)
    xlim = (-range_count * 0.85, range_count * 0.85)
    ylim = (0.0, range_count * 1.05)
    xi = np.floor((x - xlim[0]) / (xlim[1] - xlim[0]) * (n - 1)).astype(np.int32)
    yi = np.floor((y - ylim[0]) / (ylim[1] - ylim[0]) * (n - 1)).astype(np.int32)
    valid = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    out = np.zeros((n, n), dtype=np.float32)
    np.maximum.at(out, (n - 1 - yi[valid], xi[valid]), az_range.astype(np.float32)[valid])
    return out


def infer_one(
    model: torch.nn.Module, prepared: Prepared, device: torch.device
) -> dict:
    radar = torch.from_numpy(prepared.sample[None]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        y_hat = model({"radar": radar})
    if "map" not in y_hat:
        raise KeyError(f"model output keys={sorted(y_hat.keys())}, expected 'map'")
    logits_t = y_hat["map"][0].detach().cpu().float()
    logits = logits_t.numpy()
    prob = torch.sigmoid(logits_t).numpy()
    pred = {
        "logits": logits,
        "prob": prob,
        "logit_stats": {
            "min": float(logits.min()),
            "mean": float(logits.mean()),
            "max": float(logits.max()),
        },
    }
    for threshold in (-1.0, 0.0, 1.0):
        depth, valid = first_hit_depth(logits, threshold)
        pred[f"depth_logit_gt_{threshold:g}"] = depth
        pred[f"valid_frac_logit_gt_{threshold:g}"] = valid
        pred[f"invalid_mask_logit_gt_{threshold:g}"] = ~np.isfinite(depth)
        bev = (logits > threshold).max(axis=0).astype(np.float32)
        pred[f"bev_polar_logit_gt_{threshold:g}"] = bev
        pred[f"bev_cart_logit_gt_{threshold:g}"] = polar_to_cartesian(bev)
    pred["bev_polar_prob_maxe"] = prob.max(axis=0)
    pred["bev_cart_prob_maxe"] = polar_to_cartesian(pred["bev_polar_prob_maxe"])
    return pred


def imshow_mag(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(display_mag(image), cmap="magma", origin="upper", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def imshow_prob(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="inferno", origin="upper", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def imshow_depth(ax: plt.Axes, image: np.ndarray, title: str, vmax: int) -> None:
    masked = np.ma.masked_invalid(image)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    ax.imshow(masked, cmap=cmap, origin="upper", aspect="auto", vmin=0.0, vmax=float(vmax))
    ax.set_facecolor("black")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def imshow_invalid(ax: plt.Axes, mask: np.ndarray, title: str) -> None:
    ax.imshow(mask.astype(np.float32), cmap="gray", origin="upper", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title + " | white=invalid", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def summarize_frame(
    frame_path: Path,
    prepared: list[Prepared],
    preds: list[dict],
    ckpt: Path,
) -> dict:
    summary = {
        "frame": str(frame_path),
        "checkpoint": str(ckpt),
        "modes": [],
    }
    for col, (prep, pred) in enumerate(zip(prepared, preds)):
        stats = {
            "mode": prep.mode,
            "crop_start": prep.crop_start,
            "crop_source": prep.crop_source,
            "raw_crop_start": prep.raw_crop_start,
            "gt_crop_start": prep.gt_crop_start,
            "input_stats": prep.input_stats,
            "logit_stats": pred["logit_stats"],
            "valid_frac_logit_gt_-1": pred["valid_frac_logit_gt_-1"],
            "invalid_frac_logit_gt_-1": 1.0 - pred["valid_frac_logit_gt_-1"],
            "valid_frac_logit_gt_0": pred["valid_frac_logit_gt_0"],
            "invalid_frac_logit_gt_0": 1.0 - pred["valid_frac_logit_gt_0"],
            "valid_frac_logit_gt_1": pred["valid_frac_logit_gt_1"],
            "invalid_frac_logit_gt_1": 1.0 - pred["valid_frac_logit_gt_1"],
        }
        if prep.gt_polar is not None:
            stats["rads_gt_note"] = (
                "RADs_gt is sparse radar occupancy, while GRT predicts dense "
                "LiDAR occupancy; use these scores only for relative calibration.")
            for threshold in (-1.0, 0.0, 1.0):
                stats[f"rads_gt_logit_gt_{threshold:g}"] = binary_metrics(
                    pred[f"bev_polar_logit_gt_{threshold:g}"], prep.gt_polar)
        summary["modes"].append(stats)
    return summary


def render_frame(
    out_path: Path,
    frame_path: Path,
    prepared: list[Prepared],
    preds: list[dict],
    ckpt: Path,
) -> dict:
    n_cols = len(prepared)
    row_defs = [
        ("raw RA sqrt(abs), x=range", "raw_ra"),
        ("raw RD sqrt(abs), x=range", "raw_rd"),
        ("raw AD sqrt(abs), x=azimuth", "raw_ad"),
        ("model input RA mag", "input_ra"),
        ("model input RD mag", "input_rd"),
        ("model input AD mag", "input_ad"),
        ("pred BEV polar prob maxE", "bev_polar_prob"),
        ("RADs_gt polar occupancy", "gt_polar"),
        ("first-hit depth logit>-1", "depth_-1"),
        ("invalid mask logit>-1", "invalid_-1"),
        ("first-hit depth logit>0", "depth_0"),
        ("invalid mask logit>0", "invalid_0"),
        ("BEV cart prob maxE", "bev_cart_prob"),
        ("BEV cart logit>-1", "bev_cart_-1"),
    ]
    fig, axes = plt.subplots(
        len(row_defs),
        n_cols,
        figsize=(6.4 * n_cols, 2.4 * len(row_defs)),
        constrained_layout=True,
        squeeze=False,
    )
    summary = summarize_frame(frame_path, prepared, preds, ckpt)
    for col, (prep, pred) in enumerate(zip(prepared, preds)):
        col_title = (
            f"{prep.mode} crop_start={prep.crop_start} source={prep.crop_source}\n"
            f"in mean/p99/max={prep.input_stats['mag_mean']:.3g}/"
            f"{prep.input_stats['mag_p99']:.3g}/{prep.input_stats['mag_max']:.3g}\n"
            f"logit min/mean/max={pred['logit_stats']['min']:.2f}/"
            f"{pred['logit_stats']['mean']:.2f}/{pred['logit_stats']['max']:.2f}"
        )
        axes[0, col].set_title(col_title, fontsize=9)
        for row, (_label, key) in enumerate(row_defs):
            ax = axes[row, col]
            if row == 0:
                image = prep.raw_views["ra"]
                imshow_mag(ax, image, col_title)
            elif key == "raw_rd":
                imshow_mag(ax, prep.raw_views["rd"], "raw RD: y=doppler, x=range")
            elif key == "raw_ad":
                imshow_mag(ax, prep.raw_views["ad"], "raw AD: y=doppler, x=azimuth")
            elif key == "input_ra":
                imshow_mag(
                    ax,
                    prep.input_views["ra"],
                    f"model input RA: y={prep.azimuth_bins} bins, x=range",
                )
            elif key == "input_rd":
                imshow_mag(
                    ax,
                    prep.input_views["rd"],
                    "model input RD: y=doppler, x=range",
                )
            elif key == "input_ad":
                imshow_mag(
                    ax,
                    prep.input_views["ad"],
                    "model input AD: y=doppler, x=azimuth",
                )
            elif key == "bev_polar_prob":
                imshow_prob(ax, pred["bev_polar_prob_maxe"], "pred BEV polar prob maxE")
            elif key == "gt_polar":
                if prep.gt_polar is None:
                    ax.set_facecolor("black")
                    ax.text(
                        0.5, 0.5, "RADs_gt unavailable", color="white",
                        ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    imshow_prob(
                        ax,
                        prep.gt_polar.astype(np.float32),
                        "RADs_gt polar occupancy",
                    )
            elif key == "depth_-1":
                imshow_depth(
                    ax,
                    pred["depth_logit_gt_-1"],
                    f"depth logit>-1 valid={pred['valid_frac_logit_gt_-1']:.3f}",
                    vmax=pred["logits"].shape[-1] - 1,
                )
            elif key == "invalid_-1":
                imshow_invalid(
                    ax,
                    pred["invalid_mask_logit_gt_-1"],
                    f"invalid depth logit>-1 frac={1.0 - pred['valid_frac_logit_gt_-1']:.3f}",
                )
            elif key == "depth_0":
                imshow_depth(
                    ax,
                    pred["depth_logit_gt_0"],
                    f"depth logit>0 valid={pred['valid_frac_logit_gt_0']:.3f}",
                    vmax=pred["logits"].shape[-1] - 1,
                )
            elif key == "invalid_0":
                imshow_invalid(
                    ax,
                    pred["invalid_mask_logit_gt_0"],
                    f"invalid depth logit>0 frac={1.0 - pred['valid_frac_logit_gt_0']:.3f}",
                )
            elif key == "bev_cart_prob":
                imshow_prob(ax, pred["bev_cart_prob_maxe"], "cart BEV prob maxE")
            elif key == "bev_cart_-1":
                imshow_prob(ax, pred["bev_cart_logit_gt_-1"], "cart BEV occupancy logit>-1")
            if col == 0:
                ax.set_ylabel(row_defs[row][0], fontsize=9)
    fig.suptitle(
        f"DeepRadar dimension-aligned map checkpoint on RADs | "
        f"{frame_path.parent.name}/{frame_path.stem}",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="GRT repository root used to import deepradar.")
    parser.add_argument("--rads-root", type=Path, default=Path("/root/autodl-tmp/data/RADs"))
    parser.add_argument("--gt-root", type=Path, default=Path("/root/autodl-tmp/data/RADs_gt"))
    parser.add_argument(
        "--gt-manifest",
        type=Path,
        help=(
            "Optional compact manifest with per-frame GT crop starts and "
            "native azimuth-range sparse indices. Full GT cubes take precedence."
        ),
    )
    parser.add_argument(
        "--require-gt",
        action="store_true",
        help="Fail instead of falling back to raw crop when GT is unavailable.",
    )
    parser.add_argument("--frames", nargs="+", default=["100/000050", "100/000001", "101/000001"])
    parser.add_argument(
        "--modes", nargs="+", choices=["nocrop", "crop_auto"],
        default=["crop_auto"],
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/root/autodl-fs/outputs/grt/rads_map_checkpoint_infer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam-sigma", type=float, default=18.0)
    parser.add_argument(
        "--input-azimuth-bins", choices=["auto", "8", "256"], default="8")
    parser.add_argument(
        "--a8-reducer",
        choices=[
            "aperture_truncate",
            "aperture_window",
            "gaussian_coherent",
            "sector_coherent",
            "sector_energy_circular",
            "sector_max_energy",
        ],
        default="aperture_truncate",
    )
    parser.add_argument(
        "--aperture-start",
        type=int,
        default=0,
        help=(
            "Circular aperture-window start used by aperture_window; "
            "-3 selects [253,254,255,0,1,2,3,4]."
        ),
    )
    parser.add_argument("--range-smooth", type=float, default=0.0)
    parser.add_argument("--az-smooth", type=float, default=0.0)
    parser.add_argument(
        "--spatial-smooth-mode",
        choices=["none", "power_peak_phase", "legacy_amplitude"],
        default="none",
        help=(
            "Spatial smoothing contract. 'none' matches the precomputed "
            "training cache; 'power_peak_phase' preserves a valid complex "
            "signal; 'legacy_amplitude' reproduces the old non-physical path."
        ),
    )
    parser.add_argument(
        "--crop-fraction", type=float, default=1.0,
        help="Fraction of the automatically detected empty near-range prefix to shift out.",
    )
    parser.add_argument(
        "--crop-source", choices=["gt", "raw"], default="gt",
        help="Use matched RADs_gt for the crop when available; otherwise fall back to raw.",
    )
    parser.add_argument(
        "--amp-scale", type=float, default=2.6245,
        help=(
            "Fixed magnitude scale applied after the global power law; the "
            "default was selected on the fixed 48-frame RADs calibration set."
        ),
    )
    parser.add_argument("--mag-keep-frac", type=float, default=1.0)
    parser.add_argument(
        "--amp-gamma", type=float, default=0.45,
        help=(
            "Global magnitude power-law exponent; the default was selected "
            "on the fixed 48-frame RADs calibration set and is never "
            "estimated per frame."
        ),
    )
    parser.add_argument(
        "--amp-clip", type=float, default=0.0,
        help="Optional fixed post-scale magnitude ceiling; 0 disables clipping.",
    )
    parser.add_argument(
        "--doppler-keep-bins", type=int, default=11,
        help=(
            "Keep only this many centered Doppler bins before representation "
            "conversion; 0 keeps all bins. The RADs-like training cache "
            "occupies exactly the centered 11 of 64 bins."
        ),
    )
    parser.add_argument(
        "--azimuth-flip-mode",
        choices=["index", "spectral"],
        default="index",
        help=(
            "'index' reproduces the established RADs convention; 'spectral' "
            "uses exact centered-frequency reflection on an even grid."
        ),
    )
    parser.add_argument("--no-flip-azimuth", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Compute JSON metrics without rendering PNGs or saving prediction arrays.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repo = args.repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from deepradar import DeepRadar  # noqa: PLC0415

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=device,
    )
    model.eval().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt_manifest_frames: dict[str, dict] = {}
    if args.gt_manifest is not None:
        gt_manifest_payload = json.loads(args.gt_manifest.read_text())
        if gt_manifest_payload.get("native_ar_shape") != [256, 256]:
            raise ValueError(
                "GT manifest native_ar_shape must be [256,256].")
        gt_manifest_frames = gt_manifest_payload["frames"]

    if args.input_azimuth_bins == "auto":
        target_azimuth_bins = int(getattr(model.encoder, "azimuth_bins", 8))
    else:
        target_azimuth_bins = int(args.input_azimuth_bins)
    encoder_azimuth_bins = int(getattr(model.encoder, "azimuth_bins", 8))
    supports_phase_expansion = hasattr(model.encoder, "azimuth_bins")
    if target_azimuth_bins > encoder_azimuth_bins or (
        not supports_phase_expansion and target_azimuth_bins != encoder_azimuth_bins
    ):
        raise ValueError(
            f"Checkpoint encoder supports input A<={encoder_azimuth_bins}, "
            f"but requested A={target_azimuth_bins}.")

    all_summary = {
        "repo": str(repo),
        "rads_root": str(args.rads_root),
        "gt_root": str(args.gt_root),
        "gt_manifest": str(args.gt_manifest) if args.gt_manifest else None,
        "out_dir": str(args.out_dir),
        "checkpoint": str(args.checkpoint),
        "hparams": str(args.hparams),
        "device": str(device),
        "input_azimuth_bins": target_azimuth_bins,
        "crop_fraction": args.crop_fraction,
        "crop_source": args.crop_source,
        "a8_reducer": args.a8_reducer if target_azimuth_bins == 8 else None,
        "aperture_start": args.aperture_start if target_azimuth_bins == 8 else None,
        "doppler_keep_bins": args.doppler_keep_bins,
        "azimuth_flip": not args.no_flip_azimuth,
        "azimuth_flip_mode": args.azimuth_flip_mode,
        "frames": [],
        "notes": [
            "RADs arrays are treated as [range, azimuth, doppler].",
            "RADs inference uses crop_auto by default.",
            "crop_auto shifts the whole 3D RAD cube along range before any RA/RD/AD/model input derivation; matched RADs_gt is preferred and raw detection is only a fallback.",
            "RADs-like defaults keep the centered 11/64 Doppler bins, disable spatial smoothing, and use the fixed gamma=0.45/scale=2.6245 magnitude calibration.",
            f"Model input is [D,A,E,R,C]=[64,{target_azimuth_bins},1,256,2], "
            "where C=(sqrt(abs(complex)), phase).",
            "A256 checkpoints retain the cropped RADs azimuth axis directly; "
            "A8 checkpoints use the explicitly selected reducer.",
            "Prediction is the trained DeepRadar map head, output [elevation, azimuth, range]=[64,128,64].",
            "Depth invalid pixels are rendered black; separate invalid-mask rows show white where no first-hit exists.",
        ],
    }
    for frame in args.frames:
        rel = Path(frame)
        frame_path = args.rads_root / rel.parent / f"{rel.name}.npy"
        manifest_entry = gt_manifest_frames.get(rel.as_posix())
        gt_ar_manifest = None
        gt_crop_start = None
        if manifest_entry is not None:
            gt_crop_start = int(manifest_entry["crop_start"])
            gt_ar_manifest = np.zeros((256, 256), dtype=bool)
            gt_ar_manifest.flat[
                np.asarray(manifest_entry["ar_indices"], dtype=np.int64)
            ] = True
        gt_path = args.gt_root / rel.parent / f"{rel.name}.npy"
        if args.require_gt and not gt_path.exists() and manifest_entry is None:
            raise FileNotFoundError(
                f"GT required but unavailable for {rel.as_posix()}")
        prepared = prepare_modes(
            frame_path,
            args.modes,
            flip_azimuth=not args.no_flip_azimuth,
            azimuth_flip_mode=args.azimuth_flip_mode,
            target_azimuth_bins=target_azimuth_bins,
            a8_reducer=args.a8_reducer,
            beam_sigma=args.beam_sigma,
            aperture_start=args.aperture_start,
            range_smooth=args.range_smooth,
            az_smooth=args.az_smooth,
            spatial_smooth_mode=args.spatial_smooth_mode,
            amp_scale=args.amp_scale,
            amp_gamma=args.amp_gamma,
            amp_clip=args.amp_clip,
            crop_fraction=args.crop_fraction,
            crop_source=args.crop_source,
            mag_keep_frac=args.mag_keep_frac,
            doppler_keep_bins=args.doppler_keep_bins,
            gt_path=gt_path,
            gt_crop_start=gt_crop_start,
            gt_ar_manifest=gt_ar_manifest,
        )
        preds = [infer_one(model, prep, device) for prep in prepared]
        out_name = (
            f"rads_{rel.parent.name}_{rel.name}_map_ckpt_"
            f"a{target_azimuth_bins}_crop_compare.png")
        if args.summary_only:
            summary = summarize_frame(
                frame_path, prepared, preds, args.checkpoint)
        else:
            summary = render_frame(
                args.out_dir / out_name,
                frame_path,
                prepared,
                preds,
                args.checkpoint,
            )
            for prep, pred in zip(prepared, preds):
                arrays_path = args.out_dir / (
                    f"rads_{rel.parent.name}_{rel.name}_{prep.mode}_predictions.npz"
                )
                np.savez_compressed(
                    arrays_path,
                    crop_start=np.asarray(prep.crop_start, dtype=np.int16),
                    bev_polar_prob_maxe=pred["bev_polar_prob_maxe"].astype(np.float16),
                    bev_polar_logit_gt_m1=pred["bev_polar_logit_gt_-1"].astype(np.uint8),
                    bev_polar_logit_gt_0=pred["bev_polar_logit_gt_0"].astype(np.uint8),
                    bev_polar_logit_gt_1=pred["bev_polar_logit_gt_1"].astype(np.uint8),
                    depth_logit_gt_m1=pred["depth_logit_gt_-1"].astype(np.float16),
                    depth_logit_gt_0=pred["depth_logit_gt_0"].astype(np.float16),
                    depth_logit_gt_1=pred["depth_logit_gt_1"].astype(np.float16),
                    invalid_mask_logit_gt_m1=pred["invalid_mask_logit_gt_-1"].astype(np.uint8),
                    invalid_mask_logit_gt_0=pred["invalid_mask_logit_gt_0"].astype(np.uint8),
                    invalid_mask_logit_gt_1=pred["invalid_mask_logit_gt_1"].astype(np.uint8),
                    input_azimuth_bins=np.asarray(prep.azimuth_bins, dtype=np.int16),
                    gt_polar=(
                        prep.gt_polar.astype(np.uint8)
                        if prep.gt_polar is not None
                        else np.empty((0, 0), dtype=np.uint8)
                    ),
                )
                print("ARRAYS", arrays_path, flush=True)
        all_summary["frames"].append(summary)
        if not args.summary_only:
            print("SAVED", args.out_dir / out_name, flush=True)
        print("SUMMARY", json.dumps(summary["modes"]), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_path = args.out_dir / "rads_map_checkpoint_infer_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print("SUMMARY_JSON", summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
