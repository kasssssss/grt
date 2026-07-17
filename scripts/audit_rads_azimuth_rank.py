#!/usr/bin/env python3
"""Measure whether RADs azimuth signals admit a useful rank-8 linear subspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_vectors(
    path: Path,
    *,
    vectors_per_frame: int,
    doppler_keep_bins: int,
    flip_azimuth: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    cube = np.load(path, mmap_mode="r")
    if cube.shape != (256, 256, 64):
        raise ValueError(f"Expected RADs [R,A,D]=[256,256,64], got {cube.shape} at {path}")
    dar = np.transpose(cube, (2, 1, 0))
    if flip_azimuth:
        dar = dar[:, ::-1, :]
    if doppler_keep_bins > 0:
        start = (dar.shape[0] - doppler_keep_bins) // 2
        dar = dar[start : start + doppler_keep_bins]

    energy = np.sum(np.abs(dar) ** 2, axis=1)
    flat_energy = energy.reshape(-1)
    count = min(vectors_per_frame, flat_energy.size)
    top_count = count // 2
    random_count = count - top_count
    top = np.argpartition(flat_energy, -top_count)[-top_count:] if top_count else np.empty(0, int)
    random = rng.choice(flat_energy.size, size=random_count, replace=False)
    indices = np.unique(np.concatenate([top, random]))
    d, r = np.unravel_index(indices, energy.shape)
    return np.asarray(dar[d, :, r], dtype=np.complex64)


def second_moment(vectors: list[np.ndarray], normalize: bool) -> np.ndarray:
    covariance = np.zeros((256, 256), dtype=np.complex128)
    for x in vectors:
        if normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        covariance += x.conj().T @ x
    return covariance


def principal_basis(covariance: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[:rank]], eigenvalues[order]


def captured_fraction(x: np.ndarray, basis: np.ndarray) -> float:
    coefficients = x @ basis
    return float(np.sum(np.abs(coefficients) ** 2) / np.maximum(np.sum(np.abs(x) ** 2), 1e-12))


def aperture_fraction(x: np.ndarray, rank: int) -> float:
    aperture = np.fft.ifft(np.fft.ifftshift(x, axes=1), axis=1)
    return float(
        np.sum(np.abs(aperture[:, :rank]) ** 2)
        / np.maximum(np.sum(np.abs(aperture) ** 2), 1e-12)
    )


def aperture_indices_fraction(x: np.ndarray, indices: np.ndarray) -> float:
    aperture = np.fft.ifft(np.fft.ifftshift(x, axes=1), axis=1)
    return float(
        np.sum(np.abs(aperture[:, indices]) ** 2)
        / np.maximum(np.sum(np.abs(aperture) ** 2), 1e-12)
    )


def aperture_selection(vectors: list[np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.zeros(256, dtype=np.float64)
    for x in vectors:
        aperture = np.fft.ifft(np.fft.ifftshift(x, axes=1), axis=1)
        energy += np.sum(np.abs(aperture) ** 2, axis=0)
    top = np.argsort(energy)[::-1][:rank]
    window_energy = np.asarray(
        [np.sum(energy[(start + np.arange(rank)) % energy.size]) for start in range(energy.size)]
    )
    start = int(np.argmax(window_energy))
    window = (start + np.arange(rank)) % energy.size
    return top, window, energy / np.sum(energy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rads-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-frames", type=int, default=64)
    parser.add_argument("--test-frames", type=int, default=16)
    parser.add_argument("--vectors-per-frame", type=int, default=1024)
    parser.add_argument("--doppler-keep-bins", type=int, default=11)
    parser.add_argument("--flip-azimuth", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--basis-out", type=Path)
    parser.add_argument("--exclude-frames", nargs="*", default=[])
    parser.add_argument("--train-sequences", nargs="*")
    parser.add_argument("--test-sequences", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded = {str(Path(item).with_suffix("")) for item in args.exclude_frames}
    files = [
        path
        for path in sorted(args.rads_root.glob("*/*.npy"))
        if str(path.relative_to(args.rads_root).with_suffix("")) not in excluded
    ]
    needed = args.train_frames + args.test_frames
    rng = np.random.default_rng(args.seed)
    if args.train_sequences or args.test_sequences:
        if not args.train_sequences or not args.test_sequences:
            raise ValueError("Specify both --train-sequences and --test-sequences")
        train_pool = [path for path in files if path.parent.name in args.train_sequences]
        test_pool = [path for path in files if path.parent.name in args.test_sequences]
        if len(train_pool) < args.train_frames or len(test_pool) < args.test_frames:
            raise ValueError(
                f"Insufficient sequence-disjoint frames: train={len(train_pool)}, test={len(test_pool)}"
            )
        train_files = [train_pool[i] for i in rng.choice(len(train_pool), args.train_frames, replace=False)]
        test_files = [test_pool[i] for i in rng.choice(len(test_pool), args.test_frames, replace=False)]
    else:
        if len(files) < needed:
            raise ValueError(f"Need {needed} RADs frames, found {len(files)}")
        selected = rng.choice(len(files), size=needed, replace=False)
        train_files = [files[i] for i in selected[: args.train_frames]]
        test_files = [files[i] for i in selected[args.train_frames :]]

    train = [
        load_vectors(
            path,
            vectors_per_frame=args.vectors_per_frame,
            doppler_keep_bins=args.doppler_keep_bins,
            flip_azimuth=args.flip_azimuth,
            rng=rng,
        )
        for path in train_files
    ]
    test = [
        load_vectors(
            path,
            vectors_per_frame=args.vectors_per_frame,
            doppler_keep_bins=args.doppler_keep_bins,
            flip_azimuth=args.flip_azimuth,
            rng=rng,
        )
        for path in test_files
    ]
    test_matrix = np.concatenate(test)

    energy_basis, energy_spectrum = principal_basis(second_moment(train, normalize=False), args.rank)
    direction_basis, direction_spectrum = principal_basis(second_moment(train, normalize=True), args.rank)
    top_aperture, window_aperture, aperture_profile = aperture_selection(train, args.rank)
    result = {
        "rank": args.rank,
        "doppler_keep_bins": args.doppler_keep_bins,
        "flip_azimuth": args.flip_azimuth,
        "train_frames": len(train_files),
        "test_frames": len(test_files),
        "test_vectors": int(test_matrix.shape[0]),
        "heldout_energy_fraction": {
            "fixed_first_aperture": aperture_fraction(test_matrix, args.rank),
            "top_energy_aperture": aperture_indices_fraction(test_matrix, top_aperture),
            "best_contiguous_aperture": aperture_indices_fraction(test_matrix, window_aperture),
            "energy_pca": captured_fraction(test_matrix, energy_basis),
            "direction_pca": captured_fraction(test_matrix, direction_basis),
        },
        "aperture_selection": {
            "top_energy_indices": top_aperture.tolist(),
            "best_contiguous_indices": window_aperture.tolist(),
            "energy_fraction_by_index": aperture_profile.tolist(),
        },
        "spectrum_fraction_top_rank": {
            "energy_pca": float(np.sum(energy_spectrum[: args.rank]) / np.sum(energy_spectrum)),
            "direction_pca": float(np.sum(direction_spectrum[: args.rank]) / np.sum(direction_spectrum)),
        },
        "train_files": [str(path.relative_to(args.rads_root)) for path in train_files],
        "test_files": [str(path.relative_to(args.rads_root)) for path in test_files],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.basis_out is not None:
        args.basis_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.basis_out,
            energy_basis=energy_basis.astype(np.complex64),
            direction_basis=direction_basis.astype(np.complex64),
            top_aperture_indices=top_aperture,
            contiguous_aperture_indices=window_aperture,
            aperture_energy_fraction=aperture_profile.astype(np.float32),
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
