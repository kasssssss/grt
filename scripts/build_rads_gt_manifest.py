#!/usr/bin/env python3
"""Build a compact crop/occupancy manifest from RADs_gt cubes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    frames: dict[str, dict[str, int | list[int]]] = {}
    paths = sorted(args.gt_root.glob("*/*.npy"))
    if not paths:
        raise SystemExit(f"no RADs_gt cubes found under {args.gt_root}")
    for path in paths:
        cube = np.load(path, mmap_mode="r")
        if cube.shape != (256, 256, 64):
            raise ValueError(f"expected {path} shape (256,256,64), got {cube.shape}")
        occupied_ra = np.any(np.abs(cube) > 0, axis=2)
        active_ranges = np.flatnonzero(occupied_ra.any(axis=1))
        crop_start = int(active_ranges[0]) if active_ranges.size else 0
        native_ar = occupied_ra.T
        frames[f"{path.parent.name}/{path.stem}"] = {
            "crop_start": crop_start,
            "ar_indices": np.flatnonzero(native_ar).astype(int).tolist(),
        }

    payload = {
        "version": 1,
        "source_shape": [256, 256, 64],
        "source_axes": ["range", "azimuth", "doppler"],
        "native_ar_shape": [256, 256],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} frames={len(frames)} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
