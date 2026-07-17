#!/usr/bin/env python3
"""Compare raw-signal and RADs_gt near-range crop contracts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def first_raw_signal(cube: np.ndarray) -> tuple[int, float, np.ndarray]:
    profile = np.abs(cube).mean(axis=(1, 2))
    median = float(np.median(profile))
    mad = float(np.median(np.abs(profile - median)))
    maximum = float(np.max(profile))
    threshold = max(median + 6.0 * mad, maximum * 0.08)
    hits = np.flatnonzero(profile >= threshold)
    return (int(hits[0]) if hits.size else 0), threshold, profile


def first_gt_signal(gt: np.ndarray) -> int:
    hits = np.flatnonzero(np.any(gt != 0, axis=(1, 2)))
    return int(hits[0]) if hits.size else 0


def profile_quantile(profile: np.ndarray, quantile: float) -> int:
    profile = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
    total = float(profile.sum())
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(profile) / total, quantile, side="left"))


def summarize(values: list[int]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": int(data.min()),
        "q05": float(np.quantile(data, 0.05)),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "q95": float(np.quantile(data, 0.95)),
        "max": int(data.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rads-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.rads_root.rglob("*.npy"))
    rows: list[dict[str, object]] = []
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(args.rads_root)
        gt_path = args.gt_root / rel
        if not gt_path.exists():
            raise FileNotFoundError(f"missing matched GT: {gt_path}")
        cube = np.load(path, mmap_mode="r")
        gt = np.load(gt_path, mmap_mode="r")
        if cube.shape != (256, 256, 64) or gt.shape != cube.shape:
            raise ValueError(f"bad shapes for {rel}: RADs={cube.shape}, GT={gt.shape}")
        raw_start, threshold, profile = first_raw_signal(cube)
        gt_start = first_gt_signal(gt)
        row = {
            "frame": rel.as_posix(),
            "sequence": rel.parent.as_posix(),
            "raw_start": raw_start,
            "gt_start": gt_start,
            "raw_minus_gt": raw_start - gt_start,
            "power_q01": profile_quantile(np.square(profile), 0.01),
            "power_q05": profile_quantile(np.square(profile), 0.05),
            "raw_threshold": threshold,
            "raw_profile_at_gt": float(profile[gt_start]),
            "gt_points": int(np.count_nonzero(gt)),
        }
        rows.append(row)
        by_sequence[str(row["sequence"])].append(row)
        if index % 50 == 0 or index == len(files):
            print(f"processed {index}/{len(files)}", flush=True)

    delta = [int(row["raw_minus_gt"]) for row in rows]
    result = {
        "frames": len(rows),
        "raw_start": summarize([int(row["raw_start"]) for row in rows]),
        "gt_start": summarize([int(row["gt_start"]) for row in rows]),
        "raw_minus_gt": summarize(delta),
        "agreement": {
            "exact_fraction": sum(value == 0 for value in delta) / len(rows),
            "within_2_fraction": sum(abs(value) <= 2 for value in delta) / len(rows),
            "raw_before_gt_fraction": sum(value < 0 for value in delta) / len(rows),
            "raw_after_gt_fraction": sum(value > 0 for value in delta) / len(rows),
        },
        "sequence": {
            sequence: {
                "frames": len(sequence_rows),
                "raw_start": summarize([int(row["raw_start"]) for row in sequence_rows]),
                "gt_start": summarize([int(row["gt_start"]) for row in sequence_rows]),
                "raw_minus_gt": summarize([int(row["raw_minus_gt"]) for row in sequence_rows]),
            }
            for sequence, sequence_rows in sorted(by_sequence.items())
        },
        "largest_absolute_mismatches": sorted(
            rows, key=lambda row: abs(int(row["raw_minus_gt"])), reverse=True
        )[:20],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in (
        "frames", "raw_start", "gt_start", "raw_minus_gt", "agreement"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
