#!/usr/bin/env python3
"""Summarize fixed-frame RADs preprocessing ablations.

RADs_gt is sparse radar occupancy rather than dense LiDAR occupancy, so the
reported overlap metrics are intended for relative preprocessing comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


THRESHOLDS = (-1, 0, 1)


def mean(items: list[float]) -> float:
    return fmean(items) if items else float("nan")


def summarize_case(path: Path) -> dict[str, float | str | int]:
    payload = json.loads(path.read_text())
    modes = [frame["modes"][0] for frame in payload["frames"]]
    row: dict[str, float | str | int] = {
        "case": path.parent.name,
        "frames": len(modes),
        "crop_source": payload["crop_source"],
        "a8_reducer": payload["a8_reducer"],
        "aperture_start": payload["aperture_start"],
        "azimuth_flip": payload.get("azimuth_flip", True),
        "azimuth_flip_mode": payload.get("azimuth_flip_mode", "index"),
        "smooth_mode": modes[0]["input_stats"]["spatial_smooth_mode"],
        "amp_gamma": modes[0]["input_stats"]["amp_gamma"],
        "input_mag_mean": mean([m["input_stats"]["mag_mean"] for m in modes]),
        "input_mag_p99": mean([m["input_stats"]["mag_p99"] for m in modes]),
        "logit_mean": mean([m["logit_stats"]["mean"] for m in modes]),
        "logit_max": mean([m["logit_stats"]["max"] for m in modes]),
    }
    for threshold in THRESHOLDS:
        suffix = f"m{abs(threshold)}" if threshold < 0 else f"p{threshold}"
        metric_key = f"rads_gt_logit_gt_{threshold}"
        row[f"valid_{suffix}"] = mean(
            [m[f"valid_frac_logit_gt_{threshold}"] for m in modes]
        )
        for metric in ("precision", "recall", "f1", "iou", "pred_fraction"):
            row[f"{metric}_{suffix}"] = mean(
                [m[metric_key][metric] for m in modes]
            )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/rads_map_checkpoint_infer_summary.json"))
    if not paths:
        raise SystemExit(f"no summaries found under {args.root}")
    rows = [summarize_case(path) for path in paths]

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    columns = (
        "case",
        "valid_m1",
        "f1_m1",
        "recall_m1",
        "pred_fraction_m1",
        "valid_p0",
        "f1_p0",
        "recall_p0",
        "pred_fraction_p0",
        "input_mag_mean",
        "input_mag_p99",
    )
    print("\t".join(columns))
    for row in rows:
        print(
            "\t".join(
                str(row[column]) if column == "case" else f"{float(row[column]):.6f}"
                for column in columns
            )
        )


if __name__ == "__main__":
    main()
