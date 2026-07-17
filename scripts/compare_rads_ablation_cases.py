#!/usr/bin/env python3
"""Paired frame-level comparison for RADs preprocessing ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "f1_m1": ("rads_gt_logit_gt_-1", "f1"),
    "f1_p0": ("rads_gt_logit_gt_0", "f1"),
    "recall_m1": ("rads_gt_logit_gt_-1", "recall"),
    "recall_p0": ("rads_gt_logit_gt_0", "recall"),
    "pred_fraction_m1": ("rads_gt_logit_gt_-1", "pred_fraction"),
    "pred_fraction_p0": ("rads_gt_logit_gt_0", "pred_fraction"),
}


def load_case(root: Path, name: str) -> dict[str, dict]:
    path = root / name / "rads_map_checkpoint_infer_summary.json"
    payload = json.loads(path.read_text())
    return {
        Path(frame["frame"]).as_posix(): frame["modes"][0]
        for frame in payload["frames"]
    }


def values(frames: dict[str, dict], metric: str) -> np.ndarray:
    group, key = METRICS[metric]
    return np.asarray(
        [frames[name][group][key] for name in sorted(frames)],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--compare", nargs="+", required=True,
        help="Pairs formatted reference:candidate.",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    report = []
    cache: dict[str, dict[str, dict]] = {}
    for pair in args.compare:
        reference, candidate = pair.split(":", 1)
        for name in (reference, candidate):
            cache.setdefault(name, load_case(args.root, name))
        if cache[reference].keys() != cache[candidate].keys():
            raise ValueError(f"frame mismatch for {pair}")
        row = {
            "reference": reference,
            "candidate": candidate,
            "frames": len(cache[reference]),
            "metrics": {},
        }
        for metric in METRICS:
            ref = values(cache[reference], metric)
            cand = values(cache[candidate], metric)
            delta = cand - ref
            indices = rng.integers(
                0, delta.size, size=(args.bootstrap, delta.size))
            boot = delta[indices].mean(axis=1)
            row["metrics"][metric] = {
                "reference_mean": float(ref.mean()),
                "candidate_mean": float(cand.mean()),
                "mean_delta": float(delta.mean()),
                "delta_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "candidate_win_fraction": float(np.mean(delta > 0)),
                "equal_fraction": float(np.mean(delta == 0)),
            }
        report.append(row)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
