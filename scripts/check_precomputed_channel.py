#!/usr/bin/env python3
"""Smoke-check the precomputed RADs-like radar channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--cache-root", type=Path, default=None,
        help="Override the cache_root from iq1m_radslike_precomputed.yaml.")
    p.add_argument("--trace", default="outdoor/baum")
    p.add_argument(
        "--atol", type=float, default=None,
        help="Maximum absolute cache error. Defaults to 1e-2 for float16 "
        "and 1e-6 for float32 caches.")
    p.add_argument(
        "--mean-atol", type=float, default=None,
        help="Maximum mean absolute cache error. Defaults to 1e-4 for "
        "float16 and 1e-7 for float32 caches.")
    return p.parse_args()


def radar_only(cfg: dict) -> dict:
    return {"radar": cfg["dataset"]["channels"]["radar"]}


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    import sys

    sys.path.insert(0, str(repo))
    from deepradar import config
    from deepradar.dataloader import RoverDataModule, RoverTrace

    raw_cfg = config.load_config(
        str(repo / "config/grt/grt.yaml"),
        str(repo / "config/grt/small.yaml"),
        str(repo / "config/data/outdoor.yaml"),
        str(repo / "config/repr/rads_like.yaml"),
    )
    cache_cfg = config.load_config(
        str(repo / "config/grt/grt.yaml"),
        str(repo / "config/grt/small.yaml"),
        str(repo / "config/data/outdoor.yaml"),
        str(repo / "config/repr/rads_like.yaml"),
        str(repo / "config/data/iq1m_radslike_precomputed.yaml"),
    )
    if args.cache_root is not None:
        cache_cfg["dataset"]["channels"]["radar"]["args"]["cache_root"] = str(
            args.cache_root)

    trace_path = Path(args.data_root) / args.trace
    raw = RoverTrace(str(trace_path), channels=radar_only(raw_cfg), bounds=(0.0, 1.0))
    cached = RoverTrace(str(trace_path), channels=radar_only(cache_cfg), bounds=(0.0, 1.0))
    if len(raw) != len(cached):
        raise SystemExit(f"length mismatch: raw={len(raw)} cached={len(cached)}")

    cache_root = Path(
        cache_cfg["dataset"]["channels"]["radar"]["args"]["cache_root"])
    with open(cache_root / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    atol = args.atol
    if atol is None:
        atol = 1e-2 if manifest.get("sample_dtype") == "float16" else 1e-6
    mean_atol = args.mean_atol
    if mean_atol is None:
        mean_atol = 1e-4 if manifest.get("sample_dtype") == "float16" else 1e-7

    positions = [0, min(123, len(raw) - 1), len(raw) // 2, len(raw) - 1]
    checks = []
    for idx in positions:
        a = raw[idx]["radar"]
        b = cached[idx]["radar"]
        diff = np.abs(a - b)
        checks.append({
            "idx": int(idx),
            "shape": list(b.shape),
            "dtype": str(b.dtype),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        })
        if (
            a.shape != b.shape
            or diff.max() > atol
            or diff.mean() > mean_atol
        ):
            raise SystemExit(json.dumps({"failed_check": checks[-1]}, indent=2))

    dm = RoverDataModule(
        **cache_cfg["dataset"],
        path=args.data_root,
        n_workers=0,
    )
    batch = next(iter(dm.train_dataloader()))
    print(json.dumps({
        "trace": args.trace,
        "length": len(raw),
        "atol": atol,
        "mean_atol": mean_atol,
        "checks": checks,
        "train_batch_radar_shape": list(batch["radar"].shape),
        "train_batch_radar_dtype": str(batch["radar"].dtype),
    }, indent=2))


if __name__ == "__main__":
    main()
