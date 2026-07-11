#!/usr/bin/env python3
"""Print recent TensorBoard scalar values for a GRT run."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


PREFERRED = (
    "loss/train_step",
    "map_loss/train_step",
    "loss/train_epoch",
    "map_loss/train_epoch",
    "loss/train",
    "loss/val",
    "map_loss/train",
    "map_loss/val",
    "map_depth/val",
    "map_depth_m1/val",
    "map_depth_0/val",
    "map_depth_p1/val",
    "map_f1/val",
    "map_m1_f1/val",
    "map_0_f1/val",
    "map_p1_f1/val",
    "map_invalid_m1/val",
    "map_invalid_0/val",
    "map_invalid_p1/val",
    "map_precision/val",
    "map_recall/val",
    "map_chamfer/val",
)


def fmt(values) -> str:
    return "; ".join(f"s={v.step}: {v.value:.6g}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=Path)
    parser.add_argument("--tail", type=int, default=8)
    args = parser.parse_args()

    ea = event_accumulator.EventAccumulator(str(args.event), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print("SCALAR_TAGS", ", ".join(tags))
    for tag in PREFERRED:
        if tag not in tags:
            continue
        vals = ea.Scalars(tag)
        if not vals:
            continue
        best = min(vals, key=lambda v: v.value) if "loss" in tag or "depth" in tag or "chamfer" in tag else max(vals, key=lambda v: v.value)
        print(f"TAG {tag} count={len(vals)} best_step={best.step} best={best.value:.8g}")
        print("  last", fmt(vals[-args.tail :]))


if __name__ == "__main__":
    main()
