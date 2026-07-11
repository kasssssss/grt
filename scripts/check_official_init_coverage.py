#!/usr/bin/env python3
"""Verify task-specific official checkpoint coverage before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepradar import DeepRadar, config
from deepradar.pretrained import load_official_grt_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--min-fraction", type=float, default=0.99)
    args = parser.parse_args()

    base_configs = [
        "config/grt/grt.yaml",
        "config/grt/small.yaml",
        "config/repr/rads_like.yaml",
    ]
    tasks = {
        "map": ("config/obj/map.yaml", "base/small", "occ3d"),
        "semseg": (
            "config/obj/segment_lidar_index_balanced.yaml",
            "semseg/small",
            "semseg",
        ),
    }
    results = {}
    for name, (objective, checkpoint, head) in tasks.items():
        model = DeepRadar(**config.load_config(*base_configs, objective))
        report = load_official_grt_base(
            model,
            Path(args.checkpoint_root) / checkpoint,
            elevation_index=0,
            decoder_head=head,
            min_loaded_fraction=args.min_fraction,
        )
        results[name] = {
            key: report[key]
            for key in (
                "model_dir", "decoder_head", "loaded_count",
                "loaded_numel", "expected_numel", "loaded_fraction",
                "missing_source", "missing_target", "shape_skipped",
            )
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
