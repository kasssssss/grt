#!/usr/bin/env python3
"""Run one augmented RADs-like forward/backward step on a real batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepradar import DeepRadar, config
from deepradar.pretrained import load_official_grt_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--task", choices=("map", "semseg"), default="map")
    args = parser.parse_args()

    objective = (
        "config/obj/map.yaml" if args.task == "map"
        else "config/obj/segment_lidar_index_balanced.yaml")
    checkpoint = "base/small" if args.task == "map" else "semseg/small"
    head = "occ3d" if args.task == "map" else "semseg"
    cfg = config.load_config(
        "config/grt/grt.yaml",
        "config/grt/small.yaml",
        "config/data/outdoor.yaml",
        "config/repr/rads_like.yaml",
        "config/data/iq1m_radslike_precomputed.yaml",
        "config/aug/full.yaml",
        "config/splits/codex_iq1m_original_like_autodl.yaml",
        "config/optim/radslike_finetune.yaml",
        objective,
    )
    # Keep the smoke bounded while exercising the same transforms and shape.
    cfg["dataset"]["batch_size"] = 2
    model = DeepRadar(**cfg)
    report = load_official_grt_base(
        model,
        Path(args.checkpoint_root) / checkpoint,
        elevation_index=0,
        decoder_head=head,
        min_loaded_fraction=0.99,
    )
    dm = model.get_dataset(args.data_root, n_workers=0)
    batch = next(iter(dm.train_dataloader()))
    device = torch.device("cuda")
    model = model.to(device).train()
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch)
        result = model.objectives[0].metrics(
            batch, output, train=True, reduce=True)
    result.loss.backward()
    print(json.dumps({
        "task": args.task,
        "loaded_fraction": report["loaded_fraction"],
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "output_shapes": {key: list(value.shape) for key, value in output.items()},
        "loss": float(result.loss.detach().cpu()),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }, indent=2))


if __name__ == "__main__":
    main()
