#!/usr/bin/env python3
"""Audit fixed-threshold sparse-occupancy F1 by output range band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepradar import DeepRadar
from scripts.evaluate_rads_azimuth_adapter import load_adapter
from scripts.train_rads_azimuth_adapter import (
    _blank,
    _finish,
    _update,
    dilate,
    load_frame,
    model_sample,
)


THRESHOLDS = (-1.0, 0.0, 1.0)
RADII = (0, 1, 2, 4)
RANGE_BANDS = {
    "all": (0, 64),
    "r00_15": (0, 16),
    "r16_31": (16, 32),
    "r32_47": (32, 48),
    "r48_63": (48, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--phase-mode",
        choices=("auto", "zero", "original"),
        default="auto",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    saved, adapter = load_adapter(args.adapter, device)
    phase_mode = (
        saved.get("phase_mode", "zero")
        if args.phase_mode == "auto"
        else args.phase_mode
    )
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=device,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    entries = json.loads(args.manifest.read_text())["frames"]
    names = [
        name for name in sorted(entries)
        if name.startswith(f"{args.sequence}/")
    ]
    if not names:
        raise ValueError(f"No frames found for sequence {args.sequence!r}.")
    stats = {
        variant: {
            band: {
                str(radius): {
                    str(threshold): _blank()
                    for threshold in THRESHOLDS
                }
                for radius in RADII
            }
            for band in RANGE_BANDS
        }
        for variant in ("baseline", "adapter")
    }

    with torch.inference_mode():
        for index, name in enumerate(names):
            dar_np, target_np = load_frame(
                args.data_root, name, entries[name])
            dar = torch.from_numpy(dar_np).to(device)
            target = torch.from_numpy(target_np[None]).to(device).bool()
            baseline = adapter.initial_forward(dar, azimuth_dim=1)
            adapted = adapter(dar, azimuth_dim=1)
            logits = model(
                {
                    "radar": torch.cat(
                        (
                            model_sample(baseline, phase_mode),
                            model_sample(adapted, phase_mode),
                        ),
                        dim=0,
                    )
                }
            )["map"].amax(dim=1)
            for radius in RADII:
                expanded = dilate(target, radius)
                for band, (start, stop) in RANGE_BANDS.items():
                    one_target = expanded[..., start:stop]
                    for variant_index, variant in enumerate(
                        ("baseline", "adapter")
                    ):
                        one_logit = logits[
                            variant_index:variant_index + 1, :, start:stop]
                        for threshold in THRESHOLDS:
                            _update(
                                stats[variant][band][str(radius)][
                                    str(threshold)
                                ],
                                one_logit > threshold,
                                one_target,
                            )
            if (index + 1) % 32 == 0:
                print(f"RANGE_AUDIT {index + 1}/{len(names)}", flush=True)

    metrics = {}
    for band in RANGE_BANDS:
        metrics[band] = {}
        for radius in RADII:
            radius_key = f"tol{radius}" if radius else "strict"
            metrics[band][radius_key] = {}
            for threshold in THRESHOLDS:
                baseline = _finish(
                    stats["baseline"][band][str(radius)][str(threshold)])
                adapted = _finish(
                    stats["adapter"][band][str(radius)][str(threshold)])
                metrics[band][radius_key][str(threshold)] = {
                    "baseline": baseline,
                    "adapter": adapted,
                    "f1_delta": adapted["f1"] - baseline["f1"],
                }

    result = {
        "adapter": str(args.adapter),
        "adapter_step": int(saved["step"]),
        "trained_sequence": saved.get("train_sequence"),
        "held_out_sequence": args.sequence,
        "phase_mode": phase_mode,
        "frame_count": len(names),
        "range_bands": RANGE_BANDS,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
