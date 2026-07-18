#!/usr/bin/env python3
"""Evaluate a RADs azimuth adapter on an entire held-out sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepradar import DeepRadar
from deepradar.modules import ComplexAzimuthProjection
from scripts.train_rads_azimuth_adapter import evaluate


def load_adapter(path: Path, device: torch.device) -> tuple[dict, ComplexAzimuthProjection]:
    saved = torch.load(path, map_location=device, weights_only=False)
    config = saved.get("adapter_config")
    if config is None:
        initial = saved["adapter"]["initial"]
        config = {
            "source_bins": int(initial.shape[0]),
            "target_bins": int(initial.shape[1]),
            "start": int(saved.get("start", 4)),
        }
    adapter = ComplexAzimuthProjection(**config).to(device)
    adapter.load_state_dict(saved["adapter"])
    return saved, adapter.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--phase-mode", choices=("auto", "zero", "original"), default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    saved, adapter = load_adapter(args.adapter, device)
    phase_mode = saved.get("phase_mode", "zero") if args.phase_mode == "auto" else args.phase_mode
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint), hparams_file=str(args.hparams), map_location=device
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    payload = json.loads(args.manifest.read_text())
    entries = payload["frames"]
    names = [
        name for name in sorted(entries)
        if name.startswith(f"{args.sequence}/")
    ]
    if not names:
        raise ValueError(f"No frames found for sequence {args.sequence!r}.")
    metrics = evaluate(
        model,
        adapter,
        args.data_root,
        entries,
        names,
        device,
        phase_mode,
    )
    result = {
        "adapter": str(args.adapter),
        "adapter_step": int(saved["step"]),
        "trained_sequence": saved.get("train_sequence"),
        "held_out_sequence": args.sequence,
        "phase_mode": phase_mode,
        "frame_count": len(names),
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"RADS_AZIMUTH_ADAPTER_EVAL_DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
