#!/usr/bin/env python3
"""Train smooth range-local residuals on a frozen global RADs adapter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deepradar import DeepRadar
from deepradar.modules import (
    ComplexAzimuthProjection,
    RangeConditionedComplexAzimuthProjection,
)
from scripts.evaluate_rads_azimuth_adapter import load_adapter
from scripts.train_rads_azimuth_adapter import (
    dilate,
    evaluate,
    load_frame,
    model_sample,
    select_evenly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-frames", type=int, default=128)
    parser.add_argument("--val-frames", type=int, default=64)
    parser.add_argument("--train-sequence", default="100")
    parser.add_argument("--val-sequence", default="101")
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument(
        "--phase-mode", choices=("zero", "original"), default="zero")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def projection_config(adapter: ComplexAzimuthProjection) -> dict:
    config = {
        "source_bins": adapter.source_bins,
        "target_bins": adapter.target_bins,
        "start": adapter.start,
    }
    if adapter.rank is not None:
        config["rank"] = adapter.rank
    return config


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=device,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _, base = load_adapter(args.base_adapter, device)
    if not isinstance(base, ComplexAzimuthProjection):
        raise ValueError("--base-adapter must be a global projection adapter.")
    base_config = projection_config(base)
    adapter = RangeConditionedComplexAzimuthProjection(
        base,
        range_bins=256,
        bands=args.bands,
        rank=args.rank,
    ).to(device)
    trainable = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0)

    entries = json.loads(args.manifest.read_text())["frames"]
    train_names = select_evenly(
        [
            name for name in sorted(entries)
            if name.startswith(f"{args.train_sequence}/")
        ],
        args.train_frames,
    )
    val_names = select_evenly(
        [
            name for name in sorted(entries)
            if name.startswith(f"{args.val_sequence}/")
        ],
        args.val_frames,
    )
    if not train_names or not val_names:
        raise ValueError(
            "Empty sequence selection: "
            f"train={args.train_sequence!r}, val={args.val_sequence!r}."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    baseline_cache: dict[str, torch.Tensor] = {}

    def save(step: int, report: dict) -> None:
        state = {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        }
        torch.save(
            {
                "format_version": 2,
                "adapter_kind": "range_conditioned",
                "adapter": state,
                "adapter_config": {
                    "base": base_config,
                    "range_bins": adapter.range_bins,
                    "bands": adapter.bands,
                    "rank": adapter.rank,
                },
                "step": step,
                "train_sequence": args.train_sequence,
                "val_sequence": args.val_sequence,
                "phase_mode": args.phase_mode,
                "training": {
                    "base_adapter": str(args.base_adapter),
                    "learning_rate": args.learning_rate,
                    "train_frames": len(train_names),
                    "val_frames": len(val_names),
                    "seed": args.seed,
                },
                "validation": report,
            },
            args.out / "best_adapter.pt",
        )

    initial = evaluate(
        model,
        adapter,
        args.data_root,
        entries,
        val_names,
        device,
        args.phase_mode,
    )
    history.append({"step": 0, "validation": initial})
    best_score = initial["adapter"]["f1_tol2"] + initial["adapter"]["f1_tol4"]
    save(0, initial)
    print(json.dumps(history[-1], indent=2), flush=True)

    adapter.train()
    adapter.base.eval()
    for step in range(1, args.steps + 1):
        offset = step * 73 + step // len(train_names)
        name = train_names[offset % len(train_names)]
        dar_np, target_np = load_frame(args.data_root, name, entries[name])
        dar = torch.from_numpy(dar_np).to(device)
        target = torch.from_numpy(target_np[None]).to(device).bool()
        positive = dilate(target, 2)

        if name not in baseline_cache:
            with torch.inference_mode():
                baseline = adapter.initial_forward(dar, azimuth_dim=1)
                baseline_cache[name] = model(
                    {"radar": model_sample(baseline, args.phase_mode)}
                )["map"].amax(dim=1).detach().cpu()
        baseline_logits = baseline_cache[name].to(device)
        logits = model(
            {
                "radar": model_sample(
                    adapter(dar, azimuth_dim=1), args.phase_mode)
            }
        )["map"].amax(dim=1)

        positive_loss = F.softplus(0.5 - logits[positive]).mean()
        preserve = ~positive
        distill_loss = F.mse_loss(
            logits[preserve], baseline_logits[preserve])
        baseline_mass = torch.sigmoid(baseline_logits).mean()
        mass = torch.sigmoid(logits).mean()
        mass_loss = F.relu(mass - baseline_mass * 1.03).square()
        residual_loss, smoothness_loss = adapter.regularization()
        loss = (
            positive_loss
            + 0.25 * distill_loss
            + 20.0 * mass_loss
            + 1e-3 * residual_loss
            + 1e-2 * smoothness_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "positive_loss": float(positive_loss.detach()),
                        "distill_loss": float(distill_loss.detach()),
                        "mass_loss": float(mass_loss.detach()),
                        "residual_loss": float(residual_loss.detach()),
                        "smoothness_loss": float(smoothness_loss.detach()),
                    }
                ),
                flush=True,
            )

        if step % args.validate_every == 0 or step == args.steps:
            report = evaluate(
                model,
                adapter,
                args.data_root,
                entries,
                val_names,
                device,
                args.phase_mode,
            )
            history.append({"step": step, "validation": report})
            score = report["adapter"]["f1_tol2"] + report["adapter"]["f1_tol4"]
            print(json.dumps(history[-1], indent=2), flush=True)
            if score > best_score:
                best_score = score
                save(step, report)
            adapter.train()
            adapter.base.eval()

    (args.out / "history.json").write_text(
        json.dumps(history, indent=2) + "\n")
    print("RADS_RANGE_CONDITIONED_ADAPTER_DONE", flush=True)


if __name__ == "__main__":
    main()
