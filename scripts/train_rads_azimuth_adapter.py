#!/usr/bin/env python3
"""Train a small RADs A256-to-A8 projection with a frozen GRT model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepradar import DeepRadar
from deepradar.modules import ComplexAzimuthProjection
from scripts.rads_map_checkpoint_infer import (
    keep_center_doppler,
    project_gt_ar,
    shift_range_cube,
    to_dar,
)


THRESHOLDS = tuple(round(-2.0 + 0.25 * index, 2) for index in range(21))
RADII = (0, 1, 2, 4)
FIXED_THRESHOLDS = (-1.0, 0.0, 1.0)


def select_evenly(names: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(names):
        return names
    indices = np.linspace(0, len(names) - 1, count, dtype=np.int64)
    return [names[index] for index in indices]


def load_frame(
    data_root: Path, name: str, entry: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Load, whole-cube crop, and map one RADs frame and sparse GT."""
    gt_ar = np.zeros((256, 256), dtype=bool)
    gt_ar.flat[np.asarray(entry["ar_indices"], dtype=np.int64)] = True
    crop_start = int(entry["crop_start"])
    cube = np.load(
        data_root / "RADs" / Path(name).with_suffix(".npy")
    ).astype(np.complex64, copy=False)
    shifted = shift_range_cube(cube, crop_start)
    dar = to_dar(shifted, flip_azimuth=True, azimuth_flip_mode="index")
    dar = keep_center_doppler(dar, 11).astype(np.complex64, copy=False)
    target = project_gt_ar(
        gt_ar,
        crop_start,
        flip_azimuth=True,
        azimuth_flip_mode="index",
    )
    return dar, target


def model_sample(
    compressed: torch.Tensor,
    phase_mode: str = "zero",
) -> torch.Tensor:
    """Convert complex [D,A,R] data to the trained [D,A,1,R,2] contract."""
    magnitude = compressed.abs()
    amplitude = 2.6245 * magnitude.clamp_min(1e-12).sqrt().pow(0.45)
    amplitude = amplitude * (magnitude > 0).to(amplitude.dtype)
    if phase_mode == "zero":
        phase = torch.zeros_like(amplitude)
    elif phase_mode == "original":
        phase = torch.angle(compressed)
    else:
        raise ValueError(f"Unknown phase mode {phase_mode!r}.")
    return torch.stack((amplitude, phase), dim=-1).unsqueeze(2).unsqueeze(0)


def dilate(target: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return target
    size = radius * 2 + 1
    return F.max_pool2d(
        target.float().unsqueeze(1), size, stride=1, padding=radius
    )[:, 0] > 0.5


def _blank() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "target": 0, "total": 0}


def _update(
    stats: dict[str, int], pred: torch.Tensor, target: torch.Tensor
) -> None:
    stats["tp"] += int((pred & target).sum())
    stats["fp"] += int((pred & ~target).sum())
    stats["fn"] += int((~pred & target).sum())
    stats["pred"] += int(pred.sum())
    stats["target"] += int(target.sum())
    stats["total"] += target.numel()


def _finish(stats: dict[str, int]) -> dict[str, float]:
    precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
    recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "pred_fraction": stats["pred"] / max(stats["total"], 1),
        "target_fraction": stats["target"] / max(stats["total"], 1),
    }


def evaluate(
    model: nn.Module,
    adapter: ComplexAzimuthProjection,
    data_root: Path,
    entries: dict,
    names: list[str],
    device: torch.device,
    phase_mode: str,
) -> dict:
    """Evaluate baseline and adapter with pooled TP/FP/FN statistics."""
    stats = {
        variant: {
            str(radius): {str(threshold): _blank() for threshold in THRESHOLDS}
            for radius in RADII
        }
        for variant in ("baseline", "adapter")
    }
    adapter.eval()
    with torch.inference_mode():
        for index, name in enumerate(names):
            dar_np, target_np = load_frame(data_root, name, entries[name])
            dar = torch.from_numpy(dar_np).to(device)
            target = torch.from_numpy(target_np[None]).to(device).bool()
            baseline = adapter.initial_forward(dar, azimuth_dim=1)
            adapted = adapter(dar, azimuth_dim=1)
            samples = torch.cat(
                (
                    model_sample(baseline, phase_mode),
                    model_sample(adapted, phase_mode),
                ),
                dim=0,
            )
            logits = model({"radar": samples})["map"].amax(dim=1)
            for variant_index, variant in enumerate(("baseline", "adapter")):
                one_logit = logits[variant_index:variant_index + 1]
                for radius in RADII:
                    expanded = dilate(target, radius)
                    for threshold in THRESHOLDS:
                        _update(
                            stats[variant][str(radius)][str(threshold)],
                            one_logit > threshold,
                            expanded,
                        )
            if (index + 1) % 8 == 0:
                print(f"VALIDATE {index + 1}/{len(names)}", flush=True)

    report = {}
    for variant in ("baseline", "adapter"):
        row = {
            "variant": variant,
            "fixed_thresholds": {
                str(threshold): {} for threshold in FIXED_THRESHOLDS
            },
        }
        for radius in RADII:
            rows = {
                threshold: _finish(stats[variant][str(radius)][str(threshold)])
                for threshold in map(str, THRESHOLDS)
            }
            best = max(rows.items(), key=lambda item: item[1]["f1"])
            key = f"f1_tol{radius}" if radius else "f1_strict"
            threshold_key = (
                f"threshold_tol{radius}" if radius else "threshold_strict")
            row[key] = best[1]["f1"]
            row[threshold_key] = float(best[0])
            for threshold in FIXED_THRESHOLDS:
                row["fixed_thresholds"][str(threshold)][key] = rows[
                    str(threshold)
                ]["f1"]
        report[variant] = row
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument(
        "--rank",
        type=int,
        help=(
            "Optional complex residual rank. Omit for the full residual matrix; "
            "rank 4 is the validated compression ablation."
        ),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-frames", type=int, default=128)
    parser.add_argument("--val-frames", type=int, default=64)
    parser.add_argument("--train-sequence", default="100")
    parser.add_argument("--val-sequence", default="101")
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--phase-mode", choices=("zero", "original"), default="zero")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint), hparams_file=str(args.hparams), map_location=device
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = ComplexAzimuthProjection(
        256, 8, start=args.start, rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=0.0)
    payload = json.loads(args.manifest.read_text())
    entries = payload["frames"]
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
            f"train={args.train_sequence!r}, val={args.val_sequence!r}.")

    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    baseline_cache: dict[str, torch.Tensor] = {}
    best_score = -1.0
    initial_report = evaluate(
        model, adapter, args.data_root, entries, val_names, device, args.phase_mode)
    history.append({"step": 0, "validation": initial_report})
    print(json.dumps(history[-1], indent=2), flush=True)

    adapter.train()
    for step in range(1, args.steps + 1):
        name = train_names[(step * 73 + step // len(train_names)) % len(train_names)]
        dar_np, target_np = load_frame(args.data_root, name, entries[name])
        dar = torch.from_numpy(dar_np).to(device)
        target = torch.from_numpy(target_np[None]).to(device).bool()
        positive = dilate(target, 2)

        if name not in baseline_cache:
            with torch.inference_mode():
                baseline = adapter.initial_forward(dar, azimuth_dim=1)
                baseline_cache[name] = (
                    model({
                        "radar": model_sample(baseline, args.phase_mode)
                    })["map"].amax(dim=1).detach().cpu()
                )
        baseline_logits = baseline_cache[name].to(device)
        logits = model({
            "radar": model_sample(adapter(dar, azimuth_dim=1), args.phase_mode)
        })["map"].amax(dim=1)

        positive_loss = F.softplus(0.5 - logits[positive]).mean()
        preserve = ~positive
        distill_loss = F.mse_loss(logits[preserve], baseline_logits[preserve])
        baseline_mass = torch.sigmoid(baseline_logits).mean()
        mass = torch.sigmoid(logits).mean()
        mass_loss = F.relu(mass - baseline_mass * 1.05).square()
        delta_loss, orthogonality_loss = adapter.regularization()
        loss = (
            positive_loss
            + 0.25 * distill_loss
            + 20.0 * mass_loss
            + 1e-3 * delta_loss
            + 1e-2 * orthogonality_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach()),
                "positive_loss": float(positive_loss.detach()),
                "distill_loss": float(distill_loss.detach()),
                "mass_loss": float(mass_loss.detach()),
                "delta_loss": float(delta_loss.detach()),
                "orthogonality_loss": float(orthogonality_loss.detach()),
                "mass": float(mass.detach()),
                "baseline_mass": float(baseline_mass.detach()),
            }), flush=True)

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
                adapter_config = {
                    "source_bins": 256,
                    "target_bins": 8,
                    "start": args.start,
                }
                if args.rank is not None:
                    adapter_config["rank"] = args.rank
                torch.save({
                    "format_version": 1,
                    "adapter": adapter.state_dict(),
                    "adapter_config": adapter_config,
                    "step": step,
                    "train_sequence": args.train_sequence,
                    "val_sequence": args.val_sequence,
                    "phase_mode": args.phase_mode,
                    "training": {
                        "learning_rate": args.learning_rate,
                        "train_frames": len(train_names),
                        "val_frames": len(val_names),
                        "seed": args.seed,
                    },
                    "validation": report,
                }, args.out / "best_adapter.pt")
            adapter.train()

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print("RADS_AZIMUTH_ADAPTER_DONE", flush=True)


if __name__ == "__main__":
    main()
