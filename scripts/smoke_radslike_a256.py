#!/usr/bin/env python3
"""Validate the phase-preserving A256 GRT path on one real I/Q-1M sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/root/autodl-fs/projects/grt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deepradar import DeepRadar, config
from deepradar.pretrained import load_official_grt_base


COMMON_CONFIG = (
    "grt/grt.yaml",
    "grt/small.yaml",
    "data/outdoor.yaml",
    "repr/rads_like.yaml",
    "data/iq1m_radslike_precomputed.yaml",
    "splits/codex_iq1m_radslike_model_selection_autodl.yaml",
    "optim/radslike_adapt.yaml",
    "obj/map_radslike_balanced.yaml",
)


def load_cfg(extra: tuple[str, ...] = ()) -> dict:
    paths = tuple(str(REPO / "config" / value) for value in (*COMMON_CONFIG, *extra))
    return config.load_config(*paths)


def tensor_stats(value: torch.Tensor) -> dict[str, float | list[int]]:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "abs_p99": float(torch.quantile(value.abs(), 0.99).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def initialize(model: DeepRadar, official: Path) -> dict:
    return load_official_grt_base(
        model,
        official,
        elevation_index=0,
        load_decoder=True,
        decoder_head="occ3d",
        min_loaded_fraction=0.95,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    base = DeepRadar(**load_cfg())
    base_report = initialize(base, args.official)
    data = base.get_dataset(str(args.data_root), n_workers=0)
    sample_np = data.val_samples
    radar = torch.from_numpy(np.asarray(sample_np["radar"][:1])).to(device)
    target = torch.from_numpy(np.asarray(sample_np["map"][:1])).to(device)
    base = base.eval().to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        base_patch = base.encoder.patch(radar)
        base_prediction = base({"radar": radar})
        base_loss = base.objectives[0].metrics(
            {"map": target}, base_prediction, train=False, reduce=True).loss
    base_patch_stats = tensor_stats(base_patch)
    base_output_stats = tensor_stats(base_prediction["map"])
    del base, base_patch, base_prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = DeepRadar(**load_cfg(("repr/rads_like_a256.yaml",)))
    a256_report = initialize(model, args.official)
    model = model.eval().to(device)
    expanded = model.encoder.expand_azimuth(radar)
    with torch.no_grad():
        a256_patch = model.encoder.patch(expanded)
    a256_patch_stats = tensor_stats(a256_patch)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        prediction = model({"radar": radar})
        loss = model.objectives[0].metrics(
            {"map": target}, prediction, train=False, reduce=True).loss

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        train_prediction = model({"radar": radar})
        train_loss = model.objectives[0].metrics(
            {"map": target}, train_prediction, train=True, reduce=True).loss
    train_loss.backward()
    patch_grad = model.encoder.patch.reduction.weight.grad

    report = {
        "device": str(device),
        "input_shape": list(radar.shape),
        "expanded_shape": list(expanded.shape),
        "base_patch": base_patch_stats,
        "a256_patch": a256_patch_stats,
        "base_token_count": int(np.prod(base_patch_stats["shape"][1:-1])),
        "a256_token_count": int(np.prod(a256_patch_stats["shape"][1:-1])),
        "base_output": base_output_stats,
        "a256_output": tensor_stats(prediction["map"]),
        "base_loss": float(base_loss.detach().item()),
        "a256_loss": float(loss.detach().item()),
        "a256_train_loss": float(train_loss.detach().item()),
        "patch_gradient_norm": float(patch_grad.float().norm().item()),
        "base_init_loaded_fraction": base_report["loaded_fraction"],
        "a256_init_loaded_fraction": a256_report["loaded_fraction"],
        "a256_patch_resized": a256_report["patch_resized"],
        "a256_patch_resize_scale_exponent": a256_report[
            "patch_resize_scale_exponent"],
        "a256_source_patch": a256_report["source_patch"],
        "a256_target_patch": a256_report["target_patch"],
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"SAVED {args.output}", flush=True)


if __name__ == "__main__":
    main()
