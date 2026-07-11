#!/usr/bin/env python3
"""Estimate patch-weight scaling for the A256 encoder on fixed IQ1M samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/root/autodl-fs/projects/grt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deepradar import DeepRadar, config
from deepradar.pretrained import load_official_grt_base


CONFIGS = (
    "grt/grt.yaml",
    "grt/small.yaml",
    "data/outdoor.yaml",
    "repr/rads_like.yaml",
    "data/iq1m_radslike_precomputed.yaml",
    "splits/codex_iq1m_radslike_model_selection_autodl.yaml",
    "optim/radslike_adapt.yaml",
    "obj/map_radslike_balanced.yaml",
)


def cfg(a256: bool) -> dict:
    names = (*CONFIGS, *(("repr/rads_like_a256.yaml",) if a256 else ()))
    return config.load_config(*(str(REPO / "config" / name) for name in names))


def initialize(model: DeepRadar, official: Path) -> None:
    load_official_grt_base(
        model,
        official,
        elevation_index=0,
        load_decoder=True,
        decoder_head="occ3d",
        min_loaded_fraction=0.95,
    )


def patch_stds(
    model: DeepRadar, radar: torch.Tensor, device: torch.device, a256: bool
) -> list[float]:
    model = model.eval().to(device)
    values = []
    with torch.inference_mode():
        for sample in radar:
            x = sample[None].to(device)
            if a256:
                x = model.encoder.expand_azimuth(x)
            values.append(float(model.encoder.patch(x).float().std().item()))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = DeepRadar(**cfg(False))
    initialize(base, args.official)
    radar = torch.from_numpy(np.asarray(base.get_dataset(
        str(args.data_root), n_workers=0).val_samples["radar"]))
    base_stds = patch_stds(base, radar, device, a256=False)
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    a256 = DeepRadar(**cfg(True))
    initialize(a256, args.official)
    a256_stds = patch_stds(a256, radar, device, a256=True)
    ratios = [base_value / a256_value for base_value, a256_value in zip(base_stds, a256_stds)]
    multiplier = float(np.median(ratios))
    source_to_target = (2 * 8 * 1 * 4) / (8 * 32 * 1 * 8)
    current_scale = math.sqrt(source_to_target)
    total_scale = current_scale * multiplier
    exponent = math.log(total_scale) / math.log(source_to_target)

    report = {
        "samples": len(base_stds),
        "base_patch_std": base_stds,
        "a256_patch_std": a256_stds,
        "base_over_a256": ratios,
        "median_extra_multiplier": multiplier,
        "current_total_scale": current_scale,
        "recommended_total_scale": total_scale,
        "recommended_source_target_exponent": exponent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"SAVED {args.output}", flush=True)


if __name__ == "__main__":
    main()
