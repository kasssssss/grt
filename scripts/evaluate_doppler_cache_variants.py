#!/usr/bin/env python3
"""Evaluate aligned precomputed radar variants on the same I/Q-1M frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_variant(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("variant must be NAME=PATH")
    return name, Path(path)


def parse_named_float(value: str) -> tuple[str, float]:
    name, separator, raw = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("value must be NAME=FLOAT")
    return name, float(raw)


def power_blur_complex_phase(radar: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return radar
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, device=radar.device)
    kernel = torch.exp(-0.5 * torch.square(offsets.to(torch.float32) / sigma))
    kernel = kernel / kernel.sum()
    magnitude = torch.square(radar[..., 0])
    phase = radar[..., 1]
    complex_data = torch.polar(magnitude, phase)
    power = torch.square(torch.abs(complex_data))
    smoothed_power = torch.zeros_like(power)
    best_score = torch.zeros_like(power)
    best_phase = torch.zeros_like(phase)
    for offset, weight in zip(offsets.tolist(), kernel):
        shifted_power = torch.roll(power, shifts=offset, dims=1)
        shifted_phase = torch.roll(phase, shifts=offset, dims=1)
        if offset > 0:
            shifted_power[:, :offset] = 0
        elif offset < 0:
            shifted_power[:, offset:] = 0
        score = weight * shifted_power
        smoothed_power += score
        update = score > best_score
        best_score = torch.where(update, score, best_score)
        best_phase = torch.where(update, shifted_phase, best_phase)
    output = radar.clone()
    output[..., 0] = torch.pow(smoothed_power.clamp_min(0), 0.25)
    output[..., 1] = best_phase
    return output


def batch_slices(count: int, batch_size: int) -> list[slice]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        slice(start, min(start + batch_size, count))
        for start in range(0, count, batch_size)
    ]


def load_range(
    cache_root: Path, trace: str, start: int, count: int
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    position = 0
    stop = start + count
    for shard in sorted((cache_root / "traces" / trace).glob("radar_*.npy")):
        data = np.load(shard, mmap_mode="r")
        shard_stop = position + data.shape[0]
        if shard_stop > start and position < stop:
            local_start = max(0, start - position)
            local_stop = min(data.shape[0], stop - position)
            arrays.append(np.asarray(data[local_start:local_stop]))
        position = shard_stop
        if position >= stop:
            break
    found = sum(array.shape[0] for array in arrays)
    if found != count:
        raise ValueError(
            f"Expected samples [{start}, {stop}) for {trace} in {cache_root}, "
            f"found {found}")
    return np.concatenate(arrays, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True)
    parser.add_argument("--mag-scale", action="append", type=parse_named_float, default=[])
    parser.add_argument("--power-blur", action="append", type=parse_named_float, default=[])
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--samples-per-trace", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from deepradar import DeepRadar  # noqa: PLC0415
    from deepradar.dataloader import RoverTrace  # noqa: PLC0415

    device = torch.device(args.device)
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint), hparams_file=str(args.hparams), map_location=device)
    model.eval().to(device)
    objective = model.objectives[0]
    map_spec = model.dataset["channels"]["map"]

    manifests = {
        name: json.loads((root / "manifest.json").read_text())
        for name, root in args.variant
    }
    mag_scales = dict(args.mag_scale)
    power_blurs = dict(args.power_blur)
    trace_sets = [set(manifest["full_trace_counts"]) for manifest in manifests.values()]
    traces = sorted(set.intersection(*trace_sets))
    values: dict[str, dict[str, list[float]]] = {
        name: {} for name, _ in args.variant
    }

    with torch.inference_mode():
        for trace in traces:
            dataset = RoverTrace(
                str(args.data_root / trace),
                channels={"map": map_spec},
                augmentations={},
                bounds=(0.0, 1.0),
            )
            target = np.stack(
                [
                    dataset[index]["map"]
                    for index in range(
                        args.sample_offset,
                        args.sample_offset + args.samples_per_trace,
                    )
                ])
            for name, root in args.variant:
                radar = load_range(
                    root, trace, args.sample_offset, args.samples_per_trace)
                variant = values[name]
                for batch_slice in batch_slices(args.samples_per_trace, args.batch_size):
                    radar_t = torch.from_numpy(radar[batch_slice]).to(
                        device=device, dtype=torch.float32)
                    radar_t = power_blur_complex_phase(
                        radar_t, power_blurs.get(name, 0.0))
                    radar_t[..., 0] *= mag_scales.get(name, 1.0)
                    batch = {
                        "radar": radar_t,
                        "map": torch.from_numpy(target[batch_slice]).to(device=device),
                    }
                    prediction = model(batch)
                    metrics = objective.metrics(
                        batch, prediction, train=False, reduce=False)
                    variant.setdefault("loss", []).extend(
                        metrics.loss.detach().cpu().reshape(-1).tolist())
                    for key, tensor in metrics.metrics.items():
                        variant.setdefault(key, []).extend(
                            tensor.detach().cpu().reshape(-1).tolist())

    result = {
        "checkpoint": str(args.checkpoint),
        "sample_offset": args.sample_offset,
        "samples_per_trace": args.samples_per_trace,
        "batch_size": args.batch_size,
        "traces": traces,
        "mag_scales": mag_scales,
        "power_blurs": power_blurs,
        "variants": {},
    }
    for name, metrics in values.items():
        result["variants"][name] = {
            key: {
                "mean": float(np.mean(value)),
                "std": float(np.std(value)),
                "count": len(value),
            }
            for key, value in metrics.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    concise = {
        name: {
            key: metrics[key]["mean"]
            for key in ("map_loss", "map_f1", "map_depth", "map_chamfer")
        }
        for name, metrics in result["variants"].items()
    }
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
