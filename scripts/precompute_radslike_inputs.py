#!/usr/bin/env python3
"""Precompute RADs-like I/Q-1M radar tensors used as GRT network inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TraceSpec:
    name: str
    path: str
    count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shard-samples", type=int, default=256)
    p.add_argument("--workers-per-process", type=int, default=12)
    p.add_argument("--num-processes", type=int, default=2)
    p.add_argument("--device-prefix", default="cuda")
    p.add_argument("--cache-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--limit-per-trace", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--traces", nargs="*", default=None)
    p.add_argument(
        "--configs",
        nargs="+",
        default=[
            "grt/grt.yaml",
            "grt/small.yaml",
            "data/outdoor.yaml",
            "repr/rads_like.yaml",
        ],
    )
    return p.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_npy(path: Path, arr: np.ndarray) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)
    return path.stat().st_size


def trace_out_dir(root: Path, trace: str) -> Path:
    return root / "traces" / trace


def preprocess_fingerprint(
    cfg: dict[str, Any], cache_dtype: str, limit_per_trace: int | None
) -> str:
    """Identify cache contents independently of paths and shard layout."""
    payload = {
        "pipeline_version": 2,
        "preprocess_config": {k: v for k, v in cfg.items() if k != "traces"},
        "cache_dtype": cache_dtype,
        "limit_per_trace": limit_per_trace,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_fft_contract(axes: list[int], pad: int) -> None:
    """Reject configs the optimized GPU implementation cannot reproduce."""
    if axes != [0, 1, 2, 3]:
        raise ValueError(
            "GPU RADs-like precompute currently requires FFTArray "
            f"axes=[0, 1, 2, 3], got {axes!r}.")
    if pad != 0:
        raise ValueError(
            "GPU RADs-like precompute currently requires FFTArray pad=0, "
            f"got {pad}.")


def load_precompute_config(repo: Path, configs: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    from deepradar import config  # noqa: PLC0415

    cfg = config.load_config(*[str(repo / "config" / c) for c in configs])
    radar_channel = cfg["dataset"]["channels"]["radar"]
    transforms = radar_channel["args"]["transform"]
    rads_like = next(t["args"] for t in transforms if t["name"] == "RADsLikeDoppler")
    iiqq = next(t["args"] for t in transforms if t["name"] == "IIQQtoIQ")
    fft = next(t["args"] for t in transforms if t["name"] == "FFTArray")
    if "elevation_indices" in rads_like:
        elevation_indices = [int(i) for i in rads_like["elevation_indices"]]
    elif "elevation_index" in rads_like:
        elevation_indices = [int(rads_like["elevation_index"])]
    else:
        elevation_indices = [0]
    if not elevation_indices:
        raise ValueError("RADsLikeDoppler elevation_indices must not be empty")
    fft_axes = [int(axis) for axis in fft.get("axes", [0, 1, 2, 3])]
    fft_pad = int(fft.get("pad", 0))
    validate_fft_contract(fft_axes, fft_pad)

    return {
        "traces": cfg["dataset"]["traces"],
        "scale": float(iiqq.get("scale", 0.001)),
        "fft_axes": fft_axes,
        "fft_pad": fft_pad,
        "target_bins": int(rads_like.get("target_bins", 64)),
        "target_max_speed": float(rads_like.get("target_max_speed", 90.0)),
        "elevation_indices": elevation_indices,
        "merge": str(rads_like.get("merge", "mean")),
        "mapping": str(rads_like.get("mapping", "physical")),
        "smooth_mode": str(rads_like.get("smooth_mode", "complex")),
        "output_scale": float(rads_like.get("output_scale", 1.0)),
        "smooth_sigma_bins": float(rads_like.get("smooth_sigma_bins", 0.0)),
    }


def make_raw_trace_dataset(repo: Path, trace_path: str):
    sys.path.insert(0, str(repo))
    from deepradar.dataloader import RoverTrace  # noqa: PLC0415

    channels = {
        "radar": {
            "name": "RawChannel",
            "indices": "radar",
            "args": {
                "sensor": "radar",
                "channel": "iq",
                "transform": [],
            },
        }
    }
    return RoverTrace(trace_path, channels=channels, augmentations={}, bounds=(0.0, 1.0))


def discover_traces(repo: Path, data_root: Path, cfg: dict[str, Any]) -> list[TraceSpec]:
    traces = []
    for name in cfg["traces"]:
        ds = make_raw_trace_dataset(repo, str(data_root / name))
        traces.append(TraceSpec(name=name, path=str(data_root / name), count=len(ds)))
    return traces


def build_target_index(
    source_bins: int,
    doppler_res: float,
    target_bins: int,
    target_max_speed: float,
    mapping: str = "physical",
) -> tuple[torch.Tensor, torch.Tensor]:
    if mapping == "native_index":
        if target_bins != source_bins:
            raise ValueError("native_index mapping requires target_bins == source_bins")
        target_index = np.arange(source_bins, dtype=np.int64)
    elif mapping == "physical":
        source_velocity = (np.arange(source_bins, dtype=np.float32) - source_bins // 2) * doppler_res
        target_res = 2.0 * target_max_speed / target_bins
        target_index = np.rint(source_velocity / target_res + target_bins // 2).astype(np.int64)
    else:
        raise ValueError(f"Unsupported mapping={mapping!r}")
    counts = np.zeros(target_bins, dtype=np.int64)
    for target_idx in target_index:
        if 0 <= target_idx < target_bins:
            counts[target_idx] += 1
    return torch.from_numpy(target_index), torch.from_numpy(counts)


def make_smooth_kernel(sigma: float, device: torch.device) -> torch.Tensor | None:
    if sigma <= 0:
        return None
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * torch.square(offsets / sigma))
    return kernel / kernel.sum()


@torch.inference_mode()
def preprocess_batch(
    raw: torch.Tensor,
    *,
    device: torch.device,
    scale: float,
    target_bins: int,
    target_index_cpu: torch.Tensor,
    counts_cpu: torch.Tensor,
    elevation_indices: list[int],
    merge: str,
    smooth_kernel: torch.Tensor | None,
    smooth_mode: str,
    output_scale: float,
) -> np.ndarray:
    raw = raw.to(device=device, non_blocking=True)
    if raw.ndim != 5:
        raise ValueError(f"Expected raw [B,D,Tx,Rx,R2], got {tuple(raw.shape)}")
    b, source_bins, tx, rx, r2 = raw.shape
    if tx != 3 or rx != 4:
        raise ValueError(f"Expected 3x4 TX/RX, got tx={tx}, rx={rx}")
    ranges = r2 // 2

    raw_f = raw.to(torch.float32)
    iq = torch.empty((b, source_bins, tx, rx, ranges), dtype=torch.complex64, device=device)
    iq[..., 0::2] = torch.complex(raw_f[..., 2::4], raw_f[..., 0::4])
    iq[..., 1::2] = torch.complex(raw_f[..., 3::4], raw_f[..., 1::4])
    iq = iq * scale

    iq_daer = torch.zeros((b, source_bins, 8, 2, ranges), dtype=torch.complex64, device=device)
    iq_daer[:, :, 0:4, 0, :] = iq[:, :, 0, :, :]
    iq_daer[:, :, 4:8, 0, :] = iq[:, :, 2, :, :]
    iq_daer[:, :, 2:6, 1, :] = iq[:, :, 1, :, :]

    daer = torch.fft.fftn(iq_daer, dim=(1, 2, 3, 4))
    daer = torch.fft.fftshift(daer, dim=(1, 2, 3))

    for elevation_index in elevation_indices:
        if not 0 <= elevation_index < daer.shape[3]:
            raise ValueError(f"elevation_index={elevation_index} out of bounds for {daer.shape[3]}")
    kept = daer[:, :, :, elevation_indices, :]
    target = torch.zeros(
        (b, target_bins, kept.shape[2], len(elevation_indices), kept.shape[4]),
        dtype=torch.complex64,
        device=device,
    )

    target_index = target_index_cpu.to(device=device)
    counts = counts_cpu.to(device=device)
    if merge == "rms_peak_phase":
        for target_idx in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist():
            source = torch.nonzero(target_index == target_idx, as_tuple=False).flatten()
            group = kept.index_select(1, source)
            amplitude = torch.sqrt(torch.mean(torch.square(torch.abs(group)), dim=1))
            peak_index = torch.argmax(torch.abs(group), dim=1, keepdim=True)
            peak = torch.gather(group, 1, peak_index).squeeze(1)
            target[:, target_idx] = torch.polar(amplitude, torch.angle(peak))
    else:
        for source_idx in range(source_bins):
            target_idx = int(target_index[source_idx].item())
            if 0 <= target_idx < target_bins:
                target[:, target_idx] += kept[:, source_idx]
        if merge == "mean":
            nonzero = counts > 0
            target[:, nonzero] = target[:, nonzero] / counts[nonzero][None, :, None, None, None]
        elif merge != "sum":
            raise ValueError(f"Unsupported merge={merge!r}")

    if smooth_kernel is not None:
        radius = smooth_kernel.numel() // 2
        padded = F.pad(target, (0, 0, 0, 0, 0, 0, radius, radius))
        if smooth_mode == "complex":
            smoothed = torch.zeros_like(target)
            for i, weight in enumerate(smooth_kernel):
                smoothed += weight * padded[:, i:i + target_bins]
            target = smoothed
        elif smooth_mode == "power_peak_phase":
            smoothed_power = torch.zeros_like(target.real)
            best_score = torch.zeros_like(target.real)
            best_source = torch.zeros_like(target)
            for i, weight in enumerate(smooth_kernel):
                source = padded[:, i:i + target_bins]
                score = weight * torch.square(torch.abs(source))
                smoothed_power += score
                update = score > best_score
                best_score = torch.where(update, score, best_score)
                best_source = torch.where(update, source, best_source)
            target = torch.polar(torch.sqrt(smoothed_power), torch.angle(best_source))
        else:
            raise ValueError(f"Unsupported smooth_mode={smooth_mode!r}")

    target *= output_scale

    magnitude = torch.sqrt(torch.abs(target))
    phase = torch.angle(target)
    out = torch.stack((magnitude, phase), dim=-1).to(torch.float32)
    return out.cpu().numpy()


def read_radar_json(trace_path: Path) -> dict[str, Any]:
    with open(trace_path / "radar" / "radar.json", "r", encoding="utf-8") as f:
        return json.load(f)


def worker_main(rank: int, args_dict: dict[str, Any], traces_payload: list[dict[str, Any]]) -> None:
    args = argparse.Namespace(**args_dict)
    repo = Path(args.repo)
    out_root = Path(args.out_root)
    device = torch.device(f"{args.device_prefix}:{rank}" if args.device_prefix == "cuda" and torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None

    cfg = load_precompute_config(repo, args.configs)
    fingerprint = preprocess_fingerprint(
        cfg, args.cache_dtype, args.limit_per_trace)
    total_processed = 0
    total_bytes = 0
    start_all = time.perf_counter()

    for trace_payload in traces_payload:
        trace = TraceSpec(**trace_payload)
        out_dir = trace_out_dir(out_root, trace.name)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("complete") is True:
                    if old.get("preprocess_fingerprint") != fingerprint:
                        raise RuntimeError(
                            f"Refusing to reuse stale cache for {trace.name}: "
                            f"manifest fingerprint="
                            f"{old.get('preprocess_fingerprint')!r}, requested="
                            f"{fingerprint!r}. Use a new cache root or "
                            "--overwrite explicitly.")
                    print(json.dumps({"event": "skip_complete", "rank": rank, "trace": trace.name}), flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        radar_meta = read_radar_json(Path(trace.path))
        target_index_cpu, counts_cpu = build_target_index(
            source_bins=int(radar_meta.get("doppler_bins", radar_meta["shape"][0])),
            doppler_res=float(radar_meta["doppler_resolution"]),
            target_bins=cfg["target_bins"],
            target_max_speed=cfg["target_max_speed"],
            mapping=cfg["mapping"],
        )
        smooth_kernel = make_smooth_kernel(cfg["smooth_sigma_bins"], device)
        ds = make_raw_trace_dataset(repo, trace.path)
        limit = len(ds) if args.limit_per_trace is None else min(len(ds), args.limit_per_trace)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers_per_process,
            pin_memory=(device.type == "cuda"),
        )

        trace_start = time.perf_counter()
        pending: list[np.ndarray] = []
        pending_count = 0
        processed = 0
        bytes_written = 0
        shards = []
        shard_idx = 0

        atomic_json(manifest_path, {
            "complete": False,
            "trace": trace.name,
            "trace_path": trace.path,
            "rank": rank,
            "device": str(device),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "count": len(ds),
            "limit": limit,
            "preprocess_fingerprint": fingerprint,
        })

        for batch in loader:
            if processed >= limit:
                break
            raw = batch["radar"]
            remaining = limit - processed
            if raw.shape[0] > remaining:
                raw = raw[:remaining]
            arr = preprocess_batch(
                raw,
                device=device,
                scale=cfg["scale"],
                target_bins=cfg["target_bins"],
                target_index_cpu=target_index_cpu,
                counts_cpu=counts_cpu,
                elevation_indices=cfg["elevation_indices"],
                merge=cfg["merge"],
                smooth_kernel=smooth_kernel,
                smooth_mode=cfg["smooth_mode"],
                output_scale=cfg["output_scale"],
            )
            pending.append(arr)
            pending_count += arr.shape[0]
            processed += arr.shape[0]

            while pending_count >= args.shard_samples or (processed >= limit and pending_count > 0):
                shard_take = min(args.shard_samples, pending_count)
                pieces = []
                need = shard_take
                while need > 0:
                    head = pending[0]
                    if head.shape[0] <= need:
                        pieces.append(head)
                        need -= head.shape[0]
                        pending.pop(0)
                    else:
                        pieces.append(head[:need])
                        pending[0] = head[need:]
                        need = 0
                shard_start = processed - pending_count
                shard_arr = np.concatenate(pieces, axis=0)
                if args.cache_dtype != "float32":
                    shard_arr = shard_arr.astype(args.cache_dtype, copy=False)
                rel = f"radar_{shard_idx:06d}_{shard_start:09d}.npy"
                shard_path = out_dir / rel
                shard_bytes = atomic_npy(shard_path, shard_arr)
                shards.append({
                    "file": rel,
                    "start": int(shard_start),
                    "samples": int(shard_arr.shape[0]),
                    "shape": list(shard_arr.shape),
                    "dtype": str(shard_arr.dtype),
                    "bytes": int(shard_bytes),
                })
                bytes_written += shard_bytes
                total_bytes += shard_bytes
                pending_count -= shard_take
                shard_idx += 1

            total_processed += arr.shape[0]
            elapsed = time.perf_counter() - trace_start
            print(json.dumps({
                "event": "progress",
                "rank": rank,
                "trace": trace.name,
                "processed": processed,
                "limit": limit,
                "trace_samples_per_s": processed / elapsed if elapsed > 0 else None,
                "total_processed_rank": total_processed,
            }), flush=True)

        trace_elapsed = time.perf_counter() - trace_start
        trace_manifest = {
            "complete": True,
            "trace": trace.name,
            "trace_path": trace.path,
            "rank": rank,
            "device": str(device),
            "count": len(ds),
            "processed": processed,
            "limit": limit,
            "seconds": trace_elapsed,
            "samples_per_s": processed / trace_elapsed if trace_elapsed > 0 else None,
            "bytes_written": bytes_written,
            "bytes_per_sample": bytes_written / processed if processed else None,
            "sample_shape": [cfg["target_bins"], 8, len(cfg["elevation_indices"]), 256, 2],
            "sample_dtype": args.cache_dtype,
            "preprocess_fingerprint": fingerprint,
            "shards": shards,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(manifest_path, trace_manifest)
        print(json.dumps({"event": "trace_complete", **trace_manifest}), flush=True)

    elapsed_all = time.perf_counter() - start_all
    print(json.dumps({
        "event": "worker_complete",
        "rank": rank,
        "processed": total_processed,
        "bytes_written": total_bytes,
        "seconds": elapsed_all,
        "samples_per_s": total_processed / elapsed_all if elapsed_all > 0 else None,
    }), flush=True)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_precompute_config(repo, args.configs)
    if args.traces:
        selected = set(args.traces)
        configured = set(cfg["traces"])
        missing = selected - configured
        if missing:
            raise ValueError(f"Unknown traces requested: {sorted(missing)}")
        cfg = {
            **cfg,
            "traces": [name for name in cfg["traces"] if name in selected],
        }
    fingerprint = preprocess_fingerprint(
        cfg, args.cache_dtype, args.limit_per_trace)
    traces = discover_traces(repo, data_root, cfg)
    total = sum(t.count for t in traces)

    root_manifest = {
        "complete": False,
        "repo": str(repo),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "configs": args.configs,
        "pipeline": "RawChannel iq -> IIQQtoIQ -> FFTArray -> RADsLikeDoppler -> ComplexPhase",
        "format": "per-trace numpy shards",
        "sample_shape": [cfg["target_bins"], 8, len(cfg["elevation_indices"]), 256, 2],
        "sample_dtype": args.cache_dtype,
        "total_samples": total if args.limit_per_trace is None else sum(min(t.count, args.limit_per_trace) for t in traces),
        "full_trace_counts": {t.name: t.count for t in traces},
        "preprocess_config": {k: v for k, v in cfg.items() if k != "traces"},
        "preprocess_fingerprint": fingerprint,
        "num_processes": args.num_processes,
        "batch_size": args.batch_size,
        "shard_samples": args.shard_samples,
        "workers_per_process": args.workers_per_process,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(out_root / "manifest.json", root_manifest)

    assignments = [[] for _ in range(args.num_processes)]
    for i, trace in enumerate(sorted(traces, key=lambda t: t.count, reverse=True)):
        assignments[i % args.num_processes].append(trace.__dict__)

    print(json.dumps({
        "event": "start",
        "out_root": str(out_root),
        "total_samples": root_manifest["total_samples"],
        "assignments": [[t["name"] for t in a] for a in assignments],
    }), flush=True)

    ctx = get_context("spawn")
    procs = []
    args_dict = vars(args)
    for rank, payload in enumerate(assignments):
        p = ctx.Process(target=worker_main, args=(rank, args_dict, payload))
        p.start()
        procs.append(p)
    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"Worker failures: {exit_codes}")

    trace_manifests = []
    total_processed = 0
    total_bytes = 0
    for trace in traces:
        manifest_path = trace_out_dir(out_root, trace.name) / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        trace_manifests.append(m)
        total_processed += int(m["processed"])
        total_bytes += int(m["bytes_written"])

    root_manifest.update({
        "complete": True,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "processed_samples": total_processed,
        "bytes_written": total_bytes,
        "bytes_per_sample": total_bytes / total_processed if total_processed else None,
        "trace_manifests": [
            str((trace_out_dir(out_root, m["trace"]) / "manifest.json").relative_to(out_root))
            for m in trace_manifests
        ],
    })
    atomic_json(out_root / "manifest.json", root_manifest)
    print("FINAL_MANIFEST " + json.dumps(root_manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
