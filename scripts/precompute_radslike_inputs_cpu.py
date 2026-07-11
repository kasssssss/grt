#!/usr/bin/env python3
"""Precompute exact RADs-like radar tensors with the training transform chain."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-samples", type=int, default=256)
    p.add_argument("--workers-per-process", type=int, default=12)
    p.add_argument("--num-processes", type=int, default=4)
    p.add_argument("--device-prefix", choices=["cpu"], default="cpu")
    p.add_argument("--cache-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--limit-per-trace", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
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


def preprocessing_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the path-independent transform contract encoded in this cache."""
    return {
        "radar_channel": cfg["dataset"]["channels"]["radar"],
    }


def preprocess_fingerprint(
    cfg: dict[str, Any], cache_dtype: str, limit_per_trace: int | None
) -> str:
    payload = {
        "pipeline_version": 2,
        "implementation": "cpu_training_transform_chain",
        "preprocess_config": preprocessing_spec(cfg),
        "cache_dtype": cache_dtype,
        "limit_per_trace": limit_per_trace,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_reusable_manifest(
    manifest: dict[str, Any], fingerprint: str, trace: str
) -> bool:
    """Return whether a complete cache is reusable or reject it as stale."""
    if manifest.get("complete") is not True:
        return False
    if manifest.get("preprocess_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Refusing to reuse stale cache for {trace}: manifest "
            f"fingerprint={manifest.get('preprocess_fingerprint')!r}, "
            f"requested={fingerprint!r}. Use a new cache root or "
            "--overwrite explicitly.")
    return True


def load_config(repo: Path, configs: list[str]) -> dict[str, Any]:
    sys.path.insert(0, str(repo))
    from deepradar import config  # noqa: PLC0415

    return config.load_config(*[str(repo / "config" / c) for c in configs])


def make_trace_dataset(repo: Path, data_root: Path, trace: str, cfg: dict[str, Any]):
    sys.path.insert(0, str(repo))
    from deepradar.dataloader import RoverTrace  # noqa: PLC0415

    radar_channel = cfg["dataset"]["channels"]["radar"]
    return RoverTrace(
        str(data_root / trace),
        channels={"radar": radar_channel},
        augmentations={},
        bounds=(0.0, 1.0),
    )


def discover_traces(repo: Path, data_root: Path, cfg: dict[str, Any]) -> list[TraceSpec]:
    traces = []
    for trace in cfg["dataset"]["traces"]:
        ds = make_trace_dataset(repo, data_root, trace, cfg)
        traces.append(TraceSpec(name=trace, path=str(data_root / trace), count=len(ds)))
    return traces


def flush_shard(out_dir: Path, shard_idx: int, start_idx: int, pieces: list[np.ndarray]) -> dict[str, Any]:
    arr = np.concatenate(pieces, axis=0)
    rel = f"radar_{shard_idx:06d}_{start_idx:09d}.npy"
    size = atomic_npy(out_dir / rel, arr)
    return {
        "file": rel,
        "start": int(start_idx),
        "samples": int(arr.shape[0]),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "bytes": int(size),
    }


def worker_main(rank: int, args_dict: dict[str, Any], trace_payloads: list[dict[str, Any]]) -> None:
    args = argparse.Namespace(**args_dict)
    repo = Path(args.repo)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    cfg = load_config(repo, args.configs)
    fingerprint = preprocess_fingerprint(
        cfg, args.cache_dtype, args.limit_per_trace)
    rank_start = time.perf_counter()
    rank_samples = 0
    rank_bytes = 0

    for payload in trace_payloads:
        trace = TraceSpec(**payload)
        out_dir = trace_out_dir(out_root, trace.name)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if validate_reusable_manifest(
                    old, fingerprint, trace.name
                ):
                    print(json.dumps({"event": "skip_complete", "rank": rank, "trace": trace.name}), flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = make_trace_dataset(repo, data_root, trace.name, cfg)
        limit = len(ds) if args.limit_per_trace is None else min(len(ds), args.limit_per_trace)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers_per_process,
            pin_memory=False,
        )

        atomic_json(manifest_path, {
            "complete": False,
            "trace": trace.name,
            "rank": rank,
            "count": len(ds),
            "limit": limit,
            "preprocess_fingerprint": fingerprint,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

        trace_start = time.perf_counter()
        processed = 0
        bytes_written = 0
        shards: list[dict[str, Any]] = []
        pending: list[np.ndarray] = []
        pending_count = 0
        pending_start = 0
        shard_idx = 0
        first_shape = None
        first_dtype = None

        for batch in loader:
            if processed >= limit:
                break
            arr = batch["radar"]
            if isinstance(arr, torch.Tensor):
                arr_np = arr.numpy()
            else:
                arr_np = np.asarray(arr)
            if arr_np.shape[0] > limit - processed:
                arr_np = arr_np[:limit - processed]
            arr_np = arr_np.astype(args.cache_dtype, copy=False)
            if first_shape is None:
                first_shape = list(arr_np.shape[1:])
                first_dtype = str(arr_np.dtype)
            pending.append(arr_np)
            pending_count += int(arr_np.shape[0])
            processed += int(arr_np.shape[0])

            while pending_count >= args.shard_samples or (processed >= limit and pending_count > 0):
                take = min(args.shard_samples, pending_count)
                pieces = []
                need = take
                while need:
                    head = pending[0]
                    if head.shape[0] <= need:
                        pieces.append(head)
                        need -= head.shape[0]
                        pending.pop(0)
                    else:
                        pieces.append(head[:need])
                        pending[0] = head[need:]
                        need = 0
                shard = flush_shard(out_dir, shard_idx, pending_start, pieces)
                shards.append(shard)
                shard_idx += 1
                pending_start += take
                pending_count -= take
                bytes_written += int(shard["bytes"])
                rank_bytes += int(shard["bytes"])

            rank_samples += int(arr_np.shape[0])
            elapsed = time.perf_counter() - trace_start
            print(json.dumps({
                "event": "progress",
                "rank": rank,
                "trace": trace.name,
                "processed": processed,
                "limit": limit,
                "trace_samples_per_s": processed / elapsed if elapsed > 0 else None,
                "rank_samples": rank_samples,
            }), flush=True)

        trace_seconds = time.perf_counter() - trace_start
        manifest = {
            "complete": True,
            "trace": trace.name,
            "trace_path": trace.path,
            "rank": rank,
            "count": len(ds),
            "processed": processed,
            "limit": limit,
            "seconds": trace_seconds,
            "samples_per_s": processed / trace_seconds if trace_seconds > 0 else None,
            "bytes_written": bytes_written,
            "bytes_per_sample": bytes_written / processed if processed else None,
            "sample_shape": first_shape,
            "sample_dtype": first_dtype,
            "preprocess_fingerprint": fingerprint,
            "shards": shards,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(manifest_path, manifest)
        print(json.dumps({"event": "trace_complete", **manifest}), flush=True)

    seconds = time.perf_counter() - rank_start
    print(json.dumps({
        "event": "worker_complete",
        "rank": rank,
        "processed": rank_samples,
        "bytes_written": rank_bytes,
        "seconds": seconds,
        "samples_per_s": rank_samples / seconds if seconds > 0 else None,
    }), flush=True)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo, args.configs)
    fingerprint = preprocess_fingerprint(
        cfg, args.cache_dtype, args.limit_per_trace)
    traces = discover_traces(repo, data_root, cfg)
    requested_total = sum(t.count if args.limit_per_trace is None else min(t.count, args.limit_per_trace) for t in traces)
    assignments = [[] for _ in range(args.num_processes)]
    for i, trace in enumerate(sorted(traces, key=lambda t: t.count, reverse=True)):
        assignments[i % args.num_processes].append(trace.__dict__)

    root_manifest = {
        "complete": False,
        "repo": str(repo),
        "data_root": str(data_root),
        "out_root": str(out_root),
        "configs": args.configs,
        "pipeline": "RoverTrace exact radar channel transform: Raw IQ -> IIQQtoIQ -> FFTArray -> RADsLikeDoppler -> ComplexPhase",
        "format": "per-trace numpy shards",
        "sample_dtype": args.cache_dtype,
        "preprocess_config": preprocessing_spec(cfg),
        "preprocess_fingerprint": fingerprint,
        "num_processes": args.num_processes,
        "batch_size": args.batch_size,
        "shard_samples": args.shard_samples,
        "workers_per_process": args.workers_per_process,
        "requested_total_samples": requested_total,
        "full_trace_counts": {t.name: t.count for t in traces},
        "assignments": [[t["name"] for t in a] for a in assignments],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(out_root / "manifest.json", root_manifest)
    print(json.dumps({"event": "start", **root_manifest}), flush=True)

    ctx = get_context("spawn")
    procs = []
    args_dict = vars(args)
    for rank, payloads in enumerate(assignments):
        proc = ctx.Process(target=worker_main, args=(rank, args_dict, payloads))
        proc.start()
        procs.append(proc)
    exit_codes = []
    for proc in procs:
        proc.join()
        exit_codes.append(proc.exitcode)
    if any(code != 0 for code in exit_codes):
        raise SystemExit(f"Worker failures: {exit_codes}")

    total_processed = 0
    total_bytes = 0
    trace_manifests = []
    for trace in traces:
        rel = Path("traces") / trace.name / "manifest.json"
        with open(out_root / rel, "r", encoding="utf-8") as f:
            m = json.load(f)
        trace_manifests.append(str(rel))
        total_processed += int(m["processed"])
        total_bytes += int(m["bytes_written"])
    root_manifest.update({
        "complete": True,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "processed_samples": total_processed,
        "bytes_written": total_bytes,
        "bytes_per_sample": total_bytes / total_processed if total_processed else None,
        "trace_manifests": trace_manifests,
    })
    atomic_json(out_root / "manifest.json", root_manifest)
    print("FINAL_MANIFEST " + json.dumps(root_manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
