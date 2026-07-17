from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from deepradar.channels import PrecomputedRadarChannel


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def argparse_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        call.args[0].value
        for call in ast.walk(tree)
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        )
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_autodl_launchers_are_tracked_exceptions() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("git ignore contract requires a Git worktree")
    for rel in (
        "scripts/autodl_precompute_radslike_inputs.sh",
        "scripts/autodl_train_radslike.sh",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", rel], cwd=ROOT, check=False)
        assert result.returncode == 1, f"{rel} is still ignored"


def test_launcher_defaults_and_validator_cache_override() -> None:
    train = (ROOT / "scripts/autodl_train_radslike.sh").read_text(
        encoding="utf-8")
    precompute = (
        ROOT / "scripts/autodl_precompute_radslike_inputs.sh"
    ).read_text(encoding="utf-8")

    assert "LIMIT_TRAIN=${LIMIT_TRAIN:-}" in train
    assert "LIMIT_TRAIN=${LIMIT_TRAIN:-500}" not in train
    assert '--cache-root "${CACHE}"' in precompute
    assert 'DEFAULT_DEVICE_PREFIX=cpu' in precompute


def test_cpu_and_gpu_precompute_accept_launcher_argument_set() -> None:
    expected = {
        "--repo",
        "--data-root",
        "--out-root",
        "--batch-size",
        "--shard-samples",
        "--workers-per-process",
        "--num-processes",
        "--device-prefix",
        "--cache-dtype",
    }
    for name in (
        "precompute_radslike_inputs.py",
        "precompute_radslike_inputs_cpu.py",
    ):
        assert expected <= argparse_options(ROOT / "scripts" / name)


def test_gpu_precompute_rejects_unsupported_fft_contract() -> None:
    gpu = load_script("precompute_radslike_inputs.py")
    gpu.validate_fft_contract([0, 1, 2, 3], 0)
    with pytest.raises(ValueError, match=r"axes=\[0, 1, 2, 3\]"):
        gpu.validate_fft_contract([0, 1, 2], 0)
    with pytest.raises(ValueError, match="pad=0"):
        gpu.validate_fft_contract([0, 1, 2, 3], 8)


def test_cpu_fingerprint_changes_and_refuses_stale_cache() -> None:
    cpu = load_script("precompute_radslike_inputs_cpu.py")
    cfg = {
        "dataset": {
            "channels": {
                "radar": {
                    "name": "RawChannel",
                    "args": {
                        "transform": [
                            {"name": "FFTArray", "args": {"pad": 0}}
                        ]
                    },
                }
            }
        }
    }
    fingerprint = cpu.preprocess_fingerprint(cfg, "float16", None)
    assert fingerprint == cpu.preprocess_fingerprint(cfg, "float16", None)
    assert fingerprint != cpu.preprocess_fingerprint(cfg, "float32", None)
    assert fingerprint != cpu.preprocess_fingerprint(cfg, "float16", 5)
    with pytest.raises(RuntimeError, match="Refusing to reuse stale cache"):
        cpu.validate_reusable_manifest(
            {"complete": True, "preprocess_fingerprint": "stale"},
            fingerprint,
            "outdoor/baum",
        )


@pytest.mark.parametrize(
    ("root_manifest", "message"),
    [
        ({"complete": False}, "Incomplete precomputed cache root"),
        (
            {
                "complete": True,
                "full_trace_counts": {"outdoor/baum": 5},
                "processed_samples": 2,
            },
            "Limited precomputed cache",
        ),
    ],
)
def test_channel_rejects_incomplete_or_limited_cache_root(
    tmp_path: Path, root_manifest: dict, message: str
) -> None:
    dataset = tmp_path / "data" / "outdoor" / "baum"
    cache = tmp_path / "cache"
    write_json(cache / "manifest.json", root_manifest)
    with pytest.raises(ValueError, match=message):
        PrecomputedRadarChannel(str(dataset), None, str(cache))


def test_channel_rejects_limited_trace_cache(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    dataset = data_root / "outdoor" / "baum"
    cache = tmp_path / "cache"
    write_json(cache / "manifest.json", {
        "complete": True,
        "data_root": str(data_root),
        "full_trace_counts": {"outdoor/baum": 5},
        "processed_samples": 5,
    })
    write_json(cache / "traces/outdoor/baum/manifest.json", {
        "complete": True,
        "count": 5,
        "limit": 2,
        "processed": 2,
    })
    with pytest.raises(ValueError, match="Limited precomputed trace cache"):
        PrecomputedRadarChannel(str(dataset), None, str(cache))


def test_inference_paths_and_elevation_choice_are_explicit() -> None:
    infer_options = argparse_options(
        ROOT / "scripts/rads_map_checkpoint_infer.py")
    assert {"--repo", "--rads-root", "--gt-root", "--out-dir"} <= infer_options

    config = (ROOT / "config/repr/rads_like.yaml").read_text(
        encoding="utf-8")
    assert "elevation_index: 0" in config
    assert "negative/downward elevation bin" in config
    assert "does not mean zero elevation" in config


def test_aperture_truncate_is_exact_for_iq1m_a256_subspace() -> None:
    infer = load_script("rads_map_checkpoint_infer.py")
    rng = np.random.default_rng(17)
    native = (
        rng.standard_normal((3, 8, 5))
        + 1j * rng.standard_normal((3, 8, 5))
    ).astype(np.complex64)
    aperture = np.fft.ifft(np.fft.ifftshift(native, axes=1), axis=1)
    padded = np.pad(aperture, ((0, 0), (0, 248), (0, 0)))
    expanded = np.fft.fftshift(np.fft.fft(padded, axis=1), axes=1)

    recovered = infer.reduce_azimuth(
        expanded, target_bins=8, mode="aperture_truncate", beam_sigma=18.0)

    np.testing.assert_allclose(recovered, native, rtol=2e-6, atol=2e-6)


def test_circular_aperture_window_recovers_boundary_straddling_array() -> None:
    infer = load_script("rads_map_checkpoint_infer.py")
    rng = np.random.default_rng(23)
    aperture8 = (
        rng.standard_normal((3, 8, 5))
        + 1j * rng.standard_normal((3, 8, 5))
    ).astype(np.complex64)
    aperture256 = np.zeros((3, 256, 5), dtype=np.complex64)
    indices = (-3 + np.arange(8)) % 256
    aperture256[:, indices, :] = aperture8
    expanded = np.fft.fftshift(np.fft.fft(aperture256, axis=1), axes=1)
    expected = np.fft.fftshift(np.fft.fft(aperture8, axis=1), axes=1)

    recovered = infer.reduce_azimuth(
        expanded,
        target_bins=8,
        mode="aperture_window",
        beam_sigma=18.0,
        aperture_start=-3,
    )

    np.testing.assert_allclose(recovered, expected, rtol=2e-6, atol=2e-6)
