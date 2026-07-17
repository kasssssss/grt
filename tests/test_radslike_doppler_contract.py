from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import torch

from deepradar.transforms.radar import RADsLikeDoppler


ROOT = Path(__file__).resolve().parents[1]


def load_eval_script():
    path = ROOT / "scripts" / "evaluate_doppler_cache_variants.py"
    spec = importlib.util.spec_from_file_location("evaluate_doppler_cache_variants", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_trace(tmp_path: Path) -> Path:
    radar = tmp_path / "radar"
    radar.mkdir()
    (radar / "radar.json").write_text(
        json.dumps({"doppler_resolution": 0.03810895950223435}),
        encoding="utf-8",
    )
    return tmp_path


def test_physical_iq1m_grid_maps_all_bins_to_center(tmp_path: Path) -> None:
    transform = RADsLikeDoppler(
        str(make_trace(tmp_path)),
        target_bins=64,
        target_max_speed=90.0,
        elevation_index=0,
        mapping="physical",
        merge="mean",
    )
    source = np.ones((1, 64, 1, 2, 1), dtype=np.complex64)
    target = transform(source)
    support = np.flatnonzero(np.any(np.abs(target) > 0, axis=(0, 2, 3, 4)))
    np.testing.assert_array_equal(support, np.asarray([32]))


def test_rms_peak_phase_avoids_opposite_phase_cancellation(tmp_path: Path) -> None:
    trace = make_trace(tmp_path)
    source = np.ones((1, 64, 1, 2, 1), dtype=np.complex64)
    source[:, 32:] = -1.0
    mean = RADsLikeDoppler(
        str(trace), elevation_index=0, mapping="physical", merge="mean")
    rms = RADsLikeDoppler(
        str(trace), elevation_index=0, mapping="physical", merge="rms_peak_phase")

    assert np.abs(mean(source)[0, 32, 0, 0, 0]) == 0.0
    np.testing.assert_allclose(np.abs(rms(source)[0, 32, 0, 0, 0]), 1.0)


def test_native_index_preserves_doppler_and_selected_elevation(tmp_path: Path) -> None:
    transform = RADsLikeDoppler(
        str(make_trace(tmp_path)),
        target_bins=64,
        elevation_index=0,
        mapping="native_index",
        merge="mean",
        smooth_sigma_bins=0.0,
    )
    rng = np.random.default_rng(7)
    source = (
        rng.standard_normal((1, 64, 3, 2, 4))
        + 1j * rng.standard_normal((1, 64, 3, 2, 4))
    ).astype(np.complex64)

    target = transform(source)

    np.testing.assert_allclose(target, source[:, :, :, 0:1, :], rtol=0, atol=0)


def test_power_blur_preserves_total_complex_power() -> None:
    evaluator = load_eval_script()
    sample = torch.zeros((1, 64, 1, 1, 1, 2), dtype=torch.float32)
    sample[:, 32, ..., 0] = 1.0

    blurred = evaluator.power_blur_complex_phase(sample, sigma=1.25)

    power = torch.pow(blurred[..., 0], 4)
    torch.testing.assert_close(power.sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(power) == 9


def test_evaluator_batch_slices_cover_all_samples_once() -> None:
    evaluator = load_eval_script()

    slices = evaluator.batch_slices(19, 8)

    assert slices == [slice(0, 8), slice(8, 16), slice(16, 19)]
    assert [index for part in slices for index in range(part.start, part.stop)] == list(range(19))


def test_evaluator_load_range_crosses_shard_boundaries(tmp_path: Path) -> None:
    evaluator = load_eval_script()
    trace_root = tmp_path / "traces" / "outdoor" / "sample"
    trace_root.mkdir(parents=True)
    np.save(trace_root / "radar_000.npy", np.arange(12).reshape(3, 4))
    np.save(trace_root / "radar_001.npy", np.arange(12, 28).reshape(4, 4))

    selected = evaluator.load_range(tmp_path, "outdoor/sample", start=2, count=4)

    np.testing.assert_array_equal(selected, np.arange(8, 24).reshape(4, 4))


def test_numpy_power_smoothing_preserves_total_complex_power(tmp_path: Path) -> None:
    transform = RADsLikeDoppler(
        str(make_trace(tmp_path)),
        elevation_index=0,
        mapping="physical",
        merge="rms_peak_phase",
        smooth_sigma_bins=1.25,
        smooth_mode="power_peak_phase",
    )
    source = np.ones((1, 64, 1, 2, 1), dtype=np.complex64)

    target = transform(source)

    np.testing.assert_allclose(np.sum(np.square(np.abs(target))), 1.0, rtol=1e-6, atol=1e-6)
    assert np.count_nonzero(np.abs(target)) == 9
