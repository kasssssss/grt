import numpy as np
import pytest
import torch

import models
from scripts import train_rads_azimuth_adapter as adapter_train
from deepradar.modules import (
    ComplexAzimuthProjection,
    RangeConditionedComplexAzimuthProjection,
    physical_azimuth_projection,
)
from scripts.evaluate_rads_azimuth_adapter import load_adapter
from models.grt import (
    QueryGridResidualMixer,
    QueryMixedTransformerDecoder,
    TransformerDecoder,
)


def test_physical_projection_matches_circular_aperture_window() -> None:
    rng = np.random.default_rng(7)
    value = (
        rng.normal(size=(3, 16, 5))
        + 1j * rng.normal(size=(3, 16, 5))
    ).astype(np.complex64)
    aperture = np.fft.ifft(np.fft.ifftshift(value, axes=1), axis=1)
    indices = (3 + np.arange(4)) % 16
    expected = np.fft.fftshift(
        np.fft.fft(aperture[:, indices, :], axis=1), axes=1)

    projection = ComplexAzimuthProjection(16, 4, start=3)
    actual = projection(torch.from_numpy(value)).detach().numpy()

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_complex_projection_zero_initialization_and_gradients() -> None:
    projection = ComplexAzimuthProjection(16, 4, start=2)
    expected = physical_azimuth_projection(16, 4, start=2)
    torch.testing.assert_close(projection.matrix(), expected, rtol=0.0, atol=0.0)

    value = torch.randn(2, 16, 3, dtype=torch.complex64)
    loss = projection(value).abs().mean()
    loss.backward()
    assert projection.delta_real.grad is not None
    assert projection.delta_imag.grad is not None
    assert torch.isfinite(projection.delta_real.grad).all()
    assert torch.isfinite(projection.delta_imag.grad).all()


def test_complex_projection_validates_input_contract() -> None:
    projection = ComplexAzimuthProjection(16, 4)
    with pytest.raises(ValueError, match="complex tensor"):
        projection(torch.randn(2, 16, 3))
    with pytest.raises(ValueError, match="Expected A=16"):
        projection(torch.randn(2, 8, 3, dtype=torch.complex64))
    with pytest.raises(ValueError, match="rank must be"):
        ComplexAzimuthProjection(16, 4, rank=5)


def test_low_rank_projection_is_exact_then_receives_gradients() -> None:
    projection = ComplexAzimuthProjection(16, 4, start=2, rank=2)
    expected = physical_azimuth_projection(16, 4, start=2)
    torch.testing.assert_close(projection.matrix(), expected, rtol=0.0, atol=0.0)

    value = torch.randn(2, 16, 3, dtype=torch.complex64)
    projection(value).abs().mean().backward()
    assert projection.left_real.grad is not None
    assert projection.left_imag.grad is not None
    assert torch.isfinite(projection.left_real.grad).all()
    assert torch.isfinite(projection.left_imag.grad).all()
    assert projection.left_real.grad.abs().sum() > 0


def test_range_conditioned_projection_is_exact_then_receives_gradients() -> None:
    base = ComplexAzimuthProjection(16, 4, start=2)
    with torch.no_grad():
        base.delta_real.normal_(std=0.05)
        base.delta_imag.normal_(std=0.05)
    projection = RangeConditionedComplexAzimuthProjection(
        base, range_bins=7, bands=3, rank=2)
    value = torch.randn(2, 16, 7, dtype=torch.complex64)

    torch.testing.assert_close(
        projection(value), base(value), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        projection.range_gates.sum(dim=1),
        torch.ones(7),
        rtol=0.0,
        atol=1e-7,
    )

    projection(value).abs().mean().backward()
    assert projection.left_real.grad is not None
    assert projection.left_imag.grad is not None
    assert projection.left_real.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in base.parameters())


def test_range_conditioned_projection_validates_contract() -> None:
    base = ComplexAzimuthProjection(16, 4)
    with pytest.raises(ValueError, match="range_bins"):
        RangeConditionedComplexAzimuthProjection(base, range_bins=0)
    with pytest.raises(ValueError, match="bands"):
        RangeConditionedComplexAzimuthProjection(base, bands=1)
    with pytest.raises(ValueError, match="rank"):
        RangeConditionedComplexAzimuthProjection(base, rank=5)

    projection = RangeConditionedComplexAzimuthProjection(
        base, range_bins=7, bands=3, rank=2)
    with pytest.raises(ValueError, match="complex tensor"):
        projection(torch.randn(2, 16, 7))
    with pytest.raises(ValueError, match="Expected A=16"):
        projection(torch.randn(2, 8, 7, dtype=torch.complex64))
    with pytest.raises(ValueError, match="Expected R=7"):
        projection(torch.randn(2, 16, 5, dtype=torch.complex64))


def test_load_portable_range_conditioned_checkpoint(tmp_path) -> None:
    base = ComplexAzimuthProjection(16, 4, start=2)
    projection = RangeConditionedComplexAzimuthProjection(
        base, range_bins=7, bands=3, rank=2)
    with torch.no_grad():
        projection.left_real.normal_(std=0.01)
    path = tmp_path / "range_adapter.pt"
    torch.save(
        {
            "adapter_kind": "range_conditioned",
            "adapter_config": {
                "base": {
                    "source_bins": 16,
                    "target_bins": 4,
                    "start": 2,
                },
                "range_bins": 7,
                "bands": 3,
                "rank": 2,
            },
            "adapter": projection.state_dict(),
        },
        path,
    )

    _, restored = load_adapter(path, torch.device("cpu"))
    value = torch.randn(2, 16, 7, dtype=torch.complex64)
    torch.testing.assert_close(restored(value), projection(value))


def test_load_frame_uses_matched_cube_and_gt_crop(monkeypatch, tmp_path) -> None:
    cube = np.ones((2, 3, 4), dtype=np.complex64)
    calls = []

    monkeypatch.setattr(adapter_train.np, "load", lambda path: cube)

    def shift(value, start):
        calls.append(("shift", start))
        return value + 1

    def to_dar(value, *, flip_azimuth, azimuth_flip_mode):
        calls.append(("to_dar", flip_azimuth, azimuth_flip_mode))
        return value + 2

    def keep(value, bins):
        calls.append(("doppler", bins))
        return value + 3

    def project(gt_ar, start, *, flip_azimuth, azimuth_flip_mode):
        calls.append(("gt", start, flip_azimuth, azimuth_flip_mode))
        assert gt_ar[0, 7]
        return gt_ar[:2, :4]

    monkeypatch.setattr(adapter_train, "shift_range_cube", shift)
    monkeypatch.setattr(adapter_train, "to_dar", to_dar)
    monkeypatch.setattr(adapter_train, "keep_center_doppler", keep)
    monkeypatch.setattr(adapter_train, "project_gt_ar", project)

    dar, target = adapter_train.load_frame(
        tmp_path,
        "100/000001",
        {"crop_start": 5, "ar_indices": [7]},
    )

    np.testing.assert_array_equal(dar, cube + 6)
    assert target.shape == (2, 4)
    assert calls == [
        ("shift", 5),
        ("to_dar", True, "index"),
        ("doppler", 11),
        ("gt", 5, True, "index"),
    ]


def test_query_grid_mixer_is_exact_identity_at_initialization() -> None:
    mixer = QueryGridResidualMixer(features=16, hidden=8, kernel=3)
    query = torch.randn(2, 8, 16)
    output = mixer(query, grid=(2, 2, 2))
    torch.testing.assert_close(output, query, rtol=0.0, atol=0.0)


def test_query_mixed_decoder_matches_standard_decoder_before_training() -> None:
    common = dict(
        key="map",
        layers=1,
        dim=16,
        ff_ratio=2.0,
        head_dim=4,
        dropout=0.0,
        activation="GELU",
        shape=(4, 4, 4),
        pos_scale=(1.0, 1.0, 1.0),
        patch=(2, 2, 2),
        positions="nd",
    )
    baseline = TransformerDecoder(**common)
    comparison = QueryMixedTransformerDecoder(mixer_dim=8, **common)
    incompatible = comparison.load_state_dict(
        baseline.state_dict(), strict=False)
    assert incompatible.missing_keys
    assert all(key.startswith("refiner.") for key in incompatible.missing_keys)
    assert not incompatible.unexpected_keys
    baseline.eval()
    comparison.eval()
    encoded = torch.randn(2, 9, 16)
    torch.testing.assert_close(
        comparison(encoded)["map"],
        baseline(encoded)["map"],
        rtol=0.0,
        atol=0.0,
    )
    assert models.QueryMixedTransformerDecoder is QueryMixedTransformerDecoder
