import os

import numpy as np
import pytest
import torch

from deepradar.pretrained import _adapt_official_patch_weight
from deepradar.objectives.occupancy3 import PolarOccupancy
from deepradar.transforms import lidar as lidar_transforms
from models.grt import (
    AzimuthFFTTransformerEncoder,
    ResidualRefinedTransformerDecoder,
    TransformerDecoder,
)


def make_encoder(azimuth_bins: int = 256) -> AzimuthFFTTransformerEncoder:
    return AzimuthFFTTransformerEncoder(
        azimuth_bins=azimuth_bins,
        layers=0,
        dim=32,
        ff_ratio=2.0,
        head_dim=8,
        patch=[1, 1, 1, 1],
        input_channels=2,
    )


def complex_phase(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.sqrt(torch.abs(value)), torch.angle(value)), dim=-1)


def test_a8_to_a256_matches_aperture_zero_padding() -> None:
    torch.manual_seed(7)
    source = torch.complex(
        torch.randn(1, 4, 8, 1, 16),
        torch.randn(1, 4, 8, 1, 16),
    )
    encoded = complex_phase(source)
    actual = make_encoder().expand_azimuth(encoded)

    aperture = torch.fft.ifft(torch.fft.ifftshift(source, dim=2), dim=2)
    aperture = torch.cat(
        (aperture, torch.zeros(1, 4, 248, 1, 16, dtype=aperture.dtype)),
        dim=2,
    )
    expected = complex_phase(torch.fft.fftshift(
        torch.fft.fft(aperture, dim=2), dim=2))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_a256_input_is_not_modified() -> None:
    value = torch.randn(1, 2, 256, 1, 4, 2)
    assert make_encoder().expand_azimuth(value) is value


def test_azimuth_shrink_requires_an_explicit_reducer() -> None:
    with pytest.raises(ValueError, match="explicit reducer"):
        make_encoder(azimuth_bins=8).expand_azimuth(
            torch.randn(1, 2, 256, 1, 4, 2))


def test_official_patch_resize_has_a256_shape_and_finite_values() -> None:
    official = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256)
    baseline = _adapt_official_patch_weight(official, elevation_index=0)
    resized = _adapt_official_patch_weight(
        official,
        elevation_index=0,
        target_patch=[8, 32, 1, 8],
    )
    assert baseline.shape == (3, 128)
    assert resized.shape == (3, 4096)
    assert torch.isfinite(resized).all()
    assert np.isfinite(float(resized.std()))


def make_decoder(cls: type[TransformerDecoder]) -> TransformerDecoder:
    return cls(
        key="map",
        layers=0,
        dim=32,
        ff_ratio=2.0,
        head_dim=8,
        shape=[8, 16, 8],
        patch=[2, 4, 2],
        positions="nd",
    )


def test_zero_initialized_refiner_matches_baseline() -> None:
    torch.manual_seed(11)
    baseline = make_decoder(TransformerDecoder)
    refined = make_decoder(ResidualRefinedTransformerDecoder)
    refined.load_state_dict(baseline.state_dict(), strict=False)
    encoded = torch.randn(2, 9, 32)
    torch.testing.assert_close(
        refined(encoded)["map"], baseline(encoded)["map"],
        rtol=0.0, atol=0.0,
    )


def test_refiner_receives_gradient_from_first_step() -> None:
    refined = make_decoder(ResidualRefinedTransformerDecoder)
    loss = refined(torch.randn(2, 9, 32))["map"].square().mean()
    loss.backward()
    final = refined.refiner[-1]
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0


def test_seam_loss_targets_only_surface_context_boundaries() -> None:
    objective = PolarOccupancy(
        seam_weight=0.1,
        seam_patch=[2, 2, 2],
        seam_context_radius=1,
    )
    target = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
    target[:, 1, 1, 1] = True
    continuous = torch.zeros(1, 4, 4, 4)
    assert objective.seam_loss(target, continuous).item() == 0.0

    broken = continuous.clone()
    broken[:, 2:, :, :] = 2.0
    assert objective.seam_loss(target, broken).item() > 0.0

    far_background = continuous.clone()
    far_background[:, :, :, 3] = 2.0
    assert objective.seam_loss(target, far_background).item() == 0.0


def test_destagger_restores_stdout_after_metadata_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lidar_dir = tmp_path / "lidar"
    lidar_dir.mkdir()
    (lidar_dir / "lidar.json").write_text("{}", encoding="utf-8")

    def fail_sensor_info(_metadata: str) -> None:
        raise RuntimeError("invalid test metadata")

    monkeypatch.setattr(
        lidar_transforms.client, "SensorInfo", fail_sensor_info)
    with pytest.raises(RuntimeError, match="invalid test metadata"):
        lidar_transforms.Destagger(str(tmp_path))

    duplicate = os.dup(1)
    os.close(duplicate)
