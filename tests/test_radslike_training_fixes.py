from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from deepradar.objectives.occupancy3 import PolarOccupancy
from deepradar.objectives.semantics import Segmentation
from deepradar.pretrained import _resolve_decoder_head
from deepradar.transforms.radar import PrecomputedComplexPhaseAugment
from scripts.rads_map_checkpoint_infer import first_hit_depth
from train import (
    _FrozenModulesEvalCallback,
    _SelectiveTrainModeCallback,
    _initialize_from_base_models,
)


def test_semantic_miou_reduces_spatial_axes_and_ignores_absent_classes():
    target = torch.tensor([[[0, 0], [1, 1]]])
    assert torch.allclose(Segmentation.miou(target, target, nc=3), torch.ones(1))

    pred = torch.tensor([[[0, 1], [1, 1]]])
    expected = torch.tensor([(0.5 + 2.0 / 3.0) / 2.0])
    assert torch.allclose(Segmentation.miou(target, pred, nc=3), expected)


def test_semantic_balanced_focal_loss_is_finite():
    objective = Segmentation(
        class_weights=[1.0, 2.0, 1.0], focal_gamma=1.5)
    logits = torch.randn(2, 2, 2, 1, 3)
    target = torch.randint(0, 3, (2, 2, 2))
    metrics = objective.metrics(
        {"segment": target}, {"segment": logits}, train=False)
    assert torch.isfinite(metrics.loss)
    assert torch.isfinite(metrics.metrics["seg_miou"])


def test_weighted_focal_loss_uses_unweighted_probability_and_one_class_weight():
    logits = torch.tensor([[[[[2.0, -1.0]], [[-0.5, 1.5]]]]])
    target = torch.tensor([[[0, 1]]])
    weights = torch.tensor([2.0, 5.0])
    gamma = 2.0
    objective = Segmentation(
        class_weights=weights.tolist(), focal_gamma=gamma)

    actual = objective.metrics(
        {"segment": target}, {"segment": logits}, train=False).loss
    channel_first = logits.permute(0, 4, 3, 1, 2).reshape(1, 2, 1, 2)
    unweighted_ce = torch.nn.functional.cross_entropy(
        channel_first, target, reduction="none")
    expected = torch.mean(
        (1.0 - torch.exp(-unweighted_ce)).pow(gamma)
        * unweighted_ce
        * weights[target]
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("base_model", "official_base_model"),
    [("experiment", None), (None, "official")],
)
def test_freeze_applies_after_either_base_initialization(
    monkeypatch, base_model, official_base_model
):
    model = SimpleNamespace(encoder=MagicMock(), decoder=MagicMock())
    base = SimpleNamespace(encoder=MagicMock(), decoder=MagicMock())
    experiment_loader = MagicMock(return_value=base)
    monkeypatch.setattr(
        "train.DeepRadar.load_from_experiment", experiment_loader)
    official_loader = MagicMock(return_value={"loaded": True})
    monkeypatch.setattr("train.load_official_grt_base", official_loader)
    args = SimpleNamespace(
        base_model=base_model,
        official_base_model=official_base_model,
        freeze=True,
        load_decoder=False,
        load_full_decoder=False,
        official_elevation_index=0,
        official_skip_decoder=False,
        official_decoder_head=None,
        official_min_loaded_fraction=0.95,
    )

    _initialize_from_base_models(model, args)

    model.encoder.freeze.assert_called_once_with()
    if base_model is not None:
        experiment_loader.assert_called_once_with(
            base_model, checkpoint=None)
        official_loader.assert_not_called()
    else:
        experiment_loader.assert_not_called()
        official_loader.assert_called_once()


class _SelectiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.9))
        self.decoder = torch.nn.Module()
        self.decoder.base = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.9))
        self.decoder.refiner = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.9))
        self.decoder.unpatch = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.9))


@pytest.mark.parametrize("selected", ["decoder.refiner", "decoder.unpatch"])
def test_selective_train_mode_keeps_only_selected_branch_training(selected):
    model = _SelectiveModel()
    callback = _SelectiveTrainModeCallback(selected)

    for hook in (
        callback.on_train_start,
        callback.on_train_epoch_start,
        lambda trainer, module: callback.on_train_batch_start(
            trainer, module, batch=None, batch_idx=0),
    ):
        model.train()
        hook(None, model)
        assert model.training
        assert not model.encoder.training
        assert not model.decoder.training
        assert not model.decoder.base.training
        assert model.get_submodule(selected).training
        other = (
            model.decoder.unpatch
            if selected == "decoder.refiner"
            else model.decoder.refiner
        )
        assert not other.training

        sample = torch.ones(2, 4)
        assert torch.equal(model.encoder(sample), model.encoder(sample))


@pytest.mark.parametrize("frozen", [["encoder"], ["decoder"], ["encoder", "decoder"]])
def test_explicitly_frozen_modules_remain_in_eval_mode(frozen):
    model = _SelectiveModel()
    callback = _FrozenModulesEvalCallback(frozen)

    for hook in (
        callback.on_train_start,
        callback.on_train_epoch_start,
        lambda trainer, module: callback.on_train_batch_start(
            trainer, module, batch=None, batch_idx=0),
    ):
        model.train()
        hook(None, model)
        assert model.training
        assert model.encoder.training is ("encoder" not in frozen)
        assert model.decoder.training is ("decoder" not in frozen)


def test_precomputed_augmentation_flips_expected_axes():
    data = np.zeros((2, 3, 1, 4, 2), dtype=np.float32)
    data[..., 0] = np.arange(2 * 3 * 4).reshape(2, 3, 1, 4)
    transform = PrecomputedComplexPhaseAugment("")
    out = transform(data, aug={
        "azimuth_flip": True,
        "doppler_flip": True,
        "radar_scale": 1.0,
        "radar_phase": 0.0,
        "range_scale": 1.0,
        "speed_scale": 1.0,
    })
    np.testing.assert_allclose(out[..., 0], data[::-1, ::-1, ..., 0])


def test_first_hit_has_explicit_invalid_zero():
    occupancy = torch.zeros((1, 1, 2, 4), dtype=torch.bool)
    occupancy[0, 0, 1, 0] = True
    depth, valid = PolarOccupancy._first_hit(occupancy)
    assert depth.tolist() == [[[0.0, 1.0]]]
    assert valid.tolist() == [[[False, True]]]


def test_soft_first_hit_matches_single_confident_surface():
    logits = torch.full((1, 1, 2, 4), -20.0)
    logits[0, 0, 0, 2] = 20.0
    depth, hit_mass = PolarOccupancy._soft_first_hit(
        logits, threshold=1.0, temperature=0.25)
    torch.testing.assert_close(depth[0, 0, 0], torch.tensor(3.0))
    torch.testing.assert_close(hit_mass[0, 0, 0], torch.tensor(1.0))
    assert hit_mass[0, 0, 1] < 1e-5


def test_soft_first_hit_rejects_nonpositive_temperature():
    logits = torch.zeros((1, 1, 1, 2))
    with pytest.raises(ValueError):
        PolarOccupancy._soft_first_hit(logits, temperature=0.0)


def test_rads_first_hit_uses_training_one_based_range_contract():
    logits = np.full((1, 2, 4), -10.0, dtype=np.float32)
    logits[0, 0, 2] = 10.0
    depth, valid_fraction = first_hit_depth(logits, threshold=0.0)
    assert depth[0, 0] == 3.0
    assert np.isnan(depth[0, 1])
    assert valid_fraction == 0.5


def test_decoder_head_is_inferred_or_validated():
    state = {"decoder.semseg.decoder.layers.0.norm1.weight": torch.ones(1)}
    assert _resolve_decoder_head(state, None) == "semseg"
    assert _resolve_decoder_head(state, "semseg") == "semseg"
    with pytest.raises(ValueError):
        _resolve_decoder_head(state, "occ3d")
