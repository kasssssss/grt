"""Pretrained weight initialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


# Calibrated on all 16 fixed RADs-like I/Q-1M validation samples for the
# [2,8,1,4] -> [8,32,1,8] patch conversion. The usual 0.5 fan-in exponent
# assumes independent inputs, while zero-padded azimuth FFT bins are strongly
# correlated and otherwise produce roughly 4.3x larger patch activations.
PATCH_RESIZE_SCALE_EXPONENT = 0.92


def _patch_merge_order(values: Tensor, scale: list[int]) -> Tensor:
    """Return the flattened feature order produced by deepradar PatchMerge."""
    n, *spatial, channels = values.shape
    dims: list[int] = [n]
    for dim, patch in zip(spatial, scale):
        dims.extend([dim // patch, patch])
    order = (
        [0]
        + [2 * i + 1 for i in range(len(scale))]
        + [2 * i + 2 for i in range(len(scale))]
        + [-1]
    )
    out_spatial = [dim // patch for dim, patch in zip(spatial, scale)]
    return values.reshape(dims + [channels]).permute(order).reshape(n, *out_spatial, -1).reshape(-1)


def _squeeze_official_indices(values: Tensor, squeeze_dims: list[int]) -> Tensor:
    """Apply the official SpectrumTokenizer squeeze ordering to an index tensor."""
    out = values
    for dim in sorted(squeeze_dims, reverse=True):
        out = torch.moveaxis(out, dim + 1, -1)
        out = out.reshape(*out.shape[:-2], -1)
    return out


def _adapt_official_patch_weight(
    weight: Tensor,
    elevation_index: int,
    target_patch: list[int] | None = None,
) -> Tensor:
    """Reorder official tokenizer columns for local RADs-like PatchMerge.

    Official base/small tokenization receives [T,D,E,A,R,C], squeezes E and A
    into channels, then patches [T,D,R]. Local RADs-like training receives
    [D,A,E,R,C] and patches [D,A,E,R]. This routine builds both feature orders
    with integer sentinels and maps the official columns into the local order.
    A non-negative ``elevation_index`` keeps one official elevation bin, which
    is the RADs-like I/Q-1M path used for matching RADs' single-elevation cube.
    ``-1`` keeps both official elevation bins only for explicit diagnostics.
    """
    if elevation_index < -1 or elevation_index >= 2:
        raise ValueError(
            "official base/small has two elevation bins: use -1, 0, or 1")
    elevation_indices = (0, 1) if elevation_index < 0 else (elevation_index,)

    # Official base/small patch: T=1, D=2, E=2, A=8, R=4, C=2.
    official_shape = (1, 1, 2, 2, 8, 4, 2)
    official_values = torch.arange(int(np.prod(official_shape))).reshape(official_shape)
    squeezed = _squeeze_official_indices(official_values, squeeze_dims=[2, 3])
    official_order = _patch_merge_order(squeezed, scale=[1, 2, 4])
    official_feature_pos = {int(v): i for i, v in enumerate(official_order.tolist())}

    # Local RADs-like patch: D=2, A=8, E={1,2}, R=4, C=2.
    local_values = torch.empty(
        (1, 2, 8, len(elevation_indices), 4, 2), dtype=torch.long)
    for d in range(2):
        for a in range(8):
            for local_e, official_e in enumerate(elevation_indices):
                for r in range(4):
                    for c in range(2):
                        official_idx = np.ravel_multi_index(
                            (0, d, official_e, a, r, c),
                            official_shape[1:],
                        )
                        local_values[0, d, a, local_e, r, c] = int(official_idx)

    source_patch = [2, 8, len(elevation_indices), 4]
    local_as_official = _patch_merge_order(
        local_values, scale=source_patch)
    source_columns = torch.tensor(
        [official_feature_pos[int(v)] for v in local_as_official.tolist()],
        dtype=torch.long,
        device=weight.device,
    )
    adapted = weight[:, source_columns].contiguous()
    if target_patch is not None and list(target_patch) != source_patch:
        adapted = _resize_local_patch_weight(
            adapted, source_patch=source_patch, target_patch=target_patch)
    return adapted


def _resize_local_patch_weight(
    weight: Tensor,
    source_patch: list[int],
    target_patch: list[int],
    channels: int = 2,
) -> Tensor:
    """Resize a local GRT patch projection while preserving feature order.

    PatchMerge flattens features in ``D,A,E,R,C`` order.  The official local
    adapter first constructs that exact order, after which this function
    trilinearly resizes the Doppler/azimuth/range kernel.  Elevation is kept
    discrete because the RADs-like experiment explicitly selects one physical
    elevation channel rather than averaging channels.  Fan-in scaling keeps
    the initialized projection variance comparable after resizing.
    """
    if len(source_patch) != 4 or len(target_patch) != 4:
        raise ValueError("source_patch and target_patch must be [D,A,E,R].")
    if source_patch[2] != target_patch[2]:
        raise ValueError(
            "Patch resizing does not interpolate elevation; select the "
            "required elevation channel before resizing.")

    d_model = int(weight.shape[0])
    expected = int(np.prod(source_patch)) * channels
    if int(weight.shape[1]) != expected:
        raise ValueError(
            f"Expected patch weight width {expected}, got {weight.shape[1]}.")

    source = weight.reshape(d_model, *source_patch, channels)
    volume = source.permute(0, 3, 5, 1, 2, 4).reshape(
        d_model * source_patch[2] * channels,
        1,
        source_patch[0],
        source_patch[1],
        source_patch[3],
    )
    resized = F.interpolate(
        volume.float(),
        size=(target_patch[0], target_patch[1], target_patch[3]),
        mode="trilinear",
        align_corners=False,
    ).to(weight.dtype)
    resized = resized.reshape(
        d_model,
        source_patch[2],
        channels,
        target_patch[0],
        target_patch[1],
        target_patch[3],
    ).permute(0, 3, 4, 1, 5, 2)

    fan_in_scale = float(
        (np.prod(source_patch) / np.prod(target_patch))
        ** PATCH_RESIZE_SCALE_EXPONENT)
    return resized.reshape(d_model, -1).mul_(fan_in_scale).contiguous()


def _load_official_state(model_dir: str | Path) -> dict[str, Tensor]:
    model_dir = Path(model_dir)
    state = torch.load(model_dir / "weights.pth", map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    return {k.removeprefix("model."): v for k, v in state.items()}


def _copy_if_present(
    updates: dict[str, Tensor],
    report: dict[str, Any],
    target_state: dict[str, Tensor],
    source_state: dict[str, Tensor],
    source_key: str,
    target_key: str,
    value: Tensor | None = None,
) -> None:
    if value is None:
        if source_key not in source_state:
            report["missing_source"].append(source_key)
            return
        value = source_state[source_key]

    if target_key not in target_state:
        report["missing_target"].append(target_key)
        return
    if tuple(value.shape) != tuple(target_state[target_key].shape):
        report["shape_skipped"].append(
            {
                "source": source_key,
                "target": target_key,
                "source_shape": list(value.shape),
                "target_shape": list(target_state[target_key].shape),
            }
        )
        return

    updates[target_key] = value.detach().clone()
    report["loaded"].append(target_key)


def _map_encoder_layer(
    updates: dict[str, Tensor],
    report: dict[str, Any],
    target_state: dict[str, Tensor],
    source_state: dict[str, Tensor],
    layer: int,
) -> None:
    pairs = {
        "self_attn.in_proj_weight": "attn.in_proj_weight",
        "self_attn.in_proj_bias": "attn.in_proj_bias",
        "self_attn.out_proj.weight": "attn.out_proj.weight",
        "self_attn.out_proj.bias": "attn.out_proj.bias",
        "norm1.weight": "norm.weight",
        "norm1.bias": "norm.bias",
        "norm2.weight": "feedforward.0.weight",
        "norm2.bias": "feedforward.0.bias",
        "linear1.weight": "feedforward.1.weight",
        "linear1.bias": "feedforward.1.bias",
        "linear2.weight": "feedforward.4.weight",
        "linear2.bias": "feedforward.4.bias",
    }
    for source_suffix, target_suffix in pairs.items():
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            f"encoder.layers.{layer}.{source_suffix}",
            f"encoder.layers.{layer}.{target_suffix}",
        )


def _map_decoder_layer(
    updates: dict[str, Tensor],
    report: dict[str, Any],
    target_state: dict[str, Tensor],
    source_state: dict[str, Tensor],
    layer: int,
    source_head: str,
) -> None:
    pairs = {
        "self_attn.in_proj_weight": "attn.in_proj_weight",
        "self_attn.in_proj_bias": "attn.in_proj_bias",
        "self_attn.out_proj.weight": "attn.out_proj.weight",
        "self_attn.out_proj.bias": "attn.out_proj.bias",
        "multihead_attn.in_proj_weight": "attn2.in_proj_weight",
        "multihead_attn.in_proj_bias": "attn2.in_proj_bias",
        "multihead_attn.out_proj.weight": "attn2.out_proj.weight",
        "multihead_attn.out_proj.bias": "attn2.out_proj.bias",
        "norm1.weight": "norm.weight",
        "norm1.bias": "norm.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "feedforward.0.weight",
        "norm3.bias": "feedforward.0.bias",
        "linear1.weight": "feedforward.1.weight",
        "linear1.bias": "feedforward.1.bias",
        "linear2.weight": "feedforward.4.weight",
        "linear2.bias": "feedforward.4.bias",
    }
    for source_suffix, target_suffix in pairs.items():
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            f"decoder.{source_head}.decoder.layers.{layer}.{source_suffix}",
            f"decoder.layers.{layer}.{target_suffix}",
        )


def _resolve_decoder_head(
    source_state: dict[str, Tensor], requested: str | None
) -> str:
    """Resolve the task head contained in an official GRT checkpoint."""
    available = sorted({
        key.split(".")[1]
        for key in source_state
        if key.startswith("decoder.") and len(key.split(".")) > 2
    })
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested decoder head {requested!r} is not present in the "
                f"official checkpoint; available heads: {available}.")
        return requested
    if len(available) != 1:
        raise ValueError(
            "Could not infer a unique official decoder head; pass "
            f"decoder_head explicitly. Available heads: {available}.")
    return available[0]


def load_official_grt_base(
    model: torch.nn.Module,
    model_dir: str | Path,
    elevation_index: int = 0,
    load_decoder: bool = True,
    decoder_head: str | None = None,
    min_loaded_fraction: float = 0.0,
) -> dict[str, Any]:
    """Initialize a local model from an official task-specific GRT checkpoint.

    The official checkpoints store each task under a different decoder prefix
    (for example ``occ3d`` or ``semseg``). Loading the wrong prefix silently
    leaves the local decoder random, so this function records coverage and can
    fail fast when less than ``min_loaded_fraction`` of the requested model
    scope was initialized.
    """
    if not 0.0 <= min_loaded_fraction <= 1.0:
        raise ValueError("min_loaded_fraction must be between 0 and 1.")
    source_state = _load_official_state(model_dir)
    target_state = model.state_dict()
    resolved_head = (
        _resolve_decoder_head(source_state, decoder_head)
        if load_decoder else decoder_head)
    updates: dict[str, Tensor] = {}
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "elevation_index": elevation_index,
        "load_decoder": load_decoder,
        "decoder_head": resolved_head,
        "min_loaded_fraction": min_loaded_fraction,
        "loaded": [],
        "missing_source": [],
        "missing_target": [],
        "shape_skipped": [],
    }

    target_patch = list(getattr(getattr(model.encoder, "patch", None), "scale", []))
    source_patch = [2, 8, 2 if elevation_index < 0 else 1, 4]
    report["source_patch"] = source_patch
    report["target_patch"] = target_patch
    report["patch_resized"] = bool(target_patch and target_patch != source_patch)
    report["patch_resize_scale_exponent"] = (
        PATCH_RESIZE_SCALE_EXPONENT if report["patch_resized"] else None)

    patch_key = "tokenizer.patch.linear.weight"
    if patch_key in source_state:
        adapted = _adapt_official_patch_weight(
            source_state[patch_key],
            elevation_index,
            target_patch=target_patch or None,
        )
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            patch_key,
            "encoder.patch.reduction.weight",
            value=adapted,
        )
    else:
        report["missing_source"].append(patch_key)

    _copy_if_present(
        updates,
        report,
        target_state,
        source_state,
        "tokenizer.readout.readout",
        "encoder.readout.readout",
    )

    for layer in range(len(getattr(model.encoder, "layers", []))):
        _map_encoder_layer(updates, report, target_state, source_state, layer)

    if load_decoder:
        assert resolved_head is not None
        for layer in range(len(getattr(model.decoder, "layers", []))):
            _map_decoder_layer(
                updates, report, target_state, source_state, layer,
                resolved_head)
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            f"decoder.{resolved_head}.unpatch.linear.weight",
            "decoder.unpatch.linear.weight",
        )
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            f"decoder.{resolved_head}.unpatch.linear.bias",
            "decoder.unpatch.linear.bias",
        )

    target_state.update(updates)
    model.load_state_dict(target_state)
    report["loaded_count"] = len(report["loaded"])
    report["loaded_tensors"] = int(sum(updates[k].numel() for k in updates))
    expected_prefixes = ("encoder.", "decoder.") if load_decoder else ("encoder.",)
    expected_tensors = {
        key: value for key, value in target_state.items()
        if key.startswith(expected_prefixes)}
    expected_numel = int(sum(value.numel() for value in expected_tensors.values()))
    loaded_numel = int(sum(
        target_state[key].numel() for key in updates if key in expected_tensors))
    report["expected_tensors"] = len(expected_tensors)
    report["expected_numel"] = expected_numel
    report["loaded_numel"] = loaded_numel
    report["loaded_fraction"] = (
        loaded_numel / expected_numel if expected_numel else 1.0)
    if report["loaded_fraction"] < min_loaded_fraction:
        raise RuntimeError(
            "Official checkpoint coverage is below the required threshold: "
            f"{report['loaded_fraction']:.6f} < {min_loaded_fraction:.6f}. "
            f"decoder_head={resolved_head!r}, missing_source="
            f"{len(report['missing_source'])}, shape_skipped="
            f"{len(report['shape_skipped'])}.")
    return report
