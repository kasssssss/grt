"""Pretrained weight initialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


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
) -> Tensor:
    """Select one official elevation bin and reorder columns for local PatchMerge.

    Official base/small tokenization receives [T,D,E,A,R,C], squeezes E and A
    into channels, then patches [T,D,R]. Local RADs-like training receives
    [D,A,E,R,C] and patches [D,A,E,R]. This routine builds both feature orders
    with integer sentinels and maps the chosen official elevation columns into
    the local order.
    """
    if not 0 <= elevation_index < 2:
        raise ValueError("official base/small has two elevation bins: 0 or 1")

    # Official base/small patch: T=1, D=2, E=2, A=8, R=4, C=2.
    official_shape = (1, 1, 2, 2, 8, 4, 2)
    official_values = torch.arange(int(np.prod(official_shape))).reshape(official_shape)
    squeezed = _squeeze_official_indices(official_values, squeeze_dims=[2, 3])
    official_order = _patch_merge_order(squeezed, scale=[1, 2, 4])
    official_feature_pos = {int(v): i for i, v in enumerate(official_order.tolist())}

    # Local RADs-like patch: D=2, A=8, E=1, R=4, C=2.
    local_values = torch.empty((1, 2, 8, 1, 4, 2), dtype=torch.long)
    for d in range(2):
        for a in range(8):
            for r in range(4):
                for c in range(2):
                    official_idx = np.ravel_multi_index(
                        (0, d, elevation_index, a, r, c),
                        official_shape[1:],
                    )
                    local_values[0, d, a, 0, r, c] = int(official_idx)

    local_as_official = _patch_merge_order(local_values, scale=[2, 8, 1, 4])
    source_columns = torch.tensor(
        [official_feature_pos[int(v)] for v in local_as_official.tolist()],
        dtype=torch.long,
        device=weight.device,
    )
    return weight[:, source_columns].contiguous()


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
            f"decoder.occ3d.decoder.layers.{layer}.{source_suffix}",
            f"decoder.layers.{layer}.{target_suffix}",
        )


def load_official_grt_base(
    model: torch.nn.Module,
    model_dir: str | Path,
    elevation_index: int = 0,
    load_decoder: bool = True,
) -> dict[str, Any]:
    """Initialize local training model from official GRT base/small weights."""
    source_state = _load_official_state(model_dir)
    target_state = model.state_dict()
    updates: dict[str, Tensor] = {}
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "elevation_index": elevation_index,
        "load_decoder": load_decoder,
        "loaded": [],
        "missing_source": [],
        "missing_target": [],
        "shape_skipped": [],
    }

    patch_key = "tokenizer.patch.linear.weight"
    if patch_key in source_state:
        adapted = _adapt_official_patch_weight(source_state[patch_key], elevation_index)
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
        for layer in range(len(getattr(model.decoder, "layers", []))):
            _map_decoder_layer(updates, report, target_state, source_state, layer)
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            "decoder.occ3d.unpatch.linear.weight",
            "decoder.unpatch.linear.weight",
        )
        _copy_if_present(
            updates,
            report,
            target_state,
            source_state,
            "decoder.occ3d.unpatch.linear.bias",
            "decoder.unpatch.linear.bias",
        )

    target_state.update(updates)
    model.load_state_dict(target_state)
    report["loaded_count"] = len(report["loaded"])
    report["loaded_tensors"] = int(sum(updates[k].numel() for k in updates))
    return report
