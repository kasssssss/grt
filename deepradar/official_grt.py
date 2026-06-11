"""Minimal loader for NRDK-format official GRT checkpoints.

The research-code Lightning path in this repository cannot directly load the
official NRDK checkpoint bundle. This module implements the small subset of
NRDK modules needed for inference from ``model.yaml`` + ``weights.pth``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from einops import rearrange
from torch import Tensor, nn


class Squeeze(nn.Module):
    def __init__(self, dim: Sequence[int] = (), size: Sequence[int] | None = None) -> None:
        super().__init__()
        self.dim = sorted(dim, reverse=True)
        self.size = size

    @property
    def n_channels(self) -> int:
        if self.size is None:
            raise ValueError("Cannot compute n_channels without `size`.")
        n = 1
        for dim in self.dim:
            n *= self.size[dim]
        return n

    def forward(self, data: Tensor) -> Tensor:
        for dim in self.dim:
            data = torch.moveaxis(data, dim + 1, -1)
            data = data.reshape(*data.shape[:-2], -1)
        return data


class PatchMerge(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        scale: Sequence[int],
        norm: bool = True,
        remainder: str = "crop",
    ) -> None:
        super().__init__()
        self.scale = list(scale)
        self.linear = nn.Linear(d_in * int(np.prod(scale)), d_out, bias=False)
        self.norm = nn.LayerNorm(d_in * int(np.prod(scale))) if norm else None
        self.remainder = remainder

    def _merge(self, x: Tensor) -> Tensor:
        n, *spatial, c = x.shape
        dims = sum(([d // s, s] for d, s in zip(spatial, self.scale)), start=[n])
        order = (
            [0]
            + [2 * i + 1 for i in range(len(self.scale))]
            + [2 * i + 2 for i in range(len(self.scale))]
            + [-1]
        )
        out_spatial = [d // s for d, s in zip(spatial, self.scale)]
        return x.reshape(dims + [c]).permute(order).reshape(n, *out_spatial, -1)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[1:-1]
        if any(xs % ps != 0 for xs, ps in zip(shape, self.scale)):
            remainder = [xs - ps * (xs // ps) for xs, ps in zip(shape, self.scale)]
            if self.remainder == "pad":
                pad = sum([[0, r] for r in reversed(remainder)], start=[0, 0])
                x = nn.functional.pad(x, pad, value=0.0)
            else:
                slices = [slice(None)] + [
                    slice(0, xs - r) for xs, r in zip(shape, remainder)
                ] + [slice(None)]
                x = x[tuple(slices)]
        merged = self._merge(x)
        if self.norm is not None:
            merged = self.norm(merged)
        return self.linear(merged)


class Sinusoid(nn.Module):
    def __init__(
        self,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = None,
        coef: float = 10000.0,
        channels: Sequence[int] | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.w_min = w_min
        self.coef = coef
        self.channels = channels

    def _channels(self, shape: Sequence[int]) -> list[int]:
        nd = len(shape) - 2
        if self.channels is None:
            return [shape[-1] // 2 // nd] * nd
        channels = list(self.channels)
        if any(isinstance(c, float) for c in channels):
            channels = [int(float(c) * shape[-1] // 2) for c in channels]
        return [int(c) for c in channels]

    def _scales(self, shape: Sequence[int]) -> list[float]:
        nd = len(shape) - 2
        if self.scale is None:
            scale = [1.0] * nd
        elif isinstance(self.scale, (float, int)):
            scale = [float(self.scale)] * nd
        else:
            scale = [float(s) for s in self.scale]

        if self.w_min is None:
            w_min = [1.0] * nd
        elif isinstance(self.w_min, (float, int)):
            w_min = [float(self.w_min)] * nd
        else:
            w_min = [float(w) for w in self.w_min]

        return [s * np.pi / w for s, w in zip(scale, w_min)]

    def forward(self, x: Tensor, positions: Sequence[Tensor] | None = None) -> Tensor:
        if positions is None:
            positions = [
                torch.linspace(-1.0, 1.0, steps=n, device=x.device)[None, :]
                for n in x.shape[1:-1]
            ]
        channels = self._channels(x.shape)
        scales = self._scales(x.shape)

        start_dim = 0
        for axis, (t, scale, n_channels) in enumerate(zip(positions, scales, channels)):
            w = self.coef ** (-torch.arange(n_channels, device=x.device) / n_channels)
            wt = scale * t[:, :, None] * w[None, None, :]

            p_slice: list[slice | None] = [None] * len(x.shape)
            p_slice[0] = slice(None)
            p_slice[axis + 1] = slice(None)
            p_slice[-1] = slice(None)

            sin_slice = [slice(None)] * len(x.shape)
            sin_slice[-1] = slice(start_dim, start_dim + n_channels * 2, 2)
            x[tuple(sin_slice)] = x[tuple(sin_slice)] + torch.sin(wt)[tuple(p_slice)]

            cos_slice = [slice(None)] * len(x.shape)
            cos_slice[-1] = slice(start_dim + 1, start_dim + n_channels * 2 + 1, 2)
            x[tuple(cos_slice)] = x[tuple(cos_slice)] + torch.cos(wt)[tuple(p_slice)]
            start_dim += n_channels * 2

        return x


class Readout(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.readout = nn.Parameter(data=torch.normal(0, 0.02, (d_model,)))

    def forward(self, x: Tensor) -> Tensor:
        readout = torch.tile(self.readout[None, None, :], (x.shape[0], 1, 1))
        return torch.concatenate((x, readout), dim=1)


class BasisChange(nn.Module):
    def __init__(
        self,
        shape: Sequence[int],
        flatten: bool = True,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = None,
        coef: float = 10000.0,
    ) -> None:
        super().__init__()
        self.shape = list(shape)
        self.flatten = flatten
        self.pos = Sinusoid(scale=scale, w_min=w_min, coef=coef)

    def forward(self, x: Tensor, positions: Sequence[Tensor] | None = None) -> Tensor:
        idxs = tuple([slice(None)] + [None] * len(self.shape) + [slice(None)])
        query = self.pos(torch.tile(x[idxs], (1, *self.shape, 1)), positions=positions)
        if self.flatten:
            query = query.reshape(x.shape[0], -1, x.shape[-1])
        return query


class Unpatch(nn.Module):
    def __init__(self, output_size: Sequence[int], features: int, size: Sequence[int]) -> None:
        super().__init__()
        self.output_size = list(output_size)
        self.size = list(size)
        self.linear = nn.Linear(features, output_size[-1] * int(np.prod(size)))

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.linear(x)
        if len(self.size) == 2:
            return rearrange(
                embedding,
                "n (x1 x2) (s1 s2 c) -> n (x1 s1) (x2 s2) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                s1=self.size[0],
                s2=self.size[1],
                c=self.output_size[-1],
            )
        if len(self.size) == 3:
            return rearrange(
                embedding,
                "n (x1 x2 x3) (s1 s2 s3 c) -> n (x1 s1) (x2 s2) (x3 s3) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                x3=self.output_size[2] // self.size[2],
                s1=self.size[0],
                s2=self.size[1],
                s3=self.size[2],
                c=self.output_size[-1],
            )
        if len(self.size) == 4:
            return rearrange(
                embedding,
                "n (x1 x2 x3 x4) (s1 s2 s3 s4 c) -> "
                "n (x1 s1) (x2 s2) (x3 s3) (x4 s4) c",
                x1=self.output_size[0] // self.size[0],
                x2=self.output_size[1] // self.size[1],
                x3=self.output_size[2] // self.size[2],
                x4=self.output_size[3] // self.size[3],
                s1=self.size[0],
                s2=self.size[1],
                s3=self.size[2],
                s4=self.size[3],
                c=self.output_size[-1],
            )
        raise ValueError("Unpatch is only implemented for 2/3/4D tensors.")


class SpectrumTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch: Sequence[int],
        squeeze: Sequence[int] = (),
        n_channels: int = 2,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
        positions: str = "nd",
    ) -> None:
        super().__init__()
        if squeeze:
            self.squeeze = Squeeze(dim=squeeze, size=patch)
            n_channels = n_channels * self.squeeze.n_channels
            patch = [p for i, p in enumerate(patch) if i not in squeeze]
        else:
            self.squeeze = None
        self.patch = PatchMerge(d_in=n_channels, d_out=d_model, scale=patch, norm=False)
        self.positions = positions
        self.pos = Sinusoid(scale=scale, w_min=w_min)
        self.readout = Readout(d_model=d_model)

    def forward(self, spectrum: Any) -> Tensor:
        x = spectrum.spectrum if hasattr(spectrum, "spectrum") else spectrum
        if self.squeeze is not None:
            x = self.squeeze(x)
        embedded = self.patch(x)
        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)
        return self.readout(flat)


class TransformerTensorDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.TransformerDecoderLayer,
        d_model: int,
        num_layers: int,
        shape: Sequence[int],
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
        patch: Sequence[int] = (16, 16),
        out_dim: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        query_shape = [s // p for s, p in zip(shape, patch)]
        self.query = BasisChange(shape=query_shape, scale=scale, w_min=w_min)
        self.unpatch = Unpatch(
            output_size=(*shape, max(1, out_dim)), features=d_model, size=patch
        )

    def forward(self, encoded: Tensor) -> Tensor:
        x = self.query(encoded[:, -1, :])
        enc = encoded[:, :-1, :]
        out = self.unpatch(self.decoder(x, enc))
        if self.out_dim == 0:
            out = out[..., 0]
        return out


class TokenizerEncoderDecoder(nn.Module):
    def __init__(self, tokenizer: nn.Module, encoder: nn.Module, decoder: dict[str, nn.Module]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = nn.ModuleDict(decoder)

    def forward(self, data: dict[str, Any]) -> dict[str, Tensor]:
        encoded = self.encoder(self.tokenizer(data["spectrum"]))
        return {key: decoder(encoded) for key, decoder in self.decoder.items()}


def _strip_target(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.pop("_target_", None)
    return cfg


def _build_encoder(cfg: dict[str, Any]) -> nn.TransformerEncoder:
    cfg = _strip_target(cfg)
    layer_cfg = _strip_target(cfg.pop("encoder_layer"))
    layer = nn.TransformerEncoderLayer(**layer_cfg)
    return nn.TransformerEncoder(layer, **cfg)


def _build_decoder_head(cfg: dict[str, Any]) -> TransformerTensorDecoder:
    cfg = _strip_target(cfg)
    layer_cfg = _strip_target(cfg.pop("decoder_layer"))
    layer = nn.TransformerDecoderLayer(**layer_cfg)
    return TransformerTensorDecoder(decoder_layer=layer, **cfg)


def load_official_grt(model_dir: str, device: torch.device | str = "cpu") -> TokenizerEncoderDecoder:
    """Load an official NRDK-format GRT model directory."""
    model_dir_path = __import__("pathlib").Path(model_dir)
    cfg = yaml.safe_load((model_dir_path / "model.yaml").read_text())
    model_cfg = cfg["model"]

    tokenizer = SpectrumTokenizer(**_strip_target(model_cfg["tokenizer"]))
    encoder = _build_encoder(model_cfg["encoder"])
    decoder = {
        name: _build_decoder_head(head_cfg)
        for name, head_cfg in model_cfg["decoder"].items()
    }
    model = TokenizerEncoderDecoder(tokenizer=tokenizer, encoder=encoder, decoder=decoder)

    state = torch.load(model_dir_path / "weights.pth", map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("model."): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"State dict mismatch for {model_dir}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def make_spectrum(real: np.ndarray, device: torch.device | str) -> SimpleNamespace:
    """Create the small SpectrumData-like object needed by official GRT."""
    return SimpleNamespace(
        spectrum=torch.from_numpy(real[None, None, ...]).float().to(device),
        timestamps=torch.zeros((1, 1), dtype=torch.float64, device=device),
        range_resolution=torch.full((1, 1), 0.08737720774357578, dtype=torch.float32, device=device),
        doppler_resolution=torch.full((1, 1), 0.05588182275609268, dtype=torch.float32, device=device),
    )
