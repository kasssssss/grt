"""Generalizable Radar Transformer."""

import lightning as L
import numpy as np
import torch
from beartype.typing import Literal, Optional, Sequence
from jaxtyping import Float
from torch import Tensor, nn

from deepradar import modules


class TransformerEncoder(L.LightningModule):
    """Radar 4D Doppler Transformer.

    Only a selected subset of hidden layers are passed to the decoder, and
    are specified by `dec_layers` as follows:

    - Each entry in the list indicates the index of the encoder layer output
      that will be passed to the decoder. The length must match `dec_layers` in
      the decoder.
    - Indexing starts from 0 at the output of the patch projection (e.g. the
      output of the first encoder layer is index 1).

    Args:
        layers: number of encoder layers.
        dec_layers: which layers to pass to the decoder. If `None` (default),
            only the output of the last layer is returned.
        dim: hidden dimension.
        ff_ratio: expansion ratio for feedforward blocks.
        head_dim: number of dimensions per head for multihead attention.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        patch: input (doppler, azimuth, elevation, range) patch size.
        pos_scale: position embedding scale (i.e. the spatial range that this
            axis corresponds to). If `None`, only the global scale is used.
            Note that the frequencies are automatically scaled based on the
            dimension length, so changing resolution does not require any
            modification to `pos_scale` or `global_scale`.
            global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        input_channels: number of input channels.
        positions: type of positional embedding. `flat`: flattened positional
            embeddings, similar to the original ViT; `nd`: n-dimensional
            embeddings, splitting the input features into `d` equal chunks
            encoding each axis separately.
    """

    def __init__(
        self, layers: int = 5, dec_layers: Optional[Sequence[int]] = None,
        dim: int = 768, ff_ratio: float = 4.0, head_dim: int = 64,
        dropout: float = 0.1, activation: str = 'GELU',
        patch: Sequence[int] = (16, 1, 1, 16),
        pos_scale: Optional[Sequence[float]] = None, global_scale: float = 1.0,
        input_channels: int = 2,
        positions: Literal["flat", "nd"] = "nd",
    ) -> None:
        super().__init__()

        if len(patch) not in {4, 5}:
            raise ValueError(f"Must specify a 4D or 5D patch size ({patch}).")

        self.patch = modules.PatchMerge(
            d_in=input_channels, d_out=dim, scale=patch, norm=False)

        self.positions = positions
        self.pos = modules.Sinusoid(
            scale=pos_scale, global_scale=global_scale)
        self.readout = modules.Readout(d_model=dim)

        self.dec_layers = dec_layers
        self.layers = nn.ModuleList([
            modules.TransformerLayer(
                d_feedforward=int(ff_ratio * dim), d_model=dim,
                n_head=dim // head_dim, dropout=dropout, activation=activation)
            for _ in range(layers)])

    def forward(
        self, x: Float[Tensor, "n *t d a e r c"]
    ) -> Float[Tensor, "n s c"] | list[Float[Tensor, "n s c"]]:
        """Apply radar transformer.

        Args:
            x: input batch, with batch-doppler-azimuth-elevation-range-iq (or
                batch-time-... if 5D) axis order.

        Returns:
            Encoding output.
        """
        embedded = self.patch(x)

        if self.positions == "nd":
            embedded = self.pos(embedded)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])
        if self.positions == "flat":
            flat = self.pos(flat)

        x = self.readout(flat)

        if self.dec_layers is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i in self.dec_layers:
                    out.append(x)
            return out


class AzimuthFFTTransformerEncoder(TransformerEncoder):
    """GRT encoder that preserves a dense azimuth input grid.

    The precomputed RADs-like I/Q-1M cache stores the native eight-bin
    azimuth spectrum.  Expanding that spectrum by interpolation would discard
    phase consistency.  Instead, this encoder reconstructs the complex
    spectrum, returns to the antenna-aperture domain, zero pads the aperture,
    and applies the azimuth FFT again.  Inputs that already have the requested
    azimuth size, such as a native RADs ``A=256`` tensor, pass through
    unchanged.

    This class is opt-in so existing ``TransformerEncoder`` checkpoints and
    configurations retain their original behavior.

    Args:
        azimuth_bins: target azimuth FFT size. Must not be smaller than the
            input spectrum.
        kwargs: standard :class:`TransformerEncoder` arguments.
    """

    def __init__(self, azimuth_bins: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        if azimuth_bins < 1:
            raise ValueError("azimuth_bins must be positive.")
        self.azimuth_bins = int(azimuth_bins)

    def expand_azimuth(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> Float[Tensor, "n d a2 e r c"]:
        """Expand a ComplexPhase azimuth spectrum without losing phase."""
        if x.ndim != 6 or x.shape[-1] != 2:
            raise ValueError(
                "AzimuthFFTTransformerEncoder expects [N,D,A,E,R,2], "
                f"got {tuple(x.shape)}.")

        source_bins = int(x.shape[2])
        if source_bins == self.azimuth_bins:
            return x
        if source_bins > self.azimuth_bins:
            raise ValueError(
                f"Cannot phase-preservingly shrink A={source_bins} to "
                f"A={self.azimuth_bins}; use an explicit reducer.")

        magnitude = x[..., 0].float().square()
        phase = x[..., 1].float()
        spectrum = torch.polar(magnitude, phase)
        aperture = torch.fft.ifft(
            torch.fft.ifftshift(spectrum, dim=2), dim=2)
        padding = torch.zeros(
            (*aperture.shape[:2], self.azimuth_bins - source_bins,
             *aperture.shape[3:]),
            dtype=aperture.dtype,
            device=aperture.device,
        )
        aperture = torch.cat((aperture, padding), dim=2)
        expanded = torch.fft.fftshift(
            torch.fft.fft(aperture, dim=2), dim=2)
        return torch.stack(
            (torch.sqrt(torch.abs(expanded)), torch.angle(expanded)), dim=-1)

    def forward(
        self, x: Float[Tensor, "n d a e r c"]
    ) -> Float[Tensor, "n s c"] | list[Float[Tensor, "n s c"]]:
        return super().forward(self.expand_azimuth(x))


class TransformerDecoder(L.LightningModule):
    """Radar transformer tensor decoder.

    Args:
        key: target key, e.g. `bev`, `depth`.
        layers: number of decoder layers.
        dim: hidden dimension; should be the same as the encoder.
        ff_ratio: expansion ratio for feedforward blocks.
        head_dim: number of feature dimensions per head.
        dropout: dropout during training.
        activation: activation function to use.
        shape: output shape; should be a 2 element list or tuple.
        pos_scale: position embedding scale (i.e. the spatial range that this
            axis corresponds to). If `None`, only the global scale is used.
            Note that the frequencies are automatically scaled based on the
            dimension length, so changing resolution does not require any
            modification to `pos_scale` or `global_scale`.
        global_scale: scalar constant to multiply scale by for convenience of
            representation; yields a net scale of `scale * global_scale`.
        patch: patch size to use for unpatching. Must evenly divide `shape`.
        out_dim: output channels; if `=0`, the dimension is omitted entirely,
            i.e. `(h, w)` instead of `(h, w, c)`.
        positions: type of positional embedding. `flat`: flattened positional
            embeddings, similar to the original ViT; `nd`: n-dimensional
            embeddings, splitting the input features into `d` equal chunks
            encoding each axis separately.
        mode: how to obtain the query vector. Can be `last` (use the last
            token, nominally a output token) or `pool` (average pooling).
    """

    def __init__(
        self, key: str, layers: int = 3, dim: int = 768,
        ff_ratio: float = 4.0, head_dim: int = 64, dropout: float = 0.1,
        activation: str = 'GELU', shape: Sequence[int] = (1024, 256),
        pos_scale: Optional[Sequence[float]] = None, global_scale: float = 1.0,
        patch: Sequence[int] = (16, 16), out_dim: int = 0,
        positions: Literal["flat", "nd"] = "flat",
        mode: Literal["last", "pool"] = "last",
        shift_blend: float = 0.0,
    ) -> None:
        super().__init__()

        self.key = key
        self.out_dim = out_dim
        self.mode = mode
        self.shift_blend = float(shift_blend)
        if not 0.0 <= self.shift_blend <= 1.0:
            raise ValueError("shift_blend must be between zero and one.")

        self.patch = tuple(int(value) for value in patch)
        self.query_grid = tuple(
            int(output) // patch_size
            for output, patch_size in zip(shape, self.patch)
        )
        if any(
            int(output) % patch_size != 0
            for output, patch_size in zip(shape, self.patch)
        ):
            raise ValueError("patch must evenly divide decoder shape.")
        if self.shift_blend > 0.0:
            if positions != "nd" or len(self.query_grid) != 3:
                raise ValueError(
                    "shift_blend currently requires a 3D nd-position decoder.")
            if any(size < 2 for size in self.query_grid):
                raise ValueError(
                    "shift_blend requires at least two patches per axis.")
            if any(size % 2 != 0 for size in self.patch):
                raise ValueError("shift_blend requires even patch sizes.")

        self.layers = nn.ModuleList([
            modules.TransformerDecoder(
                d_feedforward=int(ff_ratio * dim), d_model=dim,
                n_head=dim // head_dim, dropout=dropout, activation=activation)
            for _ in range(layers)])

        query_shape = list(self.query_grid)
        if positions == "flat":
            query_shape = [int(np.prod(query_shape))]
        self.query = modules.BasisChange(
            shape=query_shape, scale=pos_scale, global_scale=global_scale)

        self.unpatch = modules.Unpatch(
            output_size=(*shape, max(1, self.out_dim)),
            features=dim, size=self.patch)

    def _decode_queries(
        self, query: Float[Tensor, "n q c"],
        encoded: Float[Tensor, "n s c"],
    ) -> Float[Tensor, "n q c"]:
        for layer in self.layers:
            query = layer(query, encoded)
        return query

    def _shift_queries(
        self, query: Float[Tensor, "n q c"]
    ) -> Float[Tensor, "n q_shift c"]:
        """Average neighboring query corners to form a half-patch grid."""
        n, _, c = query.shape
        q1, q2, q3 = self.query_grid
        grid = query.reshape(n, q1, q2, q3, c)
        shifted = sum(
            grid[:, i:i + q1 - 1, j:j + q2 - 1, k:k + q3 - 1]
            for i in (0, 1) for j in (0, 1) for k in (0, 1)
        ) / 8.0
        return shifted.reshape(n, -1, c)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
    ) -> dict[str, Float[Tensor, "n h w ..."]]:
        """Apply decoder.

        Args:
            encoded: list of encoded values. Each tensor should be the same
                size, and use batch-spatial-channel order. The last spatial
                element of each tensor should correspond to a readout token.

        Returns:
            2-dimensional output; only a single key (e.g. the specified `key`)
            is decoded.
        """
        if self.mode == "last":
            x = encoded[:, -1, :]
        else:
            x = torch.mean(encoded, dim=1)

        x = self.query(x)
        enc = encoded[:, :-1, :]
        out = self.unpatch(self._decode_queries(x, enc))
        if self.shift_blend > 0.0:
            shifted_grid = tuple(size - 1 for size in self.query_grid)
            shifted = self.unpatch.forward_grid(
                self._decode_queries(self._shift_queries(x), enc), shifted_grid)
            slices = tuple(
                slice(patch // 2, patch // 2 + shifted_size)
                for patch, shifted_size in zip(self.patch, shifted.shape[1:-1])
            )
            out = out.clone()
            out[(slice(None), *slices, slice(None))] = torch.lerp(
                out[(slice(None), *slices, slice(None))],
                shifted,
                self.shift_blend,
            )
        if self.out_dim == 0:
            out = out[..., 0]

        return {self.key: out}


class ResidualRefinedTransformerDecoder(TransformerDecoder):
    """Transformer decoder with an opt-in local 3D residual refiner.

    The standard GRT unpatch projection predicts each output patch without a
    local operation spanning neighboring patch boundaries.  This lightweight
    depthwise-convolutional block operates on the reconstructed polar volume
    and can therefore repair boundary discontinuities without increasing the
    transformer token count.  Its final projection is zero-initialized, so a
    converted checkpoint initially produces exactly the baseline decoder
    output.

    Args:
        refine_dim: hidden width of the local refiner.
        refine_kernel: odd spatial kernel size for elevation, azimuth, and
            range mixing.
        kwargs: standard :class:`TransformerDecoder` arguments.  The output
            shape must be three-dimensional.
    """

    def __init__(
        self, refine_dim: int = 8, refine_kernel: int = 3, **kwargs
    ) -> None:
        shape = kwargs.get("shape", (1024, 256))
        if len(shape) != 3:
            raise ValueError(
                "ResidualRefinedTransformerDecoder requires a 3D output "
                f"shape, got {shape}.")
        if refine_dim < 1:
            raise ValueError("refine_dim must be positive.")
        if refine_kernel < 1 or refine_kernel % 2 == 0:
            raise ValueError("refine_kernel must be a positive odd integer.")

        super().__init__(**kwargs)
        channels = max(1, self.out_dim)
        padding = refine_kernel // 2
        self.refiner = nn.Sequential(
            nn.Conv3d(channels, refine_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(
                refine_dim,
                refine_dim,
                kernel_size=refine_kernel,
                padding=padding,
                groups=refine_dim,
            ),
            nn.GELU(),
            nn.Conv3d(refine_dim, channels, kernel_size=1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
    ) -> dict[str, Float[Tensor, "n h w d ..."]]:
        decoded = super().forward(encoded)
        value = decoded[self.key]
        channels_first = (
            value.unsqueeze(1) if self.out_dim == 0
            else value.movedim(-1, 1)
        )
        refined = channels_first + self.refiner(channels_first)
        decoded[self.key] = (
            refined[:, 0] if self.out_dim == 0
            else refined.movedim(1, -1)
        )
        return decoded


class VectorDecoder(L.LightningModule):
    """Generic MLP-based vector decoder without spatial dimensions.

    Always uses the first encoder output tensor, and supports the following
    reduction strategies for that tensor:

    - `last`: take the last spatial feature (nominally a readout token).
    - `max`, `avg`: max or average pooling over all spatial dimensions.

    Args:
        key: target key, e.g. `bev`, `depth`.
        layers: MLP architecture.
        dropout: dropout ratio during training.
        activation: activation function; specify as a name (i.e. corresponding
            to a class in `torch.nn`).
        dim: input features.
        out_dim: output features.
        reduce: reduction strategy.
        channels_first: whether input channels are in channels-spatial (`NCHW`)
            order instead of spatial-channels order (`NHWC`).
    """

    def __init__(
        self, key: str, layers: list[int] = [512, 512], dropout: float = 0.1,
        activation: str = 'GELU', dim: int = 768, out_dim: int = 3,
        strategy: Literal["last", "maxpool", "avgpool"] = "last",
        channels_first: bool = False
    ) -> None:
        super().__init__()

        self.key = key
        self.strategy = strategy
        self.channels_first = channels_first

        _layers = []
        for d1, d2 in zip(([dim] + layers)[:-1], layers):
            _layers += [
                nn.Linear(d1, d2, bias=True),
                getattr(nn, activation)(),
                nn.Dropout(dropout)]

        _layers.append(nn.Linear(([dim] + layers)[-1], out_dim))
        self.mlp = nn.Sequential(*_layers)

    def forward(
        self, encoded: Float[Tensor, "n s c"]
            | Sequence[Float[Tensor, "?n ?*s ?c"]]
    ) -> dict[str, Float[Tensor, "n f"]]:
        """Apply decoder.

        Args:
            encoded: encoded values. Only the last token (nominally the readout
                token) is used; if a list is passed, the last token of the last
                tensor is used.

        Returns:
            A tensor with the specified number of features.
        """
        if not isinstance(encoded, Tensor):
            encoded = encoded[0]

        if self.channels_first:
            n, c, *s = encoded.shape
            x = encoded.reshape(n, c, -1).permute(0, 2, 1)
        else:
            n, *s, c = encoded.shape
            x = encoded.reshape(n, -1, c)

        if self.strategy == "last":
            x = x[:, -1]
        elif self.strategy == "maxpool":
            x = torch.max(x, dim=1).values
        elif self.strategy == "avgpool":
            x = torch.mean(x, dim=1)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        return {self.key: self.mlp(x)}
