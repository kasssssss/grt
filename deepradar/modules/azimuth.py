"""Complex-valued azimuth projection modules."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def physical_azimuth_projection(
    source_bins: int = 256,
    target_bins: int = 8,
    start: int = 4,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Return the spectrum-to-spectrum matrix for a circular aperture window."""
    if source_bins < 1 or target_bins < 1:
        raise ValueError("source_bins and target_bins must be positive.")
    if target_bins > source_bins:
        raise ValueError("target_bins must not exceed source_bins.")
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be torch.complex64 or torch.complex128.")

    basis = torch.eye(source_bins, dtype=dtype, device=device)
    aperture = torch.fft.ifft(
        torch.fft.ifftshift(basis, dim=-1), dim=-1)
    indices = (
        int(start) + torch.arange(target_bins, device=aperture.device)
    ) % source_bins
    selected = aperture.index_select(-1, indices)
    return torch.fft.fftshift(
        torch.fft.fft(selected, dim=-1), dim=-1)


class ComplexAzimuthProjection(nn.Module):
    """Learn a residual complex projection around a physical aperture window.

    The zero-initialized module is exactly the circular aperture projection.
    Training changes only a small complex matrix while the downstream radar
    network can remain frozen.
    """

    def __init__(
        self,
        source_bins: int = 256,
        target_bins: int = 8,
        start: int = 4,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        if rank is not None and not 1 <= rank <= min(source_bins, target_bins):
            raise ValueError(
                "rank must be in [1, min(source_bins, target_bins)].")
        initial = physical_azimuth_projection(
            source_bins=source_bins,
            target_bins=target_bins,
            start=start,
        )
        self.source_bins = int(source_bins)
        self.target_bins = int(target_bins)
        self.start = int(start)
        self.rank = rank
        self.register_buffer("initial", initial)
        self.scale = float(initial.abs().mean())
        if rank is None:
            self.delta_real = nn.Parameter(torch.zeros_like(initial.real))
            self.delta_imag = nn.Parameter(torch.zeros_like(initial.real))
        else:
            left = torch.zeros(source_bins, rank, dtype=initial.real.dtype)
            generator = torch.Generator().manual_seed(20260719)
            right = torch.randn(
                target_bins, rank, generator=generator, dtype=initial.real.dtype
            )
            right = torch.linalg.qr(right, mode="reduced").Q.T.contiguous()
            self.left_real = nn.Parameter(left.clone())
            self.left_imag = nn.Parameter(left.clone())
            self.right_real = nn.Parameter(right)
            self.right_imag = nn.Parameter(torch.zeros_like(right))

    def _delta(self) -> Tensor:
        if self.rank is None:
            return torch.complex(self.delta_real, self.delta_imag)
        left = torch.complex(self.left_real, self.left_imag)
        right = torch.complex(self.right_real, self.right_imag)
        return left @ right

    def matrix(self) -> Tensor:
        return self.initial + self.scale * self._delta()

    def forward(self, spectrum: Tensor, azimuth_dim: int = 1) -> Tensor:
        if not torch.is_complex(spectrum):
            raise ValueError("ComplexAzimuthProjection expects a complex tensor.")
        azimuth_dim %= spectrum.ndim
        if spectrum.shape[azimuth_dim] != self.source_bins:
            raise ValueError(
                f"Expected A={self.source_bins} at axis {azimuth_dim}, "
                f"got shape {tuple(spectrum.shape)}.")
        moved = spectrum.movedim(azimuth_dim, -1)
        projected = moved @ self.matrix().to(dtype=spectrum.dtype)
        return projected.movedim(-1, azimuth_dim)

    def initial_forward(self, spectrum: Tensor, azimuth_dim: int = 1) -> Tensor:
        """Apply the unchanged physical projection used at initialization."""
        if not torch.is_complex(spectrum):
            raise ValueError("ComplexAzimuthProjection expects a complex tensor.")
        azimuth_dim %= spectrum.ndim
        if spectrum.shape[azimuth_dim] != self.source_bins:
            raise ValueError(
                f"Expected A={self.source_bins} at axis {azimuth_dim}, "
                f"got shape {tuple(spectrum.shape)}.")
        moved = spectrum.movedim(azimuth_dim, -1)
        projected = moved @ self.initial.to(dtype=spectrum.dtype)
        return projected.movedim(-1, azimuth_dim)

    def regularization(self) -> tuple[Tensor, Tensor]:
        """Return residual magnitude and row-orthogonality penalties."""
        delta = self._delta().abs().square().mean()
        rows = self.matrix().T
        rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        gram = rows @ rows.conj().T
        identity = torch.eye(
            self.target_bins, dtype=gram.dtype, device=gram.device)
        orthogonality = (gram - identity).abs().square().mean()
        return delta, orthogonality


class RangeConditionedComplexAzimuthProjection(nn.Module):
    """Refine a frozen global projection with smooth range-local residuals.

    The base projection remains unchanged. Each input range cell blends two
    neighboring low-rank complex residuals through fixed triangular gates.
    Zero-initialized left factors make the module exactly equal to ``base``
    before training.
    """

    def __init__(
        self,
        base: ComplexAzimuthProjection,
        range_bins: int = 256,
        bands: int = 4,
        rank: int = 2,
    ) -> None:
        super().__init__()
        if range_bins < 1:
            raise ValueError("range_bins must be positive.")
        if bands < 2:
            raise ValueError("bands must be at least 2.")
        if not 1 <= rank <= base.target_bins:
            raise ValueError("rank must be in [1, target_bins].")

        self.base = base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.source_bins = int(base.source_bins)
        self.target_bins = int(base.target_bins)
        self.range_bins = int(range_bins)
        self.bands = int(bands)
        self.rank = int(rank)
        self.scale = float(base.scale)

        position = torch.linspace(0.0, 1.0, self.range_bins)
        centers = torch.linspace(0.0, 1.0, self.bands)
        width = 1.0 / (self.bands - 1)
        gates = (
            1.0 - (position[:, None] - centers[None]).abs() / width
        ).clamp_min(0.0)
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("range_gates", gates)

        left = torch.zeros(self.bands, self.source_bins, self.rank)
        generator = torch.Generator().manual_seed(20260719)
        right = torch.randn(
            self.bands,
            self.target_bins,
            self.rank,
            generator=generator,
        )
        right = torch.linalg.qr(
            right, mode="reduced").Q.transpose(-2, -1).contiguous()
        self.left_real = nn.Parameter(left.clone())
        self.left_imag = nn.Parameter(left.clone())
        self.right_real = nn.Parameter(right)
        self.right_imag = nn.Parameter(torch.zeros_like(right))

    def _delta(self) -> Tensor:
        left = torch.complex(self.left_real, self.left_imag)
        right = torch.complex(self.right_real, self.right_imag)
        return left @ right

    def initial_forward(self, spectrum: Tensor, azimuth_dim: int = 1) -> Tensor:
        """Apply the frozen global adapter without range-local refinement."""
        return self.base(spectrum, azimuth_dim=azimuth_dim)

    def forward(self, spectrum: Tensor, azimuth_dim: int = 1) -> Tensor:
        if not torch.is_complex(spectrum):
            raise ValueError(
                "RangeConditionedComplexAzimuthProjection expects a complex "
                "tensor."
            )
        azimuth_dim %= spectrum.ndim
        moved = spectrum.movedim(azimuth_dim, -1)
        if moved.shape[-1] != self.source_bins:
            raise ValueError(
                f"Expected A={self.source_bins} at axis {azimuth_dim}, "
                f"got shape {tuple(spectrum.shape)}."
            )
        if moved.ndim < 2 or moved.shape[-2] != self.range_bins:
            raise ValueError(
                f"Expected R={self.range_bins} immediately before the moved "
                f"azimuth axis, got shape {tuple(spectrum.shape)}."
            )

        delta_projection = torch.einsum(
            "...ra,kat->...rkt",
            moved,
            self._delta().to(dtype=spectrum.dtype),
        )
        gates = self.range_gates.to(dtype=delta_projection.dtype)
        residual = (delta_projection * gates[..., None]).sum(dim=-2)
        base = self.base(spectrum, azimuth_dim=azimuth_dim).movedim(
            azimuth_dim, -1)
        return (base + self.scale * residual).movedim(-1, azimuth_dim)

    def regularization(self) -> tuple[Tensor, Tensor]:
        """Return residual magnitude and adjacent-band smoothness penalties."""
        delta = self._delta()
        magnitude = delta.abs().square().mean()
        smoothness = (delta[1:] - delta[:-1]).abs().square().mean()
        return magnitude, smoothness
