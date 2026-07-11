"""Radar transforms."""

import json
import os

import numpy as np
import torch
from beartype.typing import Any, Optional
from jaxtyping import Bool, Complex64, Float32
from torchvision import transforms

from deepradar._compat import roverd
from .base import Transform


class RadarResolution(Transform):
    """Get radar resolution metadata.

    Augmentations:

        - `range_scale`: apply scale multiplicatively to `range_resolution`.
        - `speed_scale`: apply multiplicatively to `doppler_resolution`.

    Args:
        path: path to dataset directory.
    """

    def __init__(self, path: str) -> None:

        with open(os.path.join(path, "radar", "radar.json")) as f:
            cfg = json.load(f)
        self.range_res = cfg["range_resolution"]
        self.doppler_res = cfg["doppler_resolution"]

    def __call__(
        self, data, aug: dict[str, Any] = {}, idx: int = 0
    ) -> Float32[np.ndarray, "d2"]:

        meta = np.array([
            self.range_res * aug.get("range_scale", 1.0),
            self.doppler_res * aug.get("speed_scale", 1.0)
        ], dtype=np.float32)
        return meta


class RADsLikeDoppler(Transform):
    """Project post-FFT I/Q-1M cubes onto a RADs-like Doppler/elevation grid.

    The input is expected after :class:`FFTArray`, in
    time-doppler-azimuth-elevation-range order. The selected elevation bin(s)
    are kept, and source Doppler bins are merged into a target physical
    velocity span.
    """

    def __init__(
        self, path: str, target_bins: int = 64,
        target_max_speed: float = 90.0, elevation_index: Optional[int] = None,
        elevation_indices: Optional[list[int]] = None, merge: str = "mean",
        smooth_sigma_bins: float = 0.0
    ) -> None:
        with open(os.path.join(path, "radar", "radar.json")) as f:
            cfg = json.load(f)

        if target_bins <= 0:
            raise ValueError("target_bins must be positive.")
        if target_max_speed <= 0:
            raise ValueError("target_max_speed must be positive.")
        if merge not in {"mean", "sum"}:
            raise ValueError("merge must be 'mean' or 'sum'.")
        if smooth_sigma_bins < 0:
            raise ValueError("smooth_sigma_bins must be non-negative.")

        self.source_doppler_res = float(cfg["doppler_resolution"])
        self.target_bins = int(target_bins)
        self.target_max_speed = float(target_max_speed)
        if elevation_indices is not None:
            if len(elevation_indices) == 0:
                raise ValueError("elevation_indices must not be empty.")
            self.elevation_indices = tuple(int(i) for i in elevation_indices)
        elif elevation_index is not None:
            self.elevation_indices = (int(elevation_index),)
        else:
            self.elevation_indices = None
        self.merge = merge
        self.smooth_sigma_bins = float(smooth_sigma_bins)

    def __call__(
        self, data: Complex64[np.ndarray, "T D A E R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Complex64[np.ndarray, "T D2 A E2 R"]:
        if data.ndim != 5:
            raise ValueError(
                "RADsLikeDoppler expects [time, doppler, azimuth, "
                "elevation, range] data.")

        t, source_bins, azimuth, elevation, ranges = data.shape
        if self.elevation_indices is None:
            elevation_indices = tuple(range(elevation))
        else:
            elevation_indices = self.elevation_indices
        for elevation_index in elevation_indices:
            if not 0 <= elevation_index < elevation:
                raise ValueError(
                    f"elevation_index={elevation_index} is out of bounds "
                    f"for elevation={elevation}.")

        target = np.zeros(
            (t, self.target_bins, azimuth, len(elevation_indices), ranges),
            dtype=data.dtype)
        counts = np.zeros(self.target_bins, dtype=np.int32)

        source_velocity = (
            np.arange(source_bins, dtype=np.float32) - source_bins // 2
        ) * self.source_doppler_res
        target_res = 2.0 * self.target_max_speed / self.target_bins
        target_index = np.rint(
            source_velocity / target_res + self.target_bins // 2
        ).astype(np.int32)

        kept = data[:, :, :, list(elevation_indices), :]
        for source_idx, target_idx in enumerate(target_index):
            if 0 <= target_idx < self.target_bins:
                target[:, target_idx] += kept[:, source_idx]
                counts[target_idx] += 1

        if self.merge == "mean":
            nonzero = counts > 0
            target[:, nonzero] /= counts[nonzero][None, :, None, None, None]

        if self.smooth_sigma_bins > 0:
            radius = max(1, int(np.ceil(3.0 * self.smooth_sigma_bins)))
            offsets = np.arange(-radius, radius + 1, dtype=np.float32)
            kernel = np.exp(
                -0.5 * np.square(offsets / self.smooth_sigma_bins))
            kernel = (kernel / kernel.sum()).astype(np.float32)
            padded = np.pad(
                target,
                ((0, 0), (radius, radius), (0, 0), (0, 0), (0, 0)),
                mode="constant")
            smoothed = np.zeros_like(target)
            for i, weight in enumerate(kernel):
                smoothed += weight * padded[:, i:i + self.target_bins]
            target = smoothed

        return target


class Representation(Transform):
    """Base class for radar representations."""

    @staticmethod
    def _wrap(
        x: Float32[np.ndarray, "T D A ... R"], width: int
    ) -> Float32[np.ndarray, "T D_crop A ... R"]:
        i_left = x.shape[1] // 2 - width // 2
        i_right = x.shape[1] // 2 + width // 2

        left = x[:, :i_left]
        center = x[:, i_left:i_right]
        right = x[:, i_right:]

        center[:, :right.shape[1]] += right
        center[:, -left.shape[1]:] += left

        return center

    @classmethod
    def _augment(
        cls, data: Float32[np.ndarray, "T D A ... R"],
        aug: dict[str, Any] = {}
    ) -> Float32[np.ndarray, "T D A ... R"]:
        """Apply radar representation-specific data augmentations.

        NOTE: we use `torchvision.transforms.Resize`, which requires a
        round-trip through a (cpu) `Tensor`. From some limited testing, this
        appears to be the most performant image resizing which supports
        antialiasing, with `skimage.transform.resize` being particularly slow.
        """
        T, Nd, *A, Nr = data.shape
        range_out_dim = int(aug.get("range_scale", 1.0) * Nr)
        speed_out_dim = 2 * (int(aug.get("speed_scale", 1.0) * Nd) // 2)

        if range_out_dim != Nr or speed_out_dim != Nd:
            # The leading T axis is transparently vectorized by Resize.
            # Note that we also have to do this reshape dance since Resize
            # only allows a maximum of 2 leading dimensions for some reason.
            as_tensor: Float32[torch.Tensor, "T ... R D"] = torch.Tensor(
                np.ascontiguousarray(np.moveaxis(data, 1, -1)))
            resized_t: Float32[torch.Tensor, "X R2 D2"] = transforms.Resize(
                (range_out_dim, speed_out_dim),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )(as_tensor.reshape(-1, Nr, Nd))
            resized = np.moveaxis(resized_t.reshape(
                T, *A, range_out_dim, speed_out_dim).numpy(), -1, 1)

            # Upsample -> crop
            if range_out_dim >= Nr:
                resized = resized[..., :Nr]
            # Downsample -> zero pad far ranges (high indices)
            else:
                pad = np.zeros(
                    (*data.shape[:-1], Nr - range_out_dim), dtype=np.float32)
                resized = np.concatenate([resized, pad], axis=-1)

            # Upsample -> wrap
            if speed_out_dim > data.shape[1]:
                resized = cls._wrap(resized, Nd)
            # Downsample -> zero pad high velocities (low and high indices)
            else:
                pad = np.zeros(
                    (T, (Nd - speed_out_dim) // 2, *data.shape[2:]),
                    dtype=np.float32)
                resized = np.concatenate((pad, resized, pad), axis=1)

            data = resized

        return data


class ComplexParts(Representation):
    """Convert complex numbers to (real, imag) along a new axis.

    To boost numerical stability, the parts are transformed by
    `sqrt(real(x)), sqrt(imag(x))` instead of being returned "raw".

    Augmentations:

        - `radar_scale`: radar magnitude scale factor.
        - `radar_phase`: radar phase shift.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.

    Args:
        do_sqrt: whether to do `sqrt()` transform on input magnitudes.
    """

    def __init__(self, path: str, do_sqrt: bool = True) -> None:
        self.do_sqrt = do_sqrt

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Float32[np.ndarray, "... 2"]:
        if aug.get("radar_phase"):
            data *= np.exp(-1j * aug["radar_phase"])

        if self.do_sqrt:
            real = np.sqrt(np.abs(np.real(data))) * np.sign(np.real(data))
            imag = np.sqrt(np.abs(np.imag(data))) * np.sign(np.imag(data))
        else:
            real = np.real(data)
            imag = np.imag(data)

        stretched = [
            self._augment(real, aug) * aug.get("radar_scale", 1.0),
            self._augment(imag, aug) * aug.get("radar_scale", 1.0)]
        return np.stack(stretched, axis=-1)


class ComplexAmplitude(Representation):
    """Convert complex numbers to amplitude-only.

    Augmentations:

        - `radar_scale`: radar magnitude scale factor.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.

    Args:
        cfar: if `True`, apply a pre-computed CFAR mask to the data.
        keepdim: keep the trailing axis created after converting
            `(real, complex) -> amplitude`.
    """

    def __init__(
        self, path: str, cfar: bool = False, keepdim: bool = True
    ) -> None:
        self.cfar: Optional[roverd.channels.Channel] = None
        self.keepdim = keepdim
        if cfar:
            _radar = roverd.sensors.RadarData(os.path.join(path, "_radar"))
            self.cfar = _radar["cfar"]

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Float32[np.ndarray, "... 1"] | Float32[np.ndarray, "..."]:
        data = np.sqrt(np.abs(data))

        if self.cfar is not None:
            mask: Bool[np.ndarray, "Nd Nr"] = self.cfar.read(idx, samples=1)[0]
            Nd, Nr = mask.shape
            reshape = [Nd] + [1] * (len(data.shape) - 2) + [Nr]
            data = data * mask.reshape(reshape)

        stretched = self._augment(data, aug) * aug.get("radar_scale", 1.0)
        return stretched[..., None] if self.keepdim else stretched


class ComplexPhase(Representation):
    """Convert complex numbers to (amplitude, phase) along a new axis.

    Augmentations:

        - `radar_scale`: radar magnitude scale factor.
        - `radar_phase`: radar phase shift.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.
    """

    def __call__(
        self, data: Complex64[np.ndarray, "..."], aug: dict[str, Any] = {},
        idx: int = 0
    ) -> Float32[np.ndarray, "... 2"]:
        def _normalize(x):
            return (x + np.pi) % (2 * np.pi) - np.pi

        stretched_magnitude = self._augment(np.sqrt(np.abs(data)), aug)
        stretched_phase = self._augment(np.angle(data), aug)

        return np.stack([
            stretched_magnitude * aug.get("radar_scale", 1.0),
            _normalize(stretched_phase + aug.get("radar_phase", 0.0))
        ], axis=-1)


class PrecomputedComplexPhaseAugment(Transform):
    """Apply official radar augmentations to cached ComplexPhase tensors.

    Precomputed RADs-like samples have shape ``[D, A, E, R, 2]`` with
    magnitude and phase in the final axis. The raw-IQ transforms cannot be
    replayed after caching, so this transform applies the equivalent flips,
    scale/phase shifts, and range/Doppler resizing directly to that
    representation. It must be paired with the same augmentation dictionary
    used by lidar/camera targets so spatial labels stay aligned.
    """

    def __init__(self, path: str) -> None:
        del path

    def __call__(
        self, data: Float32[np.ndarray, "D A E R C"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Float32[np.ndarray, "D A E R C"]:
        del idx
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 5 or data.shape[-1] != 2:
            raise ValueError(
                "PrecomputedComplexPhaseAugment expects [D,A,E,R,2], "
                f"got {tuple(data.shape)}.")
        if not aug:
            return data

        magnitude = data[..., 0]
        phase = data[..., 1]
        magnitude = Representation._augment(magnitude[None], aug=aug)[0]
        phase = Representation._augment(phase[None], aug=aug)[0]

        if aug.get("azimuth_flip"):
            magnitude = np.flip(magnitude, axis=1)
            phase = np.flip(phase, axis=1)
        if aug.get("doppler_flip"):
            magnitude = np.flip(magnitude, axis=0)
            phase = np.flip(phase, axis=0)

        magnitude = magnitude * aug.get("radar_scale", 1.0)
        phase = (
            phase + aug.get("radar_phase", 0.0) + np.pi
        ) % (2 * np.pi) - np.pi
        return np.ascontiguousarray(
            np.stack((magnitude, phase), axis=-1), dtype=np.float32)


class AmplitudeAOA(Representation):
    """Convert to amplitude + Azimuth AOA.

    Requires the input to be a 4D radar cube with elevation. A pre-computed
    AOA calculation is loaded, and concatenated to the norm of the amplitude
    across the azimuth axis. The AOA is scaled to [-1, 1].

    Augmentations:

        - `radar_scale`: radar magnitude scale factor.
        - `range_scale`: apply random range scale. Excess ranges are cropped;
          missing ranges are zero-filled.
        - `speed_scale`: apply random speed scale. Excess doppler bins are
          wrapped (causing ambiguous doppler velocities); missing doppler
          velocities are zero-filled.
        - `azimuth_flip`: apply azimuth flipping. :py:class:`.BaseFFT`
          typically handles this; however, since we integrate the azimuth
          axis out and use a pre-computed AOA, we must add it back in here.
    """

    def __init__(self, path: str) -> None:
        _radar = roverd.sensors.RadarData(os.path.join(path, "_radar"))
        self.aoa = _radar["aoa"]

    def __call__(
        self, data: Complex64[np.ndarray, "1 D A E R"],
        aug: dict[str, Any] = {}, idx: int = 0
    ) -> Float32[np.ndarray, "1 D 1 1 R 3"]:

        aoa_dr = self.aoa.read(idx, samples=1).astype(np.float32) / 128.0
        if aug.get("azimuth_flip"):
            aoa_dr = -aoa_dr

        # Need to divide by the azimuth axis length to maintain the same
        # magnitudes (for numerical stability)
        amplitude_der = np.sqrt(
            np.linalg.norm(np.abs(data) / data.shape[1], axis=1)
        ) * aug.get("radar_scale", 1.0)

        # Add singleton azimuth and elevation axes back.
        stretched = np.stack([
            self._augment(amplitude_der[:, :, 0, None, None, :], aug),
            self._augment(amplitude_der[:, :, 1, None, None, :], aug),
            self._augment(aoa_dr[:, :, None, None, :])
         ], axis=-1)
        return stretched
