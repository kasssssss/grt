"""3D Occupancy classification objective."""

import numpy as np
import torch
from beartype.typing import Any, Sequence
from jaxtyping import Bool, Float, Shaped
from torch import Tensor
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    max_pool3d,
    smooth_l1_loss,
)

from deepradar.utils import comparison_grid, polar3_to_bev

from .base import (
    LPObjective,
    Metrics,
    Objective,
    PointCloudObjective,
    accuracy_metrics,
    focal_loss_with_logits,
)


class PolarChamfer(PointCloudObjective):
    """Chamfer metric for 3D polar depth maps.

    See :py:class:`.PointCloudObjective`.
    """

    def __init__(
        self, az_span: float = np.pi / 2, el_span: float = np.pi / 4,
        max_range: float = 64.0
    ) -> None:
        super().__init__(on_empty=max_range, mode="chamfer")
        self.az_span = az_span
        self.el_span = el_span

    def as_points(
        self, data: Float[Tensor, "el az"]
    ) -> Float[Tensor, "n 3"]:
        """Convert elevation-azimuth-range occupancy grid to xyz points."""
        ne, na = data.shape
        el_angles = torch.linspace(
            self.el_span, -self.el_span, ne, device=data.device)
        az_angles = torch.linspace(
            -self.az_span, self.az_span, na, device=data.device)

        el, az = torch.where(data)
        _el_cos = torch.cos(el_angles)[el]
        x = _el_cos * torch.cos(az_angles)[az] * data[el, az]
        y = _el_cos * torch.sin(az_angles)[az] * data[el, az]
        z = torch.sin(el_angles)[el] * data[el, az]

        return torch.stack([x, y, z], dim=-1)


class PolarOccupancy(Objective):
    """Radar -> lidar as 3D polar occupancy.

    Args:
        weight: total objective weight.
        range_weighted: Whether to apply equal-area "range weighting", where
            the relative weight of bins is adjusted on the range axis so that
            each bin is weighted according to the area which it represents
            (i.e. multiplying the weight by `range**2`).
        positive_weight: weight to give occupied samples; nominally the number
            of range bins.
        focal_loss: focusing parameter `gamma` for a focal loss [L1]_. If
            `focal_loss = 0` (default), BCE loss is used instead, and the
            focal loss is optimized out. Note that if a focal loss is used,
            `positive_weight` should be adjusted accordingly; [L1]_ recommends
            `positive_weight = 3.0` with `focal_loss = 2.0`.
        max_range: nominal maximum range, in bins.
        el_span, az_span: max elevation, azimuth angles, in radians.
        cmap_bev, cmap_depth: colors to use for visualization.

    Metrics:

    - `map_depth`: L1 depth loss, after projecting to a depth image.
    - `map_chamfer`: chamfer point cloud loss; occluded points are excluded.
    - `map_(tpr|fpr|tnr|fnr)`: true/false positive/negative rate.
    - `map_(precision|recall)`: precision and recall.
    - `map_acc`: accuracy (`tnr + tnr`).
    - `map_f1`: F1 score (`2 precision * recall / (precision + recall)`).
    """

    def __init__(
        self, weight: float = 1.0, range_weighted: bool = True,
        positive_weight: float = 64.0, focal_loss: float = 0.0,
        max_range: float = 64.0,
        el_span: float = np.pi / 4, az_span: float = np.pi / 2,
        cmap_bev: str = 'inferno', cmap_depth: str = 'viridis',
        seam_weight: float = 0.0,
        seam_patch: Sequence[int] = (8, 8, 8),
        seam_context_radius: int = 2,
    ) -> None:
        self.weight = weight
        self.range_weighted = range_weighted
        self.positive_weight = positive_weight
        self.cmap_bev = cmap_bev
        self.cmap_depth = cmap_depth
        self.focal_loss = focal_loss
        self.seam_weight = float(seam_weight)
        self.seam_patch = tuple(int(value) for value in seam_patch)
        self.seam_context_radius = int(seam_context_radius)
        if len(self.seam_patch) != 3 or any(value < 1 for value in self.seam_patch):
            raise ValueError("seam_patch must contain three positive values.")
        if self.seam_context_radius < 0:
            raise ValueError("seam_context_radius must be non-negative.")

        self.chamfer = PolarChamfer(
            az_span=az_span, el_span=el_span, max_range=max_range)

    @staticmethod
    def _first_hit(
        occupancy: Bool[Tensor, "batch el az rng"]
    ) -> tuple[Float[Tensor, "batch el az"], Bool[Tensor, "batch el az"]]:
        """Return one-based first-hit range and a separate validity mask."""
        valid = torch.any(occupancy, dim=-1)
        depth = torch.argmax(
            occupancy.to(torch.uint8), dim=-1).to(torch.float32) + 1.0
        return torch.where(valid, depth, 0.0), valid

    def bce_loss(
        self, y_true: Bool[Tensor, "batch el az rng"],
        y_hat: Float[Tensor, "batch el az rng"], reduce: bool = True
    ) -> Float[Tensor, "*#batch"]:
        """Primary BCE loss objective."""
        batch, ne, na, nr = y_true.shape
        if self.range_weighted:
            bins = torch.arange(nr, device=y_true.device)
            weight = ((bins + 1) / nr)[None, None, None, :] ** 2
        else:
            # Mypy doesn't seem to recognize the type overloading here.
            weight = 1.0  # type: ignore

        # Class weighting
        weight = torch.where(y_true, self.positive_weight, 1.0) * weight
        total_weight = torch.sum(weight, dim=(1, 2, 3))

        if self.focal_loss == 0.0:
            loss = binary_cross_entropy_with_logits(
                y_hat, y_true.to(y_hat.dtype), reduction='none')
        else:
            loss = focal_loss_with_logits(y_hat, y_true, gamma=self.focal_loss)

        loss = torch.sum(weight * loss, dim=(1, 2, 3)) / total_weight

        return torch.mean(loss) if reduce else loss

    def seam_loss(
        self, y_true: Bool[Tensor, "batch el az rng"],
        y_hat: Float[Tensor, "batch el az rng"], reduce: bool = True,
    ) -> Float[Tensor, "*#batch"]:
        """Penalize discontinuities across decoder patches near GT surfaces.

        Pairs with different labels are excluded so real occupancy boundaries
        remain sharp. Empty pairs are considered only when either side is near
        a positive target voxel, preventing the vast empty volume from
        dominating the regularizer.
        """
        if self.seam_weight == 0.0:
            zero = y_hat.new_zeros(y_hat.shape[0])
            return zero.mean() if reduce else zero

        radius = self.seam_context_radius
        if radius == 0:
            context = y_true
        else:
            kernel = 2 * radius + 1
            context = max_pool3d(
                y_true.to(y_hat.dtype).unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=radius,
            )[:, 0] > 0

        batch_losses = []
        for axis, patch in enumerate(self.seam_patch, start=1):
            boundary = torch.arange(
                patch, y_hat.shape[axis], patch, device=y_hat.device)
            if boundary.numel() == 0:
                continue
            left_index = boundary - 1
            left_hat = torch.index_select(y_hat, axis, left_index)
            right_hat = torch.index_select(y_hat, axis, boundary)
            left_true = torch.index_select(y_true, axis, left_index)
            right_true = torch.index_select(y_true, axis, boundary)
            left_context = torch.index_select(context, axis, left_index)
            right_context = torch.index_select(context, axis, boundary)
            mask = (left_true == right_true) & (left_context | right_context)
            pair_loss = smooth_l1_loss(
                left_hat, right_hat, reduction="none") * mask.to(y_hat.dtype)
            dims = tuple(range(1, pair_loss.ndim))
            batch_losses.append(
                pair_loss.sum(dim=dims)
                / mask.sum(dim=dims).clamp_min(1).to(y_hat.dtype))

        if not batch_losses:
            result = y_hat.new_zeros(y_hat.shape[0])
        else:
            result = torch.stack(batch_losses, dim=0).mean(dim=0)
        return result.mean() if reduce else result

    def metrics(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]],
        reduce: bool = True, train: bool = True
    ) -> Metrics:
        """Get training metrics."""
        bce = self.bce_loss(y_true['map'], y_hat['map'], reduce=reduce)
        seam = self.seam_loss(y_true['map'], y_hat['map'], reduce=reduce)
        loss = bce + self.seam_weight * seam
        if train:
            return Metrics(
                loss=self.weight * loss,
                metrics={"map_loss": bce, "map_seam": seam},
            )
        else:
            depth_true, _ = self._first_hit(y_true["map"])
            metrics = {
                "map_loss": bce,
                "map_seam": seam,
            }
            for label, threshold in (("m1", -1.0), ("0", 0.0), ("p1", 1.0)):
                y_hat_occ = y_hat["map"] > threshold
                depth_hat, valid_hat = self._first_hit(y_hat_occ)
                metrics[f"map_depth_{label}"] = LPObjective(
                    ord=1, mask=0)(depth_hat, depth_true, reduce=reduce)
                invalid = torch.mean(
                    (~valid_hat).to(torch.float32), dim=(1, 2))
                metrics[f"map_invalid_{label}"] = (
                    torch.mean(invalid) if reduce else invalid)
                metrics.update(accuracy_metrics(
                    y_hat_occ, y_true["map"],
                    prefix=f"map_{label}_", reduce=reduce))
                if threshold == 0.0:
                    metrics["map_depth"] = metrics[f"map_depth_{label}"]
                    metrics["map_chamfer"] = self.chamfer(
                        depth_true, depth_hat, reduce=reduce)
                    metrics.update(accuracy_metrics(
                        y_hat_occ, y_true["map"],
                        prefix="map_", reduce=reduce))
            return Metrics(loss=self.weight * loss, metrics=metrics)

    def visualizations(
        self, y_true: dict[str, Shaped[Tensor, "..."]],
        y_hat: dict[str, Shaped[Tensor, "..."]]
    ) -> dict[str, Shaped[np.ndarray, "H W 3"]]:
        """Generate visualizations."""
        depth, _ = self._first_hit(y_true['map'])
        bev = polar3_to_bev(y_true['map'], mode='highest')
        bev[bev == 0] = torch.min(bev) - 2

        map_pred = y_hat['map'] > 0
        depth_hat, _ = self._first_hit(map_pred)
        bev_hat = polar3_to_bev(map_pred, mode="highest")
        bev_hat[bev_hat == 0] = torch.min(bev_hat) - 2

        return {
            "bev": comparison_grid(
                bev, bev_hat, cmap=self.cmap_bev, cols=8, normalize=True),
            "depth": comparison_grid(
                depth / y_true['map'].shape[-1],
                depth_hat / y_true['map'].shape[-1],
                cmap=self.cmap_depth, cols=8, normalize=False,
                invalid_zero_black=True)}

    RENDER_CHANNELS: dict[str, dict[str, Any]] = {
        "bev": {
            "format": "lzma", "type": "u1", "shape": [64, 128],
            "desc": "range-azimuth BEV from 3D polar occupancy"},
        "depth": {
            "format": "lzma", "type": "u1", "shape": [64, 128],
            "desc": "elevation-azimuth depth image from 3D polar occupancy."},
        "bev_gt": {
            "format": "lzma", "type": "u1", "shape": [64, 128],
            "desc": "ground-truth BEV from lidar"},
        "depth_gt": {
            "format": "lzma", "type": "u1", "shape": [64, 128],
            "desc": "ground-truth depth from lidar (via low-res 3D occupancy)"}
    }
    """Channels output by :py:meth:`Objective.render`.

    Each key should correspond to a channel name, and each value should
    correspond to a configuration for :py:mod:`roverd.channels`.
    """

    def render(
        self, y_true: dict[str, Shaped[Tensor, "batch ..."]],
        y_hat: dict[str, Shaped[Tensor, "batch ..."]], gt: bool = True
    ) -> dict[str, Shaped[np.ndarray, "batch ..."]]:
        """Summarize predictions to visualize later.

        Args:
            y_true, y_hat: see :py:meth:`Objective.metrics`.
            gt: render ground truth.

        Returns:
            A dict, where each key is the name of a visualization or output
            data, and the value is a quantized or packed format if possible.
        """
        map_pred = y_hat['map'] > 0
        raw = {
            "bev": 255 - polar3_to_bev(map_pred, mode="highest"),
            "depth": self._first_hit(map_pred)[0]
        }

        if gt:
            raw['bev_gt'] = 255 - polar3_to_bev(y_true['map'], mode='highest')
            raw['depth_gt'] = self._first_hit(y_true['map'])[0]

        return {k: v.to(torch.uint8).cpu().numpy() for k, v in raw.items()}
