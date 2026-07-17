#!/usr/bin/env python3
"""Render a high-resolution IQ1M hard/soft first-hit comparison panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_indices(cache_root: Path, trace: str, indices: list[int]) -> np.ndarray:
    wanted = set(indices)
    found: dict[int, np.ndarray] = {}
    position = 0
    for shard in sorted((cache_root / "traces" / trace).glob("radar_*.npy")):
        data = np.load(shard, mmap_mode="r")
        shard_stop = position + data.shape[0]
        for index in sorted(wanted):
            if position <= index < shard_stop:
                found[index] = np.asarray(data[index - position])
        position = shard_stop
        if len(found) == len(indices):
            break
    missing = [index for index in indices if index not in found]
    if missing:
        raise ValueError(f"Missing cache indices: {missing}")
    return np.stack([found[index] for index in indices])


def hard_first_hit(
    occupancy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = occupancy.any(dim=-1)
    depth = occupancy.to(torch.uint8).argmax(dim=-1).float() + 1.0
    return torch.where(valid, depth, 0.0), valid


def show_depth(
    axis: plt.Axes,
    depth: np.ndarray,
    invalid: np.ndarray | None,
    title: str,
) -> None:
    mask = (
        ~np.isfinite(depth)
        if invalid is None
        else invalid | ~np.isfinite(depth)
    )
    image = np.ma.array(depth, mask=mask)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    axis.imshow(
        image,
        cmap=cmap,
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=64.0,
        interpolation="nearest",
    )
    axis.set_facecolor("black")
    axis.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--trace", default="outdoor/forbes.east")
    parser.add_argument(
        "--indices", type=int, nargs="+", default=[0, 256, 1024, 2048]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from deepradar import DeepRadar  # noqa: PLC0415
    from deepradar.dataloader import RoverTrace  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRadar.load_from_checkpoint(
        str(args.checkpoint),
        hparams_file=str(args.hparams),
        map_location=device,
    )
    model.eval().to(device)
    objective = model.objectives[0]
    map_spec = model.dataset["channels"]["map"]
    dataset = RoverTrace(
        str(args.data_root / args.trace),
        channels={"map": map_spec},
        augmentations={},
        bounds=(0.0, 1.0),
    )
    target = np.stack([dataset[index]["map"] for index in args.indices])
    radar = load_indices(args.cache_root, args.trace, args.indices)

    with torch.inference_mode():
        batch = {
            "radar": torch.from_numpy(radar).to(
                device=device, dtype=torch.float32
            ),
            "map": torch.from_numpy(target).to(device=device),
        }
        logits = model(batch)["map"]
        depth_true, valid_true = hard_first_hit(batch["map"])
        depth_hard, valid_hard = hard_first_hit(logits > 0.0)
        depth_soft, hit_mass = objective._soft_first_hit(
            logits,
            threshold=objective.soft_depth_threshold,
            temperature=objective.soft_depth_temperature,
        )
        bev_true = batch["map"].any(dim=1).float()
        bev_probability = torch.sigmoid(logits).amax(dim=1)

    arrays = [
        value.detach().cpu().numpy()
        for value in (
            bev_true,
            bev_probability,
            depth_true,
            valid_true,
            depth_hard,
            valid_hard,
            depth_soft,
            hit_mass,
        )
    ]
    (
        bev_true_np,
        bev_probability_np,
        depth_true_np,
        valid_true_np,
        depth_hard_np,
        valid_hard_np,
        depth_soft_np,
        hit_mass_np,
    ) = arrays

    fig, axes = plt.subplots(
        len(args.indices),
        6,
        figsize=(24, 3.2 * len(args.indices)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, index in enumerate(args.indices):
        axes[row, 0].imshow(
            bev_true_np[row],
            cmap="gray",
            origin="upper",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, 0].set_title(f"BEV GT | frame {index}")
        axes[row, 1].imshow(
            bev_probability_np[row],
            cmap="inferno",
            origin="upper",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, 1].set_title("BEV predicted probability")
        show_depth(
            axes[row, 2], depth_true_np[row], ~valid_true_np[row], "Depth GT"
        )
        show_depth(
            axes[row, 3],
            depth_hard_np[row],
            ~valid_hard_np[row],
            f"Hard depth logit>0 | valid={valid_hard_np[row].mean():.3f}",
        )
        show_depth(
            axes[row, 4],
            depth_soft_np[row],
            None,
            (
                "Soft depth "
                f"t={objective.soft_depth_threshold:g} "
                f"temp={objective.soft_depth_temperature:g}"
            ),
        )
        axes[row, 5].imshow(
            hit_mass_np[row],
            cmap="gray",
            origin="upper",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[row, 5].set_title(
            f"Soft hit probability | mean={hit_mass_np[row].mean():.3f}"
        )
        for axis in axes[row]:
            axis.set_xlabel("range/azimuth output cell")
            axis.set_ylabel("azimuth/elevation output cell")

    fig.suptitle(
        f"GRT best checkpoint on RADs-like I/Q-1M | {args.trace}",
        fontsize=18,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"SAVED {args.out}")


if __name__ == "__main__":
    main()
