"""Run official GRT checkpoints on range-cropped RADs samples.

The crop is applied to the raw [range, azimuth, Doppler] cube first, then all
RA/RD/AD views and model inputs are derived from that shifted cube.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RADAR_ROOT = REPO_ROOT.parents[1]

from deepradar.official_grt import load_official_grt, make_spectrum


TARGET_LOG_MEAN = -2.463407
TARGET_LOG_STD = 0.550160
EPS = 1e-12

CLASS_COLORS = np.array(
    [
        [66, 133, 244],
        [255, 109, 1],
        [52, 168, 83],
        [136, 14, 220],
        [80, 80, 80],
        [244, 180, 0],
        [219, 68, 55],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)
CLASS_NAMES = ["sky", "structure", "nature", "vehicle", "flat", "object", "person", "void"]


def to_dar_corrected(rad: np.ndarray) -> np.ndarray:
    """RADs [range, azimuth, Doppler] -> GRT-style [Doppler, azimuth, range]."""
    dar = np.moveaxis(rad, [2, 1, 0], [0, 1, 2]).astype(np.complex64, copy=False)
    return dar[:, ::-1, :]


def first_nonzero_range(cube: np.ndarray) -> int | None:
    mask = np.any(np.abs(cube) > 0, axis=(1, 2))
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def shift_range_cube(cube: np.ndarray, start: int) -> np.ndarray:
    """Move range bin `start` to zero and zero-pad the far-range tail.

    This keeps the official GRT input shape fixed. Depth/range predictions are
    therefore in cropped coordinates; add `start` bins to recover raw RADs range
    coordinates.
    """
    out = np.zeros_like(cube)
    if start <= 0:
        out[...] = cube
    elif start < cube.shape[0]:
        out[: cube.shape[0] - start] = cube[start:]
    return out


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def smooth_axis(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return arr
    moved = np.moveaxis(arr.astype(np.float32, copy=False), axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty_like(flat)
    pad = kernel.size // 2
    for i, row in enumerate(flat):
        out[i] = np.convolve(np.pad(row, (pad, pad), mode="edge"), kernel, mode="valid")
    return np.moveaxis(out.reshape(moved.shape), -1, axis)


def soft_beam_weights(n_src: int = 256, n_beams: int = 8, sigma: float = 18.0) -> np.ndarray:
    centers = np.linspace(0, n_src - 1, n_beams, dtype=np.float32)
    x = np.arange(n_src, dtype=np.float32)
    weights = np.exp(-0.5 * ((x[None, :] - centers[:, None]) / sigma) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    return weights.astype(np.float32)


def normalize_log_amp(amp: np.ndarray) -> np.ndarray:
    amp = np.asarray(amp, dtype=np.float32)
    log_amp = np.log(np.maximum(amp, EPS))
    std = max(float(log_amp.std()), EPS)
    matched = np.exp((log_amp - log_amp.mean()) / std * TARGET_LOG_STD + TARGET_LOG_MEAN)
    return matched.astype(np.float32)


def build_grt_input(rad_shifted: np.ndarray, sigma_az: float = 18.0) -> np.ndarray:
    dar = to_dar_corrected(rad_shifted)
    weights = soft_beam_weights(sigma=sigma_az)

    amp256 = np.sqrt(np.abs(dar) * 1e-6).astype(np.float32)
    amp8 = np.einsum("ba,dar->dbr", weights, amp256, optimize=True)
    amp8 = smooth_axis(amp8, 1.4, axis=2)
    amp8 = smooth_axis(amp8, 0.45, axis=1)
    amp8 = normalize_log_amp(amp8)

    complex8 = np.einsum("ba,dar->dbr", weights, dar, optimize=True).astype(np.complex64)
    phase8 = (np.angle(complex8) % (2 * np.pi)).astype(np.float32)
    real = np.stack([amp8, phase8], axis=-1)
    return np.repeat(real[:, None, :, :, :], 2, axis=1).astype(np.float32, copy=False)


def normalize_image(arr: np.ndarray, pct: tuple[float, float] = (2, 99)) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, pct)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def polar_to_cartesian(az_range: np.ndarray, fov_deg: float = 120.0, n: int = 420) -> np.ndarray:
    az_count, range_count = az_range.shape
    angles = np.deg2rad(np.linspace(-fov_deg / 2, fov_deg / 2, az_count))
    ranges = np.arange(range_count, dtype=np.float32) + 0.5
    ranges = ranges / range_count * 256.0
    rr, aa = np.meshgrid(ranges, angles, indexing="xy")
    x = rr * np.sin(aa)
    y = rr * np.cos(aa)
    xlim = (-230.0, 230.0)
    ylim = (0.0, 260.0)
    xi = np.floor((x - xlim[0]) / (xlim[1] - xlim[0]) * (n - 1)).astype(np.int32)
    yi = np.floor((y - ylim[0]) / (ylim[1] - ylim[0]) * (n - 1)).astype(np.int32)
    valid = (xi >= 0) & (xi < n) & (yi >= 0) & (yi < n)
    out = np.zeros((n, n), dtype=np.float32)
    np.maximum.at(out, (yi[valid], xi[valid]), az_range.astype(np.float32)[valid])
    return out


def first_hit_depth(occ_logits: torch.Tensor, threshold: float = 0.5, empty_value: float | None = None) -> np.ndarray:
    prob = torch.sigmoid(occ_logits)
    occ = prob > threshold
    has_hit = occ.any(dim=-1)
    depth = torch.argmax(occ.to(torch.uint8), dim=-1).to(torch.float32)
    if empty_value is not None:
        depth[~has_hit] = empty_value
    return depth.cpu().numpy()


def empty_depth_frac(occ_logits: torch.Tensor, threshold: float = 0.5) -> float:
    prob = torch.sigmoid(occ_logits)
    occ = prob > threshold
    return float((~occ.any(dim=-1)).float().mean().item())


def semseg_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return CLASS_COLORS[np.clip(labels, 0, len(CLASS_COLORS) - 1)]


def find_image(root: Path, seq: str, frame: str) -> Path | None:
    seq_dir = root / seq
    if not seq_dir.exists():
        return None
    candidates = [
        seq_dir / f"{frame}.png",
        seq_dir / f"{frame}.jpg",
        seq_dir / f"{int(frame):06d}.png",
        seq_dir / f"{int(frame):06d}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(seq_dir.glob(f"*{int(frame):06d}*"))
    return matches[0] if matches else None


def strip_to_occ(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise RuntimeError(f"unexpected occ tensor shape {tuple(tensor.shape)}")
    return tensor


def strip_to_semseg(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 4 and tensor.shape[-2] == 1:
        tensor = tensor[..., 0, :]
    if tensor.ndim != 3:
        raise RuntimeError(f"unexpected semseg tensor shape {tuple(tensor.shape)}")
    return tensor


def strip_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    while tensor.ndim > 2:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            raise RuntimeError(f"unexpected 2D tensor shape {tuple(tensor.shape)}")
    return tensor


def official_polar_to_bev(az_range: np.ndarray, size: int = 420) -> np.ndarray:
    try:
        from nrdk.vis.voxels import bev_from_polar2

        data = torch.from_numpy(np.asarray(az_range, dtype=np.float32)[None, :, :, None])
        bev = bev_from_polar2(data, size=size, theta_min=-np.pi / 2, theta_max=np.pi / 2)
        return np.flipud(bev[0, :, :, 0].detach().cpu().numpy())
    except Exception as exc:
        print(f"OFFICIAL_BEV_FALLBACK {type(exc).__name__}: {exc}", flush=True)
        return polar_to_cartesian(np.asarray(az_range, dtype=np.float32), n=size)


def crop_gt_imgs_bev(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(2 * w / 3)
    x1 = w
    y0 = int(0.04 * h)
    y1 = int(0.96 * h)
    return img[y0:y1, x0:x1]


def show_heat(
    ax,
    arr: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    origin: str = "lower",
) -> None:
    arr = np.ma.masked_invalid(np.asarray(arr))
    ax.imshow(arr, origin=origin, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def show_rgb(ax, img: np.ndarray, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def add_semseg_legend(ax) -> None:
    handles = [
        mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.0,
        columnspacing=0.8,
    )


def run_one(args, base_model, occ2d_model, semseg_model, seq: str, frame: str, device: torch.device) -> Path:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rad_path = data_root / "RADs" / seq / f"{frame}.npy"
    gt_path = data_root / "RADs_gt" / seq / f"{frame}.npy"
    if not rad_path.exists():
        raise FileNotFoundError(rad_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    rad = np.asarray(np.load(rad_path, mmap_mode="r"), dtype=np.complex64)
    gt = np.asarray(np.load(gt_path, mmap_mode="r"))
    first_gt = first_nonzero_range(gt)
    crop_start = max(0, (first_gt or 0) - args.margin)

    rad_shifted = shift_range_cube(rad, crop_start)
    gt_shifted = shift_range_cube(gt, crop_start)
    first_gt_after = first_nonzero_range(gt_shifted)

    real = build_grt_input(rad_shifted, sigma_az=args.sigma_az)
    spectrum = make_spectrum(real, device)

    with torch.no_grad():
        occ = strip_to_occ(base_model({"spectrum": spectrum})["occ3d"])
        occ2d_logits = strip_to_2d(occ2d_model({"spectrum": spectrum})["occ2d"])
        semseg_logits = strip_to_semseg(semseg_model({"spectrum": spectrum})["semseg"])

    occ_prob = torch.sigmoid(occ).numpy().astype(np.float32)
    occ_bev_ra = occ_prob.max(axis=0)
    occ_bev_cart = polar_to_cartesian(occ_bev_ra)
    occ2d_prob = torch.sigmoid(occ2d_logits).numpy().astype(np.float32)
    occ2d_bev_cart = official_polar_to_bev(occ2d_prob)
    depth = first_hit_depth(occ, threshold=args.occ_threshold, empty_value=0.0)
    depth_missing = first_hit_depth(occ, threshold=args.occ_threshold, empty_value=float("nan"))
    depth_empty_frac = empty_depth_frac(occ, threshold=args.occ_threshold)

    semseg_labels = torch.argmax(semseg_logits, dim=-1).numpy().astype(np.uint8)
    semseg_rgb = semseg_to_rgb(semseg_labels)

    dar = to_dar_corrected(rad_shifted)
    input_amp = np.log1p(np.sqrt(np.abs(dar) * 1e-6)).astype(np.float32)
    gt_dar = to_dar_corrected(gt_shifted.astype(np.complex64, copy=False))
    native_ra = normalize_image(np.log1p(np.sqrt(np.abs(dar).sum(axis=0) * 1e-6)))
    native_rd = normalize_image(np.log1p(np.sqrt(np.abs(dar).mean(axis=1) * 1e-6)))
    native_ad = normalize_image(np.log1p(np.sqrt(np.abs(dar).mean(axis=2) * 1e-6)))
    gt_ra = (np.abs(gt_dar).max(axis=0) > 0).astype(np.float32)
    gt_bev_cart = polar_to_cartesian(gt_ra)
    gt_img_path = find_image(data_root / "GT_Imgs", seq, frame)
    stereo_path = find_image(data_root / "stereo_image", seq, frame)

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.25, 1.08, 1.08, 1.0])

    top_grid = grid[0, :].subgridspec(2, 8, wspace=0.03, hspace=0.03)
    doppler_bins = np.linspace(0, input_amp.shape[0] - 1, 16, dtype=int)
    for i, d in enumerate(doppler_bins):
        ax = fig.add_subplot(top_grid[i // 8, i % 8])
        tile = normalize_image(input_amp[d], pct=(5, 99.8))
        show_heat(ax, tile, f"D{d:02d}", cmap="viridis", vmin=0, vmax=1)

    show_heat(fig.add_subplot(grid[1, 0]), occ2d_bev_cart, "Official GRT occ2d BEV head", cmap="inferno", vmin=0, vmax=1)
    show_heat(
        fig.add_subplot(grid[1, 1]),
        depth,
        f"Official-style first-hit depth p>{args.occ_threshold:g}",
        cmap="viridis",
        origin="upper",
    )
    semseg_ax = fig.add_subplot(grid[1, 2])
    show_rgb(semseg_ax, semseg_rgb, "Official GRT semantic segmentation")
    add_semseg_legend(semseg_ax)

    show_heat(fig.add_subplot(grid[2, 0]), occ_bev_cart, "Base occ3d max-projected BEV", cmap="inferno", vmin=0, vmax=1)
    show_heat(fig.add_subplot(grid[2, 1]), gt_bev_cart, "RADs_gt shifted BEV occupancy", cmap="inferno", vmin=0, vmax=1)
    ax_img = fig.add_subplot(grid[2, 2])
    if gt_img_path is not None:
        show_rgb(ax_img, crop_gt_imgs_bev(np.asarray(Image.open(gt_img_path).convert("RGB"))), "GT_Imgs right-panel BEV crop")
    elif stereo_path is not None:
        show_rgb(ax_img, np.asarray(Image.open(stereo_path).convert("RGB")), "stereo image")
    else:
        show_heat(ax_img, depth_missing, "Missing-mask depth view (white=no hit)", cmap="viridis", origin="upper")

    show_heat(fig.add_subplot(grid[3, 0]), native_rd, "cropped raw RD (x=range, y=doppler)", cmap="magma")
    show_heat(fig.add_subplot(grid[3, 1]), native_ra, "cropped raw RA (x=range, y=azimuth)", cmap="magma")
    show_heat(fig.add_subplot(grid[3, 2]), native_ad, "cropped raw AD (x=azimuth, y=doppler)", cmap="magma")

    fig.suptitle(
        f"RADs official GRT inference seq={seq} frame={frame} "
        f"crop_start={crop_start} first_gt={first_gt} first_gt_after={first_gt_after}",
        fontsize=13,
    )
    for label, ax_pos in [
        ("4D Radar Cube Input", (0.01, 0.91)),
        ("Model Output", (0.01, 0.54)),
        ("Ground Truth / Checks", (0.01, 0.25)),
    ]:
        fig.text(*ax_pos, label, rotation=90, va="center", ha="left", fontsize=12, weight="bold")

    out_path = out_dir / f"rads_official_crop_seq{seq}_frame{frame}_margin{args.margin}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    meta = {
        "seq": seq,
        "frame": frame,
        "rad_path": str(rad_path),
        "gt_path": str(gt_path),
        "crop_start": crop_start,
        "margin": args.margin,
        "first_gt_before": first_gt,
        "first_gt_after": first_gt_after,
        "range_crop_note": "raw range bins [0:crop_start] are removed; the far-range tail is zero-padded to preserve shape",
        "zero_padded_tail_range_bins": int(crop_start),
        "kept_raw_range_bins": int(max(0, rad.shape[0] - crop_start)),
        "depth_bin_offset_to_raw_range": int(crop_start),
        "input_shape": list(real.shape),
        "occ_shape": list(occ.shape),
        "occ2d_shape": list(occ2d_logits.shape),
        "semseg_shape": list(semseg_logits.shape),
        "occ_prob_min_mean_max": [
            float(occ_prob.min()),
            float(occ_prob.mean()),
            float(occ_prob.max()),
        ],
        "occ2d_prob_min_mean_max": [
            float(occ2d_prob.min()),
            float(occ2d_prob.mean()),
            float(occ2d_prob.max()),
        ],
        "depth_first_valid_frac": float(np.isfinite(depth_missing).mean()),
        "depth_empty_ray_frac": depth_empty_frac,
        "semseg_classes": sorted(int(x) for x in np.unique(semseg_labels)),
        "class_names": CLASS_NAMES,
        "gt_imgs_path": str(gt_img_path) if gt_img_path is not None else None,
        "stereo_path": str(stereo_path) if stereo_path is not None else None,
        "output": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("SAVED", out_path, flush=True)
    print("META", json.dumps(meta, ensure_ascii=True), flush=True)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(RADAR_ROOT / "data" / "hardware_config"))
    parser.add_argument("--checkpoint-root", default=str(RADAR_ROOT / "checkpoints" / "iq1m-checkpoints"))
    parser.add_argument("--out-dir", default=str(RADAR_ROOT / "outputs" / "rads_official_crop"))
    parser.add_argument("--frames", nargs="+", default=["100:000050"])
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--sigma-az", type=float, default=18.0)
    parser.add_argument("--occ-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path(args.checkpoint_root)
    print("LOAD_BASE", checkpoint_root / "base" / "small", flush=True)
    base_model = load_official_grt(str(checkpoint_root / "base" / "small"), device=device)
    print("LOAD_OCC2D", checkpoint_root / "occ2d" / "small", flush=True)
    occ2d_model = load_official_grt(str(checkpoint_root / "occ2d" / "small"), device=device)
    print("LOAD_SEMSEG", checkpoint_root / "semseg" / "small", flush=True)
    semseg_model = load_official_grt(str(checkpoint_root / "semseg" / "small"), device=device)

    for frame_spec in args.frames:
        seq, frame = frame_spec.split(":", 1)
        run_one(args, base_model, occ2d_model, semseg_model, seq, frame, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
