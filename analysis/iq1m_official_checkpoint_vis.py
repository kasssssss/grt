"""Visualize official GRT checkpoints on native I/Q-1M samples.

This script intentionally uses the official base/occ2d/semseg checkpoint directories,
not a locally trained smoke checkpoint. It reads native I/Q-1M radar samples,
runs the AWR1843Boost RSP + PhaseAngle representation, then feeds the resulting
spectrum into the strict loader in deepradar.official_grt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RADAR_ROOT = REPO_ROOT.parents[1]

from deepradar.official_grt import load_official_grt


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
EPS = 1e-12


def phase_angle(cplx: np.ndarray) -> np.ndarray:
    amp = np.sqrt(np.abs(cplx) * 1e-6).astype(np.float32)
    phase = (np.angle(cplx) % (2 * np.pi)).astype(np.float32)
    return np.stack([amp, phase], axis=-1)


def make_spectrum(sample, rsp, device: torch.device) -> tuple[SimpleNamespace, np.ndarray]:
    flat_iq = sample.iq.reshape(-1, *sample.iq.shape[2:])
    cplx = rsp(flat_iq)
    real_flat = phase_angle(cplx)
    real = real_flat.reshape(*sample.iq.shape[:2], *real_flat.shape[1:])
    spectrum = SimpleNamespace(
        spectrum=torch.from_numpy(real).float().to(device),
        timestamps=torch.from_numpy(sample.timestamps).double().to(device),
        range_resolution=torch.from_numpy(sample.range_resolution).float().to(device),
        doppler_resolution=torch.from_numpy(sample.doppler_resolution).float().to(device),
    )
    return spectrum, real


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


def first_hit_depth(occ_logits: torch.Tensor, threshold: float, empty_value: float | None = None) -> np.ndarray:
    prob = torch.sigmoid(occ_logits)
    occ = prob > threshold
    has_hit = occ.any(dim=-1)
    depth = torch.argmax(occ.to(torch.uint8), dim=-1).to(torch.float32)
    if empty_value is not None:
        depth[~has_hit] = empty_value
    return depth.numpy()


def empty_depth_frac(occ_logits: torch.Tensor, threshold: float) -> float:
    prob = torch.sigmoid(occ_logits)
    occ = prob > threshold
    return float((~occ.any(dim=-1)).float().mean().item())


def argmax_depth(occ_logits: torch.Tensor) -> np.ndarray:
    prob = torch.sigmoid(occ_logits)
    return torch.argmax(prob, dim=-1).to(torch.float32).numpy()


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


def official_polar_to_bev(az_range: np.ndarray, size: int = 420) -> np.ndarray:
    try:
        from nrdk.vis.voxels import bev_from_polar2

        data = torch.from_numpy(np.asarray(az_range, dtype=np.float32)[None, :, :, None])
        bev = bev_from_polar2(data, size=size, theta_min=-np.pi / 2, theta_max=np.pi / 2)
        return np.flipud(bev[0, :, :, 0].detach().cpu().numpy())
    except Exception as exc:
        print(f"OFFICIAL_BEV_FALLBACK {type(exc).__name__}: {exc}", flush=True)
        return polar_to_cartesian(np.asarray(az_range, dtype=np.float32), n=size)


def normalize_image(arr: np.ndarray, pct: tuple[float, float] = (2, 99)) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, pct)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def semseg_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return CLASS_COLORS[np.clip(labels, 0, len(CLASS_COLORS) - 1)]


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
    ax.imshow(img, aspect="auto")
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


def try_lidar_occ3d_depth(trace_dir: Path, radar_ts: float, sample, occ_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    try:
        from roverd.sensors.lidar import OSLidarDepth
        from roverd.transforms.ouster import Destagger

        lidar = OSLidarDepth(str(trace_dir / "lidar"))
        lidar_ts = np.fromfile(trace_dir / "lidar" / "ts", dtype=np.float64)
        idx = int(np.abs(lidar_ts - radar_ts).argmin())
        lidar_sample = Destagger()(lidar[idx])
        rng = np.asarray(lidar_sample.rng, dtype=np.float32) * 1e-3
        batch, time, n_el_raw, n_az_raw = rng.shape
        d_el, d_az, d_rng = (2, 8, 4)
        crop_az = int(0.25 * n_az_raw)
        rng = rng[:, :, :, crop_az : n_az_raw - crop_az]

        n_el = rng.shape[2] // d_el
        n_az = rng.shape[3] // d_az
        n_rng = int(np.asarray(sample.iq).shape[-1] // 2)
        n_bins = n_rng // d_rng
        range_resolution = float(np.asarray(sample.range_resolution).reshape(-1)[0])

        bins = np.floor(rng / (range_resolution * d_rng)).astype(np.int32)
        valid = (rng > 0) & (bins > 0) & (bins < n_bins)
        bins[~valid] = 0

        bins = bins[:, :, : n_el * d_el, : n_az * d_az]
        valid = valid[:, :, : n_el * d_el, : n_az * d_az]
        bins = bins.reshape(batch, time, n_el, d_el, n_az, d_az)
        valid = valid.reshape(batch, time, n_el, d_el, n_az, d_az)
        bins = bins.transpose(0, 1, 2, 4, 3, 5).reshape(batch, time, n_el, n_az, d_el * d_az)
        valid = valid.transpose(0, 1, 2, 4, 3, 5).reshape(batch, time, n_el, n_az, d_el * d_az)

        occ = np.zeros((n_el, n_az, n_bins), dtype=bool)
        ee, aa, kk = np.nonzero(valid[0, 0])
        occ[ee, aa, bins[0, 0, ee, aa, kk]] = True
        occ[:, :, 0] = False
        depth = np.argmax(occ.astype(np.uint8), axis=-1).astype(np.float32)
        hit = occ.any(axis=-1)

        meta = {
            "lidar_idx": int(idx),
            "lidar_dt_sec": float(lidar_ts[idx] - radar_ts),
            "lidar_raw_shape": [int(x) for x in lidar_sample.rng.shape],
            "lidar_occ3d_shape": [int(x) for x in occ.shape],
            "lidar_occ3d_hit_frac": float(hit.mean()),
            "lidar_occ3d_decimate": [d_el, d_az, d_rng],
            "lidar_occ3d_range_resolution": range_resolution,
            "lidar_occ3d_shape_matches_pred": list(occ.shape) == list(occ_shape),
        }
        return depth, meta
    except Exception as exc:
        print(f"LIDAR_OCC3D_DEPTH_UNAVAILABLE {type(exc).__name__}: {exc}", flush=True)
        return None, {"lidar_occ3d_error": f"{type(exc).__name__}: {exc}"}


def try_camera_frame(trace_dir: Path, radar_ts: float) -> np.ndarray | None:
    try:
        import cv2

        camera_ts = np.fromfile(trace_dir / "_camera" / "ts", dtype=np.float64)
        idx = int(np.abs(camera_ts - radar_ts).argmin())
        cap = cv2.VideoCapture(str(trace_dir / "_camera" / "video.avi"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        print(f"CAMERA_UNAVAILABLE {type(exc).__name__}: {exc}", flush=True)
        return None


def run_one(args, radar, rsp, base_model, occ2d_model, semseg_model, frame: int, device: torch.device) -> Path:
    trace_dir = Path(args.data_root) / args.trace
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = radar[frame]
    spectrum, real = make_spectrum(sample, rsp, device)
    with torch.no_grad():
        occ = strip_to_occ(base_model({"spectrum": spectrum})["occ3d"])
        occ2d_logits = strip_to_2d(occ2d_model({"spectrum": spectrum})["occ2d"])
        semseg_logits = strip_to_semseg(semseg_model({"spectrum": spectrum})["semseg"])

    prob = torch.sigmoid(occ).numpy().astype(np.float32)
    occ_bev_ra = prob.max(axis=0)
    occ_bev_cart = polar_to_cartesian(occ_bev_ra)
    occ2d_prob = torch.sigmoid(occ2d_logits).numpy().astype(np.float32)
    occ2d_bev_cart = official_polar_to_bev(occ2d_prob)
    depth_first = first_hit_depth(occ, args.occ_threshold, empty_value=0.0)
    depth_first_nan = first_hit_depth(occ, args.occ_threshold, empty_value=float("nan"))
    depth_empty_frac = empty_depth_frac(occ, args.occ_threshold)
    depth_argmax = argmax_depth(occ)

    semseg_labels = torch.argmax(semseg_logits, dim=-1).numpy().astype(np.uint8)
    semseg_rgb = semseg_to_rgb(semseg_labels)

    radar_ts = float(sample.timestamps.reshape(-1)[0])
    lidar_depth, lidar_meta = try_lidar_occ3d_depth(trace_dir, radar_ts, sample, tuple(occ.shape))
    camera = try_camera_frame(trace_dir, radar_ts)

    # Native official RSP output after batch/time/channel strip: [D, E, A, R].
    amp = real[0, 0, ..., 0]
    input_dr = normalize_image(np.log1p(amp.mean(axis=(1, 2))))
    input_ar = normalize_image(np.log1p(amp.mean(axis=(0, 1))))

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[1.25, 1.08, 1.08, 1.0])

    top_grid = grid[0, :].subgridspec(2, 8, wspace=0.03, hspace=0.03)
    doppler_bins = np.linspace(0, amp.shape[0] - 1, 16, dtype=int)
    for i, d in enumerate(doppler_bins):
        ax = fig.add_subplot(top_grid[i // 8, i % 8])
        tile = normalize_image(np.log1p(amp[d].reshape(-1, amp.shape[-1])), pct=(2, 99.7))
        show_heat(ax, tile, f"D{d:02d}", cmap="viridis", vmin=0, vmax=1)

    show_heat(fig.add_subplot(grid[1, 0]), occ2d_bev_cart, "Official GRT occ2d BEV head", cmap="inferno", vmin=0, vmax=1)
    show_heat(
        fig.add_subplot(grid[1, 1]),
        depth_first,
        f"Official-style first-hit depth p>{args.occ_threshold:g}",
        cmap="viridis",
        origin="upper",
    )
    semseg_ax = fig.add_subplot(grid[1, 2])
    show_rgb(semseg_ax, semseg_rgb, "Official GRT semantic segmentation")
    add_semseg_legend(semseg_ax)

    show_heat(fig.add_subplot(grid[2, 0]), occ_bev_cart, "Base occ3d max-projected BEV", cmap="inferno", vmin=0, vmax=1)
    if lidar_depth is None:
        show_heat(
            fig.add_subplot(grid[2, 1]),
            depth_first_nan,
            "Missing-mask depth view (white=no hit)",
            cmap="viridis",
            origin="upper",
        )
    else:
        show_heat(
            fig.add_subplot(grid[2, 1]),
            lidar_depth,
            "Lidar occ3d-rendered depth (official GT style)",
            cmap="viridis",
            origin="upper",
        )
    if camera is None:
        show_heat(fig.add_subplot(grid[2, 2]), input_ar, "Radar input RA (x=range, y=azimuth)", cmap="magma", vmin=0, vmax=1)
    else:
        show_rgb(fig.add_subplot(grid[2, 2]), camera, "Camera")

    show_heat(fig.add_subplot(grid[3, 0]), input_dr, "Radar input RD (x=range, y=doppler)", cmap="magma", vmin=0, vmax=1)
    show_heat(fig.add_subplot(grid[3, 1]), input_ar, "Radar input RA (x=range, y=azimuth)", cmap="magma", vmin=0, vmax=1)
    show_heat(fig.add_subplot(grid[3, 2]), occ2d_prob, "Official occ2d polar logits sigmoid", cmap="inferno", vmin=0, vmax=1)

    logits_np = occ.numpy()
    meta = {
        "trace": args.trace,
        "frame": frame,
        "real_shape": list(real.shape),
        "occ_shape": list(occ.shape),
        "occ2d_shape": list(occ2d_logits.shape),
        "semseg_shape": list(semseg_logits.shape),
        "occ_logits_min_mean_max": [float(logits_np.min()), float(logits_np.mean()), float(logits_np.max())],
        "occ_prob_min_mean_max": [float(prob.min()), float(prob.mean()), float(prob.max())],
        "occ2d_prob_min_mean_max": [float(occ2d_prob.min()), float(occ2d_prob.mean()), float(occ2d_prob.max())],
        "depth_first_valid_frac": float(np.isfinite(depth_first_nan).mean()),
        "depth_empty_ray_frac": depth_empty_frac,
        **lidar_meta,
        "semseg_classes": sorted(int(x) for x in np.unique(semseg_labels)),
        "class_names": CLASS_NAMES,
    }
    fig.suptitle(
        f"Official GRT checkpoint on I/Q-1M {args.trace} frame={frame} | "
        f"prob min/mean/max={meta['occ_prob_min_mean_max']}",
        fontsize=12,
    )
    out_path = out_dir / f"iq1m_official_{args.trace.replace('/', '_')}_frame{frame:06d}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("SAVED", out_path, flush=True)
    print("META", json.dumps(meta, ensure_ascii=True), flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(RADAR_ROOT / "data" / "iq1m"))
    parser.add_argument("--checkpoint-root", default=str(RADAR_ROOT / "checkpoints" / "iq1m-checkpoints"))
    parser.add_argument("--out-dir", default=str(RADAR_ROOT / "outputs" / "iq1m_official_checkpoint_vis_20260609"))
    parser.add_argument("--trace", default="outdoor/forbes.east")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 24, 48])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--occ-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    from roverd.sensors.radar import XWRRadar
    from xwr.rsp.numpy import AWR1843Boost

    class AWR1843BoostNP(AWR1843Boost):
        """AWR1843Boost with NumPy FFTs instead of pyFFTW plans.

        The cluster pyFFTW build can fail during planner creation for this input
        shape. The official xwr processing path is otherwise preserved.
        """

        def fft(self, array, axes, size=None, shift=None):
            if size is not None:
                for axis, n in zip(axes, size):
                    array = self.pad(array, axis, n)
            out = np.fft.fftn(array, axes=axes).astype(np.complex64, copy=False)
            return np.fft.fftshift(out, axes=shift) if shift else out

    trace_dir = Path(args.data_root) / args.trace
    print("LOAD_RADAR", trace_dir / "radar", flush=True)
    radar = XWRRadar(str(trace_dir / "radar"))
    print("LOAD_RSP numpy-fft", flush=True)
    rsp = AWR1843BoostNP(window=False)

    ckpt_root = Path(args.checkpoint_root)
    print("LOAD_BASE", ckpt_root / "base" / "small", flush=True)
    base_model = load_official_grt(str(ckpt_root / "base" / "small"), device=device)
    print("LOAD_OCC2D", ckpt_root / "occ2d" / "small", flush=True)
    occ2d_model = load_official_grt(str(ckpt_root / "occ2d" / "small"), device=device)
    print("LOAD_SEMSEG", ckpt_root / "semseg" / "small", flush=True)
    semseg_model = load_official_grt(str(ckpt_root / "semseg" / "small"), device=device)

    for frame in args.frames:
        run_one(args, radar, rsp, base_model, occ2d_model, semseg_model, frame, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
