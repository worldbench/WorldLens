#!/usr/bin/env python3
"""
Render ORIGINAL per-frame RGB and Depth (no video decode) for a given clip and frame
from:
  - Training/original FRONT view
  - Novel views: front_center_interp, lateral_offset, lateral_offset_left

One output set per method (including GT) is saved as lossless PNG (+ raw depth .npy):
  data/render_frames/<clip>/<method>/
    train_rgb_fXXXX.png
    train_depth_fXXXX.png         (COLOR visualization, same as batch_eval)
    train_depth_fXXXX_raw.npy     (only if --with-raw-npy)
    train_depth_fXXXX_raw16cm.png (only if --save-raw-image; or *_raw16mm.png / _raw.exr)
    train_depth_fXXXX_vis16.png   (only if --save-raw-image and --with-depth-vis)
    front_center_interp_rgb_fXXXX.png
    front_center_interp_depth_fXXXX.png
    front_center_interp_depth_fXXXX_raw.npy      (only if --with-raw-npy and --with-front-center)
    front_center_interp_depth_fXXXX_raw16cm.png (only if --save-raw-image)
    front_center_interp_depth_fXXXX_vis16.png   (only if --save-raw-image and --with-depth-vis)
    lateral_offset_rgb_fXXXX.png
    lateral_offset_depth_fXXXX.png
    lateral_offset_depth_fXXXX_raw.npy           (only if --with-raw-npy)
    lateral_offset_depth_fXXXX_raw16cm.png (only if --save-raw-image)
    lateral_offset_depth_fXXXX_vis16.png   (only if --save-raw-image and --with-depth-vis)
    lateral_offset_left_rgb_fXXXX.png
    lateral_offset_left_depth_fXXXX.png
    lateral_offset_left_depth_fXXXX_raw.npy      (only if --with-raw-npy)
    lateral_offset_left_depth_fXXXX_raw16cm.png (only if --save-raw-image)
    lateral_offset_left_depth_fXXXX_vis16.png   (only if --save-raw-image and --with-depth-vis)

This script builds the dataset and trainer from each run's config.yaml under
work_dirs/<model>/<clip>/ and renders the requested single frame with the model,
without passing through any video encoding. It matches the evaluation utilities
in tools/batch_eval_nus.py and tools/eval.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from utils.visualization import depth_visualizer

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from utils.camera import lateral_offset_trajectory


DEFAULT_MODELS = [
    "drivestudio-nus-gt",
    "drivestudio-nus-dist4d",
    "drivestudio-nus-drivedreamer2",
    "drivestudio-nus-opendwm",
    "drivestudio-nus-dreamforge",
    "drivestudio-nus-xscene",
    "drivestudio-nus-magicdrive",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Render original per-frame RGB/Depth for FRONT+novel views")
    p.add_argument("clip_id", type=str, help="Sequence/clip id, e.g., 813")
    p.add_argument("frame", type=int, help="Zero-based frame index (0..num_timesteps-1)")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root for runs (default: work_dirs)")
    p.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS, help="Model directory names (default: all 6m+GT)")
    p.add_argument("--device", type=str, default=None, help="cuda device like 'cuda:0' or 'cpu'. Default: auto")
    p.add_argument("--output-dir", type=Path, default=Path("data/render_frames"), help="Output root directory")
    p.add_argument("--save-raw-image", action="store_true",
                   help="Also save raw depth as an image (disabled by default).")
    p.add_argument("--depth-format", type=str, choices=["png16cm", "png16mm", "exr"], default="png16cm",
                   help="Raw depth image format when --save-raw-image is set: 16-bit PNG in centimeters (png16cm) or millimeters (png16mm), or EXR float32 (exr). Default: png16cm.")
    p.add_argument("--with-depth-vis", action="store_true",
                   help="When --save-raw-image is set, also save a 16-bit grayscale visualization (_vis16.png). Disabled by default.")
    p.add_argument("--with-raw-npy", action="store_true", help="Also save raw depth as .npy (_raw.npy). Disabled by default.")
    p.add_argument("--with-front-center", action="store_true", help="Also render the 'front_center_interp' novel view. Disabled by default.")
    p.add_argument("--novel-cam-id", type=int, default=0, help="Camera id for novel-view rendering reference (front=0 for nuScenes)")
    p.add_argument("--cam-id", type=int, default=None, help="Alias for front camera id used for both train/original and novel views. If set, overrides --novel-cam-id.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--quiet", action="store_true", help="Fewer prints")
    return p.parse_args()


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    final = run_dir / "checkpoint_final.pth"
    if final.is_file():
        return final
    cands = sorted(run_dir.glob("checkpoint_*.pth"))
    return cands[-1] if cands else None


def load_cfg(run_dir: Path) -> OmegaConf:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"config.yaml not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def build_dataset_trainer(cfg: OmegaConf, device: torch.device):
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    return dataset, trainer


def save_png_uint8(rgb: np.ndarray, path: Path):
    # rgb in [0,1], HxWx3
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_depth_raw(depth_m: np.ndarray, path_stem: Path, fmt: str) -> Path:
    """Save raw depth as image.
    - png16cm: uint16 centimeters (depth[m]*100) clipped to [0,65535]
    - png16mm: uint16 millimeters (depth[m]*1000) clipped to [0,65535]
    - exr: float32 EXR (if supported by imageio)
    Returns the written file path.
    """
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    if fmt == "png16cm":
        arr = np.clip(d * 100.0, 0.0, 65535.0).astype(np.uint16)
        out = path_stem.with_suffix("")
        out = out.parent / (out.name + "_raw16cm.png")
        Image.fromarray(arr).save(out)
        return out
    if fmt == "png16mm":
        arr = np.clip(d * 1000.0, 0.0, 65535.0).astype(np.uint16)
        out = path_stem.with_suffix("")
        out = out.parent / (out.name + "_raw16mm.png")
        Image.fromarray(arr).save(out)
        return out
    if fmt == "exr":
        try:
            import imageio
            out = path_stem.with_suffix("")
            out = out.parent / (out.name + "_raw.exr")
            imageio.imwrite(out, d.astype(np.float32))
            return out
        except Exception:
            # Fallback to png16cm if EXR writer unavailable
            arr = np.clip(d * 100.0, 0.0, 65535.0).astype(np.uint16)
            out = path_stem.with_suffix("")
            out = out.parent / (out.name + "_raw16cm.png")
            Image.fromarray(arr).save(out)
            return out
    raise ValueError(f"Unknown depth format: {fmt}")


def save_depth_vis(depth_m: np.ndarray, path_stem: Path, clip_max: Optional[float] = None) -> Path:
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    if clip_max is None:
        mask = np.isfinite(d)
        clip_max = float(np.percentile(d[mask], 99.0)) if mask.any() else 50.0
    d = np.clip(d, 0.0, clip_max) / (clip_max + 1e-8)
    png16 = (d * 65535.0 + 0.5).astype(np.uint16)
    out = path_stem.with_suffix("")
    out = out.parent / (out.name + "_vis16.png")
    Image.fromarray(png16).save(out)
    return out


def save_depth_color(depth_m: np.ndarray, opacity: Optional[np.ndarray], path: Path) -> None:
    """Save colorized depth like tools/batch_eval_nus.py (turbo colormap with -log curve).
    Inputs are meters and an opacity mask in [0,1] (may be HxW or HxWx1)."""
    d = depth_m
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    w = opacity
    if w is not None and isinstance(w, np.ndarray) and w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    color = depth_visualizer(d, w)
    arr = (np.clip(color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_depth_npy(depth_m: np.ndarray, path_stem: Path) -> Path:
    out = path_stem.with_suffix("")
    out = out.parent / (out.name + "_raw.npy")
    d = depth_m.astype(np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    d[~np.isfinite(d)] = 0.0
    np.save(out, d)
    return out


@torch.no_grad()
def render_train_frame(dataset: DrivingDataset, trainer, frame_idx: int, cam_id: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    # Front camera assumed cam_id = 0 for nuScenes; image index = t * num_cams + cam_id
    num_cams = dataset.num_cams
    if not (0 <= cam_id < num_cams):
        raise SystemExit(f"Invalid cam_id {cam_id}; dataset has {num_cams} cameras")
    img_idx = int(frame_idx) * num_cams + int(cam_id)
    # render a single image via render_images with vis_indices
    from models.video_utils import render_images
    rr = render_images(trainer=trainer, dataset=dataset.full_image_set, compute_metrics=False, vis_indices=[img_idx])
    rgb = rr["rgbs"][0]
    depth = rr["depths"][0]
    opacity = rr.get("opacities", [None])[0]
    # rr stores numpy arrays already for depths/opacities; ensure numpy
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return rgb, depth, opacity


@torch.no_grad()
def render_novel_frame(dataset: DrivingDataset, trainer, traj_type: str, frame_idx: int, cam_id: int, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    # Prepare rendering data for the trajectory.
    if traj_type == "lateral_offset_left":
        # build left-shifted (-1 m) lateral trajectory explicitly
        per_cam_poses: Dict[int, torch.Tensor] = {
            c: dataset.pixel_source.camera_data[c].cam_to_worlds for c in dataset.pixel_source.camera_list
        }
        traj = lateral_offset_trajectory(
            dataset_type=dataset.type,
            per_cam_poses=per_cam_poses,
            original_frames=dataset.frame_num,
            target_frames=dataset.frame_num,
            offset_distance=-1.0,
        )
    else:
        trajs = dataset.get_novel_render_traj(traj_types=[traj_type], target_frames=dataset.frame_num)
        if traj_type not in trajs:
            raise SystemExit(f"Novel trajectory '{traj_type}' unavailable for dataset {dataset.type}")
        traj = trajs[traj_type]

    render_list = dataset.prepare_novel_view_render_data(traj, cam_id=cam_id)
    if not (0 <= frame_idx < len(render_list)):
        raise SystemExit(f"Frame {frame_idx} out of range for novel traj '{traj_type}' (0..{len(render_list)-1})")
    fd = render_list[frame_idx]
    # move to device
    for k, v in list(fd["cam_infos"].items()):
        fd["cam_infos"][k] = v.to(device, non_blocking=True)
    for k, v in list(fd["image_infos"].items()):
        fd["image_infos"][k] = v.to(device, non_blocking=True)
    # render
    out = trainer(image_infos=fd["image_infos"], camera_infos=fd["cam_infos"], novel_view=True)
    rgb = out["rgb"].detach().cpu().numpy()
    depth = out.get("depth", None)
    opacity = out.get("opacity", None)
    depth_np = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else None
    opacity_np = opacity.detach().cpu().numpy() if isinstance(opacity, torch.Tensor) else None
    return rgb, depth_np, opacity_np


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = args.output_dir / args.clip_id
    out_root.mkdir(parents=True, exist_ok=True)

    methods = list(args.models)
    if "drivestudio-nus-gt" not in methods:
        methods.insert(0, "drivestudio-nus-gt")

    # resolve camera id to use
    cam_id = args.cam_id if args.cam_id is not None else args.novel_cam_id
    # per method loop
    for m in methods:
        run_dir = args.root / m / args.clip_id
        if not run_dir.is_dir():
            print(f"[MISS] {run_dir} not found; skip {m}")
            continue
        cfg = load_cfg(run_dir)
        dataset, trainer = build_dataset_trainer(cfg, device)
        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            print(f"[MISS] checkpoint not found in {run_dir}; skip {m}")
            continue
        trainer.resume_from_checkpoint(str(ckpt), load_only_model=True)
        trainer.set_eval()

        # bounds check on frame
        if not (0 <= args.frame < int(dataset.num_img_timesteps)):
            print(f"[SKIP] {m}: frame {args.frame} out of [0,{dataset.num_img_timesteps-1}]")
            continue

        out_dir = out_root / m
        out_dir.mkdir(parents=True, exist_ok=True)
        ftag = f"f{args.frame:04d}"

        # 1) train/original FRONT
        try:
            rgb, depth, opac = render_train_frame(dataset, trainer, args.frame, cam_id, device)
            if not args.quiet:
                print(f"[OK] {m} train frame {args.frame}")
            save_png_uint8(rgb, out_dir / f"train_rgb_{ftag}.png")
            if depth is not None:
                # colorized depth (primary visualization)
                save_depth_color(depth, opac, out_dir / f"train_depth_{ftag}.png")
                # raw depth npy (default enabled)
                stem = out_dir / f"train_depth_{ftag}.png"
                if args.with_raw_npy:
                    save_depth_npy(depth, stem)
                # optional raw depth image and optional grayscale vis
                if args.save_raw_image:
                    save_depth_raw(depth.squeeze(-1), stem, args.depth_format)
                    if args.with_depth_vis:
                        save_depth_vis(depth.squeeze(-1), stem)
        except Exception as e:
            print(f"[WARN] {m} train render failed: {e}")

        # 2) novel views
        novel_views: List[str] = []
        if args.with_front_center:
            novel_views.append("front_center_interp")
        novel_views += ["lateral_offset", "lateral_offset_left"]
        for v in novel_views:
            try:
                rgb, depth, opac = render_novel_frame(dataset, trainer, v, args.frame, cam_id, device)
                save_png_uint8(rgb, out_dir / f"{v}_rgb_{ftag}.png")
                if depth is not None:
                    # colorized depth (primary visualization)
                    save_depth_color(depth, opac, out_dir / f"{v}_depth_{ftag}.png")
                    # raw npy (default)
                    stem = out_dir / f"{v}_depth_{ftag}.png"
                    if args.with_raw_npy:
                        save_depth_npy(depth, stem)
                    # optional raw image + grayscale vis
                    if args.save_raw_image:
                        save_depth_raw(depth.squeeze(-1), stem, args.depth_format)
                        if args.with_depth_vis:
                            save_depth_vis(depth.squeeze(-1), stem)
                if not args.quiet:
                    print(f"[OK] {m} {v} frame {args.frame}")
            except Exception as e:
                print(f"[WARN] {m} {v} render failed: {e}")

    print(f"[DONE] saved frames to {out_root}")


if __name__ == "__main__":
    main()
