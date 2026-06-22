#!/usr/bin/env python3
"""
Batch evaluation for nuScenes runs under work_dirs/drivestudio-nus-*.

For each run (sequence) it:
  1) Renders 3 built-in novel trajectories (front camera) to videos/novel_<step>/ at 12 fps
  2) Renders an extra left-shifted (−1 m) lateral trajectory (front camera) to the same folder
  3) Exports original-view depth video aligned with full_set_<step>_rgbs.mp4
  4) Saves raw depth (.npy) for the original view

Requirements:
  - PYTHONPATH points to repo root when running this script.

Usage:
  python tools/batch_eval_nus.py \
      --root work_dirs \
      --pattern "drivestudio-nus-*" \
      --fps 12 \
      --raw-depth-dir raw_depth

The script overwrites existing files with the same names.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from utils.camera import lateral_offset_trajectory
from models.video_utils import render_novel_views, render_images, save_videos


def find_checkpoint(run_dir: str) -> Optional[str]:
    final = os.path.join(run_dir, "checkpoint_final.pth")
    if os.path.isfile(final):
        return final
    cands = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pth")))
    return cands[-1] if cands else None


def load_cfg(log_dir: str) -> OmegaConf:
    cfg_path = os.path.join(log_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {log_dir}")
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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def remove_if_exists(path: str):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def render_three_builtins_novel(dataset: DrivingDataset, trainer, fps: int, out_dir: str, skip_existing: bool):
    # three built-ins defined in eval: s_curve, front_center_interp, lateral_offset
    trajs = dataset.get_novel_render_traj(
        traj_types=["s_curve", "front_center_interp", "lateral_offset"],
        target_frames=dataset.frame_num,
    )
    ensure_dir(out_dir)
    for tname, traj in trajs.items():
        render_data = dataset.prepare_novel_view_render_data(traj, cam_id=0)
        save_path = os.path.join(out_dir, f"{tname}.mp4")
        if skip_existing and os.path.isfile(save_path):
            print(f"[Skip] novel {tname} exists: {save_path}")
            continue
        remove_if_exists(save_path)
        render_novel_views(trainer, render_data, save_path, fps=fps)


def render_leftshift_novel(dataset: DrivingDataset, trainer, fps: int, out_dir: str, skip_existing: bool):
    # Build a left-shifted (−1 m) lateral trajectory using front camera as reference
    per_cam_poses: Dict[int, torch.Tensor] = {
        cam_id: dataset.pixel_source.camera_data[cam_id].cam_to_worlds
        for cam_id in dataset.pixel_source.camera_list
    }
    traj_left = lateral_offset_trajectory(
        dataset_type=dataset.type,
        per_cam_poses=per_cam_poses,
        original_frames=dataset.frame_num,
        target_frames=dataset.frame_num,
        offset_distance=-1.0,
    )
    render_data = dataset.prepare_novel_view_render_data(traj_left, cam_id=0)
    ensure_dir(out_dir)
    save_path = os.path.join(out_dir, "lateral_offset_left.mp4")
    if skip_existing and os.path.isfile(save_path):
        print(f"[Skip] novel lateral_offset_left exists: {save_path}")
        return
    remove_if_exists(save_path)
    render_novel_views(trainer, render_data, save_path, fps=fps)


def export_full_depth(cfg: OmegaConf, dataset: DrivingDataset, trainer, log_dir: str, step: int, raw_depth_dir: str, skip_existing: bool):
    # Depth video aligned with full_set_<step>_rgbs.mp4 naming
    videos_dir = os.path.join(log_dir, "videos")
    ensure_dir(videos_dir)
    base_mp4 = os.path.join(videos_dir, f"full_set_{step}.mp4")
    depth_mp4 = base_mp4.replace(".mp4", "_depths.mp4")
    out_base = os.path.join(log_dir, raw_depth_dir)
    raw_dir = os.path.join(out_base, f"full_set_{step}")
    # We export raw depth only for front camera frames in this script.
    expected_n = dataset.num_img_timesteps  # CAM_FRONT only
    have_depth_mp4 = os.path.isfile(depth_mp4)
    have_raw = os.path.isdir(raw_dir) and len(glob.glob(os.path.join(raw_dir, "*.npy"))) >= expected_n
    if skip_existing and have_depth_mp4 and have_raw:
        print(f"[Skip] full-set depth video and raw already exist for step {step}")
        return
    # Render if needed
    results = render_images(
        trainer=trainer,
        dataset=dataset.full_image_set,
        compute_metrics=False,
        compute_error_map=False,
        vis_indices=None,
        save_image_pairs=False,
    )
    # Save depth video unless skipping due to existing
    if not (skip_existing and have_depth_mp4):
        save_videos(
            results,
            base_mp4,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=["depths"],
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=True,
            fps=(cfg.render.fps if hasattr(cfg, "render") and "fps" in cfg.render else 10),
            verbose=True,
        )
    # Save raw depth for front camera unless skipping
    if not (skip_existing and have_raw):
        ensure_dir(raw_dir)
        if "depths" in results:
            for i, depth in enumerate(results["depths"]):
                cam_name = results["cam_names"][i]
                if cam_name == 'CAM_FRONT':
                    np.save(os.path.join(raw_dir, f"{i:05d}_{cam_name}.npy"), depth)


def main():
    parser = argparse.ArgumentParser("Batch eval for nuScenes 3DGS")
    parser.add_argument("--root", type=str, default="work_dirs", help="Root directory of work dirs")
    parser.add_argument(
        "--pattern", type=str, default="drivestudio-nus-*", help="Glob pattern to select method dirs"
    )
    parser.add_argument("--fps", type=int, default=12, help="FPS for novel view videos")
    parser.add_argument(
        "--raw-depth-dir",
        type=str,
        default="raw_depth",
        help="Directory under each run's log_dir to save raw depth npy",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip rendering if target outputs already exist")
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    args = parser.parse_args()

    root = args.root
    method_dirs = sorted(glob.glob(os.path.join(root, args.pattern)))
    if not method_dirs:
        print(f"No method dirs matched under {root} with pattern {args.pattern}")
        sys.exit(0)

    for mdir in method_dirs:
        for run in sorted(os.listdir(mdir)):
            run_dir = os.path.join(mdir, run)
            if not os.path.isdir(run_dir):
                continue

            ckpt = find_checkpoint(run_dir)
            if ckpt is None:
                continue
            log_dir = os.path.dirname(ckpt)

            step_str = ckpt.split("_")[-1].split(".")[0]
            if step_str == 'final':
                step_str = "30000"
            novel_dir = os.path.join(log_dir, "videos", f"novel_{step_str}_benchmark")
            depth_dir = os.path.join(log_dir, "videos", f"full_set_{step_str}_depths.mp4")
            raw_depth_dir = os.path.join(log_dir, args.raw_depth_dir, f"full_set_{step_str}")

            if args.skip_existing:
                novel_views = [
                    "s_curve.mp4",
                    "front_center_interp.mp4",
                    "lateral_offset.mp4",
                    "lateral_offset_left.mp4",
                ]
                novel_done = all(os.path.isfile(os.path.join(novel_dir, view)) for view in novel_views)
                depth_done = os.path.isfile(depth_dir)
                raw_done = (not args.raw_depth_dir) or (
                    os.path.isdir(raw_depth_dir) and any(fname.endswith(".npy") for fname in os.listdir(raw_depth_dir))
                )
                if novel_done and depth_done and raw_done:
                    if args.verbose:
                        print(f"[Skip] {log_dir} already processed.")
                    continue

            if args.verbose:
                print(f"[Eval] {ckpt}")

            # Load cfg and build dataset/trainer
            cfg = load_cfg(log_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dataset, trainer = build_dataset_trainer(cfg, device)
            trainer.resume_from_checkpoint(ckpt, load_only_model=True)
            step = trainer.step

            # Novel views: three built-ins + left shift (front camera)
            novel_dir = os.path.join(log_dir, "videos", f"novel_{step}_benchmark")
            render_three_builtins_novel(dataset, trainer, args.fps, novel_dir, args.skip_existing)
            render_leftshift_novel(dataset, trainer, args.fps, novel_dir, args.skip_existing)

            # Full-view depth video + raw depth
            export_full_depth(cfg, dataset, trainer, log_dir, step, args.raw_depth_dir, args.skip_existing)


if __name__ == "__main__":
    main()
