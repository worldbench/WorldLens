#!/usr/bin/env python3
"""Render nuScenes front-view depth videos for trained runs.

The script mirrors :mod:`tools.batch_eval_nus` but only exports a single
``full_set_<step>_depths_front.mp4`` per checkpoint by rendering CAM_FRONT
frames from ``full_image_set``.

Example:
    python tools/export_front_depth_nus.py \\
        --root work_dirs \\
        --pattern "drivestudio-nus-*" \\
        --fps 12 \\
        --skip-existing
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import torch
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images, save_videos
from utils.misc import import_str


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def select_cam_results(render_results: dict, cam_name: str) -> dict:
    """Filter render outputs so only frames from ``cam_name`` remain."""

    cam_names: List[str] = render_results.get("cam_names", [])
    if not cam_names:
        raise ValueError("render_results missing 'cam_names'")
    keep = [idx for idx, name in enumerate(cam_names) if name == cam_name]
    if not keep:
        raise ValueError(f"No frames found for camera {cam_name}")

    total = len(cam_names)
    filtered = {}
    for key, values in render_results.items():
        if isinstance(values, list) and len(values) == total:
            filtered[key] = [values[i] for i in keep]
        else:
            filtered[key] = values
    filtered["cam_names"] = [cam_names[i] for i in keep]
    return filtered


def export_front_depth_video(
    dataset: DrivingDataset,
    trainer,
    save_path: str,
    fps: int,
    cam_name: str,
    skip_existing: bool,
    verbose: bool,
):
    ensure_dir(os.path.dirname(save_path))
    if skip_existing and os.path.isfile(save_path):
        if verbose:
            print(f"[Skip] front depth video exists: {save_path}")
        return

    results = render_images(
        trainer=trainer,
        dataset=dataset.full_image_set,
        compute_metrics=False,
        compute_error_map=False,
        vis_indices=None,
        save_image_pairs=False,
    )
    front_results = select_cam_results(results, cam_name)
    num_frames = len(front_results["cam_names"])

    save_videos(
        front_results,
        save_path,
        layout=dataset.layout,
        num_timestamps=num_frames,
        keys=["depths"],
        num_cams=1,
        save_seperate_video=False,
        fps=fps,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser("Export nuScenes front depth videos")
    parser.add_argument("--root", type=str, default="work_dirs", help="Root of work dirs")
    parser.add_argument("--pattern", type=str, default="drivestudio-nus-*", help="Glob for method dirs")
    parser.add_argument("--fps", type=int, default=12, help="FPS for the depth video")
    parser.add_argument("--cam-name", type=str, default="CAM_FRONT", help="Camera name to render")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Log progress")
    args = parser.parse_args()

    method_dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not method_dirs:
        print(f"No method dirs matched under {args.root} with pattern {args.pattern}")
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mdir in method_dirs:
        for run in sorted(os.listdir(mdir)):
            run_dir = os.path.join(mdir, run)
            if not os.path.isdir(run_dir):
                continue

            ckpt = find_checkpoint(run_dir)
            if ckpt is None:
                continue
            log_dir = os.path.dirname(ckpt)

            step_hint = ckpt.split("_")[-1].split(".")[0]
            if step_hint == "final":
                step_hint = "30000"
            tentative_video = os.path.join(log_dir, "videos", f"full_set_{step_hint}_depths_front.mp4")
            if args.skip_existing and os.path.isfile(tentative_video):
                if args.verbose:
                    print(f"[Skip] {tentative_video} already exists (checkpoint hint)")
                continue

            if args.verbose:
                print(f"[Eval] {ckpt}")

            cfg = load_cfg(log_dir)
            dataset, trainer = build_dataset_trainer(cfg, device)
            trainer.resume_from_checkpoint(ckpt, load_only_model=True)
            step = trainer.step
            video_path = os.path.join(log_dir, "videos", f"full_set_{step}_depths_front.mp4")

            export_front_depth_video(
                dataset=dataset,
                trainer=trainer,
                save_path=video_path,
                fps=args.fps,
                cam_name=args.cam_name,
                skip_existing=args.skip_existing,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()
