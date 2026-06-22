#!/usr/bin/env python3
"""Render and export train/depth/novel results for a given model across clips.

For each clip and task (train, depth, novel) this script:

- Re-renders images via DrivingDataset + trainer (no mp4 decode).
- Saves per-camera (or per-view) JPEG frames and a video under:

    data/render_all/{task_name}/{clip_id}/{model_name}/{camera_name}/
      images/             # all frames, jpg
      {task_name}.mp4     # video composed from images

  Metrics (per clip+model, shared across cameras):

    data/render_all/train/{clip_id}/{model_name}/train_metric.json
    data/render_all/depth/{clip_id}/{model_name}/depth_metric.json

Where:
  - task_name ∈ {train, depth, novel}
  - camera_name:
      * train/depth: real camera name (e.g., CAM_FRONT)
      * novel: one of the standard novel trajectories (treated as "views"):
          - s_curve
          - front_center_interp
          - lateral_offset
          - lateral_offset_left

Metrics:
  - train_metric.json:
      taken from work_dirs_analysis_metrics.json for this model+clip
      (same LPIPS/PSNR/SSIM entries as tools/metrics_analysis.py uses).
  - depth_metric.json:
      taken from depth_tables_3/per_seq.csv (fallback depth_tables_2/per_seq.csv),
      selecting Method/Seq='{method}/{clip_id}' where method is method_label(model).
      Contains AbsRel 等所有数值列.

Novel currently has no metrics and thus no *_metric.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from utils.camera import lateral_offset_trajectory
from utils.visualization import depth_visualizer


def load_cfg(clip_dir: Path) -> OmegaConf:
    cfg_path = clip_dir / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"config.yaml not found in {clip_dir}")
    return OmegaConf.load(cfg_path)


def find_checkpoint(clip_dir: Path) -> Optional[Path]:
    final = clip_dir / "checkpoint_final.pth"
    if final.is_file():
        return final
    cands = sorted(clip_dir.glob("checkpoint_*.pth"))
    return cands[-1] if cands else None


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


def method_label(name: str) -> str:
    return name.split("drivestudio-nus-", 1)[-1].replace("_", " ").replace("-", " ")


def save_jpg(rgb: np.ndarray, path: Path) -> None:
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


def save_depth_color(depth_m: np.ndarray, opacity: Optional[np.ndarray], path: Path) -> None:
    d = depth_m
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    w = opacity
    if w is not None and isinstance(w, np.ndarray) and w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    color = depth_visualizer(d, w)
    save_jpg(color, path)


def write_video_from_images(img_dir: Path, out_path: Path, fps: int) -> None:
    files = sorted(p for p in img_dir.glob("*.jpg") if p.is_file())
    if not files:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path.as_posix(), mode="I", fps=fps)
    try:
        for p in files:
            frame = imageio.imread(p.as_posix())
            writer.append_data(frame)
    finally:
        writer.close()


def load_lpips_json(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_absrel_tables(paths: List[Path], col: str = "AbsRel") -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load per-seq depth metrics indexed by (method, seq) -> row-dict."""
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if "Method/Seq" not in reader.fieldnames:
                    continue
                for row in reader:
                    tag = row.get("Method/Seq", "").strip()
                    if not tag or "/" not in tag:
                        continue
                    method, seq = tag.split("/", 1)
                    key = (method.strip(), seq.strip())
                    # store all numeric columns
                    vals: Dict[str, float] = {}
                    for k, v in row.items():
                        if k == "Method/Seq":
                            continue
                        try:
                            vals[k] = float(v)
                        except (TypeError, ValueError):
                            continue
                    lookup[key] = vals
        except Exception:
            continue
    return lookup


@torch.no_grad()
def render_train_depth_for_clip(
    dataset: DrivingDataset,
    trainer,
    clip_id: str,
    model_name: str,
    out_root: Path,
    tasks: List[str],
    device: torch.device,
) -> None:
    need_train = "train" in tasks
    need_depth = "depth" in tasks
    if not (need_train or need_depth):
        return

    ds = dataset.full_image_set
    camera_downscale = trainer._get_downscale_factor()
    for idx in range(len(ds)):
        image_infos, cam_infos = ds.get_image(idx, camera_downscale)
        for k, v in list(image_infos.items()):
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.to(device, non_blocking=True)
        for k, v in list(cam_infos.items()):
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.to(device, non_blocking=True)
        out = trainer(image_infos=image_infos, camera_infos=cam_infos)
        rgb = out["rgb"].clamp(0.0, 1.0).detach().cpu().numpy()
        depth = out.get("depth", None)
        opacity = out.get("opacity", None)
        depth_np = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else None
        opacity_np = opacity.detach().cpu().numpy() if isinstance(opacity, torch.Tensor) else None

        cam_name = cam_infos["cam_name"]
        frame_idx = int(image_infos["frame_idx"].flatten()[0].cpu().item())

        if need_train:
            img_dir = out_root / "train" / clip_id / model_name / cam_name / "images"
            out_path = img_dir / f"{frame_idx:04d}.jpg"
            save_jpg(rgb, out_path)
            gt_pixels = image_infos.get("pixels")
            if isinstance(gt_pixels, torch.Tensor):
                gt_dir = out_root / "train" / clip_id / model_name / cam_name / "gt_images"
                gt_path = gt_dir / f"{frame_idx:04d}.jpg"
                save_jpg(gt_pixels.detach().cpu().numpy(), gt_path)
        if need_depth and depth_np is not None:
            img_dir = out_root / "depth" / clip_id / model_name / cam_name / "images"
            out_path = img_dir / f"{frame_idx:04d}.jpg"
            save_depth_color(depth_np, opacity_np, out_path)


@torch.no_grad()
def render_novel_for_clip(
    dataset: DrivingDataset,
    trainer,
    clip_id: str,
    model_name: str,
    out_root: Path,
    device: torch.device,
    cam_id: int = 0,
) -> None:
    """Render multiple standard novel trajectories for the front camera.

    We treat each trajectory type as a "camera_name" under the 'novel' task.
    Trajectories: s_curve, front_center_interp, lateral_offset, lateral_offset_left.
    """
    ds = dataset
    traj_types = ["s_curve", "front_center_interp", "lateral_offset"]

    # First three trajectories from dataset helper
    trajs = ds.get_novel_render_traj(traj_types=traj_types, target_frames=ds.frame_num)

    # Lateral offset left (-1m) trajectory constructed explicitly, mirroring batch_eval_nus
    per_cam_poses: Dict[int, torch.Tensor] = {
        c: ds.pixel_source.camera_data[c].cam_to_worlds for c in ds.pixel_source.camera_list
    }
    traj_left = lateral_offset_trajectory(
        dataset_type=ds.type,
        per_cam_poses=per_cam_poses,
        original_frames=ds.frame_num,
        target_frames=ds.frame_num,
        offset_distance=-1.0,
    )
    trajs["lateral_offset_left"] = traj_left

    for traj_type, traj in trajs.items():
        base_dir = out_root / "novel" / clip_id / model_name / traj_type / "images"
        render_list = ds.prepare_novel_view_render_data(traj, cam_id=cam_id)
        for frame_idx, frame_data in enumerate(render_list):
            for k, v in list(frame_data["cam_infos"].items()):
                frame_data["cam_infos"][k] = v.to(device, non_blocking=True)
            for k, v in list(frame_data["image_infos"].items()):
                frame_data["image_infos"][k] = v.to(device, non_blocking=True)
            out = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
            )
            rgb = out["rgb"].detach().cpu().numpy()
            out_path = base_dir / f"{frame_idx:04d}.jpg"
            save_jpg(rgb, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Render all clips for a model into per-camera images/videos + metrics.")
    p.add_argument("model", type=str, help="Model directory name under work_dirs (e.g., drivestudio-nus-dist4d)")
    p.add_argument("clip_id", nargs="*", help="Optional list of clip IDs. If empty, auto-discover clips under the model.")
    p.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=["train", "depth", "novel"],
        choices=["train", "depth", "novel"],
        help="Tasks to render (default: train depth novel)",
    )
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root for runs (default: work_dirs)")
    p.add_argument("--output-root", type=Path, default=Path("data/render_all"), help="Output root directory")
    p.add_argument("--device", type=str, default=None, help="Device (cuda:0 or cpu; default: auto)")
    p.add_argument("--fps", type=int, default=10, help="FPS for output videos (default: 10)")
    p.add_argument("--metrics-json", type=Path, default=Path("work_dirs_analysis_metrics.json"), help="LPIPS metrics JSON")
    p.add_argument(
        "--depth-csv",
        type=Path,
        nargs="*",
        default=None,
        help="Depth per-seq CSVs (default: depth_tables_3/per_seq.csv, depth_tables_2/per_seq.csv)",
    )
    p.add_argument("--absrel-column", type=str, default="AbsRel", help="Column name for AbsRel in depth CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_root = args.root / args.model
    if not model_root.is_dir():
        raise SystemExit(f"Model directory not found: {model_root}")

    # Discover clips if not provided
    if args.clip_id:
        clips = list(args.clip_id)
    else:
        clips = sorted(d.name for d in model_root.iterdir() if d.is_dir() and (d / "config.yaml").is_file())
        if not clips:
            raise SystemExit(f"No clips with config.yaml found under {model_root}")

    # Load metrics sources once
    lpips_json = load_lpips_json(args.metrics_json)
    depth_csvs = list(args.depth_csv) if args.depth_csv else [
        Path("depth_tables_3/per_seq.csv"),
        Path("depth_tables_2/per_seq.csv"),
    ]
    absrel_table = load_absrel_tables(depth_csvs, args.absrel_column)

    out_root = args.output_root

    for clip in clips:
        clip_dir = model_root / clip
        if not clip_dir.is_dir():
            print(f"[SKIP] {clip_dir} is not a directory; skip")
            continue
        cfg = load_cfg(clip_dir)
        dataset, trainer = build_dataset_trainer(cfg, device)

        ckpt = find_checkpoint(clip_dir)
        if ckpt is None:
            print(f"[SKIP] clip {clip}: no checkpoint found in {clip_dir}")
            continue
        trainer.resume_from_checkpoint(str(ckpt), load_only_model=True)
        trainer.set_eval()

        # Render train/depth
        render_train_depth_for_clip(
            dataset=dataset,
            trainer=trainer,
            clip_id=clip,
            model_name=args.model,
            out_root=out_root,
            tasks=args.tasks,
            device=device,
        )

        # Render novel (front camera only)
        if "novel" in args.tasks:
            render_novel_for_clip(
                dataset=dataset,
                trainer=trainer,
                clip_id=clip,
                model_name=args.model,
                out_root=out_root,
                device=device,
                cam_id=0,
            )

        # Build videos per camera for each task
        for task in args.tasks:
            task_root = out_root / task / clip / args.model
            if not task_root.is_dir():
                continue
            for cam_dir in task_root.iterdir():
                if not cam_dir.is_dir():
                    continue
                img_dir = cam_dir / "images"
                if not img_dir.is_dir():
                    continue
                video_path = cam_dir / f"{task}.mp4"
                write_video_from_images(img_dir, video_path, fps=args.fps)

        # Metrics per task (one per clip+model, shared across cameras)
        # Train metrics from JSON cache
        if "train" in args.tasks:
            train_metrics = lpips_json.get(args.model, {}).get(clip, {})
            if isinstance(train_metrics, dict) and train_metrics:
                task_root = out_root / "train" / clip / args.model
                task_root.mkdir(parents=True, exist_ok=True)
                metric_path = task_root / "train_metric.json"
                metric_path.write_text(json.dumps(train_metrics, indent=2))

        # Depth metrics from per_seq tables
        if "depth" in args.tasks:
            meth = method_label(args.model).replace(" ", "")
            key1 = (meth, clip)
            key2 = (meth.lower(), clip)
            depth_metrics = absrel_table.get(key1) or absrel_table.get(key2)
            if isinstance(depth_metrics, dict) and depth_metrics:
                task_root = out_root / "depth" / clip / args.model
                task_root.mkdir(parents=True, exist_ok=True)
                metric_path = task_root / "depth_metric.json"
                metric_path.write_text(json.dumps(depth_metrics, indent=2))

        print(f"[DONE] clip {clip} for model {args.model} -> {out_root}")


if __name__ == "__main__":
    main()
