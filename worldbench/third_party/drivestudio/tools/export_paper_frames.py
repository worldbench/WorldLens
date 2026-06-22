#!/usr/bin/env python3
"""
Export per-camera frames and metrics for paper figures.

For a given clip + model + render types (train/depth/novel), this script:

- Renders original per-camera RGB/depth (no mp4 decode) using the same
  dataset/trainer pipeline as tools/eval.py and tools/batch_eval_nus.py.
- Saves images under:

  data/metric_compare/paper/{render_type}/{clip_id}/{model_name}/{camera_name}/{frame_id}.png

  where:
    - render_type ∈ {train, depth, novel}
    - camera_name:
        - train/depth: real camera name, e.g., CAM_FRONT, CAM_FRONT_LEFT, ...
        - novel: one of {front_center_interp, lateral_offset, lateral_offset_left}
    - frame_id: frame_idx (0-based) for train/depth; novel trajectory frame index for novel.

- Saves metrics under:

  data/metric_compare/paper/train/{clip_id}/{model_name}/train_metric.json
  data/metric_compare/paper/depth/{clip_id}/{model_name}/depth_metric.json

  train_metric.json:
    - latest metrics/*.json for this run and clip (same source as tools/metrics_analysis.py),
      e.g., image_metrics/full/lpips, psnr, ssim, etc.

  depth_metric.json:
    - row from depth_tables_3/per_seq.csv (fallback depth_tables_2/per_seq.csv)
      for Method/Seq='{method_label}/{clip_id}', including AbsRel, MAE, RMSE, etc.

Novel currently has no metrics and is skipped for metric export.
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from utils.visualization import depth_visualizer
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from utils.camera import lateral_offset_trajectory


def load_latest_metric_file(metrics_dir: Path) -> Optional[Path]:
    json_files = sorted(
        (p for p in metrics_dir.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return json_files[-1] if json_files else None


def load_cfg(run_dir: Path) -> OmegaConf:
    """Load config.yaml from a run directory (per-clip run dir)."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"config.yaml not found in {run_dir}")
    return OmegaConf.load(cfg_path)


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find checkpoint within a run directory (per-clip run dir)."""
    final = run_dir / "checkpoint_final.pth"
    if final.is_file():
        return final
    cands = sorted(run_dir.glob("checkpoint_*.pth"))
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


def save_png_uint8(rgb: np.ndarray, path: Path) -> None:
    """Save float RGB in [0,1] as 8-bit PNG."""
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_depth_color(depth_m: np.ndarray, opacity: Optional[np.ndarray], path: Path) -> None:
    """Colorize depth using depth_visualizer (same as eval videos)."""
    d = depth_m
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    w = opacity
    if w is not None and isinstance(w, np.ndarray) and w.ndim == 3 and w.shape[-1] == 1:
        w = w[..., 0]
    color = depth_visualizer(d, w)
    arr = (np.clip(color, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def load_train_metrics(run_dir: Path, clip_id: str) -> Dict[str, float]:
    """Load latest metrics JSON for a given model/clip (train metrics)."""
    metrics_dir = run_dir / clip_id / "metrics"
    latest = load_latest_metric_file(metrics_dir)
    if latest is None:
        return {}
    try:
        payload = json.loads(latest.read_text())
    except Exception:
        return {}
    return payload


def method_label(name: str) -> str:
    return name.split("drivestudio-nus-", 1)[-1].replace("_", " ").replace("-", " ")


def load_depth_metrics(
    depth_csv_paths: List[Path],
    model_name: str,
    clip_id: str,
    absrel_col: str = "AbsRel",
) -> Dict[str, float]:
    """Load depth metrics for a given model+clip from per_seq CSVs.

    Uses Method/Seq='{method}/{clip_id}', where method is method_label(model)
    with spaces removed (same logic as compose_depth_grid.py).
    """
    meth = method_label(model_name).replace(" ", "")
    key1 = (meth, clip_id)
    key2 = (meth.lower(), clip_id)
    result: Dict[str, float] = {}
    for p in depth_csv_paths:
        if not p.is_file():
            continue
        try:
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if "Method/Seq" not in reader.fieldnames:
                    continue
                for row in reader:
                    tag = row.get("Method/Seq", "").strip()
                    if "/" not in tag:
                        continue
                    method, seq = tag.split("/", 1)
                    key = (method.strip(), seq.strip())
                    if key not in (key1, key2):
                        continue
                    # dump all numeric columns for this entry
                    for col, val in row.items():
                        if col == "Method/Seq":
                            continue
                        try:
                            result[col] = float(val)
                        except (TypeError, ValueError):
                            continue
                    return result
        except Exception:
            continue
    return result


@torch.no_grad()
def render_train_and_depth_frames(
    dataset: DrivingDataset,
    trainer,
    clip_id: str,
    model_name: str,
    out_root: Path,
    types: List[str],
    device: torch.device,
) -> None:
    """Render all train/depth frames for all cameras.

    Saves:
      train/{cam_name}/{frame_idx}.png (RGB)
      depth/{cam_name}/{frame_idx}.png (colorized depth)
    """
    need_train = "train" in types
    need_depth = "depth" in types
    if not (need_train or need_depth):
        return

    ds = dataset.full_image_set
    camera_downscale = trainer._get_downscale_factor()
    for idx in range(len(ds)):
        image_infos, cam_infos = ds.get_image(idx, camera_downscale)
        # move tensors to device
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
            base_dir_train = out_root / "train" / clip_id / model_name
            out_path = base_dir_train / cam_name / f"{frame_idx:04d}.png"
            save_png_uint8(rgb, out_path)
        if need_depth and depth_np is not None:
            base_dir_depth = out_root / "depth" / clip_id / model_name
            out_path = base_dir_depth / cam_name / f"{frame_idx:04d}.png"
            save_depth_color(depth_np, opacity_np, out_path)


@torch.no_grad()
def render_novel_frames(
    dataset: DrivingDataset,
    trainer,
    clip_id: str,
    model_name: str,
    out_root: Path,
    device: torch.device,
    traj_types: List[str] = None,
    cam_id: int = 0,
) -> None:
    """Render novel views for standard trajectories.

    Saves:
      novel/{traj_type}/{frame_id}.png
    """
    if traj_types is None:
        traj_types = ["front_center_interp", "lateral_offset", "lateral_offset_left"]

    base_dir = out_root / "novel" / clip_id / model_name
    for traj_type in traj_types:
        # build trajectory
        if traj_type == "lateral_offset_left":
            per_cam_poses: Dict[int, torch.Tensor] = {
                c: dataset.pixel_source.camera_data[c].cam_to_worlds
                for c in dataset.pixel_source.camera_list
            }
            traj = lateral_offset_trajectory(
                dataset_type=dataset.type,
                per_cam_poses=per_cam_poses,
                original_frames=dataset.frame_num,
                target_frames=dataset.frame_num,
                offset_distance=-1.0,
            )
        else:
            trajs = dataset.get_novel_render_traj(
                traj_types=[traj_type],
                target_frames=dataset.frame_num,
            )
            if traj_type not in trajs:
                print(f"[WARN] clip {clip_id} traj '{traj_type}' unavailable; skip")
                continue
            traj = trajs[traj_type]

        render_list = dataset.prepare_novel_view_render_data(traj, cam_id=cam_id)
        for frame_idx, frame_data in enumerate(render_list):
            # move to device
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
            out_path = base_dir / traj_type / f"{frame_idx:04d}.png"
            save_png_uint8(rgb, out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export per-camera frames + metrics for paper demos")
    p.add_argument("clip_id", type=str, help="Clip ID (e.g., 813)")
    p.add_argument("model", type=str, help="Model directory name under work_dirs (e.g., drivestudio-nus-dist4d)")
    p.add_argument(
        "--types",
        type=str,
        nargs="*",
        default=["train", "depth", "novel"],
        choices=["train", "depth", "novel"],
        help="Render types to export (default: train depth novel)",
    )
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root for runs (default: work_dirs)")
    p.add_argument("--step", type=int, default=30000, help="Training step (for depth metrics naming; default: 30000)")
    p.add_argument("--output-root", type=Path, default=Path("data/metric_compare/paper"), help="Output root directory")
    p.add_argument("--device", type=str, default=None, help="Device to use (e.g., cuda:0 or cpu; default: auto)")
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

    run_root = args.root / args.model
    clip_dir = run_root / args.clip_id
    if not clip_dir.is_dir():
        raise SystemExit(f"Run directory for clip not found: {clip_dir}")

    cfg = load_cfg(clip_dir)
    dataset, trainer = build_dataset_trainer(cfg, device)

    ckpt = find_checkpoint(clip_dir)
    if ckpt is None:
        raise SystemExit(f"No checkpoint found in {clip_dir}")
    trainer.resume_from_checkpoint(str(ckpt), load_only_model=True)
    trainer.set_eval()

    out_root = args.output_root

    # --- render train/depth ---
    render_train_and_depth_frames(
        dataset=dataset,
        trainer=trainer,
        clip_id=args.clip_id,
        model_name=args.model,
        out_root=out_root,
        types=args.types,
        device=device,
    )

    # --- render novel ---
    if "novel" in args.types:
        render_novel_frames(
            dataset=dataset,
            trainer=trainer,
            clip_id=args.clip_id,
            model_name=args.model,
            out_root=out_root,
            device=device,
            traj_types=["front_center_interp", "lateral_offset", "lateral_offset_left"],
            cam_id=0,
        )

    # --- metrics ---
    # train metrics JSON
    train_metrics = load_train_metrics(run_root, args.clip_id) if "train" in args.types else {}
    if train_metrics:
        metric_path = out_root / "train" / args.clip_id / args.model / "train_metric.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(train_metrics, indent=2))

    # depth metrics JSON
    depth_csvs = list(args.depth_csv) if args.depth_csv else [
        Path("depth_tables_3/per_seq.csv"),
        Path("depth_tables_2/per_seq.csv"),
    ]
    depth_metrics = load_depth_metrics(depth_csvs, args.model, args.clip_id, args.absrel_column) if "depth" in args.types else {}
    if depth_metrics:
        metric_path = out_root / "depth" / args.clip_id / args.model / "depth_metric.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(depth_metrics, indent=2))

    print(f"[DONE] Exported frames and metrics under {out_root}")


if __name__ == "__main__":
    main()
