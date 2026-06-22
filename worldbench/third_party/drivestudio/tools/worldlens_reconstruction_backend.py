#!/usr/bin/env python3
"""DriveStudio backend entrypoint for WorldLens Reconstruction.

This script is intentionally small: WorldLens calls it as an external process
when Reconstruction uses ``backend=drivestudio``. Heavy DriveStudio imports and
CUDA dependencies stay in the DriveStudio environment, while WorldLens reads the
JSON contract written here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

NOVEL_VIEWS = [
    "front_center_interp",
    "s_curve",
    "lateral_offset",
    "lateral_offset_left",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DriveStudio backend for WorldLens Reconstruction")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs"))
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes_trainval/raw"))
    parser.add_argument("--target-dir", type=Path, default=Path("data/nuscenes_trainval/processed"))
    parser.add_argument("--config-file", default="configs/omnire.yaml")
    parser.add_argument("--step", type=int, default=30000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gt-method", default="gt")
    parser.add_argument("--cameras", default="CAM_FRONT")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-depth-metrics", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def method_root(reconstruction_root: Path, method_name: str) -> Path:
    return reconstruction_root / method_name


def write_manifest(args: argparse.Namespace) -> None:
    write_json(
        method_root(args.reconstruction_root, args.method_name) / "manifest.json",
        {
            "schema_version": 1,
            "method_name": args.method_name,
            "backend": "drivestudio",
            "mode": "dry-run" if args.dry_run else "reconstruct",
            "contract": "assets",
            "clips": [str(clip) for clip in args.clips],
            "train_step": args.step,
            "config_file": args.config_file,
            "geometric_mask_source": args.gt_method,
            "novel_views": NOVEL_VIEWS,
        },
    )


def write_dry_run_outputs(reconstruction_root: Path, method_name: str, clips: Iterable[str]) -> None:
    root = method_root(reconstruction_root, method_name)
    for clip in clips:
        clip_id = str(clip)
        for relative_dir in (
            f"ckpt/{clip_id}",
            f"train/{clip_id}/pred",
            f"train/{clip_id}/gt",
            f"depth/{clip_id}/pred",
            f"depth/{clip_id}/gt",
            f"depth/{clip_id}/masks",
            f"novel/{clip_id}",
        ):
            (root / relative_dir).mkdir(parents=True, exist_ok=True)


def run(command: List[str]) -> None:
    subprocess.run(command, check=True)


def run_preprocess(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            "datasets/preprocess.py",
            "--data_root",
            str(args.data_root),
            "--target_dir",
            str(args.target_dir),
            "--dataset",
            "nuscenes_gen",
            "--split",
            "advanced_12Hz_trainval",
            "--process_keys",
            "images",
            "lidar",
            "calib",
            "dynamic_masks",
            "objects",
            "--data_source",
            args.method_name,
            "--scene_ids",
            *[str(int(clip)) for clip in args.clips],
        ]
    )


def run_train(args: argparse.Namespace) -> None:
    # This mirrors train_nus_gen.sh while staying shell-independent.
    run(
        [
            sys.executable,
            "tools/train.py",
            "--config_file",
            args.config_file,
            "--output_root",
            str(args.work_root),
            "--project",
            f"drivestudio-nus-{args.method_name}",
            "--run_name",
            "-1",
            "--start_idx",
            str(min(int(clip) for clip in args.clips)),
            "--end_idx",
            str(max(int(clip) for clip in args.clips)),
            "dataset=nuscenes_gen/6cams",
            "data.scene_idx=-1",
            "data.start_timestep=0",
            "data.end_timestep=-1",
            f"data.data_root={processed_data_root(args)}",
        ]
    )


def processed_data_root_for_method(args: argparse.Namespace, method_name: str) -> Path:
    processed_root = str(args.target_dir).replace("processed", f"processed_{method_name}", 1)
    return Path(processed_root) / "advanced_12Hz_trainval"


def processed_data_root(args: argparse.Namespace) -> Path:
    return processed_data_root_for_method(args, args.method_name)


def gt_processed_data_root(args: argparse.Namespace) -> Path:
    return processed_data_root_for_method(args, args.gt_method)


def validate_training_inputs(data_root: Path, clips: Iterable[str]) -> None:
    missing = []
    for clip in clips:
        scene_dir = data_root / f"{int(clip):03d}"
        images_dir = scene_dir / "images"
        fine_masks_dir = scene_dir / "fine_dynamic_masks"
        humanpose_path = scene_dir / "humanpose" / "smpl.pkl"
        if not images_dir.is_dir():
            missing.append(f"{scene_dir}: images")
            continue
        image_count = len(list(images_dir.iterdir()))
        for mask_name in ("all", "human", "vehicle"):
            mask_dir = fine_masks_dir / mask_name
            if not mask_dir.is_dir():
                missing.append(f"{scene_dir}: fine_dynamic_masks/{mask_name}")
            elif len(list(mask_dir.iterdir())) != image_count:
                missing.append(f"{scene_dir}: fine_dynamic_masks/{mask_name} count mismatch")
        if not humanpose_path.is_file():
            missing.append(f"{scene_dir}: humanpose/smpl.pkl")
    if missing:
        details = "\n".join(f"- {item}" for item in missing[:20])
        raise RuntimeError(
            "DriveStudio 4DGS training inputs are incomplete. "
            "Run generated-data segmentation and human-pose preprocessing first.\n"
            f"{details}"
        )


def run_render_and_export(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            "tools/render_all_for_model.py",
            f"drivestudio-nus-{args.method_name}",
            *[str(clip) for clip in args.clips],
            "--tasks",
            "train",
            "depth",
            "novel",
            "--root",
            str(args.work_root),
            "--output-root",
            str(args.reconstruction_root / "_drivestudio_render_all"),
            "--fps",
            str(args.fps),
        ]
    )


def find_checkpoint(work_root: Path, method_name: str, clip_id: str) -> Optional[Path]:
    run_dir = work_root / f"drivestudio-nus-{method_name}" / clip_id
    final = run_dir / "checkpoint_final.pth"
    if final.is_file():
        return final
    candidates = sorted(run_dir.glob("checkpoint_*.pth"))
    return candidates[-1] if candidates else None


def run_eval_for_existing_checkpoints(args: argparse.Namespace) -> None:
    for clip in args.clips:
        checkpoint = find_checkpoint(args.work_root, args.method_name, str(clip))
        if checkpoint is None:
            raise FileNotFoundError(
                f"No checkpoint found for method={args.method_name}, clip={clip} "
                f"under {args.work_root}"
            )
        run(
            [
                sys.executable,
                "tools/eval.py",
                "--resume_from",
                str(checkpoint),
                "--include_depth",
                "--save_raw_depth",
                "raw_depth",
            ]
        )


def copy_file_if_exists(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def camera_id_from_name(camera_name: str) -> Optional[int]:
    names = {
        "CAM_FRONT": 0,
        "CAM_FRONT_LEFT": 1,
        "CAM_FRONT_RIGHT": 2,
        "CAM_BACK_LEFT": 3,
        "CAM_BACK_RIGHT": 4,
        "CAM_BACK": 5,
    }
    return names.get(camera_name)


def export_checkpoint_assets(args: argparse.Namespace, clip_id: str) -> None:
    checkpoint = find_checkpoint(args.work_root, args.method_name, clip_id)
    run_dir = args.work_root / f"drivestudio-nus-{args.method_name}" / clip_id
    out_dir = method_root(args.reconstruction_root, args.method_name) / "ckpt" / clip_id
    copy_file_if_exists(run_dir / "config.yaml", out_dir / "config.yaml")
    if checkpoint is not None:
        copy_file_if_exists(checkpoint, out_dir / checkpoint.name)
        final_target = out_dir / "checkpoint_final.pth"
        if checkpoint.name != final_target.name and not final_target.exists():
            shutil.copy2(checkpoint, final_target)


def export_train_assets(args: argparse.Namespace, clip_id: str) -> None:
    render_root = (
        args.reconstruction_root
        / "_drivestudio_render_all"
        / "train"
        / clip_id
        / f"drivestudio-nus-{args.method_name}"
    )
    out_root = method_root(args.reconstruction_root, args.method_name) / "train" / clip_id

    camera_dirs = sorted(path for path in render_root.iterdir() if path.is_dir()) if render_root.is_dir() else []
    for camera_dir in camera_dirs:
        camera_name = camera_dir.name
        for pred_path in sorted((camera_dir / "images").glob("*.jpg")):
            frame_id = pred_path.stem
            target_name = f"{frame_id}_{camera_name}.jpg"
            copy_file_if_exists(pred_path, out_root / "pred" / target_name)
        for gt_path in sorted((camera_dir / "gt_images").glob("*.jpg")):
            frame_id = gt_path.stem
            target_name = f"{frame_id}_{camera_name}.jpg"
            copy_file_if_exists(gt_path, out_root / "gt" / target_name)


def export_depth_assets(args: argparse.Namespace, clip_id: str) -> None:
    out_root = method_root(args.reconstruction_root, args.method_name) / "depth" / clip_id
    pred_dir = args.work_root / f"drivestudio-nus-{args.method_name}" / clip_id / "raw_depth" / f"full_set_{args.step}"
    gt_dir = args.work_root / f"drivestudio-nus-{args.gt_method}" / clip_id / "raw_depth" / f"full_set_{args.step}"
    mask_root = gt_processed_data_root(args) / f"{int(clip_id):03d}" / "sam_mask"

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing predicted raw depth directory: {pred_dir}")
    pred_paths = sorted(pred_dir.glob("*.npy"))
    if not pred_paths:
        raise FileNotFoundError(f"No predicted raw depth files found under {pred_dir}")

    for pred_path in pred_paths:
        copy_file_if_exists(pred_path, out_root / "pred" / pred_path.name)
        gt_path = gt_dir / pred_path.name
        if not copy_file_if_exists(gt_path, out_root / "gt" / pred_path.name):
            raise FileNotFoundError(f"Missing GT raw depth for geometric discrepancy: {gt_path}")
        try:
            idx_str, cam_name = pred_path.stem.split("_CAM_")
        except ValueError:
            raise ValueError(f"Unexpected raw depth filename, expected <frame>_CAM_<name>.npy: {pred_path}")
        cam_name = f"CAM_{cam_name}" if not cam_name.startswith("CAM_") else cam_name
        camera_id = camera_id_from_name(cam_name)
        if camera_id is None:
            raise ValueError(f"Unsupported camera name in raw depth filename: {pred_path}")
        frame_idx = int(idx_str) // 6
        mask_path = mask_root / f"{frame_idx:03d}_{camera_id}.png"
        if not copy_file_if_exists(mask_path, out_root / "masks" / f"{pred_path.stem}.png"):
            raise FileNotFoundError(
                f"Missing Grounded-SAM2 evaluation mask for geometric discrepancy: {mask_path}. "
                "Generate GT masks with worldbench/videogen/reconstruction/generate_eval_masks.py."
            )


def copy_novel_videos(args: argparse.Namespace, clip_id: str) -> None:
    render_root = (
        args.reconstruction_root
        / "_drivestudio_render_all"
        / "novel"
        / clip_id
        / f"drivestudio-nus-{args.method_name}"
    )
    out_dir = method_root(args.reconstruction_root, args.method_name) / "novel" / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for view_name in NOVEL_VIEWS:
        source = render_root / view_name / "novel.mp4"
        target = out_dir / f"{view_name}.mp4"
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def export_contract_assets(args: argparse.Namespace) -> None:
    for clip in args.clips:
        clip_id = str(clip)
        export_checkpoint_assets(args, clip_id)
        export_train_assets(args, clip_id)
        export_depth_assets(args, clip_id)
        copy_novel_videos(args, clip_id)


def run_reconstruct(args: argparse.Namespace) -> None:
    if not args.skip_preprocess:
        run_preprocess(args)
    if not args.skip_train:
        validate_training_inputs(processed_data_root(args), args.clips)
        run_train(args)
    if not args.skip_eval:
        run_eval_for_existing_checkpoints(args)
        run_render_and_export(args)
    export_contract_assets(args)


def main() -> None:
    args = parse_args()
    write_manifest(args)
    if args.dry_run:
        write_dry_run_outputs(args.reconstruction_root, args.method_name, args.clips)
        return
    run_reconstruct(args)


if __name__ == "__main__":
    main()
