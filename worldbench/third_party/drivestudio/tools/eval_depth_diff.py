#!/usr/bin/env python3
"""Compute depth reconstruction error under semantic masks for nuScenes runs.

For each method (e.g. gt, dreamforge, drivedreamer2, magicdrive), sequence, and
rendered frame we compare the method's depth against the GT method's depth under
road/vehicle masks stored in `sam_mask`.

Assumptions:
  - Method work directories follow `work_dirs/drivestudio-nus-{method}/{seq}`.
  - Depth files are stored at `.../raw_depth/full_set_{step}/<index>_CAM_*.npy`.
  - GT depths are in the same structure with `{method}` = `gt` (default).
  - Semantic masks reside at
        data/nuscenes_trainval/processed_{method}/advanced_12Hz_trainval/{seq}/sam_mask/
    falling back to `processed_{gt_method}` or `processed` if missing.
  - Mask file naming convention: `{frame:03d}_{camera_id}.png`.

The script outputs a CSV with per-image MAE/MSE under the mask, plus summary
statistics per method/sequence.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import glob
import math

import numpy as np
import imageio.v2 as imageio

LOG_EPS = 1e-6

try:
    from tqdm import tqdm  # type: ignore

    def progress_iter(iterable, **kwargs):
        return tqdm(iterable, **kwargs)

except ImportError:  # pragma: no cover - fallback when tqdm is unavailable

    def progress_iter(iterable, **kwargs):
        return iterable


# Camera ordering for nuScenes 6-camera configuration
CAMERA_NAME_TO_ID: Dict[str, int] = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 1,
    "CAM_FRONT_RIGHT": 2,
    "CAM_BACK_LEFT": 3,
    "CAM_BACK_RIGHT": 4,
    "CAM_BACK": 5,
}
NUM_CAMERAS = len(CAMERA_NAME_TO_ID)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate depth differences under road/vehicle masks")
    p.add_argument("--methods", nargs="*", default=None,
                   help="List of methods to compare against GT. Default: auto-discover all methods under work_dirs.")
    p.add_argument("--gt-method", default="gt", help="Method name used as GT reference (default: gt)")
    p.add_argument("--work-root", type=Path, default=Path("work_dirs"), help="Root of drivestudio run directories")
    p.add_argument("--depth-subdir", default="raw_depth/full_set_{step}",
                   help="Relative depth subdirectory pattern (default: raw_depth/full_set_{step})")
    p.add_argument("--step", type=int, default=30000, help="Training step used in depth filenames (default: 30000)")
    p.add_argument("--mask-root-template", default="data/nuscenes_trainval/processed_{method}/advanced_12Hz_trainval",
                   help="Mask root template with {method} placeholder (default matches nuScenes)")
    p.add_argument("--mask-method", default=None,
                   help="Force using masks from this method (overrides per-method masks). Use 'gt' to always use GT masks")
    p.add_argument("--cameras", default="CAM_FRONT",
                   help="Comma-separated camera names to evaluate (e.g. CAM_FRONT,CAM_BACK). Use 'all' to include every camera. Default: CAM_FRONT")
    p.add_argument("--table-csv-prefix", type=Path, default=None,
                   help="If set, save summary tables to CSV. Pass a directory (e.g. results/depth_tables) or a base file path.")
    p.add_argument("--output", type=Path, default=Path("depth_mask_metrics.csv"),
                   help="CSV file to write per-image metrics (default: depth_mask_metrics.csv)")
    p.add_argument("--summary", type=Path, default=Path("depth_mask_metrics_summary.csv"),
                   help="CSV file to write aggregated summaries (default: depth_mask_metrics_summary.csv)")
    p.add_argument("--save-csv", action="store_true",
                   help="Explicitly save CSV outputs (--output/--summary). Default: do not write CSV files.")
    p.add_argument("--verbose", action="store_true", help="Print progress information")
    return p.parse_args()


def depth_dir(work_root: Path, method: str, seq: str, depth_subdir: str, step: int) -> Path:
    sub = depth_subdir.format(step=step)
    return work_root / f"drivestudio-nus-{method}" / seq / sub


def list_sequences(work_root: Path, method: str, depth_subdir: str, step: int) -> List[str]:
    method_root = work_root / f"drivestudio-nus-{method}"
    if not method_root.is_dir():
        return []
    seqs = []
    for seq_dir in sorted(d for d in method_root.iterdir() if d.is_dir()):
        if depth_dir(work_root, method, seq_dir.name, depth_subdir, step).is_dir():
            seqs.append(seq_dir.name)
    return seqs


def resolve_mask_dir(mask_root_template: str, method: str, seq: str, fallback_methods: Sequence[str], force_method: Optional[str]) -> Optional[Tuple[Path, str]]:
    candidates = []
    if force_method:
        candidates.append(force_method)
    for token in [method, *fallback_methods]:
        if token not in candidates:
            candidates.append(token)
    for token in candidates:
        template = mask_root_template.format(method=token)
        if token == "":
            template = template.replace("processed_", "processed", 1)
        template = template.replace("processed_//", "processed/")
        mask_dir = Path(template) / seq / "sam_mask"
        if mask_dir.is_dir():
            return mask_dir, token or "processed"
    return None


def load_mask(mask_path: Path) -> np.ndarray:
    mask_img = imageio.imread(mask_path)
    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]
    return mask_img > 0


def resize_mask_nearest(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to target shape using nearest-neighbor sampling."""
    if mask.shape == target_shape:
        return mask
    src_h, src_w = mask.shape
    tgt_h, tgt_w = target_shape
    if tgt_h <= 0 or tgt_w <= 0:
        raise ValueError(f"Invalid target shape {target_shape} when resizing mask.")
    y_idx = np.linspace(0, src_h - 1, tgt_h).round().astype(int)
    x_idx = np.linspace(0, src_w - 1, tgt_w).round().astype(int)
    resized = mask[y_idx][:, x_idx]
    return resized


def camera_id_from_name(cam_name: str) -> Optional[int]:
    return CAMERA_NAME_TO_ID.get(cam_name)


def normalize_camera_name(name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None
    name = name.upper()
    if not name.startswith("CAM_"):
        name = "CAM_" + name
    return name if name in CAMERA_NAME_TO_ID else None


def frame_index_from_depth_name(idx_str: str) -> int:
    return int(idx_str) // NUM_CAMERAS


def compute_metrics(diff: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    return mae, mse


def collect_depth_files(depth_directory: Path) -> List[Path]:
    return sorted(depth_directory.glob("*_CAM_*.npy"))


def discover_methods(work_root: Path) -> List[str]:
    methods = []
    for path in glob.glob(str(work_root / "drivestudio-nus-*")):
        name = Path(path).name
        prefix = "drivestudio-nus-"
        if name.startswith(prefix):
            methods.append(name[len(prefix):])
    return sorted(set(methods))


def format_table(header: Sequence[str], rows: Sequence[Tuple[str, Sequence[str]]]) -> str:
    widths = [len(col) for col in header]
    for _, cells in rows:
        for idx, cell in enumerate(cells):
            if idx + 1 >= len(widths):
                widths.append(len(cell))
            else:
                widths[idx + 1] = max(widths[idx + 1], len(cell))
    line_parts = ["{:<{w}}".format(header[0], w=widths[0])]
    for idx, col in enumerate(header[1:], start=1):
        line_parts.append("{:>{w}}".format(col, w=widths[idx]))
    output = [" ".join(line_parts)]
    output.append(" ".join("-" * w for w in widths))
    for label, cells in rows:
        row_parts = ["{:<{w}}".format(label, w=widths[0])]
        for idx, cell in enumerate(cells, start=1):
            row_parts.append("{:>{w}}".format(cell, w=widths[idx]))
        output.append(" ".join(row_parts))
    return "\n".join(output)


def make_table_csv_path(base: Path, kind: str) -> Path:
    if base.suffix:
        parent = base.parent if str(base.parent) != "" else Path(".")
        return parent / f"{base.stem}_{kind}{base.suffix}"
    directory = base if str(base) != "" else Path(".")
    return directory / f"{kind}.csv"


def write_table_csv(path: Path, header: Sequence[str], rows: Sequence[Tuple[str, Sequence[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, cells in rows:
            writer.writerow([label] + list(cells))


def main() -> None:
    args = parse_args()
    work_root: Path = args.work_root
    methods = list(args.methods) if args.methods else discover_methods(work_root)
    if not methods:
        raise SystemExit(f"No methods found under {work_root} (looking for drivestudio-nus-*)")
    gt_method = args.gt_method
    if gt_method in methods:
        methods = [m for m in methods if m != gt_method]
    if args.verbose and args.methods is None:
        print(f"Discovered methods: {', '.join(methods)}")

    if args.cameras.lower() == "all":
        allowed_cam_names = set(CAMERA_NAME_TO_ID.keys())
    else:
        allowed_cam_names = set()
        for token in args.cameras.split(','):
            norm = normalize_camera_name(token)
            if norm is None:
                if args.verbose:
                    print(f"[WARN] Unknown camera '{token.strip()}', ignoring")
                continue
            allowed_cam_names.add(norm)
        if not allowed_cam_names:
            allowed_cam_names = {"CAM_FRONT"}
    allowed_cam_ids = {CAMERA_NAME_TO_ID[name] for name in allowed_cam_names}
    if args.verbose:
        cams_str = ', '.join(sorted(allowed_cam_names))
        print(f"Evaluating cameras: {cams_str}")

    depth_subdir = args.depth_subdir
    fallback_mask_methods = (gt_method, "gt", "magicdrive", "")

    gt_sequences = set(list_sequences(work_root, gt_method, depth_subdir, args.step))
    if not gt_sequences:
        raise SystemExit(f"No GT sequences found under {work_root}/drivestudio-nus-{gt_method}")

    rows: List[Dict[str, object]] = []
    summary: Dict[Tuple[str, str], Dict[str, float]] = {}

    method_iter = progress_iter(methods, desc="Methods", total=len(methods))
    for method in method_iter:
        method_sequences = set(list_sequences(work_root, method, depth_subdir, args.step))
        common_seqs = sorted(gt_sequences & method_sequences)
        if args.verbose:
            print(f"Method {method}: {len(common_seqs)} common sequences with GT")
        seq_iter = progress_iter(common_seqs, desc=f"{method} seqs", leave=False, total=len(common_seqs))
        for seq in seq_iter:
            depth_dir_method = depth_dir(work_root, method, seq, depth_subdir, args.step)
            depth_dir_gt = depth_dir(work_root, gt_method, seq, depth_subdir, args.step)
            mask_info = resolve_mask_dir(args.mask_root_template, method, seq, fallback_mask_methods, args.mask_method)
            if mask_info is None:
                print(f"[WARN] No sam_mask found for method={method}, seq={seq}; skipping")
                continue
            mask_dir, mask_source = mask_info
            depth_files = collect_depth_files(depth_dir_method)
            if not depth_files:
                print(f"[WARN] No depth files in {depth_dir_method}; skipping")
                continue
            depth_iter = progress_iter(depth_files, desc=f"{method}/{seq}", leave=False, total=len(depth_files))
            for depth_path in depth_iter:
                stem = depth_path.stem  # e.g. 00036_CAM_FRONT
                try:
                    idx_str, cam_name = stem.split("_CAM_")
                    cam_name = "CAM_" + cam_name if not cam_name.startswith("CAM_") else cam_name
                except ValueError:
                    print(f"[WARN] Unexpected depth filename format: {depth_path.name}; skipping")
                    continue
                cam_id = camera_id_from_name(cam_name)
                if cam_id is None:
                    continue  # skip cameras without mask mapping

                if cam_name not in allowed_cam_names or cam_id not in allowed_cam_ids:
                    continue

                frame_idx = frame_index_from_depth_name(idx_str)
                mask_path = mask_dir / f"{frame_idx:03d}_{cam_id}.png"
                if not mask_path.is_file():
                    print(f"[WARN] Missing mask {mask_path}; skipping frame")
                    continue

                gt_path = depth_dir_gt / depth_path.name
                if not gt_path.is_file():
                    print(f"[WARN] Missing GT depth {gt_path}; skipping frame")
                    continue

                depth_pred = np.load(depth_path)
                depth_gt = np.load(gt_path)
                if depth_pred.shape != depth_gt.shape:
                    print(f"[WARN] Shape mismatch pred {depth_pred.shape} vs gt {depth_gt.shape} for {depth_path}")
                    continue

                mask = load_mask(mask_path)
                if mask.shape != depth_pred.shape:
                    if args.verbose:
                        print(f"[INFO] Resizing mask {mask.shape} -> {depth_pred.shape} for {mask_path}")
                    try:
                        mask = resize_mask_nearest(mask, depth_pred.shape)
                    except ValueError as exc:
                        print(f"[WARN] Failed to resize mask {mask_path}: {exc}; skipping frame")
                        continue
                if not mask.any():
                    bucket = summary.setdefault(
                        (method, seq),
                        {
                            "abs_sum": 0.0,
                            "sq_sum": 0.0,
                            "pixels": 0,
                            "frames": 0,
                            "abs_rel_sum": 0.0,
                            "sq_rel_sum": 0.0,
                            "log_sq_sum": 0.0,
                            "log_pixels": 0,
                            "rel_pixels": 0,
                            "delta1_count": 0,
                            "delta2_count": 0,
                            "delta3_count": 0,
                        },
                    )
                    bucket["frames"] += 1
                    rows.append({
                        "method": method,
                        "sequence": seq,
                        "frame_index": frame_idx,
                        "camera": cam_name,
                        "depth_file": depth_path.name,
                        "mask_file": mask_path.name,
                        "mask_source": mask_source,
                        "pixels": 0,
                        "abs_sum": float("nan"),
                        "sq_sum": float("nan"),
                        "mae": float("nan"),
                        "mse": float("nan"),
                        "rmse": float("nan"),
                        "abs_rel": float("nan"),
                        "sq_rel": float("nan"),
                        "delta1": float("nan"),
                        "delta2": float("nan"),
                        "delta3": float("nan"),
                        "log_rmse": float("nan"),
                    })
                    continue

                diff = depth_pred[mask] - depth_gt[mask]
                abs_sum = float(np.abs(diff).sum())
                sq_sum = float((diff ** 2).sum())
                pixels = diff.size
                if pixels == 0:
                    continue
                mae = abs_sum / pixels
                mse = sq_sum / pixels
                rmse = math.sqrt(mse)

                gt_vals = depth_gt[mask]
                pred_vals = depth_pred[mask]
                rel_mask = gt_vals > 1e-6
                rel_pixels = int(rel_mask.sum())
                abs_rel_sum = 0.0
                sq_rel_sum = 0.0
                delta1_cnt = delta2_cnt = delta3_cnt = 0
                abs_rel = float("nan")
                sq_rel = float("nan")
                delta1 = float("nan")
                delta2 = float("nan")
                delta3 = float("nan")
                log_rmse = float("nan")
                if rel_pixels > 0:
                    gt_rel = gt_vals[rel_mask]
                    pred_rel = pred_vals[rel_mask]
                    rel_errors = np.abs(pred_rel - gt_rel) / gt_rel
                    abs_rel_sum = float(rel_errors.sum())
                    sq_rel_sum = float(((pred_rel - gt_rel) ** 2 / gt_rel).sum())
                    abs_rel = abs_rel_sum / rel_pixels
                    sq_rel = sq_rel_sum / rel_pixels
                    pred_safe = np.maximum(pred_rel, 1e-6)
                    ratios = np.maximum(pred_safe / gt_rel, gt_rel / pred_safe)
                    delta1_cnt = int((ratios < 1.25).sum())
                    delta2_cnt = int((ratios < 1.25 ** 2).sum())
                    delta3_cnt = int((ratios < 1.25 ** 3).sum())
                    delta1 = delta1_cnt / rel_pixels
                    delta2 = delta2_cnt / rel_pixels
                    delta3 = delta3_cnt / rel_pixels

                bucket = summary.setdefault(
                    (method, seq),
                    {
                        "abs_sum": 0.0,
                        "sq_sum": 0.0,
                        "pixels": 0,
                        "frames": 0,
                        "abs_rel_sum": 0.0,
                        "sq_rel_sum": 0.0,
                        "log_sq_sum": 0.0,
                        "log_pixels": 0,
                        "rel_pixels": 0,
                        "delta1_count": 0,
                        "delta2_count": 0,
                        "delta3_count": 0,
                    },
                )
                bucket["abs_sum"] += abs_sum
                bucket["sq_sum"] += sq_sum
                bucket["pixels"] += pixels
                bucket["frames"] += 1
                if rel_pixels > 0:
                    bucket["abs_rel_sum"] += abs_rel_sum
                    bucket["sq_rel_sum"] += sq_rel_sum
                    bucket["rel_pixels"] += rel_pixels
                    bucket["delta1_count"] += delta1_cnt
                    bucket["delta2_count"] += delta2_cnt
                    bucket["delta3_count"] += delta3_cnt
                    diff_log = (np.log(np.maximum(pred_rel, LOG_EPS)) - np.log(np.maximum(gt_rel, LOG_EPS)))**2
                    log_rmse = math.sqrt(float(diff_log.mean())) if diff_log.size > 0 else float("nan")
                    bucket["log_sq_sum"] = bucket.get("log_sq_sum", 0.0) + float(diff_log.sum())
                    bucket["log_pixels"] = bucket.get("log_pixels", 0) + int(diff_log.size)

                rows.append({
                    "method": method,
                    "sequence": seq,
                    "frame_index": frame_idx,
                    "camera": cam_name,
                    "depth_file": depth_path.name,
                    "mask_file": mask_path.name,
                    "mask_source": mask_source,
                    "pixels": pixels,
                    "abs_sum": abs_sum,
                    "sq_sum": sq_sum,
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "abs_rel": abs_rel,
                    "sq_rel": sq_rel,
                    "delta1": delta1,
                    "delta2": delta2,
                    "delta3": delta3,
                    "log_rmse": log_rmse,
                })

    if not rows:
        print("No metrics computed; check paths and inputs.")
        return

    # Write per-image metrics
    fieldnames = [
        "method",
        "sequence",
        "frame_index",
        "camera",
        "depth_file",
        "mask_file",
        "mask_source",
        "pixels",
        "abs_sum",
        "sq_sum",
        "mae",
        "mse",
        "rmse",
        "log_rmse",
        "abs_rel",
        "sq_rel",
        "delta1",
        "delta2",
        "delta3",
    ]
    if args.save_csv:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    summary_fieldnames = [
        "method",
        "sequence",
        "frames",
        "mean_mae",
        "mean_mse",
        "mean_rmse",
        "mean_log_rmse",
        "mean_abs_rel",
        "mean_sq_rel",
        "delta1",
        "delta2",
        "delta3",
        "total_pixels",
        "rel_pixels",
        "log_pixels",
    ]
    if args.save_csv:
        with args.summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            for (method, seq), stats in sorted(summary.items()):
                frames = stats["frames"]
                pixels = stats["pixels"]
                rel_pixels = stats["rel_pixels"]
                mean_mae = stats["abs_sum"] / pixels if pixels else float("nan")
                mean_mse = stats["sq_sum"] / pixels if pixels else float("nan")
                mean_rmse = math.sqrt(mean_mse) if math.isfinite(mean_mse) else float("nan")
                log_pixels = stats["log_pixels"]
                mean_log_rmse = math.sqrt(stats["log_sq_sum"] / log_pixels) if log_pixels else float("nan")
                mean_abs_rel = stats["abs_rel_sum"] / rel_pixels if rel_pixels else float("nan")
                mean_sq_rel = stats["sq_rel_sum"] / rel_pixels if rel_pixels else float("nan")
                delta1 = stats["delta1_count"] / rel_pixels if rel_pixels else float("nan")
                delta2 = stats["delta2_count"] / rel_pixels if rel_pixels else float("nan")
                delta3 = stats["delta3_count"] / rel_pixels if rel_pixels else float("nan")
                writer.writerow({
                    "method": method,
                    "sequence": seq,
                    "frames": frames,
                    "mean_mae": mean_mae,
                    "mean_mse": mean_mse,
                    "mean_rmse": mean_rmse,
                    "mean_log_rmse": mean_log_rmse,
                    "mean_abs_rel": mean_abs_rel,
                    "mean_sq_rel": mean_sq_rel,
                    "delta1": delta1,
                    "delta2": delta2,
                    "delta3": delta3,
                    "total_pixels": pixels,
                    "rel_pixels": rel_pixels,
                    "log_pixels": log_pixels,
                })

    if args.verbose and args.save_csv:
        print(f"Wrote per-image metrics to {args.output}")
        print(f"Wrote summaries to {args.summary}")


    def weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
        if not values or not weights or len(values) != len(weights):
            return float("nan")
        acc = 0.0
        total = 0
        for v, w in zip(values, weights):
            if not math.isfinite(v) or w <= 0:
                continue
            acc += v * w
            total += w
        return acc / total if total else float("nan")

    def weighted_std(values: Sequence[float], weights: Sequence[int], mean_val: Optional[float] = None) -> float:
        if not values or not weights or len(values) != len(weights):
            return float("nan")
        if mean_val is None:
            mean_val = weighted_mean(values, weights)
        if not math.isfinite(mean_val):
            return float("nan")
        acc = 0.0
        total = 0
        for v, w in zip(values, weights):
            if not math.isfinite(v) or w <= 0:
                continue
            acc += w * ((v - mean_val) ** 2)
            total += w
        return math.sqrt(acc / total) if total else float("nan")

# Console tables
    def fmt_float(val: float) -> str:
        return f"{val:.4f}" if math.isfinite(val) else "nan"

    def fmt_mean_std(mean_val: float, std_val: float) -> str:
        if not math.isfinite(mean_val):
            return "nan"
        if math.isfinite(std_val):
            return f"{mean_val:.4f}±{std_val:.4f}"
        return f"{mean_val:.4f}"

    per_seq_rows: List[Tuple[str, Sequence[str]]] = []
    aggregated: Dict[str, Dict[str, float]] = {}
    method_metrics: Dict[str, Dict[str, float]] = {}

    seq_mae: Dict[str, List[float]] = {}
    seq_rmse: Dict[str, List[float]] = {}
    seq_log_rmse: Dict[str, List[float]] = {}
    seq_abs_rel: Dict[str, List[float]] = {}
    seq_sq_rel: Dict[str, List[float]] = {}
    seq_delta1: Dict[str, List[float]] = {}
    seq_delta2: Dict[str, List[float]] = {}
    seq_delta3: Dict[str, List[float]] = {}

    pixel_weights: Dict[str, List[int]] = {}
    rel_weights: Dict[str, List[int]] = {}
    log_weights: Dict[str, List[int]] = {}

    for (method, seq), stats in sorted(summary.items()):
        frames = stats["frames"]
        pixels = stats["pixels"]
        rel_pixels = stats["rel_pixels"]
        mae_seq = stats["abs_sum"] / pixels if pixels else float("nan")
        mse_seq = stats["sq_sum"] / pixels if pixels else float("nan")
        rmse_seq = math.sqrt(mse_seq) if math.isfinite(mse_seq) else float("nan")
        log_rmse_seq = math.sqrt(stats["log_sq_sum"] / stats["log_pixels"]) if stats["log_pixels"] else float("nan")
        abs_rel_seq = stats["abs_rel_sum"] / rel_pixels if rel_pixels else float("nan")
        sq_rel_seq = stats["sq_rel_sum"] / rel_pixels if rel_pixels else float("nan")
        delta1_seq = stats["delta1_count"] / rel_pixels if rel_pixels else float("nan")
        delta2_seq = stats["delta2_count"] / rel_pixels if rel_pixels else float("nan")
        delta3_seq = stats["delta3_count"] / rel_pixels if rel_pixels else float("nan")

        per_seq_rows.append((
            f"{method}/{seq}",
            [
                str(frames),
                str(pixels),
                fmt_float(mae_seq),
                fmt_float(rmse_seq),
                fmt_float(log_rmse_seq),
                fmt_float(abs_rel_seq),
                fmt_float(sq_rel_seq),
                fmt_float(delta1_seq),
                fmt_float(delta2_seq),
                fmt_float(delta3_seq),
            ],
        ))

        bucket = aggregated.setdefault(
            method,
            {
                "abs_sum": 0.0,
                "sq_sum": 0.0,
                "pixels": 0,
                "frames": 0,
                "abs_rel_sum": 0.0,
                "sq_rel_sum": 0.0,
                "log_sq_sum": 0.0,
                "log_pixels": 0,
                "rel_pixels": 0,
                "delta1_count": 0,
                "delta2_count": 0,
                "delta3_count": 0,
            },
        )
        bucket["abs_sum"] += stats["abs_sum"]
        bucket["sq_sum"] += stats["sq_sum"]
        bucket["pixels"] += pixels
        bucket["frames"] += frames
        bucket["abs_rel_sum"] += stats["abs_rel_sum"]
        bucket["sq_rel_sum"] += stats["sq_rel_sum"]
        bucket["log_sq_sum"] += stats["log_sq_sum"]
        bucket["log_pixels"] += stats["log_pixels"]
        bucket["rel_pixels"] += rel_pixels
        bucket["delta1_count"] += stats["delta1_count"]
        bucket["delta2_count"] += stats["delta2_count"]
        bucket["delta3_count"] += stats["delta3_count"]

        seq_mae.setdefault(method, []).append(mae_seq)
        seq_rmse.setdefault(method, []).append(rmse_seq)
        seq_log_rmse.setdefault(method, []).append(log_rmse_seq)
        seq_abs_rel.setdefault(method, []).append(abs_rel_seq)
        seq_sq_rel.setdefault(method, []).append(sq_rel_seq)
        seq_delta1.setdefault(method, []).append(delta1_seq)
        seq_delta2.setdefault(method, []).append(delta2_seq)
        seq_delta3.setdefault(method, []).append(delta3_seq)

        pixel_weights.setdefault(method, []).append(pixels)
        rel_weights.setdefault(method, []).append(rel_pixels)
        log_weights.setdefault(method, []).append(stats["log_pixels"])

    for method in sorted(aggregated):
        stats = aggregated[method]
        pixels = stats["pixels"]
        rel_pixels = stats["rel_pixels"]
        global_mae = stats["abs_sum"] / pixels if pixels else float("nan")
        global_mse = stats["sq_sum"] / pixels if pixels else float("nan")
        global_rmse = math.sqrt(global_mse) if math.isfinite(global_mse) else float("nan")
        global_log_rmse = math.sqrt(stats["log_sq_sum"] / stats["log_pixels"]) if stats["log_pixels"] else float("nan")
        global_abs_rel = stats["abs_rel_sum"] / rel_pixels if rel_pixels else float("nan")
        global_sq_rel = stats["sq_rel_sum"] / rel_pixels if rel_pixels else float("nan")
        global_delta1 = stats["delta1_count"] / rel_pixels if rel_pixels else float("nan")
        global_delta2 = stats["delta2_count"] / rel_pixels if rel_pixels else float("nan")
        global_delta3 = stats["delta3_count"] / rel_pixels if rel_pixels else float("nan")

        mae_list = seq_mae.get(method, [])
        rmse_list = seq_rmse.get(method, [])
        log_list = seq_log_rmse.get(method, [])
        abs_rel_list = seq_abs_rel.get(method, [])
        sq_rel_list = seq_sq_rel.get(method, [])
        delta1_list = seq_delta1.get(method, [])
        delta2_list = seq_delta2.get(method, [])
        delta3_list = seq_delta3.get(method, [])

        pixel_w = pixel_weights.get(method, [])
        rel_w = rel_weights.get(method, [])
        log_w = log_weights.get(method, [])

        seq_mae_mean = weighted_mean(mae_list, pixel_w)
        seq_mae_std = weighted_std(mae_list, pixel_w, seq_mae_mean)
        seq_rmse_mean = weighted_mean(rmse_list, pixel_w)
        seq_rmse_std = weighted_std(rmse_list, pixel_w, seq_rmse_mean)
        seq_log_rmse_mean = weighted_mean(log_list, log_w)
        seq_log_rmse_std = weighted_std(log_list, log_w, seq_log_rmse_mean)
        seq_abs_rel_mean = weighted_mean(abs_rel_list, rel_w)
        seq_abs_rel_std = weighted_std(abs_rel_list, rel_w, seq_abs_rel_mean)
        seq_sq_rel_mean = weighted_mean(sq_rel_list, rel_w)
        seq_sq_rel_std = weighted_std(sq_rel_list, rel_w, seq_sq_rel_mean)
        seq_delta1_mean = weighted_mean(delta1_list, rel_w)
        seq_delta1_std = weighted_std(delta1_list, rel_w, seq_delta1_mean)
        seq_delta2_mean = weighted_mean(delta2_list, rel_w)
        seq_delta2_std = weighted_std(delta2_list, rel_w, seq_delta2_mean)
        seq_delta3_mean = weighted_mean(delta3_list, rel_w)
        seq_delta3_std = weighted_std(delta3_list, rel_w, seq_delta3_mean)

        method_metrics[method] = {
            "frames": stats["frames"],
            "pixels": pixels,
            "global_mae": global_mae,
            "global_mse": global_mse,
            "global_rmse": global_rmse,
            "global_log_rmse": global_log_rmse,
            "global_abs_rel": global_abs_rel,
            "global_sq_rel": global_sq_rel,
            "global_delta1": global_delta1,
            "global_delta2": global_delta2,
            "global_delta3": global_delta3,
            "seq_mae_mean": seq_mae_mean,
            "seq_mae_std": seq_mae_std,
            "seq_rmse_mean": seq_rmse_mean,
            "seq_rmse_std": seq_rmse_std,
            "seq_log_rmse_mean": seq_log_rmse_mean,
            "seq_log_rmse_std": seq_log_rmse_std,
            "seq_abs_rel_mean": seq_abs_rel_mean,
            "seq_abs_rel_std": seq_abs_rel_std,
            "seq_sq_rel_mean": seq_sq_rel_mean,
            "seq_sq_rel_std": seq_sq_rel_std,
            "seq_delta1_mean": seq_delta1_mean,
            "seq_delta1_std": seq_delta1_std,
            "seq_delta2_mean": seq_delta2_mean,
            "seq_delta2_std": seq_delta2_std,
            "seq_delta3_mean": seq_delta3_mean,
            "seq_delta3_std": seq_delta3_std,
        }

    if method_metrics:
        method_order = sorted(method_metrics)
        metric_specs = [
            ("Frames", "frames", True),
            ("Pixels", "pixels", True),
            ("Global MAE", "global_mae", False),
            ("Global MSE", "global_mse", False),
            ("Global RMSE", "global_rmse", False),
            ("Global Log RMSE", "global_log_rmse", False),
            ("Global AbsRel", "global_abs_rel", False),
            ("Global SqRel", "global_sq_rel", False),
            ("Global δ1", "global_delta1", False),
            ("Global δ2", "global_delta2", False),
            ("Global δ3", "global_delta3", False),
            ("Seq MAE", ("seq_mae_mean", "seq_mae_std"), False),
            ("Seq RMSE", ("seq_rmse_mean", "seq_rmse_std"), False),
            ("Seq Log RMSE", ("seq_log_rmse_mean", "seq_log_rmse_std"), False),
            ("Seq AbsRel", ("seq_abs_rel_mean", "seq_abs_rel_std"), False),
            ("Seq SqRel", ("seq_sq_rel_mean", "seq_sq_rel_std"), False),
            ("Seq δ1", ("seq_delta1_mean", "seq_delta1_std"), False),
            ("Seq δ2", ("seq_delta2_mean", "seq_delta2_std"), False),
            ("Seq δ3", ("seq_delta3_mean", "seq_delta3_std"), False),
        ]
        method_rows: List[Tuple[str, Sequence[str]]] = []
        for label, key, is_int in metric_specs:
            cells = []
            if isinstance(key, tuple):
                mean_key, std_key = key
                for method in method_order:
                    data = method_metrics[method]
                    mean_val = data.get(mean_key)
                    std_val = data.get(std_key)
                    cells.append(fmt_mean_std(mean_val, std_val))
            else:
                for method in method_order:
                    value = method_metrics[method].get(key)
                    if value is None:
                        cells.append("n/a")
                    else:
                        cells.append(str(int(value)) if is_int and math.isfinite(value) else fmt_float(value))
            method_rows.append((label, cells))
        method_header = ["Metric"] + method_order
        print("Method depth metrics (mean ± std where applicable):")
        print(format_table(method_header, method_rows))
        print()
        if args.table_csv_prefix:
            write_table_csv(
                make_table_csv_path(args.table_csv_prefix, "method"),
                method_header,
                method_rows,
            )

    if per_seq_rows:
        seq_header = [
            "Method/Seq",
            "Frames",
            "Pixels",
            "MAE",
            "RMSE",
            "Log RMSE",
            "AbsRel",
            "SqRel",
            "δ1",
            "δ2",
            "δ3",
        ]
        print("Per-sequence depth error under mask:")
        print(format_table(seq_header, per_seq_rows))
        if args.table_csv_prefix:
            write_table_csv(
                make_table_csv_path(args.table_csv_prefix, "per_seq"),
                seq_header,
                per_seq_rows,
            )


if __name__ == "__main__":
    main()
