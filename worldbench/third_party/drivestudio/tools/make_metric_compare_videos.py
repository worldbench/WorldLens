#!/usr/bin/env python3
"""Make per-clip metric-compare demo videos for LPIPS and AbsRel.

For each clip and each metric type (LPIPS for RGB, AbsRel for depth),
pick the best-2 and worst-2 methods (excluding GT) and compose a 2x2
grid video:

  rows: 1st row = best two methods (left: best, right: 2nd best)
        2nd row = worst two methods (left: 2nd worst, right: worst)

Each tile is the original 6-view training/depth video for that method
and clip (full_set_<step>_rgbs.mp4 / full_set_<step>_depths.mp4).
Metric values are overlaid by compose_*_grid.py via drawtext.

Output layout:
  data/metric_compare/LPIPS/<clip_id>.mp4
  data/metric_compare/AbsRel/<clip_id>.mp4

This script reuses:
  - work_dirs_analysis_metrics.json       (LPIPS per-seq)
  - depth_tables_3/per_seq.csv (fallback depth_tables_2/per_seq.csv) (AbsRel per-seq)
  - tools/compose_train_view_grid.py      (RGB grid with LPIPS labels)
  - tools/compose_depth_grid.py           (Depth grid with AbsRel labels)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_MODELS = [
    "drivestudio-nus-dist4d",
    "drivestudio-nus-drivedreamer2",
    "drivestudio-nus-opendwm",
    "drivestudio-nus-dreamforge",
    "drivestudio-nus-xscene",
    "drivestudio-nus-magicdrive",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compose LPIPS/AbsRel metric-compare videos per clip")
    # If no clip IDs are given, we will auto-discover all clips present in the
    # depth per-seq CSVs (i.e., all clips with AbsRel metrics).
    p.add_argument("clip_id", nargs="*", help="Clip IDs (e.g., 813 694). If omitted, auto-discover from depth tables.")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root directory containing model runs")
    p.add_argument("--models", type=str, nargs="*", default=None,
                   help="Model directory names under root (excluding GT). Default: common drivestudio-nus-* models.")
    p.add_argument("--step", type=int, default=30000, help="Training step used in file names (default: 30000)")
    p.add_argument("--metrics-json", type=Path, default=Path("work_dirs_analysis_metrics.json"),
                   help="LPIPS metrics JSON (default: work_dirs_analysis_metrics.json)")
    p.add_argument("--metric-key", type=str, default="image_metrics/full/lpips",
                   help="Metric key in JSON for LPIPS (default: image_metrics/full/lpips)")
    p.add_argument("--depth-csv", type=Path, nargs="*", default=None,
                   help="Depth per-seq CSVs (default: depth_tables_3/per_seq.csv, depth_tables_2/per_seq.csv)")
    p.add_argument("--absrel-column", type=str, default="AbsRel", help="Column name for AbsRel in depth CSV")
    p.add_argument("--output-root", type=Path, default=Path("data/metric_compare"),
                   help="Root directory for metric-compare videos")
    p.add_argument("--ffmpeg-bin", type=str, default=None, help="Optional ffmpeg binary (propagated to compose_* scripts)")
    p.add_argument("--font", type=Path, default=None, help="Optional TTF/OTF font for labels (propagated)")
    p.add_argument("--fontsize", type=int, default=26, help="Font size for labels")
    p.add_argument("--gap", type=int, default=8, help="Gap in pixels between tiles (passed to compose scripts)")
    p.add_argument("--tile-width", type=int, default=1280, help="Tile width (default: 1280; 0=auto)")
    p.add_argument("--tile-height", type=int, default=720, help="Tile height (default: 720; 0=auto)")
    p.add_argument("--fit", type=str, choices=["contain", "cover"], default="contain",
                   help="Resize mode for tiles (default: contain)")
    p.add_argument("--fps", type=int, default=0, help="Optional FPS override for composed videos (0=keep source)")
    p.add_argument("--frames", type=int, default=0, help="Optional trim to N frames (0=full length)")
    p.add_argument("--debug", action="store_true", help="Print detailed commands and selection info")
    p.add_argument("--dry-run", action="store_true", help="Only print planned commands, do not run")
    return p.parse_args()


def method_label(name: str) -> str:
    return name.split("drivestudio-nus-", 1)[-1].replace("_", " ").replace("-", " ")


def load_lpips(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.is_file():
        raise SystemExit(f"LPIPS metrics JSON not found: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse {path}: {e}")


def load_absrel(paths: Sequence[Path], col: str) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if "Method/Seq" not in reader.fieldnames or col not in reader.fieldnames:
                    continue
                for row in reader:
                    tag = row.get("Method/Seq", "").strip()
                    val = row.get(col)
                    if not tag or val is None:
                        continue
                    if "/" not in tag:
                        continue
                    method, seq = tag.split("/", 1)
                    try:
                        lookup[(method.strip(), seq.strip())] = float(val)
                    except ValueError:
                        continue
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")
    return lookup


def has_video(root: Path, model: str, clip: str, step: int, kind: str) -> bool:
    base = root / model / clip / "videos"
    if kind == "rgb":
        return (base / f"full_set_{step}_rgbs.mp4").is_file()
    if kind == "depth":
        return (base / f"full_set_{step}_depths.mp4").is_file()
    return False


def run(cmd: List[str], dry_run: bool = False, debug: bool = False) -> int:
    if debug or dry_run:
        print("RUN:", " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[WARN] command failed (exit {e.returncode}):", " ".join(str(c) for c in cmd))
        return e.returncode


def select_best_worst(values: Dict[str, float]) -> Optional[List[str]]:
    """Given model -> metric (lower is better), return [best1, best2, worst2, worst1]."""
    if len(values) < 4:
        return None
    # sort ascending by metric (lower is better)
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    best1, best2 = ordered[0][0], ordered[1][0]
    worst1, worst2 = ordered[-1][0], ordered[-2][0]
    return [best1, best2, worst2, worst1]


def main() -> None:
    args = parse_args()

    lpips_json = load_lpips(args.metrics_json)
    depth_paths: List[Path] = list(args.depth_csv) if args.depth_csv else [
        Path("depth_tables_3/per_seq.csv"),
        Path("depth_tables_2/per_seq.csv"),
    ]
    absrel_lookup = load_absrel(depth_paths, args.absrel_column)

    models_base: List[str] = list(args.models) if args.models else list(DEFAULT_MODELS)
    # ensure no GT in base list
    models_base = [m for m in models_base if not m.endswith("-gt")]

    # shared compose args
    tile_args: List[str] = ["--gap", str(args.gap)]
    if args.tile_height and args.tile_height > 0:
        tile_args += ["--tile-height", str(args.tile_height)]
    if args.tile_width and args.tile_width > 0:
        tile_args += ["--tile-width", str(args.tile_width), "--fit", args.fit]
    fps_args: List[str] = []
    if args.fps and args.fps > 0:
        fps_args += ["--fps", str(args.fps)]
    if args.frames and args.frames > 0:
        fps_args += ["--frames", str(args.frames)]

    font_args: List[str] = []
    if args.font:
        font_args = ["--font", str(args.font), "--fontsize", str(args.fontsize)]
    else:
        font_args = ["--fontsize", str(args.fontsize)]

    ffmpeg_args: List[str] = []
    if args.ffmpeg_bin:
        ffmpeg_args = ["--ffmpeg-bin", args.ffmpeg_bin]

    out_lpips_root = args.output_root / "LPIPS"
    out_absrel_root = args.output_root / "AbsRel"
    out_lpips_root.mkdir(parents=True, exist_ok=True)
    out_absrel_root.mkdir(parents=True, exist_ok=True)

    # Determine clips: use provided list, or auto-discover from depth tables.
    if args.clip_id:
        clips = list(args.clip_id)
    else:
        clips = sorted({seq for (_, seq) in absrel_lookup.keys()})
        if not clips:
            raise SystemExit("No clips found in depth tables; please provide clip_id explicitly.")

    for clip in clips:
        # Collect per-model LPIPS and AbsRel for this clip.
        # LPIPS only requires RGB video + LPIPS metric; AbsRel only requires depth video + AbsRel.
        lpips_vals: Dict[str, float] = {}
        absrel_vals: Dict[str, float] = {}
        for model in models_base:
            # LPIPS (train RGB)
            if has_video(args.root, model, clip, args.step, "rgb"):
                seq_map = lpips_json.get(model, {})
                if isinstance(seq_map, dict):
                    clip_metrics = seq_map.get(clip, {})
                    if isinstance(clip_metrics, dict):
                        v = clip_metrics.get(args.metric_key)
                        if isinstance(v, (int, float)):
                            lpips_vals[model] = float(v)
            # AbsRel (depth)
            if has_video(args.root, model, clip, args.step, "depth"):
                meth = method_label(model).replace(" ", "")
                absrel_val: Optional[float] = None
                key1 = (meth, clip)
                key2 = (meth.lower(), clip)
                if key1 in absrel_lookup:
                    absrel_val = absrel_lookup[key1]
                elif key2 in absrel_lookup:
                    absrel_val = absrel_lookup[key2]
                if absrel_val is not None:
                    absrel_vals[model] = absrel_val

        if args.debug:
            print(f"\n[DEBUG] clip {clip} candidates:")
            for m in models_base:
                print(f"  {m}: LPIPS={lpips_vals.get(m)} AbsRel={absrel_vals.get(m)}")

        # LPIPS selection
        order_lpips = select_best_worst(lpips_vals)
        if order_lpips is None:
            print(f"[WARN] clip {clip}: not enough models with LPIPS to select 4 (have {len(lpips_vals)})")
        else:
            out_path = out_lpips_root / f"{clip}.mp4"
            cmd = [
                "python3", "tools/compose_train_view_grid.py", clip,
                "--root", str(args.root),
                "--models", *order_lpips,
                "--step", str(args.step),
                "--output", str(out_path),
                *tile_args,
                *ffmpeg_args,
                *font_args,
                *fps_args,
            ]
            rc = run(cmd, dry_run=args.dry_run, debug=args.debug)
            if rc == 0:
                print(f"[OK] LPIPS compare for clip {clip} -> {out_path}")

        # AbsRel selection
        order_absrel = select_best_worst(absrel_vals)
        if order_absrel is None:
            print(f"[WARN] clip {clip}: not enough models with AbsRel to select 4 (have {len(absrel_vals)})")
        else:
            out_path = out_absrel_root / f"{clip}.mp4"
            cmd = [
                "python3", "tools/compose_depth_grid.py", clip,
                "--root", str(args.root),
                "--models", *order_absrel,
                "--step", str(args.step),
                "--output", str(out_path),
                *tile_args,
                *ffmpeg_args,
                *font_args,
                *fps_args,
            ]
            rc = run(cmd, dry_run=args.dry_run, debug=args.debug)
            if rc == 0:
                print(f"[OK] AbsRel compare for clip {clip} -> {out_path}")


if __name__ == "__main__":
    main()
