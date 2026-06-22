#!/usr/bin/env python3
"""
Collect nuScenes novel-view benchmark videos into a unified delivery tree.

Source layout assumption (per run):
  work_dirs/
    drivestudio-nus-<method>/
      <clip_name>/
        noval_30000_benchmark/               # user-customized output folder
          front_center_interp.mp4
          s_curve.mp4
          lateral_offset.mp4
          lateral_offset_left.mp4

Target layout:
  nus_novel_view_benchmark/
    <method>/
      <view_name>/
        <clip_name>.mp4

Steps:
  1) Enumerate methods matching --pattern under --root (default: work_dirs/drivestudio-nus-*)
  2) For each method, find the set of clips (run subdirs) that contain all required view mp4s
     specifically under: <run_dir>/videos/<src-folder>/
  3) Compute intersection across methods
  4) For the common clips, copy/symlink/Hardlink each view mp4 to target tree

Safety:
  - Never deletes anything; only creates target directories as needed
  - Overwrites existing target files if --overwrite

Usage:
  python tools/collect_nus_benchmark.py \
      --root work_dirs \
      --pattern "drivestudio-nus-*" \
      --src-folder "noval_30000_benchmark" \
      --dest nus_novel_view_benchmark \
      --views front_center_interp,s_curve,lateral_offset,lateral_offset_left \
      --mode copy --overwrite
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import shutil


DEFAULT_VIEWS = [
    "front_center_interp",
    "s_curve",
    "lateral_offset",
    "lateral_offset_left",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Collect nuScenes novel-view benchmark videos")
    ap.add_argument("--root", type=str, default="work_dirs", help="Root containing method dirs")
    ap.add_argument("--pattern", type=str, default="drivestudio-nus-*", help="Method dir glob pattern")
    ap.add_argument("--src-folder", type=str, default="novel_30000_benchmark", help="Subfolder under run_dir/videos/ containing view mp4s")
    ap.add_argument("--dest", type=str, default="data/nus_novel_view_benchmark", help="Destination root for collected videos")
    ap.add_argument("--views", type=str, default=",".join(DEFAULT_VIEWS), help="Comma-separated view names to collect")
    ap.add_argument("--mode", type=str, choices=["copy", "symlink", "hardlink"], default="copy", help="File materialization mode")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files at destination")
    ap.add_argument("--strip-prefix", type=str, default="drivestudio-nus-", help="Prefix to strip from method dir name when naming method in dest")
    return ap.parse_args()


def list_method_dirs(root: Path, pattern: str) -> List[Path]:
    return sorted(root.glob(pattern))


def find_src_view_file(run_dir: Path, src_folder: str, view: str) -> Optional[Path]:
    """Return path to a view mp4 strictly under <run_dir>/videos/<src_folder>/.

    Success is defined only if the file exists in this directory; no other fallbacks are considered.
    """
    base = run_dir / "videos" / src_folder
    if not base.exists():
        return None
    # prefer explicit cam0 if present, else plain view
    for name in [f"{view}_cam0.mp4", f"{view}.mp4"]:
        p = base / name
        if p.is_file():
            return p
    return None


def collect_runs_with_all_views(method_dir: Path, src_folder: str, views: Sequence[str]) -> Set[str]:
    """Return set of clip names (run subdir names) that contain all requested view files."""
    ok: Set[str] = set()
    for run in sorted(d for d in method_dir.iterdir() if d.is_dir()):
        has_all = True
        for view in views:
            if find_src_view_file(run, src_folder, view) is None:
                has_all = False
                break
        if has_all:
            ok.add(run.name)
    return ok


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        # Overwrite by replacing file atomically
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    dest_root = Path(args.dest)
    views = [v for v in (args.views.split(",") if args.views else []) if v]

    method_dirs = list_method_dirs(root, args.pattern)
    if not method_dirs:
        print(f"No method dirs under {root} matching pattern {args.pattern}")
        return

    # Compute intersection of clips across methods (and that contain all views per method)
    per_method_clips: Dict[str, Set[str]] = {}
    for mdir in method_dirs:
        clips = collect_runs_with_all_views(mdir, args.src_folder, views)
        method_name = mdir.name
        per_method_clips[method_name] = clips
        print(f"Method {method_name}: {len(clips)} eligible clips")

    if not per_method_clips:
        print("No eligible clips found")
        return

    # Intersection
    common: Optional[Set[str]] = None
    for clips in per_method_clips.values():
        common = clips if common is None else (common & clips)
    common = common or set()
    print(f"Common clips across methods: {len(common)}")

    # Materialize into target tree
    for mdir in method_dirs:
        method_label = mdir.name
        if args.strip_prefix and method_label.startswith(args.strip_prefix):
            method_label = method_label[len(args.strip_prefix):]
        for clip in sorted(common):
            run_dir = mdir / clip
            for view in views:
                src = find_src_view_file(run_dir, args.src_folder, view)
                if src is None:
                    # Should not happen given we filtered above; skip defensively
                    print(f"[WARN] Missing view {view} for {mdir.name}/{clip}")
                    continue
                dst = dest_root / method_label / view / f"{clip}.mp4"
                materialize(src, dst, args.mode, args.overwrite)

    print(f"Done. Collected into: {dest_root}")


if __name__ == "__main__":
    main()
