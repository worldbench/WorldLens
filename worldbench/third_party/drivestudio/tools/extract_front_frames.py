#!/usr/bin/env python3
"""
Extract per-frame RGB and (if available) depth images for a given clip and frame
from: (1) training views, and (2) specific novel trajectories (front_center_interp,
lateral_offset, lateral_offset_left) across multiple methods (including GT).

Outputs a simple folder layout under --output-dir:
  <out>/<clip>/<method>/
      train_rgb_f<idx>.png
      train_depth_f<idx>.png  (if available)
      front_center_interp_rgb_f<idx>.png
      front_center_interp_depth_f<idx>.png  (if available)
      lateral_offset_rgb_f<idx>.png
      lateral_offset_depth_f<idx>.png      (if available)
      lateral_offset_left_rgb_f<idx>.png
      lateral_offset_left_depth_f<idx>.png (if available)

Assumptions:
  - Training videos live at work_dirs/<model>/<clip>/videos/full_set_<step>_rgbs.mp4 (and *_depths.mp4).
  - Novel videos live at work_dirs/<model>/<clip>/videos/<novel_dir>/<view>.mp4, where
    <novel_dir> defaults to the lexicographically last 'novel_*' under videos/.
  - Some runs may not carry novel depths; we skip them with a warning.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_MODELS = [
    "drivestudio-nus-gt",
    "drivestudio-nus-dist4d",
    "drivestudio-nus-drivedreamer2",
    "drivestudio-nus-opendwm",
    "drivestudio-nus-dreamforge",
    "drivestudio-nus-xscene",
    "drivestudio-nus-magicdrive",
]

DEFAULT_VIEWS = [
    "front_center_interp",
    "lateral_offset",
    "lateral_offset_left",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Extract RGB/Depth images for selected frame across methods and views")
    p.add_argument("clip_id", type=str, help="Clip identifier (e.g., 813)")
    p.add_argument("frame", type=int, help="Zero-based frame index to extract")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root containing per-model runs (default: work_dirs)")
    p.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS, help="Model directory names under root (default: 7 incl. GT)")
    p.add_argument("--step", type=int, default=30000, help="Training step in filenames (default: 30000)")
    p.add_argument("--novel-dir", type=str, default=None, help="Explicit novel dir name (e.g., novel_30000_benchmark). Default: latest novel_* under videos.")
    p.add_argument("--views", type=str, nargs="*", default=DEFAULT_VIEWS, help="Novel view names to extract (default: front_center_interp lateral_offset lateral_offset_left)")
    p.add_argument("--output-dir", type=Path, default=Path("data/demo_frames"), help="Directory to write PNGs (default: data/demo_frames)")
    p.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="FFmpeg executable (default: ffmpeg)")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--debug", action="store_true", help="Print debug info (paths, nb_frames)")
    return p.parse_args()


def resolve_bin(preferred: str, fallback_env: str, default: str) -> str:
    for cand in [preferred, os.environ.get(fallback_env), default]:
        if cand and shutil.which(cand):
            return cand
    raise SystemExit(f"Executable not found: {preferred} / ${fallback_env} / {default}")


def ffprobe_bin(ffmpeg_bin: str) -> Optional[str]:
    p = Path(ffmpeg_bin)
    cand = p.parent / "ffprobe" if p.parent else None
    if cand and cand.exists():
        return str(cand)
    auto = shutil.which("ffprobe")
    return auto


def probe_nb_frames(ffprobe: Optional[str], video: Path) -> Optional[int]:
    if not ffprobe:
        return None
    try:
        out = subprocess.check_output([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video),
        ], text=True).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def select_novel_dir(videos_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        d = videos_dir / explicit
        if not d.is_dir():
            raise SystemExit(f"Novel dir '{explicit}' not found under {videos_dir}")
        return d
    cands = sorted([p for p in videos_dir.iterdir() if p.is_dir() and p.name.startswith("novel_")])
    if not cands:
        raise SystemExit(f"No novel_* directory found inside {videos_dir}")
    return cands[-1]


def extract_frame(ffmpeg: str, src: Path, frame_idx: int, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # select=eq(n,frame_idx) picks the exact decoded frame; setsar=1 for square pixels.
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(src),
        "-vf", f"select='eq(n\,{frame_idx})',setsar=1",
        "-vframes", "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ffmpeg = resolve_bin(args.ffmpeg_bin, "FFMPEG_BIN", "ffmpeg")
    ffprobe = ffprobe_bin(ffmpeg)

    out_root = args.output_dir / args.clip_id
    methods = list(args.models)
    if "drivestudio-nus-gt" not in methods:
        methods.insert(0, "drivestudio-nus-gt")

    # Train view sources
    train_srcs: Dict[str, Tuple[Path, Optional[Path]]] = {}
    for m in methods:
        base = args.root / m / args.clip_id / "videos"
        rgb = base / f"full_set_{args.step}_rgbs.mp4"
        dpt = base / f"full_set_{args.step}_depths.mp4"
        if not rgb.is_file():
            print(f"[MISS] Train RGB not found: {rgb}")
            continue
        train_srcs[m] = (rgb, dpt if dpt.is_file() else None)

    # Novel view sources
    view_map: Dict[str, Dict[str, Path]] = {v: {} for v in args.views}
    view_depth_map: Dict[str, Dict[str, Path]] = {v: {} for v in args.views}
    for m in methods:
        videos_dir = args.root / m / args.clip_id / "videos"
        if not videos_dir.is_dir():
            print(f"[MISS] videos dir for {m}: {videos_dir}")
            continue
        try:
            novel_dir = select_novel_dir(videos_dir, args.novel_dir)
        except SystemExit as e:
            print(f"[MISS] {e}")
            continue
        files = {p.stem: p for p in novel_dir.glob("*.mp4")}
        # try both exact and relaxed matches
        for v in args.views:
            # exact
            if v in files:
                view_map[v][m] = files[v]
            else:
                # relaxed: case-insensitive and underscore-insensitive contains
                key = None
                vn = v.lower().replace("_", "")
                for stem, path in files.items():
                    sn = stem.lower().replace("_", "")
                    if vn == sn:
                        key = stem; break
                if key is None:
                    for stem, path in files.items():
                        sn = stem.lower().replace("_", "")
                        if vn in sn:
                            key = stem; break
                if key:
                    view_map[v][m] = files[key]
            # depth candidate names
            # common patterns: <view>_depths.mp4 or <view>_depth.mp4
            for suf in ("_depths", "_depth"):
                cand = novel_dir / f"{v}{suf}.mp4"
                if cand.is_file():
                    view_depth_map[v][m] = cand
                    break

    # Extract
    for m in methods:
        # Train RGB
        if m in train_srcs:
            rgb, dpt = train_srcs[m]
            dst_rgb = out_root / m / f"train_rgb_f{args.frame:04d}.png"
            try:
                nb = probe_nb_frames(ffprobe, rgb)
                if args.debug:
                    print(f"[DBG] {m} train rgb nb_frames={nb}")
                if nb is not None and args.frame >= nb:
                    print(f"[SKIP] {m} train rgb frame {args.frame} out of range (nb={nb})")
                else:
                    extract_frame(ffmpeg, rgb, args.frame, dst_rgb, args.force)
            except subprocess.CalledProcessError:
                print(f"[WARN] ffmpeg failed extracting train RGB for {m}")
            if dpt and dpt.is_file():
                dst_d = out_root / m / f"train_depth_f{args.frame:04d}.png"
                try:
                    nb = probe_nb_frames(ffprobe, dpt)
                    if args.debug:
                        print(f"[DBG] {m} train depth nb_frames={nb}")
                    if nb is not None and args.frame >= nb:
                        print(f"[SKIP] {m} train depth frame {args.frame} out of range (nb={nb})")
                    else:
                        extract_frame(ffmpeg, dpt, args.frame, dst_d, args.force)
                except subprocess.CalledProcessError:
                    print(f"[WARN] ffmpeg failed extracting train DEPTH for {m}")
        # Novel views
        for v in args.views:
            src = view_map.get(v, {}).get(m)
            if src is None:
                print(f"[MISS] {m} novel view '{v}' not found")
                continue
            dst_rgb = out_root / m / f"{v}_rgb_f{args.frame:04d}.png"
            try:
                nb = probe_nb_frames(ffprobe, src)
                if args.debug:
                    print(f"[DBG] {m} {v} rgb nb_frames={nb}")
                if nb is not None and args.frame >= nb:
                    print(f"[SKIP] {m} {v} rgb frame {args.frame} out of range (nb={nb})")
                else:
                    extract_frame(ffmpeg, src, args.frame, dst_rgb, args.force)
            except subprocess.CalledProcessError:
                print(f"[WARN] ffmpeg failed extracting novel RGB for {m}:{v}")
            dsrc = view_depth_map.get(v, {}).get(m)
            if dsrc and dsrc.is_file():
                dst_d = out_root / m / f"{v}_depth_f{args.frame:04d}.png"
                try:
                    nb = probe_nb_frames(ffprobe, dsrc)
                    if args.debug:
                        print(f"[DBG] {m} {v} depth nb_frames={nb}")
                    if nb is not None and args.frame >= nb:
                        print(f"[SKIP] {m} {v} depth frame {args.frame} out of range (nb={nb})")
                    else:
                        extract_frame(ffmpeg, dsrc, args.frame, dst_d, args.force)
                except subprocess.CalledProcessError:
                    print(f"[WARN] ffmpeg failed extracting novel DEPTH for {m}:{v}")

    print(f"[DONE] Wrote images under {out_root}")


if __name__ == "__main__":
    main()
