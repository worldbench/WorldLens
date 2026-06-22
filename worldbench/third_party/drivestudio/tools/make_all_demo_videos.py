#!/usr/bin/env python3
"""
Batch compose RGB/depth grid videos for many sequences, saved under data/.

It discovers sequences that exist for all specified models (and have both
RGB and depth videos), then calls:
  - tools/compose_train_view_grid.py (RGB)
  - tools/compose_depth_grid.py (Depth)

Default models: drivestudio-nus-(dist4d, drivedreamer2, opendwm, dreamforge, xscene, magicdrive)
Default step: 30000
Default output dir: data/demo_videos/all_6m/

Examples:
  python3 tools/make_all_demo_videos.py                      # discover ~150 seqs, make both RGB+Depth
  python3 tools/make_all_demo_videos.py --seqs 813 694 614   # only selected seqs
  python3 tools/make_all_demo_videos.py --step 40000         # use different training step suffix

You need ffmpeg installed (or set env FFMPEG_BIN).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Set


DEFAULT_MODELS = [
    "drivestudio-nus-dist4d",
    "drivestudio-nus-drivedreamer2",
    "drivestudio-nus-opendwm",
    "drivestudio-nus-dreamforge",
    "drivestudio-nus-xscene",
    "drivestudio-nus-magicdrive",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch compose RGB+Depth grid videos across many sequences.")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root directory containing model runs.")
    p.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS, help="Model directory names under root.")
    p.add_argument("--with-gt", action="store_true", help="Include drivestudio-nus-gt as the first model if present.")
    p.add_argument("--discover", type=str, choices=["union", "common"], default="union", help="Discover mode for sequences across models (default: union).")
    p.add_argument("--step", type=int, default=30000, help="Training step used in file naming (default: 30000).")
    p.add_argument("--seqs", type=str, nargs="*", default=None, help="Optional explicit list of clip IDs (e.g., 002 013 ...). If omitted, auto-discover by --discover mode.")
    p.add_argument("--output-dir", type=Path, default=Path("data/demo_videos/all_6m"), help="Directory to write outputs.")
    p.add_argument("--ffmpeg-bin", type=str, default=None, help="ffmpeg binary (overrides compose_* scripts' auto-detect).")
    p.add_argument("--font", type=Path, default=None, help="Optional TTF/OTF font path for labels (propagated).")
    p.add_argument("--fontsize", type=int, default=26, help="Label font size.")
    p.add_argument("--gap", type=int, default=8, help="Gap in pixels between tiles.")
    p.add_argument("--tile-width", type=int, default=0, help="Per-tile width; 0 disables resizing (default: 0).")
    p.add_argument("--tile-height", type=int, default=0, help="Per-tile height; 0 disables resizing (default: 0).")
    p.add_argument("--fit", type=str, choices=["contain", "cover"], default="contain", help="Resize mode when tile size is set: contain (scale+pad) or cover (scale+crop).")
    p.add_argument("--fps", type=int, default=0, help="Optional FPS override (0 = keep source).")
    p.add_argument("--frames", type=int, default=0, help="Optional trim to N frames (0 = keep full length).")
    p.add_argument("--debug", action="store_true", help="Pass --debug to compose scripts for alignment prints.")
    p.add_argument("--novel", action="store_true", help="Also compose novel-view grid (no metrics). Requires novel_* dirs.")
    p.add_argument("--dry-run", action="store_true", help="Only print planned commands, do not run.")
    return p.parse_args()


def has_both_videos(root: Path, model: str, seq: str, step: int) -> bool:
    base = root / model / seq / "videos"
    rgb = base / f"full_set_{step}_rgbs.mp4"
    dpt = base / f"full_set_{step}_depths.mp4"
    return rgb.is_file() and dpt.is_file()


def discover_seqs(root: Path, models: Sequence[str], step: int, mode: str) -> List[str]:
    found_sets: List[Set[str]] = []
    for m in models:
        seqs: Set[str] = set()
        model_dir = root / m
        if not model_dir.is_dir():
            continue
        for d in sorted(model_dir.iterdir()):
            if not d.is_dir():
                continue
            seq = d.name
            if has_both_videos(root, m, seq, step):
                seqs.add(seq)
        found_sets.append(seqs)
    if not found_sets:
        return []
    if mode == "common":
        acc = set.intersection(*found_sets)
    else:
        acc = set.union(*found_sets)
    return sorted(acc)


def run(cmd: List[str], dry_run: bool = False) -> int:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return 0
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[WARN] command failed (exit {e.returncode}):", " ".join(cmd))
        return e.returncode


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg presence hint (compose_* will also check internally)
    if args.ffmpeg_bin is None and shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        print("[WARN] ffmpeg not found in PATH; set --ffmpeg-bin or environment FFMPEG_BIN if compose_* fails.")

    models = list(args.models)
    if args.with_gt and "drivestudio-nus-gt" not in models:
        models.insert(0, "drivestudio-nus-gt")
    if args.seqs is None or len(args.seqs) == 0:
        seqs = discover_seqs(args.root, models, args.step, args.discover)
        if not seqs:
            raise SystemExit(f"No sequences discovered across models (mode={args.discover}).")
        print(f"[INFO] Discovered {len(seqs)} sequences (mode={args.discover}).")
    else:
        seqs = list(args.seqs)
        print(f"[INFO] Using user-provided {len(seqs)} sequences.")

    ffmpeg_args: List[str] = []
    if args.ffmpeg_bin:
        ffmpeg_args = ["--ffmpeg-bin", args.ffmpeg_bin]

    font_args: List[str] = []
    if args.font:
        font_args = ["--font", str(args.font), "--fontsize", str(args.fontsize)]
    else:
        font_args = ["--fontsize", str(args.fontsize)]
    # Always pass gap; pass tile args when provided (support height-only to mirror compose_* behavior)
    tile_args: List[str] = ["--gap", str(args.gap)]
    if args.tile_height and args.tile_height > 0:
        # Respect height-only mode; do not force a width to avoid introducing internal letterbox.
        tile_args += ["--tile-height", str(args.tile_height)]
    if args.tile_width and args.tile_width > 0:
        tile_args += ["--tile-width", str(args.tile_width)]
    if (args.tile_height and args.tile_height > 0) or (args.tile_width and args.tile_width > 0):
        tile_args += ["--fit", args.fit]
    fps_args: List[str] = []
    if args.fps and args.fps > 0:
        fps_args += ["--fps", str(args.fps)]
    if args.frames and args.frames > 0:
        fps_args += ["--frames", str(args.frames)]
    debug_args: List[str] = ["--debug"] if args.debug else []

    ok = 0
    failed = 0
    for seq in seqs:
        # Per-sequence available models (in case some method is missing videos for this seq)
        models_this_seq = [m for m in models if has_both_videos(args.root, m, seq, args.step)]
        if len(models_this_seq) == 0:
            print(f"[SKIP] {seq}: no available models with both RGB+Depth videos.")
            continue
        if len(models_this_seq) < len(models):
            print(f"[WARN] {seq}: missing some methods; composing with {len(models_this_seq)} models: {', '.join(models_this_seq)}")
        print(f"[RGB ] composing clip {seq}")
        # dynamic suffix reflecting model count and GT presence
        has_gt = args.with_gt and ("drivestudio-nus-gt" in models_this_seq)
        suffix = f"{len(models_this_seq)}m" + ("_gt" if has_gt else "")
        rc1 = run([
            "python3", "tools/compose_train_view_grid.py", seq,
            "--root", str(args.root),
            "--models", *models_this_seq,
            "--step", str(args.step),
            "--output", str(args.output_dir / f"clip_{seq}_train_grid_{suffix}.mp4"),
            *tile_args,
            *ffmpeg_args,
            *font_args,
            *fps_args,
            *debug_args,
        ], dry_run=args.dry_run)

        print(f"[DEPTH] composing clip {seq}")
        rc2 = run([
            "python3", "tools/compose_depth_grid.py", seq,
            "--root", str(args.root),
            "--models", *models_this_seq,
            "--step", str(args.step),
            "--output", str(args.output_dir / f"clip_{seq}_depth_grid_{suffix}.mp4"),
            *tile_args,
            *ffmpeg_args,
            *font_args,
            *fps_args,
            *debug_args,
        ], dry_run=args.dry_run)

        if rc1 == 0: ok += 1
        else: failed += 1
        if rc2 == 0: ok += 1
        else: failed += 1

        # Optional: compose novel-view grid (with GT if requested & available)
        if args.novel and args.with_gt and "drivestudio-nus-gt" in models_this_seq:
            print(f"[NOVEL+GT] composing clip {seq}")
            rc3 = run([
                "python3", "tools/compose_novel_view_grid.py", seq,
                "--root", str(args.root),
                "--models", *models_this_seq,
                "--output", str(args.output_dir / f"clip_{seq}_novel_grid_{suffix}.mp4"),
                "--fontsize", str(args.fontsize),
            ], dry_run=args.dry_run)
            if rc3 == 0: ok += 1
            else: failed += 1

    print(f"[DONE] Wrote videos to {args.output_dir} (ok={ok}, failed={failed})")


if __name__ == "__main__":
    main()
