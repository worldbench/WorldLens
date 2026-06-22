#!/usr/bin/env python3
"""
Compose a 2x2 grid depth visualization video across multiple methods for a given clip.

Inputs (per method/run):
  work_dirs/
    <model>/
      <clip_id>/
        videos/
          full_set_<step>_depths.mp4

Output:
  A single MP4 that tiles the per-method depth videos in a 2x2 layout (for 4 methods).
  If the number of methods differs, the script lays them out in a near-square grid.

This follows the style of tools/compose_novel_view_grid.py, but specializes to depth videos.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict
import csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compose depth grid video for a given nuScenes clip")
    p.add_argument("clip_id", type=str, help="Clip identifier (e.g., 084)")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root directory containing per-model runs")
    p.add_argument("--models", type=str, nargs="*", help="Explicit list of model directory names; default: all that contain the clip")
    p.add_argument("--step", type=int, default=30000, help="Training step used in file naming (default: 30000)")
    p.add_argument("--output", type=Path, default=None, help="Output MP4 path (default: work_dirs/grids/clip_<id>_depth_grid.mp4)")
    p.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="FFmpeg executable (default: ffmpeg)")
    p.add_argument("--codec", type=str, default=None, help="Preferred FFmpeg encoder (e.g., libx264)")
    p.add_argument("--bitrate", type=str, default="10M", help="Bitrate for bitrate-based encoders (default: 10M)")
    p.add_argument("--font", type=Path, default=None, help="Optional TrueType/OpenType font for labels")
    p.add_argument("--fontsize", type=int, default=28, help="Label font size (default: 28)")
    p.add_argument("--gap", type=int, default=8, help="Gap (pixels) between tiles (default: 8)")
    p.add_argument("--tile-height", type=int, default=0, help="Optional per-tile fixed height (px); 0 disables scaling.")
    p.add_argument("--tile-width", type=int, default=0, help="Optional per-tile fixed width (px); when height or width >0, tiles are resized to this WxH.")
    p.add_argument("--fit", type=str, choices=["contain", "cover"], default="contain", help="Resize mode when both tile width/height are set: contain (scale+pad) or cover (scale+crop). Default: contain.")
    # Optional manual crop to remove baked-in letterbox bars before resizing (pixels on source frames)
    p.add_argument("--crop-top", type=int, default=0, help="Crop N pixels from top before resizing (default: 0)")
    p.add_argument("--crop-bottom", type=int, default=0, help="Crop N pixels from bottom before resizing (default: 0)")
    p.add_argument("--crop-left", type=int, default=0, help="Crop N pixels from left before resizing (default: 0)")
    p.add_argument("--crop-right", type=int, default=0, help="Crop N pixels from right before resizing (default: 0)")
    p.add_argument("--verbose", action="store_true", help="Print ffmpeg command")
    p.add_argument("--fps", type=int, default=0, help="Target output FPS override; 0 = keep source FPS (default)")
    p.add_argument("--frames", type=int, default=0, help="Trim output to this many frames after stacking (0=disable)")
    # Metric overlay (AbsRel for Depth by default)
    p.add_argument("--depth-per-seq", type=Path, nargs="*", default=None, help="Per-sequence depth CSV(s) (per_seq.csv) for AbsRel overlay. If omitted, tries depth_tables_3/per_seq.csv then depth_tables_2/per_seq.csv.")
    p.add_argument("--metric-label", type=str, default="AbsRel", help="Display label for the metric (default: AbsRel)")
    p.add_argument("--no-metric", action="store_true", help="Disable metric overlay on labels")
    p.add_argument("--debug", action="store_true", help="Print detailed alignment info (models, inputs, probed FPS/frames, metric values)")
    return p.parse_args()


def resolve_ffmpeg(binary: str) -> str:
    for cand in [binary, os.environ.get("FFMPEG_BIN"), "ffmpeg"]:
        if cand and shutil.which(cand):
            return cand
    raise SystemExit("FFmpeg not found; install ffmpeg or set --ffmpeg-bin/FFMPEG_BIN")


def resolve_ffprobe(ffmpeg_bin: str) -> str:
    p = Path(ffmpeg_bin)
    cand = p.parent / "ffprobe" if p.parent else None
    if cand and cand.exists():
        return str(cand)
    probed = shutil.which("ffprobe")
    if probed:
        return probed
    raise SystemExit("ffprobe not found; install ffmpeg (includes ffprobe) or add to PATH")


def probe_fps(ffprobe_bin: str, video: Path) -> Optional[float]:
    try:
        out = subprocess.check_output([
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1",
            str(video),
        ], text=True)
        fr = out.strip()
        if not fr:
            return None
        if "/" in fr:
            num, den = fr.split("/", 1)
            num = float(num); den = float(den)
            return num / den if den else None
        return float(fr)
    except Exception:
        return None


def probe_nb_frames(ffprobe_bin: str, video: Path) -> Optional[int]:
    try:
        out = subprocess.check_output([
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video),
        ], text=True)
        s = out.strip()
        return int(s) if s.isdigit() else None
    except Exception:
        return None


CANDIDATE_FONT_NAMES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Regular.ttf",
    "LiberationSans-Regular.ttf",
    "NotoSans-Regular.ttf",
]


def resolve_font_path(user_font: Optional[Path]) -> Optional[Path]:
    """Try best-effort to locate a usable TrueType/OpenType font for drawtext.

    Returns a Path if found; otherwise None (caller may choose to disable labels).
    """
    if user_font:
        if user_font.is_file():
            return user_font
        raise SystemExit(f"Specified font file '{user_font}' does not exist.")

    env_font = os.environ.get("FFMPEG_FONT")
    if env_font and Path(env_font).is_file():
        return Path(env_font)

    candidates: List[Path] = []
    system_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]
    for base in system_dirs:
        for name in CANDIDATE_FONT_NAMES:
            candidates.append(base / "truetype" / name)
            candidates.append(base / "truetype" / "dejavu" / name)
            candidates.append(base / name)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cp = Path(conda_prefix)
        for name in CANDIDATE_FONT_NAMES:
            candidates.append(cp / "lib" / "python3.10" / "site-packages" / name)
            candidates.append(cp / "lib" / "python3.10" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf" / name)
        pkgs_dir = cp / "pkgs"
        if pkgs_dir.is_dir():
            for name in CANDIDATE_FONT_NAMES:
                candidates.extend(pkgs_dir.glob(f"pillow*/info/test/Tests/fonts/{name}"))

    for cand in candidates:
        if cand.is_file() and "Symbols" not in cand.name and "Color" not in cand.name:
            return cand

    return None


def discover_models(root: Path, clip_id: str, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return list(explicit)
    models = []
    for d in sorted(root.iterdir()):
        if (d / clip_id).is_dir():
            models.append(d.name)
    if not models:
        raise SystemExit(f"No models containing clip '{clip_id}' found under {root}")
    models.sort()
    if "drivestudio-nus-gt" in models:
        models.remove("drivestudio-nus-gt")
        models.insert(0, "drivestudio-nus-gt")
    return models


def method_label(name: str) -> str:
    return name.split("drivestudio-nus-", 1)[-1].replace("_", " ").replace("-", " ")


def find_depth_video(root: Path, model: str, clip_id: str, step: int) -> Path:
    path = root / model / clip_id / "videos" / f"full_set_{step}_depths.mp4"
    if not path.is_file():
        raise SystemExit(f"Missing depth video: {path}")
    return path


def grid_shape(n: int) -> Tuple[int, int]:
    # Prefer fewer rows to reduce perceived row spacing. Use r=floor(sqrt(n)), c=ceil(n/r).
    if n <= 0:
        return 0, 0
    r = int(n ** 0.5)
    if r < 1:
        r = 1
    from math import ceil
    c = int(ceil(n / r))
    return r, c


def drawtext_filter(font: Optional[Path], fontsize: int, text: str) -> str:
    if not font:
        return ""  # no labeling when no font is available
    return (
        f"drawtext=fontfile={font.as_posix()}:text='{text}':fontcolor=white:fontsize={fontsize}"
        ":box=1:boxcolor=black@0.6:boxborderw=10:x=20:y=20"
    )


def build_filter_complex(
    inputs: List[Path],
    labels: List[str],
    rows: int,
    cols: int,
    font: Optional[Path],
    fontsize: int,
    gap: int,
    tile_height: int,
    tile_width: int,
    fps: int = 30,
    frames: int = 0,
    fit: str = "contain",
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    placeholder_height: int = 0,
    row_target_widths: Optional[List[int]] = None,
) -> Tuple[str, str]:
    if len(inputs) != len(labels):
        raise ValueError("inputs and labels length mismatch")
    n = len(inputs)
    cmds: List[str] = []
    out_labels: List[str] = []
    th = tile_height if tile_height and tile_height > 0 else 0
    # Height-only mode (tw==0) will use scale=-2:th (no internal vertical padding). If tw>0, use contain/cover path.
    tw = tile_width if tile_width and tile_width > 0 else 0

    for i in range(n):
        label_text = labels[i].replace(":", "\\:").replace("'", "\\'")
        filt = drawtext_filter(font, fontsize, label_text)
        out_name = f"v{i}"
        chain = f"[{i}:v]"
        if filt:
            chain += filt
        else:
            chain += "null"
        ct, cb, cl, cr = crop
        if any(v > 0 for v in (ct, cb, cl, cr)):
            chain += f",crop=iw-{cl}-{cr}:ih-{ct}-{cb}:{cl}:{ct}"
        if th:
            if fit == "cover":
                chain += f",scale={tw if tw else -2}:{th}:force_original_aspect_ratio=increase"
                if tw:
                    chain += f",crop={tw}:{th}:(iw-{tw})/2:(ih-{th})/2"
            else:
                chain += f",scale={tw if tw else -2}:{th}:force_original_aspect_ratio=decrease"
                if tw:
                    chain += f",pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black"
        cmds.append(f"{chain}[{out_name}]")
        out_labels.append(out_name)

    # pad with placeholder frames if necessary
    pads = 0
    total = rows * cols
    while len(out_labels) < total:
        pad_label = f"pad{pads}"
        if th:
            size_str = f"size={(tw if tw else 64)}x{th}"
        else:
            ph = placeholder_height if placeholder_height and placeholder_height > 0 else 16
            size_str = f"size=16x{ph}"
        if fps and fps > 0:
            cmds.append(f"color=c=black:{size_str}:rate={fps},format=rgb24[{pad_label}]")
        else:
            cmds.append(f"color=c=black:{size_str},format=rgb24[{pad_label}]")
        out_labels.append(pad_label)
        pads += 1

    # stack per row with hstack, then vstack rows
    row_outs: List[str] = []
    for r in range(rows):
        seg = out_labels[r * cols : (r + 1) * cols]
        # pad right gap for all but last column to create horizontal spacing
        padded_cols: List[str] = []
        for c_idx, s in enumerate(seg):
            cur = s
            if gap > 0 and c_idx < cols - 1:
                pad_label = f"{s}_pad"
                cmds.append(f"[{s}]pad=iw+{gap}:ih:0:0:color=black[{pad_label}]")
                cur = pad_label
            padded_cols.append(cur)
        tmp_row = f"row{r}_tmp"
        if len(padded_cols) == 1:
            cmds.append(f"[{padded_cols[0]}]null[{tmp_row}]")
        else:
            inputs_concat = "".join(f"[{s}]" for s in padded_cols)
            cmds.append(f"{inputs_concat}hstack=inputs={len(padded_cols)}:shortest=1[{tmp_row}]")
        # pad to target row width if in height-only mode (tw==0) and provided
        row_label = f"row{r}"
        if tw == 0 and row_target_widths is not None and r < len(row_target_widths) and row_target_widths[r] > 0:
            target_w = row_target_widths[r]
            cmds.append(f"[{tmp_row}]pad={target_w}:ih:0:0:color=black[{row_label}_padw]")
            tmp_row = f"{row_label}_padw"
        # pad bottom gap for all but last row to create vertical spacing
        if gap > 0 and r < rows - 1:
            cmds.append(f"[{tmp_row}]pad=iw:ih+{gap}:0:0:color=black[{row_label}]")
        else:
            cmds.append(f"[{tmp_row}]null[{row_label}]")
        row_outs.append(row_label)

    if len(row_outs) == 1:
        final = row_outs[0]
    else:
        final = "grid"
        inputs_concat = "".join(f"[{s}]" for s in row_outs)
        cmds.append(f"{inputs_concat}vstack=inputs={len(row_outs)}:shortest=1[{final}]")

    tail_parts = []
    if fps and fps > 0:
        tail_parts.append(f"fps={fps}")
    if frames and frames > 0:
        tail_parts.append(f"trim=end_frame={frames}")
    tail_parts.append("setsar=1")
    tail = ",".join(tail_parts)
    cmds.append(f"[{final}]{tail}[outv]")
    return ";".join(cmds), "outv"


def build_ffmpeg_cmd(binary: str, inputs: List[Path], filter_complex: str, final_label: str, output: Path, codec: Optional[str], bitrate: str) -> List[str]:
    cmd: List[str] = [binary, "-y"]
    for p in inputs:
        cmd.extend(["-i", str(p)])
    cmd.extend(["-filter_complex", filter_complex, "-map", f"[{final_label}]", "-pix_fmt", "yuv420p", "-vsync", "2", "-shortest", "-an"])
    if codec:
        cmd.extend(["-c:v", codec])
        if codec == "libx264":
            cmd.extend(["-preset", "medium", "-crf", "18"])
        elif codec in {"libopenh264", "mpeg4"}:
            cmd.extend(["-b:v", bitrate])
    cmd.append(str(output))
    return cmd


def main() -> None:
    args = parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg_bin)
    models = discover_models(args.root, args.clip_id, args.models)

    inputs: List[Path] = []
    labels: List[str] = []
    # Prepare metric overlay map: method -> value for this clip
    metric_map: Dict[str, Optional[float]] = {m: None for m in models}
    if not args.no_metric:
        csv_paths: List[Path] = []
        if args.depth_per_seq:
            csv_paths = [p for p in args.depth_per_seq if p and p.is_file()]
        else:
            for cand in [Path("depth_tables_3/per_seq.csv"), Path("depth_tables_2/per_seq.csv"), Path("depth_tables/per_seq.csv")]:
                if cand.is_file():
                    csv_paths.append(cand)
        # Load rows and build lookup
        lookup: Dict[Tuple[str, str], float] = {}
        for path in csv_paths:
            try:
                with path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    if "Method/Seq" in reader.fieldnames and "AbsRel" in reader.fieldnames:
                        for row in reader:
                            label = row.get("Method/Seq", "").strip()
                            absrel = row.get("AbsRel")
                            if "/" in label and absrel:
                                method, seq = label.split("/", 1)
                                try:
                                    v = float(absrel)
                                except ValueError:
                                    continue
                                lookup[(method.strip(), seq.strip())] = v
            except Exception:
                continue
        for m in models:
            meth = method_label(m).replace(" ", "")  # our labels may have spaces removed elsewhere
            key1 = (meth, args.clip_id)
            key2 = (meth.lower(), args.clip_id)
            if key1 in lookup:
                metric_map[m] = lookup[key1]
            elif key2 in lookup:
                metric_map[m] = lookup[key2]

    for m in models:
        vid = find_depth_video(args.root, m, args.clip_id, args.step)
        inputs.append(vid)
        base = method_label(m)
        v = metric_map.get(m)
        if v is not None:
            labels.append(f"{base} | {args.metric_label} {v:.3f}")
        else:
            labels.append(base)
    rows, cols = grid_shape(len(inputs))

    out_path = args.output
    if out_path is None:
        out_dir = args.root / "grids"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"clip_{args.clip_id}_depth_grid.mp4"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font_path(args.font)
    if args.font and not font_path:
        raise SystemExit(f"Font file not found: {args.font}")

    # Determine placeholder height when not resizing tiles
    ph = 0
    if (not args.tile_height or args.tile_height == 0) and inputs:
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg)
            info = subprocess.check_output([
                ffprobe_bin,
                "-v","error",
                "-select_streams","v:0",
                "-show_entries","stream=height",
                "-of","csv=p=0",
                str(inputs[0])
            ], text=True).strip()
            ph = int(info) if info.isdigit() else 0
        except Exception:
            ph = 0
    # Height-only mode: compute per-row target widths so rows align and avoid vertical bars
    tile_width_eff = args.tile_width or 0
    row_targets: Optional[List[int]] = None
    if (args.tile_height and args.tile_height > 0) and (not args.tile_width or args.tile_width == 0) and inputs:
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg)
        except Exception:
            ffprobe_bin = None
        wh_list: List[Tuple[int,int]] = []
        for pth in inputs:
            w = h = 0
            if ffprobe_bin:
                try:
                    s = subprocess.check_output([
                        ffprobe_bin,
                        "-v","error",
                        "-select_streams","v:0",
                        "-show_entries","stream=width,height",
                        "-of","csv=p=0",
                        str(pth)
                    ], text=True).strip()
                    parts = s.split(',')
                    if len(parts) >= 2:
                        w = int(parts[0]); h = int(parts[1])
                except Exception:
                    w = h = 0
            wh_list.append((w, h))
        sw: List[int] = []
        th = args.tile_height
        for (w, h) in wh_list:
            if w > 0 and h > 0:
                ww = int(round(w * (th / float(h))))
                if ww % 2 != 0:
                    ww += 1
            else:
                ww = 1280
            sw.append(ww)
        row_widths: List[int] = []
        for r in range(rows):
            seg = sw[r * cols : (r + 1) * cols]
            while len(seg) < cols:
                seg.append(64)
            row_widths.append(sum(seg) + (args.gap * (cols - 1 if args.gap > 0 else 0)))
        max_w = max(row_widths) if row_widths else 0
        row_targets = [max_w for _ in range(rows)]

    filter_complex, final_label = build_filter_complex(
        inputs, labels, rows, cols, font_path, args.fontsize, args.gap,
        args.tile_height, tile_width_eff, fps=args.fps, frames=args.frames, fit=args.fit,
        crop=(args.crop_top, args.crop_bottom, args.crop_left, args.crop_right), placeholder_height=ph,
        row_target_widths=row_targets)
    # Debug alignment info
    if args.debug:
        print("[DEBUG] clip:", args.clip_id, "step:", args.step)
        print("[DEBUG] grid:", f"rows={rows} cols={cols} gap={args.gap} tile=({('auto' if tile_width_eff==0 else tile_width_eff)}x{args.tile_height or 'auto'})")
        print("[DEBUG] models order:", ", ".join(models))
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg)
        except Exception:
            ffprobe_bin = None
        for m, path, lab in zip(models, inputs, labels):
            fps_val = probe_fps(ffprobe_bin, path) if ffprobe_bin else None
            frames_val = probe_nb_frames(ffprobe_bin, path) if ffprobe_bin else None
            wh = None
            sar = dar = None
            if ffprobe_bin:
                try:
                    whs = subprocess.check_output([
                        ffprobe_bin,
                        "-v","error",
                        "-select_streams","v:0",
                        "-show_entries","stream=width,height,sample_aspect_ratio,display_aspect_ratio",
                        "-of","csv=p=0",
                        str(path)
                    ], text=True).strip()
                    parts = whs.split('\n')[0].split(',')
                    if len(parts) >= 2:
                        wh = f"{parts[0]}x{parts[1]}"
                    if len(parts) >= 3:
                        sar = parts[2]
                    if len(parts) >= 4:
                        dar = parts[3]
                except Exception:
                    pass
            print(f"[DEBUG] {m}: video={path} size={wh} sar={sar} dar={dar} fps={fps_val} frames={frames_val} label=\"{lab}\"")
        # Metric source for Depth
        print("[DEBUG] metric source candidates:", end=" ")
        if args.depth_per_seq:
            print(", ".join(str(p) for p in args.depth_per_seq))
        else:
            print("depth_tables_3/per_seq.csv, depth_tables_2/per_seq.csv (fallback)")
    cmd = build_ffmpeg_cmd(ffmpeg, inputs, filter_complex, final_label, out_path, args.codec, args.bitrate)
    if args.verbose:
        print("FFmpeg command:")
        print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved depth grid video to {out_path}")


if __name__ == "__main__":
    main()
