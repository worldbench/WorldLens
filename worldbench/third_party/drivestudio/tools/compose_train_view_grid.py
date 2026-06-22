#!/usr/bin/env python3
"""
Compose a tiled RGB (training view) grid video across multiple methods for a given clip.

Source layout (per method/run):
  work_dirs/
    <model>/
      <clip_id>/
        videos/
          full_set_<step>.mp4

The script mirrors tools/compose_depth_grid.py but uses the RGB videos instead of depth.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compose RGB training-view grid video for a given nuScenes clip")
    p.add_argument("clip_id", type=str, help="Clip identifier (e.g., 084)")
    p.add_argument("--root", type=Path, default=Path("work_dirs"), help="Root directory containing per-model runs")
    p.add_argument("--models", type=str, nargs="*", help="Explicit list of model directory names; default: all containing the clip")
    p.add_argument("--step", type=int, default=30000, help="Training step used in file naming (default: 30000)")
    p.add_argument("--output", type=Path, default=None, help="Output MP4 path (default: work_dirs/grids/clip_<id>_train_grid.mp4)")
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
    p.add_argument("--debug", action="store_true", help="Print detailed alignment info (models, inputs, probed FPS/frames, metric values)")
    # Metric overlay options (LPIPS for RGB by default)
    p.add_argument("--metrics-json", type=Path, default=Path("work_dirs_analysis_metrics.json"), help="Path to metrics JSON for LPIPS overlay (default: work_dirs_analysis_metrics.json)")
    p.add_argument("--metric-key", type=str, default="image_metrics/full/lpips", help="Metric key in JSON to overlay (default: image_metrics/full/lpips)")
    p.add_argument("--metric-label", type=str, default="LPIPS", help="Display label for the metric (default: LPIPS)")
    p.add_argument("--no-metric", action="store_true", help="Disable metric overlay on labels")
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


def find_rgb_video(root: Path, model: str, clip_id: str, step: int) -> Path:
    base = root / model / clip_id / "videos"
    target = base / f"full_set_{step}_rgbs.mp4"
    if target.is_file():
        return target
    raise SystemExit(f"Missing RGB training video: {target}")


def grid_shape(n: int) -> Tuple[int, int]:
    # Prefer fewer rows to reduce perceived row spacing and placeholders at the last row.
    # Choose r=floor(sqrt(n)), c=ceil(n/r). Ensures r*c>=n and r<=c.
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
        return ""
    safe_text = text.replace(":", "\\:").replace("'", "\\'")
    return (
        f"drawtext=fontfile={font.as_posix()}:text='{safe_text}':fontcolor=white:fontsize={fontsize}"
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
    fps: int = 0,
    frames: int = 0,
    fit: str = "contain",
    crop: Tuple[int, int, int, int] = (0, 0, 0, 0),
    placeholder_height: int = 0,
    row_target_widths: Optional[List[int]] = None,
) -> Tuple[str, str]:
    if len(inputs) != len(labels):
        raise ValueError("inputs and labels length mismatch")
    cmds: List[str] = []
    out_labels: List[str] = []
    # Determine target tile size (letterbox to exact WxH if provided)
    th = tile_height if tile_height and tile_height > 0 else 0
    # Height-only mode (tw==0) will use scale=-2:th (no internal vertical padding). If tw>0, use contain/cover path.
    tw = tile_width if tile_width and tile_width > 0 else 0

    for i, lbl in enumerate(labels):
        filt = drawtext_filter(font, fontsize, lbl)
        out_name = f"v{i}"
        chain = f"[{i}:v]"
        if filt:
            chain += filt
        else:
            chain += "null"
        # Optional manual crop (remove baked-in bars): crop=iw-left-right:ih-top-bottom:left:top
        ct, cb, cl, cr = crop
        if any(v > 0 for v in (ct, cb, cl, cr)):
            chain += f",crop=iw-{cl}-{cr}:ih-{ct}-{cb}:{cl}:{ct}"
        if th:
            if fit == "cover":
                # Fill the tile: scale up to cover, then center-crop to exact WxH (no internal bars)
                chain += f",scale={tw if tw else -2}:{th}:force_original_aspect_ratio=increase"
                if tw:
                    chain += f",crop={tw}:{th}:(iw-{tw})/2:(ih-{th})/2"
            else:
                # contain: fit inside tile, then pad to exact WxH (may have bars)
                chain += f",scale={tw if tw else -2}:{th}:force_original_aspect_ratio=decrease"
                if tw:
                    chain += f",pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black"
        cmds.append(f"{chain}[{out_name}]")
        out_labels.append(out_name)

    total = rows * cols
    pads = 0
    while len(out_labels) < total:
        pad_label = f"pad{pads}"
        if th:
            # In height-only mode (tw==0) use a small placeholder width; rows will be padded to target width later.
            size_str = f"size={(tw if tw else 64)}x{th}"
        else:
            # match row video heights by using a reasonable placeholder height
            ph = placeholder_height if placeholder_height and placeholder_height > 0 else 16
            size_str = f"size=16x{ph}"
        if fps and fps > 0:
            cmds.append(f"color=c=black:{size_str}:rate={fps},format=rgb24[{pad_label}]")
        else:
            cmds.append(f"color=c=black:{size_str},format=rgb24[{pad_label}]")
        out_labels.append(pad_label)
        pads += 1

    row_outs: List[str] = []
    for r in range(rows):
        seg = out_labels[r * cols : (r + 1) * cols]
        padded: List[str] = []
        for c_idx, s in enumerate(seg):
            cur = s
            if gap > 0 and c_idx < cols - 1:
                pad_label = f"{s}_pad"
                cmds.append(f"[{s}]pad=iw+{gap}:ih:0:0:color=black[{pad_label}]")
                cur = pad_label
            padded.append(cur)
        tmp_row = f"row{r}_tmp"
        if len(padded) == 1:
            cmds.append(f"[{padded[0]}]null[{tmp_row}]")
        else:
            inputs_concat = "".join(f"[{s}]" for s in padded)
            cmds.append(f"{inputs_concat}hstack=inputs={len(padded)}:shortest=1[{tmp_row}]")
        # If using height-only mode (tw==0) and we computed target widths, right-pad the row to the exact width so vstack matches.
        row_label = f"row{r}"
        if tw == 0 and row_target_widths is not None and r < len(row_target_widths) and row_target_widths[r] > 0:
            target_w = row_target_widths[r]
            cmds.append(f"[{tmp_row}]pad={target_w}:ih:0:0:color=black[{row_label}_padw]")
            tmp_row = f"{row_label}_padw"
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
    root = args.root
    clip_id = args.clip_id
    models = discover_models(root, clip_id, args.models)
    inputs = [find_rgb_video(root, m, clip_id, args.step) for m in models]

    # Optional metric overlay per method from JSON
    metrics: Dict[str, Optional[float]] = {}
    if (not args.no_metric) and args.metrics_json and args.metrics_json.is_file():
        try:
            data = json.loads(args.metrics_json.read_text())
            for m in models:
                seq_map = data.get(m, {})
                val = None
                if isinstance(seq_map, dict):
                    clip_metrics = seq_map.get(clip_id, {})
                    if isinstance(clip_metrics, dict):
                        v = clip_metrics.get(args.metric_key)
                        if isinstance(v, (int, float)):
                            val = float(v)
                metrics[m] = val
        except Exception:
            metrics = {m: None for m in models}
    else:
        metrics = {m: None for m in models}

    labels = []
    for m in models:
        base = method_label(m)
        v = metrics.get(m)
        if v is not None:
            labels.append(f"{base} | {args.metric_label} {v:.3f}")
        else:
            labels.append(base)

    rows, cols = grid_shape(len(inputs))
    font = resolve_font_path(args.font)
    ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
    # detect source fps from the first input if user didn't override
    if args.fps > 0:
        fps_for_graph = args.fps
    else:
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg_bin)
            fps_detected = probe_fps(ffprobe_bin, inputs[0]) if inputs else None
            fps_for_graph = int(fps_detected) if fps_detected and fps_detected > 0 else 0
        except Exception:
            fps_for_graph = 0
    # Determine placeholder height when not resizing tiles
    ph = 0
    if (not args.tile_height or args.tile_height == 0) and inputs:
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg_bin)
            # probe dimensions via ffprobe
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

    # If only --tile-height is provided, switch to height-only mode (scale=-2:th) and pad each row to max width.
    tile_width_eff = args.tile_width or 0
    row_targets: Optional[List[int]] = None
    if (args.tile_height and args.tile_height > 0) and (not args.tile_width or args.tile_width == 0) and inputs:
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg_bin)
        except Exception:
            ffprobe_bin = None
        # probe widths/heights to predict scaled widths
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
        # compute scaled widths per input for height-only scaling; -2 makes width even
        sw: List[int] = []
        th = args.tile_height
        for (w, h) in wh_list:
            if w > 0 and h > 0:
                ww = int(round(w * (th / float(h))))
                if ww % 2 != 0:
                    ww += 1
            else:
                ww = 1280  # reasonable fallback
            sw.append(ww)
        # derive row target widths
        row_widths: List[int] = []
        for r in range(rows):
            seg = sw[r * cols : (r + 1) * cols]
            while len(seg) < cols:
                seg.append(64)
            row_widths.append(sum(seg) + (args.gap * (cols - 1 if args.gap > 0 else 0)))
        max_w = max(row_widths) if row_widths else 0
        row_targets = [max_w for _ in range(rows)]

    filter_str, final_label = build_filter_complex(
        inputs, labels, rows, cols, font, args.fontsize, args.gap,
        args.tile_height, tile_width_eff, fps=fps_for_graph, frames=args.frames, fit=args.fit,
        crop=(args.crop_top, args.crop_bottom, args.crop_left, args.crop_right), placeholder_height=ph,
        row_target_widths=row_targets)

    # ffmpeg_bin already resolved above
    if args.output is None:
        out_dir = root / "grids"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"clip_{clip_id}_train_grid.mp4"
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)

    # Debug printout: alignment verification
    if args.debug:
        print("[DEBUG] clip:", clip_id, "step:", args.step)
        print("[DEBUG] grid:", f"rows={rows} cols={cols} gap={args.gap} tile=({('auto' if tile_width_eff==0 else tile_width_eff)}x{args.tile_height or 'auto'})")
        print("[DEBUG] models order:", ", ".join(models))
        try:
            ffprobe_bin = resolve_ffprobe(ffmpeg_bin)
        except Exception:
            ffprobe_bin = None
        for m, path, lab in zip(models, inputs, labels):
            fps = probe_fps(ffprobe_bin, path) if ffprobe_bin else None
            nbf = probe_nb_frames(ffprobe_bin, path) if ffprobe_bin else None
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
            print(f"[DEBUG] {m}: video={path} size={wh} sar={sar} dar={dar} fps={fps} frames={nbf} label=\"{lab}\"")
        # Metric source for RGB
        print("[DEBUG] metric source:", args.metrics_json, "key:", args.metric_key)

    cmd = build_ffmpeg_cmd(ffmpeg_bin, inputs, filter_str, final_label, output, args.codec, args.bitrate)

    if args.verbose:
        print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
