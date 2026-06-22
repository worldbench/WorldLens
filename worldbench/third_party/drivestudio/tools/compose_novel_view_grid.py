#!/usr/bin/env python3
"""
Compose a grid video of novel-view renderings across multiple models.
Each row corresponds to a novel trajectory (e.g., s_curve) and
each column corresponds to a model (including the GT reference).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import os
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a model-by-view grid video for a given clip."
    )
    parser.add_argument("clip_id", type=str, help="Clip identifier (e.g., 084).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("work_dirs"),
        help="Root directory containing per-model results (default: work_dirs).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Explicit list of model directory names. Defaults to all models that contain the clip.",
    )
    parser.add_argument(
        "--novel-dir",
        type=str,
        default=None,
        help="Name of the novel-view directory (e.g., novel_30000). Defaults to the latest 'novel_*' folder found per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to work_dirs/grids/clip_<id>_novel_grid.mp4.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Font file for drawtext. If omitted, the script will attempt to auto-detect a suitable TrueType/OpenType font.",
    )
    parser.add_argument(
        "--fontsize", type=int, default=28, help="Font size for labels (default: 28)."
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Preferred FFmpeg video encoder (e.g., libx264, libopenh264). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="10M",
        help="Fallback bitrate when CRF is unavailable (default: 10M).",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="FFmpeg executable to use (default: ffmpeg).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed ffmpeg command."
    )
    return parser.parse_args()


CANDIDATE_FONT_NAMES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Regular.ttf",
    "LiberationSans-Regular.ttf",
    "NotoSans-Regular.ttf",
]

MODEL_NAME_MAP = {
    "drivestudio-nus-gt": "GT",
    "drivestudio-nus-dreamforge": "dreamforge",
    "drivestudio-nus-drivedreamer2": "drivedreamer2",
    "drivestudio-nus-magicdrive": "magicdrive",
}


def resolve_ffmpeg_binary(preferred: str) -> str:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        candidates.append(env_bin)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(str(Path(conda_prefix) / "bin" / "ffmpeg"))
    candidates.append("ffmpeg")

    for cand in candidates:
        if shutil.which(cand):
            return cand
    raise SystemExit(
        "Could not find an FFmpeg executable. Consider installing ffmpeg or set FFMPEG_BIN/path via --ffmpeg-bin."
    )


def detect_available_codecs(binary: str) -> List[str]:
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []
    encoders = []
    for line in result.stdout.splitlines():
        if not line or line.startswith(" "):
            continue
        encoders.append(line.strip())
    # actual encoder names are at fixed position (last token)
    available = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if name.startswith("lib") or name in {"h264", "mpeg4"}:
            available.append(parts[-1])
    return available


def resolve_font_path(user_font: Optional[Path]) -> Path:
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

    raise SystemExit(
        "No suitable font file found. Please install a TrueType/OpenType font or provide --font."
    )


def choose_codec(binary: str, user_codec: Optional[str]) -> str:
    if user_codec:
        return user_codec
    preferred_order = ["libx264", "libopenh264", "libx265", "mpeg4"]
    available = detect_available_codecs(binary)
    for codec in preferred_order:
        if codec in available:
            return codec
    # fallback to ffmpeg default (no codec flag)
    return ""


def discover_models(root: Path, clip_id: str) -> List[str]:
    models = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        clip_dir = model_dir / clip_id
        if clip_dir.is_dir():
            models.append(model_dir.name)
    if not models:
        raise SystemExit(f"No model directories containing clip '{clip_id}' were found under {root}.")
    # Prefer GT first if present
    models.sort()
    if "drivestudio-nus-gt" in models:
        models.remove("drivestudio-nus-gt")
        models.insert(0, "drivestudio-nus-gt")
    return models


def select_novel_dir(clip_dir: Path, explicit: str | None) -> Path:
    videos_dir = clip_dir / "videos"
    if not videos_dir.is_dir():
        raise SystemExit(f"No videos directory found at {videos_dir}")
    if explicit:
        candidate = videos_dir / explicit
        if not candidate.is_dir():
            raise SystemExit(f"Specified novel directory '{explicit}' not found at {candidate}")
        return candidate
    candidates = sorted(p for p in videos_dir.iterdir() if p.is_dir() and p.name.startswith("novel_"))
    if not candidates:
        raise SystemExit(f"No novel_* directory found inside {videos_dir}")
    return candidates[-1]


def load_novel_videos(novel_dir: Path) -> Dict[str, Path]:
    videos = {}
    for vid in sorted(novel_dir.glob("*.mp4")):
        if vid.is_file():
            videos[vid.stem] = vid
    if not videos:
        raise SystemExit(f"No MP4 files found in {novel_dir}")
    return videos


def prettify_model_name(name: str) -> str:
    label = MODEL_NAME_MAP.get(name, name)
    if label == name and name.startswith("drivestudio-nus-"):
        label = name.split("drivestudio-nus-", 1)[1]
    return label.replace("_", " ").replace("-", " ")


def prettify_view_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def sanitize_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
    )


def build_filter_complex(
    inputs: List[Path],
    models: Sequence[str],
    views: Sequence[str],
    font: Path | None,
    fontsize: int,
) -> Tuple[str, str]:
    num_models = len(models)
    filter_cmds: List[str] = []
    stream_labels: List[str] = []

    if font is None:
        raise SystemExit(
            "No font file available for drawtext. Please provide --font pointing to a .ttf or .otf file."
        )
    font_opt = f"fontfile={font.as_posix()}:"

    for idx, (view, model) in enumerate([(v, m) for v in views for m in models]):
        col = idx % num_models
        row = idx // num_models
        in_label = f"{idx}:v"
        current = f"{in_label}"
        filters: List[str] = []

        model_label = prettify_model_name(model)
        model_text = sanitize_text(model_label)
        filters.append(
            f"drawtext={font_opt}text='{model_text}':fontcolor=white:fontsize={fontsize}"
            f":box=1:boxcolor=black@0.6:boxborderw=10:x=(w-tw)/2:y=20"
        )

        if col == 0:
            view_label = prettify_view_name(view)
            view_text = sanitize_text(view_label)
            filters.append(
                f"drawtext={font_opt}text='{view_text}':fontcolor=white:fontsize={fontsize}"
                f":box=1:boxcolor=black@0.6:boxborderw=10:x=20:y=h-th-40"
            )

        prev = f"[{in_label}]"
        final_label = f"v{idx}"
        if filters:
            for f_idx, filt in enumerate(filters):
                out_label = final_label if f_idx == len(filters) - 1 else f"v{idx}_{f_idx}"
                filter_cmds.append(f"{prev}{filt}[{out_label}]")
                prev = f"[{out_label}]"
        else:
            filter_cmds.append(f"{prev}null[{final_label}]")
        stream_labels.append(final_label)

    row_labels: List[str] = []
    for r, view in enumerate(views):
        cols = stream_labels[r * num_models : (r + 1) * num_models]
        if len(cols) == 1:
            out_label = f"row{r}"
            filter_cmds.append(f"[{cols[0]}]null[{out_label}]")
            row_labels.append(out_label)
        else:
            out_label = f"row{r}"
            inputs_concat = "".join(f"[{label}]" for label in cols)
            filter_cmds.append(f"{inputs_concat}hstack=inputs={len(cols)}[{out_label}]")
            row_labels.append(out_label)

    if len(row_labels) == 1:
        final_label = row_labels[0]
    else:
        final_label = "grid"
        inputs_concat = "".join(f"[{label}]" for label in row_labels)
        filter_cmds.append(f"{inputs_concat}vstack=inputs={len(row_labels)}[{final_label}]")

    filter_complex = ";".join(filter_cmds)
    return filter_complex, final_label


def build_ffmpeg_command(
    binary: str,
    inputs: List[Path],
    filter_complex: str,
    final_label: str,
    output: Path,
    codec: str,
    bitrate: str,
) -> List[str]:
    cmd = [binary, "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{final_label}]",
        ]
    )
    if codec:
        cmd.extend(["-c:v", codec])
        if codec == "libx264":
            cmd.extend(["-preset", "medium", "-crf", "18"])
        elif codec.startswith("libx26"):
            cmd.extend(["-preset", "medium", "-crf", "20"])
        elif codec in {"libopenh264", "mpeg4"}:
            cmd.extend(["-b:v", bitrate])
    cmd.extend(["-pix_fmt", "yuv420p", "-an", str(output)])
    return cmd


def main() -> None:
    args = parse_args()
    ffmpeg_bin = resolve_ffmpeg_binary(args.ffmpeg_bin)

    clip_id = args.clip_id.strip()
    models = args.models if args.models else discover_models(args.root, clip_id)

    per_model_videos: Dict[str, Dict[str, Path]] = {}
    per_model_paths: Dict[str, Path] = {}
    for model in models:
        clip_dir = args.root / model / clip_id
        if not clip_dir.is_dir():
            raise SystemExit(f"Clip '{clip_id}' not found under {clip_dir.parent}")
        novel_dir = select_novel_dir(clip_dir, args.novel_dir)
        per_model_paths[model] = novel_dir
        per_model_videos[model] = load_novel_videos(novel_dir)

    common_views = set.intersection(*(set(v.keys()) for v in per_model_videos.values()))
    if not common_views:
        raise SystemExit("No common novel-view videos across the selected models.")
    view_list = sorted(common_views)

    inputs: List[Path] = []
    for view in view_list:
        for model in models:
            inputs.append(per_model_videos[model][view])

    output = args.output
    if output is None:
        output_dir = args.root / "grids"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"clip_{clip_id}_novel_grid.mp4"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font_path(args.font)

    if args.verbose:
        print(f"Using font: {font_path}")

    filter_complex, final_label = build_filter_complex(
        inputs=inputs,
        models=models,
        views=view_list,
        font=font_path,
        fontsize=args.fontsize,
    )

    codec = choose_codec(ffmpeg_bin, args.codec)
    if codec == "":
        if args.verbose:
            print("No specific encoder selected; FFmpeg default will be used.")
    cmd = build_ffmpeg_command(ffmpeg_bin, inputs, filter_complex, final_label, output, codec, args.bitrate)

    if args.verbose:
        print("Running ffmpeg command:")
        print(" ".join(cmd))

    subprocess.run(cmd, check=True)
    print(f"Saved grid video to {output}")
    print("Models:", ", ".join(models))
    print("Views:", ", ".join(view_list))


if __name__ == "__main__":
    main()
