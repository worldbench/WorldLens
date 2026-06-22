#!/usr/bin/env python3
"""Prepare WorldLens Reconstruction assets with DriveStudio."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping


OPTION_FLAGS = {
    "work_root": "--work-root",
    "data_root": "--data-root",
    "target_dir": "--target-dir",
    "config_file": "--config-file",
    "step": "--step",
    "fps": "--fps",
    "gt_method": "--gt-method",
    "cameras": "--cameras",
}

PATH_OPTIONS = {"work_root", "data_root", "target_dir"}

BOOLEAN_FLAGS = {
    "skip_preprocess": "--skip-preprocess",
    "skip_train": "--skip-train",
    "skip_eval": "--skip-eval",
    "dry_run": "--dry-run",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare WorldLens Reconstruction assets with DriveStudio")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--reconstruction-root", default="generated_results/reconstruction")
    parser.add_argument(
        "--drivestudio-root",
        default=os.environ.get("DRIVESTUDIO_ROOT"),
        help="DriveStudio source root. Defaults to worldbench/third_party/drivestudio.",
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("DRIVESTUDIO_PYTHON", sys.executable),
        help="Python executable used to run the vendored DriveStudio backend. Defaults to the current WorldLens Python.",
    )
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--step", default=None)
    parser.add_argument("--fps", default=None)
    parser.add_argument("--gt-method", default=None)
    parser.add_argument("--cameras", default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def append_options(command: List[str], args: argparse.Namespace, option_flags: Mapping[str, str]) -> None:
    for name, flag in option_flags.items():
        value = getattr(args, name)
        if value is not None:
            if name in PATH_OPTIONS:
                value = Path(value).resolve()
            command.extend([flag, str(value)])


def append_booleans(command: List[str], args: argparse.Namespace, boolean_flags: Mapping[str, str]) -> None:
    for name, flag in boolean_flags.items():
        if getattr(args, name):
            command.append(flag)


def main() -> None:
    args = parse_args()
    worldbench_root = Path(__file__).resolve().parents[2]
    drivestudio_root = (
        Path(args.drivestudio_root).resolve()
        if args.drivestudio_root
        else worldbench_root / "third_party" / "drivestudio"
    )
    backend_script = drivestudio_root / "tools" / "worldlens_reconstruction_backend.py"
    if not backend_script.is_file():
        raise SystemExit(f"Missing DriveStudio WorldLens backend script: {backend_script}")

    command = [
        str(args.python_bin),
        str(backend_script.relative_to(drivestudio_root)),
        "--method-name",
        args.method_name,
        "--clips",
        *[str(clip) for clip in args.clips],
        "--reconstruction-root",
        str(Path(args.reconstruction_root).resolve()),
    ]
    append_options(command, args, OPTION_FLAGS)
    append_booleans(command, args, BOOLEAN_FLAGS)
    subprocess.run(command, cwd=str(drivestudio_root), check=True)


if __name__ == "__main__":
    main()
