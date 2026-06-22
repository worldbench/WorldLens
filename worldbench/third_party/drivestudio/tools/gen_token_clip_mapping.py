#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mapping from first_sample_token to clip folder id."
    )
    parser.add_argument(
        "--scene-json",
        default="data/nuscenes_trainval/raw/v1.0-trainval/scene.json",
        help="Path to NuScenes scene.json.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split name defined in nuscenes.utils.splits (e.g., val, mini_val).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Path to a text file listing scene names to include (one per line).",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=150,
        help="Max number of scenes to keep (mirrors preprocess default).",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to save the mapping JSON. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--pickle-output",
        default=None,
        help=(
            "Optional path to save a pickle containing both token→clip and clip→token mappings."
        ),
    )
    parser.add_argument(
        "--pad-width",
        type=int,
        default=3,
        help="Zero padding width for clip ids (folders).",
    )
    return parser.parse_args()


def load_split(split_name: str) -> List[str]:
    try:
        from nuscenes.utils import splits as nusc_splits
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nuscenes-devkit not available; install it or provide --split-file."
        ) from exc

    if not hasattr(nusc_splits, split_name):
        available = [attr for attr in dir(nusc_splits) if not attr.startswith("_")]
        raise ValueError(f"Unknown split '{split_name}'. Available: {available}")
    return getattr(nusc_splits, split_name)


def load_split_from_file(split_file: Path) -> List[str]:
    with Path(split_file).open("r") as fp:
        return [line.strip() for line in fp if line.strip()]


def build_mapping(
    scene_json: Path,
    split_names: Optional[List[str]],
    num_scenes: Optional[int],
    pad_width: int,
) -> Dict[str, str]:
    with Path(scene_json).open("r") as fp:
        scenes = json.load(fp)

    if split_names is None:
        selected_indices = list(range(len(scenes)))
    else:
        selected_indices = [
            idx for idx, scene in enumerate(scenes) if scene["name"] in split_names
        ]
    if num_scenes is not None:
        selected_indices = selected_indices[:num_scenes]

    mapping = {}
    for scene_idx in selected_indices:
        scene = scenes[scene_idx]
        token = scene["first_sample_token"]
        clip_id = str(scene_idx).zfill(pad_width)
        mapping[token] = clip_id
    return mapping


def invert_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Return clip_id -> token dict ensuring uniqueness."""
    inverted: Dict[str, str] = {}
    for token, clip_id in mapping.items():
        if clip_id in inverted:
            raise ValueError(
                f"Duplicate clip id '{clip_id}' for tokens '{inverted[clip_id]}' and '{token}'."
            )
        inverted[clip_id] = token
    return inverted


def main():
    args = parse_args()
    if args.split_file:
        split_names = load_split_from_file(args.split_file)
    elif args.split:
        split_names = load_split(args.split)
    else:
        split_names = None

    mapping = build_mapping(
        args.scene_json,
        split_names,
        args.num_scenes,
        args.pad_width,
    )
    clip_to_token = invert_mapping(mapping)
    output_str = json.dumps(mapping, indent=2, ensure_ascii=False)

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.json_output).open("w") as fp:
            fp.write(output_str)
    else:
        print(output_str)

    if args.pickle_output:
        Path(args.pickle_output).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_clip": mapping,
            "clip_to_token": clip_to_token,
        }
        with Path(args.pickle_output).open("wb") as fp:
            pickle.dump(payload, fp)


if __name__ == "__main__":
    main()
