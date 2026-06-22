#!/usr/bin/env python3
"""Generate GT Grounded-SAM2 masks for Reconstruction geometric evaluation."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Generate Grounded-SAM2 road/vehicle masks for WorldLens Reconstruction geometric evaluation"
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/nuscenes_trainval/processed_gt/advanced_12Hz_trainval"),
        help="GT processed split root that contains <clip>/images.",
    )
    parser.add_argument("--clips", nargs="+", help="Clip ids to process, for example 081 082.")
    parser.add_argument("--start", type=int, default=None, help="Start clip index in sorted processed-root clips.")
    parser.add_argument("--end", type=int, default=None, help="Inclusive end clip index in sorted processed-root clips.")
    parser.add_argument("--output-subdir", default="sam_mask", help="Output directory name inside each clip.")
    parser.add_argument("--prompts", nargs="+", default=["road surface", "vehicles"])
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--grounded-sam-root",
        type=Path,
        default=None,
        help="Grounded-SAM-2 checkout root. Relative model/config paths are resolved from this root.",
    )
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--grounding-dino-config",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    parser.add_argument("--grounding-dino-checkpoint", default="gdino_checkpoints/groundingdino_swinb_cogcoor.pth")
    return parser.parse_args()


def resolve_path(root: Optional[Path], value: str) -> str:
    path = Path(value)
    if path.is_absolute() or root is None:
        return str(path)
    return str(root / path)


def configure_import_path(grounded_sam_root: Optional[Path]) -> None:
    if grounded_sam_root is None:
        return
    root = grounded_sam_root.resolve()
    sys.path.insert(0, str(root))


def load_models(args: argparse.Namespace):
    configure_import_path(args.grounded_sam_root)

    from grounding_dino.groundingdino.util.inference import load_model
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = resolve_path(args.grounded_sam_root, args.sam2_checkpoint)
    grounding_config = resolve_path(args.grounded_sam_root, args.grounding_dino_config)
    grounding_checkpoint = resolve_path(args.grounded_sam_root, args.grounding_dino_checkpoint)

    sam2_model = build_sam2(args.sam2_config, sam2_checkpoint, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    grounding_model = load_model(
        model_config_path=grounding_config,
        model_checkpoint_path=grounding_checkpoint,
        device=args.device,
    )
    grounding_model.eval()
    return grounding_model, sam2_predictor


def discover_clips(
    processed_root: Path,
    clips: Optional[Sequence[str]],
    start: Optional[int],
    end: Optional[int],
) -> List[Path]:
    if clips:
        selected = [processed_root / f"{int(clip):03d}" for clip in clips]
    else:
        selected = sorted(path for path in processed_root.iterdir() if path.is_dir())
        if start is not None or end is not None:
            selected = selected[start : None if end is None else end + 1]
    missing = [str(path) for path in selected if not (path / "images").is_dir()]
    if missing:
        details = "\n".join(f"- {path}/images" for path in missing[:20])
        raise FileNotFoundError(f"Missing clip image directories under {processed_root}:\n{details}")
    return selected


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def run_grounded_sam2(
    image_path: Path,
    prompts: Sequence[str],
    grounding_model,
    sam2_predictor,
    box_threshold: float,
    text_threshold: float,
) -> np.ndarray:
    import torch
    from grounding_dino.groundingdino.util.inference import load_image, predict
    from torchvision.ops import box_convert

    image_source, image = load_image(str(image_path))
    sam2_predictor.set_image(image_source)
    h, w = image_source.shape[:2]
    combined = np.zeros((h, w), dtype=bool)

    with torch.no_grad():
        for prompt in prompts:
            boxes, _, _ = predict(
                model=grounding_model,
                image=image,
                caption=prompt.rstrip(".") + ".",
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if boxes.shape[0] == 0:
                continue
            boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            masks, _, _ = sam2_predictor.predict(box=input_boxes, multimask_output=False)
            if masks.ndim == 4:
                masks = np.squeeze(masks, axis=1)
            combined |= masks.astype(bool).any(axis=0)

    return combined


def progress(items: Iterable[object], desc: str):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items


def generate_masks(args: argparse.Namespace) -> Tuple[int, int]:
    processed_root = args.processed_root.resolve()
    clips = discover_clips(processed_root, args.clips, args.start, args.end)

    written = 0
    skipped = 0
    pending_by_clip = defaultdict(list)
    for clip_dir in clips:
        image_paths = iter_images(clip_dir / "images")
        if not image_paths:
            raise FileNotFoundError(f"No images found in {clip_dir / 'images'}")
        output_dir = clip_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            output_path = output_dir / f"{image_path.stem}.png"
            if output_path.exists() and args.skip_existing:
                skipped += 1
                continue
            pending_by_clip[clip_dir.name].append((image_path, output_path))

    if not pending_by_clip:
        return written, skipped

    grounding_model, sam2_predictor = load_models(args)
    for clip_name, pending_items in pending_by_clip.items():
        for image_path, output_path in progress(pending_items, desc=clip_name):
            mask = run_grounded_sam2(
                image_path=image_path,
                prompts=args.prompts,
                grounding_model=grounding_model,
                sam2_predictor=sam2_predictor,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)
            written += 1
    return written, skipped


def main() -> None:
    args = parse_args()
    written, skipped = generate_masks(args)
    print(f"Generated {written} masks; skipped {skipped} existing masks.")


if __name__ == "__main__":
    main()
