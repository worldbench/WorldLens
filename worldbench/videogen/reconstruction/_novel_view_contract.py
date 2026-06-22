from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _clip_ids(clips: Optional[Iterable[str]]) -> Optional[List[str]]:
    if clips is None:
        return None
    return [str(clip) for clip in clips]


def novel_view_groups(
    method_name: str,
    generated_data_path: str,
    reconstruction_root: Optional[str],
    clips: Optional[Iterable[str]],
) -> Dict[str, List[str]]:
    if reconstruction_root is not None:
        root = Path(reconstruction_root) / method_name / "novel"
        clip_list = _clip_ids(clips)
        if clip_list is None:
            clip_list = sorted(path.name for path in root.iterdir() if path.is_dir())
        groups = {}
        for clip in clip_list:
            video_dir = root / str(clip)
            videos = sorted(str(path) for path in video_dir.glob("*.mp4"))
            if videos:
                groups[str(clip)] = videos
        if not groups:
            raise FileNotFoundError(f"No novel-view videos found under {root}")
        return groups

    legacy_root = Path(generated_data_path) / method_name / "reconstruction"
    groups = {}
    for dim_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
        videos = sorted(str(path) for path in dim_dir.rglob("*.mp4"))
        if videos:
            groups[dim_dir.name] = videos
    if not groups:
        raise FileNotFoundError(f"No novel-view videos found under {legacy_root}")
    return groups


def novel_metric_output_dir(
    method_name: str,
    generated_data_path: str,
    reconstruction_root: Optional[str],
    metric_name: str,
) -> Path:
    if reconstruction_root is not None:
        return Path(reconstruction_root) / method_name / metric_name
    return Path(generated_data_path) / method_name / metric_name
