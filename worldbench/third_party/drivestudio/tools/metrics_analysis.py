#!/usr/bin/env python3
"""
Aggregate DriveStudio evaluation metrics and report per-model statistics.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_METRICS = OrderedDict(
    [
        ("image_metrics/full/psnr", "PSNR"),
        ("image_metrics/full/ssim", "SSIM"),
        ("image_metrics/full/lpips", "LPIPS"),
        ("image_metrics/full/masked_psnr", "Masked PSNR"),
        ("image_metrics/full/masked_ssim", "Masked SSIM"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DriveStudio reconstruction metrics across models."
    )
    parser.add_argument(
        "--work-dirs",
        type=Path,
        default=Path("work_dirs"),
        help="Root directory that contains per-model subdirectories (default: work_dirs).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="drivestudio-nus-gt",
        help="Reference model used for delta statistics (default: drivestudio-nus-gt).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("work_dirs_analysis_metrics.json"),
        help="Optional cache file to store raw metrics JSON (default: work_dirs_analysis_metrics.json).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=list(DEFAULT_METRICS.keys()),
        help="Metric keys to include (defaults to a standard set).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Subset of model directory names to analyze (default: all under work_dirs).",
    )
    return parser.parse_args()


def load_latest_metric_file(metrics_dir: Path) -> Optional[Path]:
    json_files = sorted(
        (p for p in metrics_dir.glob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return json_files[-1] if json_files else None


def collect_metrics(root: Path, metric_keys: Iterable[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        model_metrics: Dict[str, Dict[str, float]] = {}
        for clip_dir in sorted(model_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            latest = load_latest_metric_file(clip_dir / "metrics")
            if latest is None:
                continue
            try:
                payload = json.loads(latest.read_text())
            except json.JSONDecodeError:
                continue
            filtered = {k: payload[k] for k in metric_keys if k in payload}
            if filtered:
                model_metrics[clip_dir.name] = filtered
        if model_metrics:
            data[model_name] = model_metrics
    return data


def compute_stats(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (mean(values), pstdev(values))


def summarize_common_clips(
    data: Mapping[str, Mapping[str, Mapping[str, float]]],
    models: Sequence[str],
    chosen_metrics: Mapping[str, str],
) -> Tuple[List[str], Dict[str, Dict[str, Tuple[float, float, int]]]]:
    clip_sets = [set(data[m].keys()) for m in models if m in data]
    if not clip_sets:
        return [], {}
    common = sorted(set.intersection(*clip_sets))
    summary: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for model in models:
        model_summary: Dict[str, Tuple[float, float, int]] = {}
        clips = data.get(model, {})
        for metric_key, metric_label in chosen_metrics.items():
            values = [clips[c][metric_key] for c in common if metric_key in clips.get(c, {})]
            if values:
                m, s = compute_stats(values)
                model_summary[metric_label] = (m, s, len(values))
        if model_summary:
            summary[model] = model_summary
    return common, summary


def summarize_deltas(
    data: Mapping[str, Mapping[str, Mapping[str, float]]],
    models: Sequence[str],
    reference: str,
    clip_subset: Sequence[str],
    chosen_metrics: Mapping[str, str],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    deltas: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    ref_clips = data.get(reference, {})
    for model in models:
        if model == reference or model not in data:
            continue
        model_clips = data[model]
        metric_summary: Dict[str, Tuple[float, float, int]] = {}
        for metric_key, metric_label in chosen_metrics.items():
            diff_vals: List[float] = []
            for clip in clip_subset:
                if metric_key in model_clips.get(clip, {}) and metric_key in ref_clips.get(clip, {}):
                    diff_vals.append(model_clips[clip][metric_key] - ref_clips[clip][metric_key])
            if diff_vals:
                m, s = compute_stats(diff_vals)
                metric_summary[metric_label] = (m, s, len(diff_vals))
        if metric_summary:
            deltas[model] = metric_summary
    return deltas


def write_raw_cache(path: Path, payload: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def format_table(
    header: Sequence[str],
    rows: Sequence[Tuple[str, Sequence[str]]],
) -> str:
    widths = [len(col) for col in header]
    for _, cells in rows:
        for idx, cell in enumerate(cells):
            widths[idx + 1] = max(widths[idx + 1], len(cell))
    line_parts = ["{:<{w}}".format(header[0], w=widths[0])]
    for idx, col in enumerate(header[1:], start=1):
        line_parts.append("{:>{w}}".format(col, w=widths[idx]))
    output = [" ".join(line_parts)]
    output.append(" ".join("-" * w for w in widths))
    for label, cells in rows:
        row_parts = ["{:<{w}}".format(label, w=widths[0])]
        for idx, cell in enumerate(cells, start=1):
            row_parts.append("{:>{w}}".format(cell, w=widths[idx]))
        output.append(" ".join(row_parts))
    return "\n".join(output)


def main() -> None:
    args = parse_args()

    metric_map = OrderedDict((k, DEFAULT_METRICS.get(k, k)) for k in args.metrics)
    data = collect_metrics(args.work_dirs, metric_map.keys())
    if not data:
        raise SystemExit("No metrics found under {}".format(args.work_dirs))

    models = args.models if args.models else sorted(data.keys())
    if args.reference not in data:
        raise SystemExit(f"Reference model '{args.reference}' not found in metrics.")

    write_raw_cache(args.metrics_json, data)

    common_clips, summary = summarize_common_clips(data, models, metric_map)
    if not common_clips:
        raise SystemExit("No common clips across selected models; aborting.")

    print(f"Common clips across models ({len(common_clips)}): {', '.join(sorted(common_clips))}")
    print()

    header = ["Metric"] + [m for m in models if m in summary]
    rows: List[Tuple[str, Sequence[str]]] = []
    for metric_label in metric_map.values():
        cells = []
        for model in models:
            stats = summary.get(model, {}).get(metric_label)
            if stats:
                mean_val, std_val, count = stats
                cells.append(f"{mean_val:.3f}±{std_val:.3f} (n={count})")
            else:
                cells.append("n/a")
        rows.append((metric_label, cells))
    print("Per-model statistics (mean±std over common clips):")
    print(format_table(header, rows))
    print()

    delta_summary = summarize_deltas(
        data=data,
        models=models,
        reference=args.reference,
        clip_subset=common_clips,
        chosen_metrics=metric_map,
    )
    if delta_summary:
        header = ["Metric"] + [m for m in models if m != args.reference and m in delta_summary]
        rows = []
        for metric_label in metric_map.values():
            cells = []
            for model in models:
                if model == args.reference:
                    continue
                stats = delta_summary.get(model, {}).get(metric_label)
                if stats:
                    mean_val, std_val, count = stats
                    cells.append(f"{mean_val:+.3f}±{std_val:.3f} (n={count})")
                else:
                    cells.append("n/a")
            rows.append((metric_label, cells))
        print(f"Deltas relative to reference '{args.reference}':")
        print(format_table(header, rows))


if __name__ == "__main__":
    main()
