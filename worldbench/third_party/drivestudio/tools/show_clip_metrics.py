#!/usr/bin/env python3
"""Show per-clip train/depth metrics (LPIPS, AbsRel) for selected models."""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_MODELS = [
    "drivestudio-nus-gt",
    "drivestudio-nus-dist4d",
    "drivestudio-nus-drivedreamer2",
    "drivestudio-nus-opendwm",
    "drivestudio-nus-dreamforge",
    "drivestudio-nus-xscene",
    "drivestudio-nus-magicdrive",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Print LPIPS/AbsRel metrics for specified clip IDs")
    p.add_argument("clip_id", nargs="+", help="Clip IDs (e.g., 813 694)")
    p.add_argument("--models", type=str, nargs="*", default=None, help="Model directory names. Default: all known models containing the clip in metrics/depth tables.")
    p.add_argument("--metrics-json", type=Path, default=Path("work_dirs_analysis_metrics.json"), help="Path to LPIPS metrics JSON (default: work_dirs_analysis_metrics.json)")
    p.add_argument("--metric-key", type=str, default="image_metrics/full/lpips", help="Metric key inside JSON (default: image_metrics/full/lpips)")
    p.add_argument("--depth-csv", type=Path, nargs="*", default=None, help="Candidate depth per-seq CSVs. Default: depth_tables_3/per_seq.csv, depth_tables_2/per_seq.csv")
    p.add_argument("--absrel-column", type=str, default="AbsRel", help="Column name for AbsRel in depth CSV (default: AbsRel)")
    return p.parse_args()


def method_label(name: str) -> str:
    return name.split("drivestudio-nus-", 1)[-1].replace("_", " ").replace("-", " ")


def load_lpips(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not path.is_file():
        raise SystemExit(f"LPIPS metrics JSON not found: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse {path}: {e}")


def load_absrel(paths: List[Path], col: str) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    for p in paths:
        if not p.is_file():
            continue
        try:
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if "Method/Seq" not in reader.fieldnames or col not in reader.fieldnames:
                    continue
                for row in reader:
                    tag = row.get("Method/Seq", "").strip()
                    val = row.get(col)
                    if not tag or val is None:
                        continue
                    if "/" not in tag:
                        continue
                    method, seq = tag.split("/", 1)
                    try:
                        lookup[(method.strip(), seq.strip())] = float(val)
                    except ValueError:
                        continue
        except Exception as e:
            print(f"[WARN] failed reading {p}: {e}")
    return lookup


def main() -> None:
    args = parse_args()
    metrics = load_lpips(args.metrics_json)
    depth_paths = args.depth_csv
    if not depth_paths:
        depth_paths = [Path("depth_tables_3/per_seq.csv"), Path("depth_tables_2/per_seq.csv")]
    absrel_lookup = load_absrel(depth_paths, args.absrel_column)

    for clip in args.clip_id:
        # determine models
        if args.models:
            models = list(args.models)
        else:
            models = set()
            for model, seqs in metrics.items():
                if isinstance(seqs, dict) and clip in seqs:
                    models.add(model)
            for (method, seq), _ in absrel_lookup.items():
                if seq == clip:
                    models.add(f"drivestudio-nus-{method.lower()}")
            if not models:
                models = set(DEFAULT_MODELS)
            models = sorted(models)

        print(f"\nClip {clip}")
        print(f"{'Model':<25} {'LPIPS':>10} {'AbsRel':>10}")
        print("-" * 50)
        for model in models:
            lpips = None
            seq_map = metrics.get(model, {}) if isinstance(metrics.get(model), dict) else {}
            if isinstance(seq_map, dict):
                val = seq_map.get(clip, {}).get(args.metric_key)
                if isinstance(val, (int, float)):
                    lpips = val
            meth_key = method_label(model).replace(" ", "")
            absrel = None
            key1 = (meth_key, clip)
            key2 = (meth_key.lower(), clip)
            if key1 in absrel_lookup:
                absrel = absrel_lookup[key1]
            elif key2 in absrel_lookup:
                absrel = absrel_lookup[key2]
            lpips_str = f"{lpips:.3f}" if lpips is not None else "-"
            absrel_str = f"{absrel:.3f}" if absrel is not None else "-"
            print(f"{method_label(model):<25} {lpips_str:>10} {absrel_str:>10}")


if __name__ == "__main__":
    main()
