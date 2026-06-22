#!/usr/bin/env python3
"""
Select demo-friendly sequences where methods differ most on LPIPS (RGB render,
original view) and AbsRel (depth), computed per-sequence.

Inputs the aggregated image metrics JSON (from tools/metrics_analysis.py
--metrics-json) and per-sequence depth CSV tables (from tools/eval_depth_diff.py
--table-csv-prefix <dir>/per_seq.csv). Produces ranked lists and CSVs.

Usage example:
  python3 tools/select_demo_sequences.py \
    --metrics-json work_dirs_analysis_metrics.json \
    --depth-per-seq depth_tables_3/per_seq.csv depth_tables_2/per_seq.csv \
    --top-k 15 \
    --output-dir demo_tables

This prints Top-K tables for LPIPS and AbsRel, and writes CSVs under output-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PREFIX = "drivestudio-nus-"


@dataclass
class Spread:
    seq: str
    best_method: str
    best_value: float
    worst_method: str
    worst_value: float
    delta: float
    n_methods: int


def load_lpips_per_seq(metrics_json: Path, include_methods: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Return mapping: method -> {seq -> LPIPS}.

    Expects the top-level keys like 'drivestudio-nus-<method>'.
    """
    payload = json.loads(metrics_json.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for top_key, clip_map in payload.items():
        if not top_key.startswith(PREFIX):
            continue
        method = top_key[len(PREFIX) :]
        if include_methods is not None and method not in include_methods:
            continue
        seq_to_lpips: Dict[str, float] = {}
        for seq, metrics in clip_map.items():
            v = metrics.get("image_metrics/full/lpips")
            if v is not None:
                seq_to_lpips[seq] = float(v)
        if seq_to_lpips:
            out[method] = seq_to_lpips
    return out


def _read_csv(path: Path) -> List[List[str]]:
    with path.open("r", newline="") as f:
        return list(csv.reader(f))


def load_absrel_per_seq(depth_per_seq_files: Sequence[Path], include_methods: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Return mapping: method -> {seq -> AbsRel} from one or more per_seq.csv files."""
    out: Dict[str, Dict[str, float]] = {}
    for csv_path in depth_per_seq_files:
        rows = _read_csv(csv_path)
        if not rows:
            continue
        header = rows[0]
        col_map = {name: idx for idx, name in enumerate(header)}
        try:
            idx_method_seq = col_map["Method/Seq"]
            idx_absrel = col_map["AbsRel"]
        except KeyError:
            # Not a per_seq.csv table; skip
            continue
        for row in rows[1:]:
            if not row or len(row) <= max(idx_method_seq, idx_absrel):
                continue
            label = row[idx_method_seq].strip()
            if "/" not in label:
                continue
            method, seq = label.split("/", 1)
            if include_methods is not None and method not in include_methods:
                continue
            try:
                v = float(row[idx_absrel])
            except ValueError:
                continue
            out.setdefault(method, {})[seq] = v
    return out


def _quantiles(sorted_vals: List[float], q: float) -> float:
    """Linear interpolation quantile for small sample size without numpy.

    q in [0,1]. Assumes sorted_vals non-empty and sorted.
    """
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def compute_spread(
    per_seq: Mapping[str, Mapping[str, float]],
    min_methods: int = 2,
    score: str = "range",
    trim: float = 0.1,
) -> List[Spread]:
    """Compute a sequence-level dispersion score across methods.

    score ∈ {"range", "iqr", "std", "mad", "pairwise", "trimmed"}:
      - range: max - min
      - iqr: Q3 - Q1 (robust to single outliers)
      - std: population std
      - mad: median absolute deviation (unscaled)
      - pairwise: mean absolute difference across all method pairs
      - trimmed: P(1-trim) - P(trim), e.g., 90th - 10th percentiles
    """
    # sequences -> list of (method, value)
    seq_to_vals: Dict[str, List[Tuple[str, float]]] = {}
    for method, seq_map in per_seq.items():
        for seq, v in seq_map.items():
            seq_to_vals.setdefault(seq, []).append((method, v))
    spreads: List[Spread] = []
    for seq, items in seq_to_vals.items():
        if len(items) < min_methods:
            continue
        items_sorted = sorted(items, key=lambda x: x[1])
        best_method, best_val = items_sorted[0]
        worst_method, worst_val = items_sorted[-1]
        vals = [v for _, v in items_sorted]
        m = len(vals)
        # default: range
        if score == "range":
            delta = worst_val - best_val
        elif score == "iqr":
            q1 = _quantiles(vals, 0.25)
            q3 = _quantiles(vals, 0.75)
            delta = q3 - q1
        elif score == "std":
            mu = sum(vals) / m
            delta = math.sqrt(sum((v - mu) ** 2 for v in vals) / m)
        elif score == "mad":
            med = _quantiles(vals, 0.5)
            devs = sorted(abs(v - med) for v in vals)
            delta = _quantiles(devs, 0.5)
        elif score == "pairwise":
            if m == 1:
                delta = 0.0
            else:
                s = 0.0
                cnt = 0
                for i in range(m):
                    for j in range(i + 1, m):
                        s += abs(vals[i] - vals[j])
                        cnt += 1
                delta = s / cnt
        elif score == "trimmed":
            lo = _quantiles(vals, trim)
            hi = _quantiles(vals, 1.0 - trim)
            delta = hi - lo
        else:
            raise SystemExit(f"Unknown score '{score}'.")
        spreads.append(
            Spread(
                seq=seq,
                best_method=best_method,
                best_value=best_val,
                worst_method=worst_method,
                worst_value=worst_val,
                delta=delta,
                n_methods=m,
            )
        )
    spreads.sort(key=lambda s: (s.delta, s.seq), reverse=True)
    return spreads


def intersect_methods(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]]) -> List[str]:
    return sorted(set(a.keys()) & set(b.keys()))


def filter_methods(per_seq: Dict[str, Dict[str, float]], allowed: Sequence[str]) -> Dict[str, Dict[str, float]]:
    return {m: d for m, d in per_seq.items() if m in allowed}


def print_table(title: str, spreads: Sequence[Spread], top_k: int) -> None:
    print(title)
    header = ["Seq", "Δ", "Best(method,val)", "Worst(method,val)", "#M"]
    print(" ".join(header))
    for s in spreads[:top_k]:
        best = f"{s.best_method},{s.best_value:.4f}"
        worst = f"{s.worst_method},{s.worst_value:.4f}"
        print(f"{s.seq} {s.delta:.4f} {best} {worst} {s.n_methods}")
    print()


def write_csv(path: Path, spreads: Sequence[Spread]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "delta", "best_method", "best_value", "worst_method", "worst_value", "n_methods"])
        for s in spreads:
            w.writerow([s.seq, f"{s.delta:.6f}", s.best_method, f"{s.best_value:.6f}", s.worst_method, f"{s.worst_value:.6f}", s.n_methods])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select sequences with largest per-seq metric gaps across methods.")
    p.add_argument("--metrics-json", type=Path, default=Path("work_dirs_analysis_metrics.json"))
    p.add_argument("--depth-per-seq", type=Path, nargs="*", default=[Path("depth_tables_3/per_seq.csv")])
    p.add_argument("--methods", type=str, nargs="*", default=None, help="Restrict to these method names.")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--score-lpips", type=str, default="range", choices=["range","iqr","std","mad","pairwise","trimmed"], help="Dispersion score for LPIPS.")
    p.add_argument("--score-absrel", type=str, default="range", choices=["range","iqr","std","mad","pairwise","trimmed"], help="Dispersion score for AbsRel.")
    p.add_argument("--trim", type=float, default=0.1, help="Trim fraction for 'trimmed' score (e.g., 0.1 → P90-P10).")
    p.add_argument("--min-methods", type=int, default=2, help="Min methods per sequence to consider.")
    p.add_argument("--output-dir", type=Path, default=Path("demo_tables"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load raw per-seq metrics
    lpips_map = load_lpips_per_seq(args.metrics_json, include_methods=args.methods)
    absrel_map = load_absrel_per_seq(args.depth_per_seq, include_methods=args.methods)

    if not lpips_map:
        raise SystemExit("No LPIPS data found (check --metrics-json).")
    if not absrel_map:
        raise SystemExit("No AbsRel per-seq data found (check --depth-per-seq).")

    # Use common methods if not specified
    if args.methods is None:
        common_methods = intersect_methods(lpips_map, absrel_map)
        if not common_methods:
            raise SystemExit("No common methods between LPIPS JSON and AbsRel CSVs.")
        lpips_map = filter_methods(lpips_map, common_methods)
        absrel_map = filter_methods(absrel_map, common_methods)

    # Compute spreads with requested scores
    lpips_spreads = compute_spread(lpips_map, min_methods=args.min_methods, score=args.score_lpips, trim=args.trim)
    absrel_spreads = compute_spread(absrel_map, min_methods=args.min_methods, score=args.score_absrel, trim=args.trim)

    print_table(f"Top sequences by LPIPS gap [{args.score_lpips}] (lower better):", lpips_spreads, args.top_k)
    print_table(f"Top sequences by AbsRel gap [{args.score_absrel}] (lower better):", absrel_spreads, args.top_k)

    # Write CSVs
    outdir = args.output_dir
    write_csv(outdir / "lpips_spread.csv", lpips_spreads)
    write_csv(outdir / "absrel_spread.csv", absrel_spreads)

    # Intersection candidates: rank by min rank across both lists
    rank_lp = {s.seq: i for i, s in enumerate(lpips_spreads)}
    rank_ab = {s.seq: i for i, s in enumerate(absrel_spreads)}
    both = []
    for seq in set(rank_lp) & set(rank_ab):
        r = max(rank_lp[seq], rank_ab[seq])  # conservative: good in both lists
        both.append((r, seq))
    both.sort()
    top_both = [seq for _, seq in both[: args.top_k]]
    if top_both:
        print("Top sequences that are high gap in both LPIPS and AbsRel:")
        print(", ".join(top_both))
        print()
        # Write a small CSV joining both spreads for convenience
        join_rows: List[List[str]] = [["seq", "lpips_delta", "absrel_delta"]]
        lp_map = {s.seq: s for s in lpips_spreads}
        ab_map = {s.seq: s for s in absrel_spreads}
        for seq in top_both:
            join_rows.append([
                seq,
                f"{lp_map[seq].delta:.6f}",
                f"{ab_map[seq].delta:.6f}",
            ])
        out = outdir / "both_spread_top.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            csv.writer(f).writerows(join_rows)


if __name__ == "__main__":
    main()
