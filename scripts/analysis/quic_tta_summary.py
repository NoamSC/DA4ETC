"""
Summarize CESNET-QUIC22 week-44 -> weeks 44-47 inference: vanilla vs TENT vs CoTTA.

Reads the per-week .npz files produced by scripts/inference/run_inference.py for each
method, computes per-week accuracy and macro-F1, prints a comparison table with
``method - vanilla`` deltas, and saves a JSON summary + an accuracy-vs-week plot.

Usage:
    python scripts/analysis/quic_tta_summary.py \
        --results_dir results/inference \
        --out_dir results/inference/quic_w44_summary
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import os
import re
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method name -> inference output dir (relative to --results_dir).
# Diagnostic variants are included automatically when their dirs exist (missing ones
# are skipped): bnstats = BN batch-stats no-adaptation; *_bs256 = TTA at batch_size 256.
METHOD_DIRS = {
    "vanilla":     "quic_w44_inference",
    "bnstats":     "quic_w44_inference_bnstats",
    "tent":        "quic_w44_inference_tent",
    "cotta":       "quic_w44_inference_cotta",
    "tent_bs256":  "quic_w44_inference_tent_bs256",
    "cotta_bs256": "quic_w44_inference_cotta_bs256",
}


def _week(path):
    m = re.search(r"WEEK-2022-(\d+)", path)
    return int(m.group(1)) if m else None


def macro_f1(true, pred):
    """Unweighted mean per-class F1 over classes present in the union of true/pred."""
    labels = np.unique(np.concatenate([true, pred]))
    f1s = []
    for c in labels:
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def collect(results_dir):
    """results[method][week] = {'acc':..., 'macro_f1':..., 'n':...}"""
    results = {}
    for method, sub in METHOD_DIRS.items():
        d = os.path.join(results_dir, sub)
        per_week = {}
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            w = _week(f)
            if w is None:
                continue
            z = np.load(f, allow_pickle=True)
            if "true_labels" not in z.files or "pred_labels" not in z.files:
                continue
            t, p = z["true_labels"], z["pred_labels"]
            if t.size == 0:
                continue
            per_week[w] = {
                "acc": float((t == p).mean()),
                "macro_f1": macro_f1(t, p),
                "n": int(t.size),
            }
        if per_week:
            results[method] = per_week
    return results


def print_table(results, metric):
    methods = [m for m in METHOD_DIRS if m in results]
    weeks = sorted(set().union(*[set(results[m]) for m in methods]))
    print(f"\n=== {metric} by week (week-44 model) ===")
    print(f"{'wk':>4} " + " ".join(f"{m:>9}" for m in methods))
    for w in weeks:
        print(f"{w:>4} " + " ".join(
            f"{results[m][w][metric]:>9.3f}" if w in results[m] else f"{'-':>9}"
            for m in methods))
    # averages over weeks common to all methods
    common = sorted(set.intersection(*[set(results[m]) for m in methods]))
    if common:
        print("-" * (5 + 10 * len(methods)))
        print(f"{'avg':>4} " + " ".join(
            f"{np.mean([results[m][w][metric] for w in common]):>9.3f}" for m in methods))
        if "vanilla" in results:
            for m in methods:
                if m == "vanilla":
                    continue
                delta = np.mean([results[m][w][metric] - results["vanilla"][w][metric]
                                 for w in common])
                print(f"  {m} - vanilla ({metric}) = {delta:+.3f}  "
                      f"(negative => worse than no adaptation)")


def save_plot(results, out_dir):
    methods = [m for m in METHOD_DIRS if m in results]
    plt.figure(figsize=(7, 4.5))
    for m in methods:
        weeks = sorted(results[m])
        accs = [results[m][w]["acc"] for w in weeks]
        plt.plot(weeks, accs, marker="o", label=m)
    plt.xlabel("CESNET-QUIC22 week (2022)")
    plt.ylabel("Accuracy")
    plt.title("Week-44 multimodal model — temporal generalization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "quic_w44_accuracy_by_week.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/inference",
                    help="Dir containing quic_w44_inference{,_tent,_cotta}/")
    ap.add_argument("--out_dir", default="results/inference/quic_w44_summary")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = collect(args.results_dir)
    if not results:
        raise SystemExit(f"No inference .npz found under {args.results_dir}/{{{','.join(METHOD_DIRS.values())}}}")

    found = ", ".join(f"{m}({len(results[m])}wk)" for m in results)
    print(f"Methods found: {found}")
    print_table(results, "acc")
    print_table(results, "macro_f1")

    summary_path = os.path.join(args.out_dir, "quic_w44_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")
    save_plot(results, args.out_dir)


if __name__ == "__main__":
    main()
