#!/usr/bin/env python
"""Assemble the full proprietary closed-world negative-transfer table for the paper.

Anonymized: the dataset is referred to ONLY as "a third, proprietary closed-world
dataset (anonymized for review)". Its real name never appears in any output.

Matched protocol (mirrors the CESNET tables): one frozen source model on the clean
"quarter" slice (Week-16 equivalent, source window 13), frozen FORWARD-only eval.
Three reported blocks:
  A. bs64 / seed 42  : Source-only / AdaBN / TENT / CoTTA  (DANN was not run at bs64)
  B. bs256 / seed 42 : Source-only / AdaBN / TENT / CoTTA / DANN  (fair operating point)
  C. bs256, 5-seed mean +/- std over seeds {42,1,2,3,4}: Source-only/AdaBN/TENT/CoTTA
     (DANN was trained+evaluated single-seed; its bs256/seed42 point is in block B.)

CAVEAT: 10%-sample protocol (data_sample_frac=0.1). Absolute numbers are NOT
comparable to the full-data monitor figures elsewhere in the paper.

Read-only on the npz; writes only the --json-out file.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closedworld_negtransfer_table import summarize, METHOD_LABEL  # noqa: E402

EXP_DIR = "/home/anatbr/students/noamshakedc/da4etc/exps/allot_multimodal/quarter_eq"
SRC_WINDOW = 13
SEEDS = [42, 1, 2, 3, 4]          # 42 = base bs256 run (no _seed suffix)
AGG_KEYS = ["mean_acc", "delta", "macro_f1", "indist", "far", "stable_recall"]


def _rows_by_method(summary):
    return {r["method"]: r for r in summary["rows"]}


def block(methods, suffix_tag, seed):
    """One summarize() call -> {method: row} (rows may be missing=True)."""
    s = summarize(EXP_DIR, methods, SRC_WINDOW, suffix_tag=suffix_tag, seed=seed)
    return s, _rows_by_method(s)


def multiseed(methods, seeds):
    """mean +/- std of each metric across seeds, per method. Skips missing rows."""
    per_method = {m: {k: [] for k in AGG_KEYS} for m in methods}
    n_seen = {m: 0 for m in methods}
    for sd in seeds:
        seed_arg = None if sd == 42 else sd     # seed 42 = the unsuffixed bs256 dir
        _, rows = block(methods, "_bs256", seed_arg)
        for m in methods:
            r = rows.get(m, {"missing": True})
            if r.get("missing"):
                continue
            n_seen[m] += 1
            for k in AGG_KEYS:
                v = r.get(k, float("nan"))
                if v == v:   # not NaN
                    per_method[m][k].append(v)
    out = {}
    for m in methods:
        agg = {}
        for k in AGG_KEYS:
            vals = per_method[m][k]
            if vals:
                agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                          "n": len(vals)}
            else:
                agg[k] = None
        out[m] = {"label": METHOD_LABEL[m], "n_seeds": n_seen[m], "metrics": agg}
    return out


def fmt_pm(d):
    if d is None:
        return "—"
    return f"{d['mean']:.3f}±{d['std']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="results/analysis/closedworld_negtransfer_quarter.json")
    args = ap.parse_args()

    methods_bs64 = ["vanilla", "bnstats", "tent", "cotta"]
    methods_bs256 = ["vanilla", "bnstats", "tent", "cotta", "dann"]

    s_a, rows_a = block(methods_bs64, "", None)
    s_b, rows_b = block(methods_bs256, "_bs256", None)
    ms = multiseed(methods_bs64, SEEDS)

    result = {
        "dataset": "a third, proprietary closed-world dataset (anonymized for review)",
        "protocol": "frozen source (clean 'quarter' slice, src window 13), forward-only eval, data_sample_frac=0.1",
        "caveat": "10%-sample protocol; absolute numbers not comparable to full-data monitor figures",
        "n_fwd_windows": s_b["n_fwd_windows"], "n_stable": s_b["n_stable"],
        "far_cut": s_b["far_cut"], "stable_recall_threshold": 0.70,
        "block_A_bs64_seed42": s_a,
        "block_B_bs256_seed42": s_b,
        "block_C_bs256_multiseed": {"seeds": SEEDS, "per_method": ms},
    }
    with open(args.json_out, "w") as f:
        json.dump(result, f, indent=2)

    # human-readable echo
    print(f"\n# Closed-world (anonymized) forward-only from clean source (window {SRC_WINDOW}); "
          f"{s_b['n_fwd_windows']} fwd windows; {s_b['n_stable']} stable classes\n")
    print("## A. bs64 / seed 42")
    print("| Method | Mean acc | Δ | Macro-F1 | Far | Stable-recall |")
    print("|---|--:|--:|--:|--:|--:|")
    for m in methods_bs64:
        r = rows_a.get(m, {"missing": True})
        if r.get("missing"):
            print(f"| {METHOD_LABEL[m]} | — | — | — | — | — |"); continue
        d = "—" if m == "vanilla" else f"{r['delta']:+.3f}"
        print(f"| {r['label']} | {r['mean_acc']:.3f} | {d} | {r['macro_f1']:.3f} | {r['far']:.3f} | {r['stable_recall']:.3f} |")
    print("\n## B. bs256 / seed 42 (fair operating point)")
    print("| Method | Mean acc | Δ | Macro-F1 | Far | Stable-recall |")
    print("|---|--:|--:|--:|--:|--:|")
    for m in methods_bs256:
        r = rows_b.get(m, {"missing": True})
        if r.get("missing"):
            print(f"| {METHOD_LABEL[m]} | — | — | — | — | — |"); continue
        d = "—" if m == "vanilla" else f"{r['delta']:+.3f}"
        print(f"| {r['label']} | {r['mean_acc']:.3f} | {d} | {r['macro_f1']:.3f} | {r['far']:.3f} | {r['stable_recall']:.3f} |")
    print("\n## C. bs256, 5-seed mean±std (seeds 42,1,2,3,4)")
    print("| Method | n | Mean acc | Δ | Macro-F1 | Stable-recall |")
    print("|---|--:|--:|--:|--:|--:|")
    for m in methods_bs64:
        e = ms[m]; mt = e["metrics"]
        d = "—" if m == "vanilla" else fmt_pm(mt["delta"])
        print(f"| {e['label']} | {e['n_seeds']} | {fmt_pm(mt['mean_acc'])} | {d} | {fmt_pm(mt['macro_f1'])} | {fmt_pm(mt['stable_recall'])} |")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
