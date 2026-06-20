#!/usr/bin/env python
"""Forward-only negative-transfer benchmark on the proprietary closed-world dataset.

Anonymized: the proprietary closed-world dataset is referred to ONLY as
"a third, proprietary closed-world dataset (anonymized for review)". Its real
name never appears in any emitted artifact.

Matched protocol (mirrors the CESNET tables in UDA_BENCHMARK_STATUS.md):
  - one frozen source model (the clean-regime "quarter" slice = Week-16 equivalent),
  - frozen FORWARD-only evaluation (window >= source window, future only),
  - methods: Source-only (vanilla) / AdaBN (bnstats) / TENT / CoTTA / DANN,
  - per batch size and per seed; mean +/- std reported across seeds by the caller.

Metrics, computed forward-only on the windows SHARED by a method and vanilla:
  - Mean acc, Macro-F1 (per-window sklearn macro then averaged across windows),
  - Delta vs Source-only on shared windows,
  - In-dist (source window accuracy),
  - Far (mean acc over the last 1/4 of forward windows),
  - Stable-class Macro-F1 (negative-transfer-on-the-majority guard): classes whose
    source-window recall >= STABLE_R are "stable"; we report their mean per-window
    recall to check adaptation does not harm the majority it should leave alone.

Read-only: loads npz, prints markdown / JSON. Writes nothing unless --json-out.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
SUFFIX = {"vanilla": "inference", "bnstats": "inference_bnstats",
          "tent": "inference_tent", "cotta": "inference_cotta",
          "dann": "inference_dann"}
METHOD_LABEL = {"vanilla": "Source-only", "bnstats": "AdaBN",
                "tent": "TENT", "cotta": "CoTTA", "dann": "DANN"}
STABLE_R = 0.70   # a class is "stable" if source-window recall >= this


def _win_idx(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])


def _method_dir(exp_dir, method, suffix_tag="", seed=None):
    """Resolve the inference directory for a method/batch-size/seed variant.

    DANN lives in the parallel "<exp_dir>_dann" experiment dir and was evaluated
    once at bs256/seed42, so suffix_tag/seed do not apply to it. All other methods
    read "<exp_dir>/inference_<method><suffix_tag>[_seed<N>]" (seed=None => the
    base seed-42 run, which carries no _seed suffix)."""
    if method == "dann":
        return os.path.join(exp_dir + "_dann", "inference_dann")
    sub = SUFFIX[method] + suffix_tag
    if seed is not None:
        sub = f"{sub}_seed{seed}"
    return os.path.join(exp_dir, sub)


def load_group(exp_dir, method, src_window, suffix_tag="", seed=None):
    """Return {window_idx: (true, pred)} for forward windows only (>= src_window)."""
    d = _method_dir(exp_dir, method, suffix_tag, seed)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "window_*.npz"))):
        w = _win_idx(f)
        if w < src_window:
            continue
        z = np.load(f, allow_pickle=True)
        t, p = z["true_labels"].astype(int), z["pred_labels"].astype(int)
        if t.size:
            out[w] = (t, p)
    return out


def per_window_metrics(group):
    accs, f1s, wins = {}, {}, sorted(group)
    for w in wins:
        t, p = group[w]
        accs[w] = float((t == p).mean())
        f1s[w] = float(f1_score(t, p, average="macro", zero_division=0))
    return accs, f1s


def stable_classes(group, src_window):
    """Classes with source-window recall >= STABLE_R (the 'majority to protect')."""
    if src_window not in group:
        return set()
    t, p = group[src_window]
    stable = set()
    for c in np.unique(t):
        m = t == c
        if m.sum() >= 20 and (p[m] == c).mean() >= STABLE_R:
            stable.add(int(c))
    return stable


def stable_recall(group, stable_set):
    """Mean per-window recall on the stable classes (forward windows)."""
    vals = []
    for w in sorted(group):
        t, p = group[w]
        rs = []
        for c in stable_set:
            m = t == c
            if m.sum() >= 5:
                rs.append((p[m] == c).mean())
        if rs:
            vals.append(np.mean(rs))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(exp_dir, methods, src_window, seed_tag="", suffix_tag="", seed=None):
    van = load_group(exp_dir, "vanilla", src_window, suffix_tag, seed)
    if not van:
        raise SystemExit(f"No vanilla forward windows in {exp_dir}")
    van_acc, _ = per_window_metrics(van)
    stable_set = stable_classes(van, src_window)
    fwd_wins = sorted(van)
    far_cut = fwd_wins[int(len(fwd_wins) * 0.75)] if len(fwd_wins) >= 4 else fwd_wins[-1]

    rows = []
    for m in methods:
        try:
            g = load_group(exp_dir, m, src_window, suffix_tag, seed)
        except Exception:
            g = {}
        if not g:
            rows.append(dict(method=m, label=METHOD_LABEL[m], n=0, missing=True))
            continue
        acc, f1 = per_window_metrics(g)
        shared = sorted(set(g) & set(van))
        mean_acc = float(np.mean([acc[w] for w in g]))
        mean_f1 = float(np.mean([f1[w] for w in g]))
        delta = float(np.mean([acc[w] for w in shared]) - np.mean([van_acc[w] for w in shared])) if shared else float("nan")
        indist = acc.get(src_window, float("nan"))
        far = [acc[w] for w in g if w >= far_cut]
        far_acc = float(np.mean(far)) if far else float("nan")
        srec = stable_recall(g, stable_set)
        rows.append(dict(method=m, label=METHOD_LABEL[m], n=len(g),
                         mean_acc=mean_acc, delta=delta, macro_f1=mean_f1,
                         indist=indist, far=far_acc, stable_recall=srec,
                         missing=False))
    return dict(src_window=src_window, n_stable=len(stable_set),
                far_cut=far_cut, n_fwd_windows=len(fwd_wins), rows=rows,
                seed_tag=seed_tag, suffix_tag=suffix_tag, seed=seed)


def print_markdown(summary):
    print(f"\n# Closed-world (anonymized) — forward-only from clean source "
          f"(window {summary['src_window']}); {summary['n_fwd_windows']} forward windows; "
          f"{summary['n_stable']} stable classes (src recall>={STABLE_R})")
    print(f"\n| Method | Windows | Mean acc | Δ | Macro-F1 | In-dist | Far | Stable-class recall |")
    print(f"|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in summary["rows"]:
        if r["missing"]:
            print(f"| {r['label']} | 0 | — | — | — | — | — | — |")
            continue
        d = "—" if r["method"] == "vanilla" else f"{r['delta']:+.3f}"
        print(f"| {r['label']} | {r['n']} | {r['mean_acc']:.3f} | {d} | "
              f"{r['macro_f1']:.3f} | {r['indist']:.3f} | {r['far']:.3f} | {r['stable_recall']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=os.path.join(ROOT, "exps/allot_multimodal/quarter_eq"))
    ap.add_argument("--src-window", type=int, default=13)
    ap.add_argument("--methods", nargs="+", default=["vanilla", "bnstats", "tent", "cotta", "dann"])
    ap.add_argument("--suffix-tag", default="", help='"" for bs64 dirs, "_bs256" for the bs256 re-runs')
    ap.add_argument("--seed", type=int, default=None, help="per-seed bs256 dir (omit for the base seed-42 run)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    s = summarize(args.exp_dir, args.methods, args.src_window,
                  suffix_tag=args.suffix_tag, seed=args.seed)
    print_markdown(s)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
