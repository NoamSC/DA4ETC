#!/usr/bin/env python
"""Significance analysis for the multi-seed TTA study (audit red flags #7/#8).

Reads results/inference_seedvar/seed<S>_<week>_<method>_bs<B>/*.npz (acc_only), and
for each source week computes, per method, the mean per-week accuracy at each seed, the
per-seed delta vs vanilla (same seed/population), and the mean +/- std across seeds.

Verdict per method: if |mean delta| > 2 * std(delta) the effect is distinguishable from
run-to-run noise; otherwise it is within noise (the sign could flip) -> the relevant
audit concern (AdaBN "neutral control" reliability, TENT bs64 magnitude) is confirmed.

Usage: python scripts/analysis/analyze_seedvar.py
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("results/inference_seedvar")
SEEDS = [42, 1, 2, 3, 4]
SOURCES = ["week_16", "WEEK-2022-44"]
METHODS = ["vanilla", "bnstats", "tent"]
BATCH = 64
WANT = {"week_16": 53, "WEEK-2022-44": 4}


def group_mean_acc(d):
    """Mean per-week accuracy over all .npz in a group dir, or None if absent/partial."""
    files = sorted(d.glob("*.npz"))
    if not files:
        return None, 0
    accs = []
    for f in files:
        z = np.load(f)
        accs.append((z["pred_labels"] == z["true_labels"]).mean())
    return float(np.mean(accs)), len(files)


def main():
    for src in SOURCES:
        print(f"\n{'='*70}\nSOURCE {src}  (need {WANT[src]} weeks/seed)\n{'='*70}")
        # per method: list of (seed, mean_acc); also per-seed deltas vs vanilla
        per_method = {m: {} for m in METHODS}
        for s in SEEDS:
            for m in METHODS:
                d = ROOT / f"seed{s}_{src}_{m}_bs{BATCH}"
                acc, n = group_mean_acc(d)
                if acc is not None and n >= WANT[src]:
                    per_method[m][s] = acc
        # report mean acc +/- std per method
        for m in METHODS:
            accs = list(per_method[m].values())
            seeds_done = sorted(per_method[m].keys())
            if accs:
                print(f"  {m:8s}: mean_acc={np.mean(accs):.4f} +/- {np.std(accs):.4f} "
                      f"(n_seeds={len(accs)} {seeds_done})")
            else:
                print(f"  {m:8s}: (no complete seeds yet)")
        # deltas vs vanilla, paired by seed
        for m in ("bnstats", "tent"):
            paired = [(per_method[m][s] - per_method["vanilla"][s])
                      for s in SEEDS if s in per_method[m] and s in per_method["vanilla"]]
            if len(paired) >= 2:
                md, sd = np.mean(paired), np.std(paired)
                sig = abs(md) > 2 * sd
                verdict = "SIGNIFICANT (outside noise)" if sig else "WITHIN NOISE (sign may flip)"
                print(f"    delta {m}-vanilla: {md:+.4f} +/- {sd:.4f}  over {len(paired)} seeds  -> {verdict}")
            elif paired:
                print(f"    delta {m}-vanilla: {paired[0]:+.4f} (only 1 paired seed; need >=2)")
    print()


if __name__ == "__main__":
    main()
