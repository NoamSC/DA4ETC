#!/usr/bin/env python
"""
Ablation sweep: all 16 combinations of 4 SLD-EM robustness knobs on real data.

Four options (ON values used in the sweep):
  1. clamp      — clip importance ratio to max_weight=10
  2. alpha      — raise ratio to power 0.5
  3. reg        — regularize toward train_prior (lam=0.3)
  4. naive_init — initialize from argmax frequencies

Run one combo (SLURM array task):
    python test_sld_em_sweep.py --combo-idx $SLURM_ARRAY_TASK_ID

Aggregate results after all tasks finish:
    python test_sld_em_sweep.py --aggregate
"""

import re
import itertools
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from plot_paper_figures import sld_em_estimation, build_bbse, estimate_label_dist, regularized_bbse

INFERENCE_DIR = Path('figs/week_1_inference')
REF_WN        = 1
ON_VALS       = dict(max_weight=10.0, alpha=0.5, lam=0.3, naive_init=True)
RESULTS_DIR   = Path('logs/sweep_results')

all_combos = list(itertools.product([False, True], repeat=4))


def l1(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _run_week(args):
    """Pool worker: load one week file and return (wn, l1_value)."""
    filepath, K, train_prior, kwargs = args
    d = np.load(filepath)
    wn = int(re.search(r'(\d+)$', Path(filepath).stem).group(1))
    true_counts = np.bincount(d['true_labels'], minlength=K).astype(float)
    true_prior  = true_counts / true_counts.sum()
    est = sld_em_estimation(train_prior, d['softmax'], **kwargs)
    return wn, l1(est, true_prior)


def run_combo(combo_idx, week_files, K, train_prior, n_workers):
    flags = all_combos[combo_idx]
    clamp, alp, reg, naive = flags
    kwargs = dict(
        max_weight = ON_VALS['max_weight'] if clamp else None,
        alpha      = ON_VALS['alpha']      if alp   else 1.0,
        lam        = ON_VALS['lam']        if reg   else 0.0,
        naive_init = naive,
    )
    tasks = [(str(f), K, train_prior, kwargs) for f in week_files]
    with Pool(processes=n_workers) as pool:
        week_results = pool.map(_run_week, tasks)
    week_results.sort(key=lambda x: x[0])
    l1s = [r[1] for r in week_results]
    return {
        'combo_idx': combo_idx,
        'flags':     list(flags),
        'mean_l1':   float(np.mean(l1s)),
        'per_week':  [{'wn': wn, 'l1': v} for wn, v in week_results],
    }


def print_table(results, baselines=None):
    """
    baselines: dict of {label: {'mean_l1': float, 'per_week': {wn: l1}}}
    """
    base_flags = [False, False, False, False]
    base       = next(r for r in results if r['flags'] == base_flags)
    base_mean  = base['mean_l1']

    results_sorted = sorted(results, key=lambda r: r['mean_l1'])

    HDR = f"{'#':>3}  {'clamp':>5} {'alpha':>5} {'reg':>5} {'naive':>5}  {'mean_L1':>9}  {'vs_base_EM':>11}"
    SEP = "─" * (len(HDR) + 4)
    print()
    print(SEP)
    print("  SLD-EM ablation sweep on real data  (sorted best→worst by mean L1)")
    print(f"  ON values: clamp=10, alpha=0.5, reg_lam=0.3, naive_init=True")
    print(SEP)
    print(HDR)
    print(SEP)

    for i, r in enumerate(results_sorted):
        flags = r['flags']
        tag   = lambda f: " ON " if f else " off"
        row   = f"  {i+1:>2}  {tag(flags[0]):>5} {tag(flags[1]):>5} {tag(flags[2]):>5} {tag(flags[3]):>5}"
        delta = f"{(r['mean_l1'] - base_mean) / base_mean * 100:+.1f}%"
        row  += f"  {r['mean_l1']:>9.5f}  {delta:>11}"
        marker = "  ◀ best" if i == 0 else ("  (base)" if flags == base_flags else "")
        print(row + marker)

    print(SEP)
    print(f"  {'baseline EM (all OFF)':<35} {base_mean:.5f}")
    if baselines:
        for label, bdata in baselines.items():
            delta = f"{(bdata['mean_l1'] - base_mean) / base_mean * 100:+.1f}%"
            print(f"  {label:<35} {bdata['mean_l1']:.5f}  {delta:>11}")
    print(SEP)

    # per-week breakdown for the best combo vs baselines
    best       = results_sorted[0]
    base_      = next(r for r in results if r['flags'] == base_flags)
    weeks_best = {d['wn']: d['l1'] for d in best['per_week']}
    weeks_base = {d['wn']: d['l1'] for d in base_['per_week']}
    all_wns    = sorted(weeks_best)

    bl_headers = list(baselines.keys()) if baselines else []
    hdr2 = f"  {'week':>5}  {'best_EM':>9}  {'base_EM':>9}"
    for bl in bl_headers:
        hdr2 += f"  {bl[:9]:>9}"
    print()
    print(f"  Per-week L1 — best combo {best['flags']} vs baselines")
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for wn in all_wns:
        row = f"  {wn:>5}  {weeks_best[wn]:>9.5f}  {weeks_base.get(wn, float('nan')):>9.5f}"
        for bl in bl_headers:
            v = baselines[bl]['per_week'].get(wn, float('nan'))
            row += f"  {v:>9.5f}"
        print(row)
    print()


def _compute_baselines(ref_file, week_files, K, train_prior):
    """Compute BBSE and reg-BBSE L1 for all test weeks. Returns baselines dict."""
    ref      = np.load(ref_file)
    C_T_pinv, C_T, *_ = build_bbse(ref['true_labels'], ref['pred_labels'], ref['softmax'], K)

    l1_bbse, l1_reg, l1_naive, l1_prior = {}, {}, {}, {}
    for f in week_files:
        d   = np.load(f)
        wn  = int(re.search(r'(\d+)$', Path(f).stem).group(1))
        true_counts = np.bincount(d['true_labels'], minlength=K).astype(float)
        p_true      = true_counts / true_counts.sum()

        p_bbse, q_hat = estimate_label_dist(d['pred_labels'], K, C_T_pinv)
        p_reg         = regularized_bbse(C_T, q_hat)

        l1_bbse[wn]  = l1(p_bbse,      p_true)
        l1_reg[wn]   = l1(p_reg,       p_true)
        l1_naive[wn] = l1(q_hat,       p_true)
        l1_prior[wn] = l1(train_prior, p_true)

    def _pack(per_week):
        return {'mean_l1': float(np.mean(list(per_week.values()))), 'per_week': per_week}

    return {
        'BBSE':         _pack(l1_bbse),
        'Reg-BBSE':     _pack(l1_reg),
        'Naive argmax': _pack(l1_naive),
        'Static prior': _pack(l1_prior),
    }


def aggregate():
    result_files = sorted(RESULTS_DIR.glob('combo_*.json'))
    if not result_files:
        print(f"No results found in {RESULTS_DIR}")
        return
    results = [json.loads(f.read_text()) for f in result_files]
    print(f"Loaded {len(results)}/16 combos")

    ref_file = INFERENCE_DIR / f'WEEK-2022-{REF_WN:02d}.npz'
    ref      = np.load(ref_file)
    K        = ref['softmax'].shape[1]
    counts   = np.bincount(ref['true_labels'], minlength=K).astype(float)
    train_prior = counts / counts.sum()

    week_files = sorted(
        [f for f in INFERENCE_DIR.glob('WEEK-2022-*.npz')
         if int(re.search(r'(\d+)$', f.stem).group(1)) != REF_WN],
        key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    )
    print(f"Computing BBSE/reg-BBSE baselines over {len(week_files)} test weeks...")
    baselines = _compute_baselines(ref_file, week_files, K, train_prior)

    print_table(results, baselines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo-idx', type=int, default=None,
                        help='Combo index 0-15 (SLURM_ARRAY_TASK_ID). Omit to run all.')
    parser.add_argument('--aggregate', action='store_true',
                        help='Load saved per-combo JSONs and print the combined table.')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Pool size. Defaults to SLURM_CPUS_PER_TASK or cpu_count().')
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return

    n_workers = args.n_workers or int(os.environ.get('SLURM_CPUS_PER_TASK', None) or os.cpu_count() or 4)

    ref_file = INFERENCE_DIR / f'WEEK-2022-{REF_WN:02d}.npz'
    ref      = np.load(ref_file)
    K        = ref['softmax'].shape[1]
    counts   = np.bincount(ref['true_labels'], minlength=K).astype(float)
    train_prior = counts / counts.sum()

    week_files = sorted(
        [f for f in INFERENCE_DIR.glob('WEEK-2022-*.npz')
         if int(re.search(r'(\d+)$', f.stem).group(1)) != REF_WN],
        key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    )
    print(f"Found {len(week_files)} test weeks, K={K}, workers={n_workers}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    combo_indices = [args.combo_idx] if args.combo_idx is not None else range(len(all_combos))

    for ci in combo_indices:
        flags = all_combos[ci]
        print(f"Running combo {ci} {flags}...")
        result   = run_combo(ci, week_files, K, train_prior, n_workers)
        out_file = RESULTS_DIR / f'combo_{ci:02d}.json'
        out_file.write_text(json.dumps(result, indent=2))
        print(f"  → mean_L1={result['mean_l1']:.5f}  saved to {out_file}")

    if args.combo_idx is None:
        results   = [json.loads(f.read_text()) for f in sorted(RESULTS_DIR.glob('combo_*.json'))]
        print(f"Computing BBSE/reg-BBSE baselines...")
        baselines = _compute_baselines(ref_file, week_files, K, train_prior)
        print_table(results, baselines)

    print("Done")


if __name__ == '__main__':
    main()
