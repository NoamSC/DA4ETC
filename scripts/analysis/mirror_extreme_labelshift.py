#!/usr/bin/env python
"""Mirror Effect under EXTREME injected label shift (estimator variants).

For each test week, resample the week's flows to a synthetic LogUniform
prior tilt (severity 1/s..s, same protocol as the isolation contamination
stage / Fig H), then track the hidden Macro F1 of the resampled window
against the entropy-gap residuals:

  uncorrected, BBSE (trunc. pinv), RLLS (constrained LS), SLD-EM, oracle.

Pure label shift by construction (rows resampled by TRUE class -> P(X|Y)
intact), so a detector that stays glued to F1 here is volume-robust; one
that decouples is fooled by the injected shift.

Defaults target the Week-16 source, forward weeks only (17-52).

Outputs (under --output_dir)
  fig_mirror_extreme_fwd_{sev}x.png   one mirror figure per severity
  mirror_extreme_results.json         per-severity rho table
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'archive'))
from plot_paper_figures import (build_bbse, estimate_label_dist,
                                regularized_bbse, sld_em_estimation,
                                entropy_gap_raw, entropy_gap_from_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_16_inference')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--week_lo', type=int, default=17,
                    help='first test week (forward-only span)')
    ap.add_argument('--severities', type=int, nargs='+', default=[10, 100])
    ap.add_argument('--n_resample', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='figs/ref16')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.inference_dir).glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    wn_of = lambda p: int(re.search(r'(\d+)$', p.stem).group(1))

    ref_file = next(p for p in files if wn_of(p) == args.reference_week)
    print(f"Reference week: {args.reference_week} ({ref_file.name})")
    d = np.load(ref_file)
    ref_true, ref_pred, ref_soft = d['true_labels'], d['pred_labels'], d['softmax']
    K = ref_soft.shape[1]
    C_T_pinv, C_T, p_train, h_ref, _ = build_bbse(ref_true, ref_pred, ref_soft, K)

    test_files = [p for p in files if wn_of(p) >= args.week_lo]
    sevs = args.severities
    rows = {s: [] for s in sevs}   # per week: dict of f1 + gaps

    for p in test_files:
        wn = wn_of(p)
        d = np.load(p)
        true, pred, soft = d['true_labels'], d['pred_labels'], d['softmax']
        present = np.unique(true)
        rows_by_true = {c: np.where(true == c)[0] for c in present}
        for s in sevs:
            logw = rng.uniform(np.log(1.0 / s), np.log(float(s)), size=K)
            p_synth = np.exp(logw)
            mask = np.zeros(K, bool); mask[present] = True
            p_synth *= mask; p_synth /= p_synth.sum()
            idx = []
            for c in present:
                n_c = int(round(p_synth[c] * args.n_resample))
                if n_c > 0:
                    idx.append(rng.choice(rows_by_true[c], size=n_c, replace=True))
            idx = np.concatenate(idx)
            tw, pw, sw = true[idx], pred[idx], soft[idx]

            p_true_win = np.bincount(tw, minlength=K).astype(float)
            p_true_win /= p_true_win.sum()
            p_bbse, q_hat = estimate_label_dist(pw, K, C_T_pinv)
            p_reg = regularized_bbse(C_T, q_hat, ftol=1e-8, maxiter=300)
            p_em = sld_em_estimation(p_train, sw)

            rows[s].append(dict(
                week=wn,
                f1=f1_score(tw, pw, labels=list(range(K)), average='macro',
                            zero_division=0),
                raw=entropy_gap_raw(sw, h_ref, p_train),
                bbse=entropy_gap_from_p(sw, p_bbse, h_ref),
                rlls=entropy_gap_from_p(sw, p_reg, h_ref),
                em=entropy_gap_from_p(sw, p_em, h_ref),
                oracle=entropy_gap_from_p(sw, p_true_win, h_ref),
            ))
        print(f"  week {wn:2d}: done", flush=True)

    TRACKS = [('bbse',   'Entropy gap (BBSE-corrected)', '#d7191c', 's', '-'),
              ('raw',    'Entropy gap (uncorrected)',    '#e08020', '^', '--'),
              ('oracle', 'Perfect correction (true prior from the labels)',
                                                         '#27ae60', 'D', ':'),
              ('rlls',   'Entropy gap (RLLS-corrected)', '#0e7c7b', 'v', '-.'),
              ('em',     'Entropy gap (SLD-EM-corrected)', '#9b59b6', 'P', '--')]

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.22,
                         'axes.spines.top': False, 'axes.spines.right': False})
    results = {}
    for s in sevs:
        wk = np.array([r['week'] for r in rows[s]])
        f1 = np.array([r['f1'] for r in rows[s]])
        fig, ax_l = plt.subplots(figsize=(10, 4))
        ax_r = ax_l.twinx(); ax_r.spines['top'].set_visible(False)
        l1, = ax_l.plot(wk, f1, color='#2c7bb6', marker='o', markersize=4,
                        linewidth=1.8, label='Macro F1 (true labels, shifted window)')
        handles = [l1]
        res = {}
        for key, lab, col, mk, ls in TRACKS:
            g = np.array([r[key] for r in rows[s]])
            rho = pearsonr(f1, g)[0]
            res[key] = float(rho)
            ln, = ax_r.plot(wk, g, color=col, marker=mk, markersize=4,
                            linewidth=1.6, linestyle=ls,
                            label=f'{lab}  (ρ = {rho:.3f})')
            handles.append(ln)
            print(f"  [{s}x] {lab:48s} rho = {rho:+.4f}")
        results[f'{s}x'] = res
        ax_l.set_xlabel('Week Number', fontsize=12)
        ax_l.set_ylabel('Macro F1', fontsize=12, color='#2c7bb6')
        ax_r.set_ylabel('Entropy Gap\n(actual − expected)', fontsize=12,
                        color='#d7191c')
        ax_l.tick_params(axis='y', labelcolor='#2c7bb6')
        ax_r.tick_params(axis='y', labelcolor='#d7191c')
        ax_l.set_xticks(wk[::max(1, len(wk) // 12)])
        ax_l.set_title(
            f'Mirror Effect under EXTREME ({s}×) injected label shift — '
            f'Week-{args.reference_week} source, forward\n'
            f'(weeks {wk[0]}–{wk[-1]}; windows resampled to LogUniform '
            f'[1/{s}, {s}] prior tilts; P(X|Y) intact)',
            fontsize=11.5, fontweight='bold')
        ax_l.legend(handles=handles, fontsize=8.5, loc='center left',
                    framealpha=0.92)
        plt.tight_layout()
        out = out_dir / f'fig_mirror_extreme_fwd_{s}x.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved -> {out}")

    with open(out_dir / 'mirror_extreme_results.json', 'w') as f:
        json.dump(dict(config=vars(args) | dict(output_dir=str(out_dir)),
                       rho=results), f, indent=2, default=str)
    print(f"Saved -> {out_dir / 'mirror_extreme_results.json'}")


if __name__ == '__main__':
    main()
