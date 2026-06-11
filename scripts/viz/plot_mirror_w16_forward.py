#!/usr/bin/env python
"""Forward-only Mirror Effect for the Week-16 reference.

Replots Fig B (mirror effect) from the cached per-week metrics of the
Week-16 source, restricted to weeks AFTER the reference (>= --week_lo),
matching the paper's forward-tracking convention for the Week-16 regime
(cf. Figs 1-2, "from Week 16 onward").  Correlations are recomputed on the
restricted span.  Pure cache -> plot; no npz inference files are touched.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='results/inference/week_16_inference/'
                                       'metrics_cache_ref16_grid9.npz')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--week_lo', type=int, default=17,
                    help='first test week to include (forward-only span)')
    ap.add_argument('--output', default='figs/ref16/fig_mirror_effect_fwd.png')
    args = ap.parse_args()

    z = np.load(args.cache)
    wk = z['week_nums']
    m = wk >= args.week_lo
    wk = wk[m]
    f1 = z['macro_f1s'][m]

    tracks = [
        ('ent_gaps',        'Entropy gap (BBSE-corrected)', '#d7191c', 's', '-'),
        ('ent_gaps_raw',    'Entropy gap (uncorrected)',    '#e08020', '^', '--'),
        ('ent_gaps_oracle', 'Perfect correction (true prior from the labels)',
                                                            '#27ae60', 'D', ':'),
        ('ent_gaps_reg',    'Entropy gap (RLLS-corrected)', '#0e7c7b', 'v', '-.'),
        ('ent_gaps_em',     'Entropy gap (SLD-EM-corrected)', '#9b59b6', 'P', '--'),
    ]

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11,
                         'axes.grid': True, 'grid.alpha': 0.22,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax_l = plt.subplots(figsize=(10, 4))
    ax_r = ax_l.twinx()
    ax_r.spines['top'].set_visible(False)
    l1, = ax_l.plot(wk, f1, color='#2c7bb6', marker='o', markersize=4,
                    linewidth=1.8, label='Macro F1 (true labels)')
    handles = [l1]
    for key, lab, col, mk, ls in tracks:
        if key not in z:
            continue
        g = z[key][m]
        rho = pearsonr(f1, g)[0]
        ln, = ax_r.plot(wk, g, color=col, marker=mk, markersize=4,
                        linewidth=1.6, linestyle=ls,
                        label=f'{lab}  (ρ = {rho:.3f})')
        handles.append(ln)
        print(f'  {lab:50s} rho = {rho:+.4f}')

    ax_l.set_xlabel('Week Number', fontsize=12)
    ax_l.set_ylabel('Macro F1', fontsize=12, color='#2c7bb6')
    ax_r.set_ylabel('Entropy Gap\n(actual − expected)', fontsize=12, color='#d7191c')
    ax_l.tick_params(axis='y', labelcolor='#2c7bb6')
    ax_r.tick_params(axis='y', labelcolor='#d7191c')
    ax_l.set_xticks(wk[::max(1, len(wk) // 12)])
    ax_l.set_title(
        'Macro F1 vs Entropy Gap — The Mirror Effect, forward from the Week-16 source\n'
        f'(reference: week {args.reference_week},  test weeks {wk[0]}–{wk[-1]})',
        fontsize=12, fontweight='bold')
    ax_l.legend(handles=handles, fontsize=9, loc='center left', framealpha=0.9)
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
