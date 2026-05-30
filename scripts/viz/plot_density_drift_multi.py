#!/usr/bin/env python
"""
Density-drift ridgeline for MANY apps overlaid on one plot.

Each app has its own drift direction in the 600-d space, so we put them on a
*shared normalized drift axis*:
    z = (proj - early_location) / (late_location - early_location)
so z=0 is every app's early-weeks location and z=1 its late-weeks location. On
that common axis we draw, per app (one colour each), the per-week centroid as a
track (rows top->bottom = weeks). The *mass concentration* that week is encoded
into the line itself:
  - line WIDTH  ∝ concentration (1 / spread along the axis)
  - line ALPHA  ∝ concentration  (faint = mass is spread out)
so a tight single location draws a thick, opaque segment, while a spread-out /
bimodal / transitional week draws a thin, faint one.

Thesis read-out: if drift is discrete-in-space / continuous-in-time, then for
*every* app the track is thick+opaque near z=0 early and near z=1 late, and it
THINS and FADES as it crosses the empty middle (z~0.5) — i.e. the centroid spends
its transit in a place where no concentrated mass actually sits.

Usage:
    python plot_density_drift_multi.py                       # top 6 drift apps
    python plot_density_drift_multi.py --classes 98 102 29 50 99 167
    python plot_density_drift_multi.py --top 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config, data_utils, ...

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from plot_density_drift import (load_all, class_names, drift_axis,
                                gather_class, rank_classes, inference_week_tag)


def normalized_proj(embs, labs, c, n_edge=8):
    """Per-week projections of class c onto its drift axis, rescaled so the
    early-weeks location is 0 and the late-weeks location is 1."""
    v = drift_axis(embs, labs, c, n_edge)
    _, per_week = gather_class(embs, labs, c)
    proj = [pw @ v for pw in per_week]
    a = np.concatenate(proj[:n_edge]).mean()
    b = np.concatenate(proj[-n_edge:]).mean()
    scale = (b - a) if abs(b - a) > 1e-9 else 1.0
    return [(p - a) / scale for p in proj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--out_dir', default='figs/drift')
    ap.add_argument('--classes', type=int, nargs='+', default=None,
                    help='explicit class indices to overlay')
    ap.add_argument('--top', type=int, default=6,
                    help='if --classes not given, overlay the N highest-drift apps')
    ap.add_argument('--min_week', type=int, default=None,
                    help='only use weeks >= this (e.g. 16 for the week-16 model going forward)')
    ap.add_argument('--max_week', type=int, default=None)
    args = ap.parse_args()

    weeks, embs, labs = load_all(args.inference_dir, args.min_week, args.max_week)
    idx2name = class_names(args.dataset_root)
    print(f'loaded {len(weeks)} weeks: {weeks[0]}-{weeks[-1]}')

    if args.classes:
        targets = args.classes
    else:
        ranked = rank_classes(weeks, embs, labs)
        targets = [c for c, _ in ranked[:args.top]]
    names = [idx2name.get(c, f'class{c}') for c in targets]
    print('overlaying:', list(zip(targets, names)))

    nW = len(weeks)
    n = len(targets)
    # palette that scales to any number of apps
    if n <= 10:
        colors = cm.tab10(np.linspace(0, 1, 10))[:n]
    elif n <= 20:
        colors = cm.tab20(np.linspace(0, 1, 20))[:n]
    else:
        colors = cm.turbo(np.linspace(0.02, 0.98, n))
    offset_step = 1.0

    # normalized projections, and per-week mean + spread (std) along the axis
    projz = {c: normalized_proj(embs, labs, c) for c in targets}
    zmean, zstd = {}, {}
    for c in targets:
        zmean[c] = np.array([p.mean() if len(p) else np.nan for p in projz[c]])
        zstd[c]  = np.array([p.std()  if len(p) > 1 else np.nan for p in projz[c]])

    # concentration = 1/spread; map to line width + alpha via robust global range.
    # concentrated mass -> thick & opaque; spread-out mass -> thin & faint.
    all_std = np.concatenate([zstd[c] for c in targets])
    all_std = all_std[np.isfinite(all_std)]
    conc = 1.0 / (all_std + 1e-3)
    c_lo, c_hi = np.percentile(conc, [5, 95])
    W_MIN, W_MAX = 0.4, 5.0
    A_MIN, A_MAX = 0.35, 1.0

    def conc_t(std):
        """spread -> t in [0,1]; 1 = most concentrated."""
        cc = 1.0 / (np.asarray(std) + 1e-3)
        return np.clip((cc - c_lo) / (c_hi - c_lo + 1e-9), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(11, 13))

    # discrete-location guides
    ax.axvline(0.0, color='gray', ls='-', lw=1.0, alpha=0.5, zorder=0)
    ax.axvline(1.0, color='gray', ls='-', lw=1.0, alpha=0.5, zorder=0)
    ax.axvspan(0.35, 0.65, color='gray', alpha=0.08, zorder=0)
    ax.text(0.0, nW + 0.6, 'early\nlocation', ha='center', va='bottom',
            fontsize=8, color='gray')
    ax.text(1.0, nW + 0.6, 'late\nlocation', ha='center', va='bottom',
            fontsize=8, color='gray')
    ax.text(0.5, nW + 0.6, 'empty\nmiddle', ha='center', va='bottom',
            fontsize=8, color='gray')

    bases = np.array([(nW - 1 - i) * offset_step for i in range(nW)])  # week 0 at top
    width_scale = 1.0 if n <= 12 else (0.7 if n <= 24 else 0.5)
    for ci, c in enumerate(targets):
        col = colors[ci]
        pts = np.column_stack([zmean[c], bases])           # (nW, 2)
        t = conc_t(zstd[c])                                # per-week concentration
        segs = np.stack([pts[:-1], pts[1:]], axis=1)       # (nW-1, 2, 2)
        t_seg = np.nanmean(np.stack([t[:-1], t[1:]]), axis=0)
        t_seg = np.nan_to_num(t_seg, nan=0.0)
        lws = (W_MIN + t_seg * (W_MAX - W_MIN)) * width_scale
        rgba = np.tile(np.array(to_rgba(col)), (len(segs), 1))
        rgba[:, 3] = A_MIN + t_seg * (A_MAX - A_MIN)
        # drop segments with a missing endpoint
        ok = np.isfinite(segs).all(axis=(1, 2))
        lc = LineCollection(segs[ok], linewidths=lws[ok], colors=rgba[ok],
                            capstyle='round', joinstyle='round', zorder=20 + ci)
        ax.add_collection(lc)

    ax.set_yticks([(nW - 1 - i) * offset_step for i in range(0, nW, 4)])
    ax.set_yticklabels([f'wk {weeks[i]}' for i in range(0, nW, 4)], fontsize=7)
    ax.set_xlabel('normalized position along each app\'s drift axis '
                  '(0 = early location, 1 = late location)')
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.5, nW + 2.5)
    ax.set_title(f'Density drift across {len(targets)} apps on a shared drift axis\n'
                 'line width ∝ mass concentration (thick+opaque = tight; thin+faint = spread)\n'
                 'every app: mass 0→1, and the track thins/fades through the empty middle',
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    # width/alpha legend (concentrated vs spread)
    demo = [Line2D([0], [0], color='dimgray', lw=W_MAX, alpha=A_MAX,
                   solid_capstyle='round', label='concentrated (one tight location)'),
            Line2D([0], [0], color='dimgray', lw=W_MIN + 0.4, alpha=A_MIN + 0.15,
                   solid_capstyle='round', label='spread out (transitional / bimodal)')]
    leg1 = ax.legend(handles=demo, loc='upper left', fontsize=7, framealpha=0.9,
                     title='line encoding', title_fontsize=7)
    ax.add_artist(leg1)

    handles = [Line2D([0], [0], color=colors[i], lw=2.5,
                      label=f'{targets[i]}: {names[i]}') for i in range(len(targets))]
    if n <= 10:
        ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
    else:
        # too many to sit inside — park it outside on the right
        ncol = 1 if n <= 24 else 2
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, framealpha=0.9, ncol=ncol, handlelength=1.2,
                  borderaxespad=0.0)

    fig.tight_layout()
    inf_tag = inference_week_tag(args.inference_dir)
    if args.min_week is not None or args.max_week is not None:
        inf_tag += f'_{weeks[0]}to{weeks[-1]}'
    out = Path(args.out_dir) / f'density_drift_multi_conc_{inf_tag}_{len(targets)}apps.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
