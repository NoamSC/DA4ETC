#!/usr/bin/env python
"""
Density-drift visualization — is class drift *continuous motion* or *discrete
mass-transfer between fixed spatial modes*?

Thesis: a class centroid appears to move smoothly through feature space, but that
smooth motion is an artifact. The samples actually live in a few *fixed* spatial
locations (modes); what changes continuously over time is the *density/mass* on
each location. The centroid, being a weighted mean of fixed modes, then slides
smoothly through the (near-empty) valley between them — even though essentially no
sample ever sits where the centroid is.

We draw a ridgeline of the 1-D sample density along the drift axis, one row per
week:
  - x = position along the *fixed* drift direction (a spatial coordinate)
  - rows top->bottom = time (weeks)
  - overlaid: per-week centroid, connected into a track
CONTINUOUS-MOTION predicts: a single peak that slides with the centroid.
DISCRETE-MODES predicts: peaks pinned at fixed x; only their heights swap; the
centroid track crosses the empty valley between them.

(For the global context view — focal class drift over a PCA map of all weeks —
see the companion script plot_pca_temporal.py.)

Usage:
    python plot_density_drift.py --class_idx 98
    python plot_density_drift.py --top 6           # batch the 6 highest-drift classes
    python plot_density_drift.py --class_idx 102 --n_modes 3
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: config, data_utils, ...

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def week_num(path):
    return int(re.search(r'(\d+)$', Path(path).stem).group(1))


def inference_week_tag(inference_dir):
    """Training/inference week parsed from the dir name, e.g. 'week_1_inference' -> 'w1'."""
    m = re.search(r'week[_-]?(\d+)', Path(inference_dir).name, re.I)
    return f'w{m.group(1)}' if m else 'wNA'


def load_all(inference_dir, min_week=None, max_week=None):
    fs = sorted(Path(inference_dir).glob('WEEK-2022-*.npz'), key=week_num)
    weeks, embs, labs = [], [], []
    for f in fs:
        wn = week_num(f)
        if (min_week is not None and wn < min_week) or \
           (max_week is not None and wn > max_week):
            continue
        d = np.load(f)
        emb = d['embeddings']
        lab = d['true_labels'][d['embedding_indices']]   # labels aligned to embeddings
        weeks.append(wn); embs.append(emb); labs.append(lab)
    return weeks, embs, labs


def class_names(dataset_root):
    try:
        from data_utils.cesnet_labels import load_label_mapping
        mapping, _ = load_label_mapping(Path(dataset_root))
        return {v: k for k, v in mapping.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Per-class geometry
# ---------------------------------------------------------------------------

def drift_axis(embs, labs, c, n_edge=8):
    """Unit vector from the early-weeks mean to the late-weeks mean of class c."""
    early = np.concatenate([e[l == c] for e, l in zip(embs[:n_edge], labs[:n_edge])])
    late  = np.concatenate([e[l == c] for e, l in zip(embs[-n_edge:], labs[-n_edge:])])
    v = late.mean(0) - early.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def gather_class(embs, labs, c):
    """All embeddings of class c, plus per-week index slices."""
    per_week = [e[l == c] for e, l in zip(embs, labs)]
    pooled = np.concatenate(per_week)
    return pooled, per_week


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(weeks, embs, labs, c, idx2name, n_modes=2, out_dir='figs', inf_tag='wNA'):
    name = idx2name.get(c, f'class{c}')
    pooled, per_week = gather_class(embs, labs, c)
    v = drift_axis(embs, labs, c)

    # ---- 1-D projections along the fixed drift axis ----
    proj_week = [pw @ v for pw in per_week]                  # list of 1-D arrays
    proj_all  = np.concatenate(proj_week)
    lo, hi = np.percentile(proj_all, [0.5, 99.5])
    xs = np.linspace(lo, hi, 400)
    centroids = np.array([p.mean() for p in proj_week])      # per-week mean along axis

    # global antimode (valley) between the two dominant 1-D modes
    gm1 = GaussianMixture(n_modes, covariance_type='full', random_state=0).fit(proj_all[:, None])
    dens_all = np.exp(gm1.score_samples(xs[:, None]))
    order = np.argsort(gm1.means_.ravel())
    m_lo, m_hi = gm1.means_.ravel()[order[0]], gm1.means_.ravel()[order[-1]]
    valley_mask = (xs > m_lo) & (xs < m_hi)
    valley_x = xs[valley_mask][np.argmin(dens_all[valley_mask])] if valley_mask.any() else None

    # =====================================================================
    fig, ax_ridge = plt.subplots(figsize=(9, 11))

    nW = len(weeks)
    tcolors = cm.viridis(np.linspace(0, 1, nW))

    # ---- Ridgeline: sample density along the fixed drift axis, per week ----
    offset_step = 1.0
    peak = max(np.max(gaussian_kde(p)(xs)) if len(p) > 2 else 0 for p in proj_week)
    yscale = 0.9 * offset_step / (peak + 1e-9)
    cent_pts = []
    for i, p in enumerate(proj_week):
        base = (nW - 1 - i) * offset_step      # week 0 at top
        if len(p) > 2:
            dens = gaussian_kde(p)(xs) * yscale
            ax_ridge.fill_between(xs, base, base + dens, color=tcolors[i],
                                  alpha=0.75, lw=0.4, edgecolor='white', zorder=i)
        # centroid marker on this ridge
        ci = centroids[i]
        ax_ridge.plot([ci], [base], 'o', color='black', ms=3, zorder=nW + 2)
        cent_pts.append((ci, base))
    # centroid track
    cp = np.array(cent_pts)
    ax_ridge.plot(cp[:, 0], cp[:, 1], '-', color='crimson', lw=2.2, zorder=nW + 1,
                  label='centroid track')
    if valley_x is not None:
        ax_ridge.axvline(valley_x, color='gray', ls='--', lw=1.2, alpha=0.8, zorder=0)
        ax_ridge.text(valley_x, nW * offset_step * 0.5, '  empty valley\n  (centroid lives here)',
                      rotation=90, va='center', ha='left', fontsize=8, color='gray')
    ax_ridge.set_yticks([(nW - 1 - i) * offset_step for i in range(0, nW, 4)])
    ax_ridge.set_yticklabels([f'wk {weeks[i]}' for i in range(0, nW, 4)], fontsize=7)
    ax_ridge.set_xlabel('position along fixed drift axis  (early-mean → late-mean direction)')
    ax_ridge.set_title(f'Sample density along the drift axis, week by week — class {c}: {name}\n'
                       'fixed peaks + shifting heights = discrete modes; '
                       'centroid (red) slides through the empty valley', fontsize=10)
    ax_ridge.legend(loc='lower right', fontsize=8)
    ax_ridge.set_ylim(-0.5, nW * offset_step + 0.5)
    ax_ridge.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    out = Path(out_dir) / f'density_drift_{inf_tag}_{c:03d}_{name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out}')
    return out


# ---------------------------------------------------------------------------
# Class ranking (highest centroid-travel relative to within-week spread)
# ---------------------------------------------------------------------------

def rank_classes(weeks, embs, labs, min_per_week=20):
    common = None
    for l in labs:
        s = {u for u, n in Counter(l).items() if n >= min_per_week}
        common = s if common is None else (common & s)
    scores = []
    for c in sorted(common):
        v = drift_axis(embs, labs, c)
        means, allp = [], []
        for e, l in zip(embs, labs):
            p = e[l == c] @ v; means.append(p.mean()); allp.append(p)
        means = np.array(means); allp = np.concatenate(allp)
        scores.append((c, (means.max() - means.min()) / (allp.std() + 1e-9)))
    scores.sort(key=lambda x: -x[1])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--out_dir', default='figs/drift')
    ap.add_argument('--class_idx', type=int, default=None)
    ap.add_argument('--top', type=int, default=None,
                    help='batch the N highest drift-travel classes')
    ap.add_argument('--n_modes', type=int, default=2)
    ap.add_argument('--min_week', type=int, default=None,
                    help='only use weeks >= this (e.g. 16 for the week-16 model going forward)')
    ap.add_argument('--max_week', type=int, default=None)
    args = ap.parse_args()

    weeks, embs, labs = load_all(args.inference_dir, args.min_week, args.max_week)
    print(f'loaded {len(weeks)} weeks: {weeks[0]}-{weeks[-1]}')
    idx2name = class_names(args.dataset_root)
    inf_tag = inference_week_tag(args.inference_dir)
    if args.min_week is not None or args.max_week is not None:
        inf_tag += f'_{weeks[0]}to{weeks[-1]}'

    if args.class_idx is not None:
        targets = [args.class_idx]
    else:
        ranked = rank_classes(weeks, embs, labs)
        n = args.top or 1
        targets = [c for c, _ in ranked[:n]]
        print('top drift classes:', [(c, idx2name.get(c, c), round(s, 2)) for c, s in ranked[:n]])

    for c in targets:
        make_figure(weeks, embs, labs, c, idx2name, n_modes=args.n_modes,
                    out_dir=args.out_dir, inf_tag=inf_tag)


if __name__ == '__main__':
    main()
