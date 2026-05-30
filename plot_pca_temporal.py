#!/usr/bin/env python
"""
Focal-class drift over a global PCA map of *all weeks* — static PNG.

Companion to plot_density_drift.py. Where the ridgeline shows the drift along a
single axis, this shows it in the full latent context: where the focal class sits
relative to *the rest of the data*, and how it migrates across that map over time.

Layout (dark theme, à la figs/domain_gap_tsne.png):
  - PCA(2) is fit on the data from ALL weeks (focal class every week + a sample of
    every other class) so the projection is a single, fixed global map.
  - Background (grey): all other classes — the latent cluster structure / context.
  - Focal class: samples from each week, coloured by week (plasma), so you can see
    the cloud move across the map. A faint track connects the weekly centroids.

Output: static PNG.

Usage:
    python plot_pca_temporal.py --focal_class 98
    python plot_pca_temporal.py --focal_class 102 --max_bg_per_class 40 \
        --max_focal_per_week 150 --output figs/pca_temporal_microsoft-settings.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# ── label mapping ───────────────────────────────────────────────────────────

def load_class_names(dataset_root):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_per_week_cesnet import load_label_mapping
    mapping, num_classes = load_label_mapping(Path(dataset_root))
    return {v: k for k, v in mapping.items()}, num_classes


# ── data loading ────────────────────────────────────────────────────────────

def week_num(path):
    return int(re.search(r'(\d+)$', Path(path).stem).group(1))


def load_npz(path):
    d = np.load(path)
    return d['embeddings'], d['true_labels'][d['embedding_indices']]


def sample_class(emb, lab, c, max_n, rng):
    idx = np.where(lab == c)[0]
    if max_n and len(idx) > max_n:
        idx = rng.choice(idx, max_n, replace=False)
    return emb[idx]


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='figs/week_1_inference')
    ap.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--focal_class', type=int, default=98)
    ap.add_argument('--max_bg_per_class', type=int, default=25,
                    help='Background samples per other class, pooled across all weeks')
    ap.add_argument('--max_focal_per_week', type=int, default=120,
                    help='Focal-class samples drawn per week')
    ap.add_argument('--output', default=None)
    ap.add_argument('--out_dir', default='figs/drift')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_week', type=int, default=None,
                    help='only use weeks >= this (e.g. 16 for the week-16 model going forward)')
    ap.add_argument('--max_week', type=int, default=None)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    inference_dir = Path(args.inference_dir)
    npz_files = sorted(inference_dir.glob('WEEK-2022-*.npz'), key=week_num)
    npz_files = [f for f in npz_files
                 if (args.min_week is None or week_num(f) >= args.min_week)
                 and (args.max_week is None or week_num(f) <= args.max_week)]
    weeks = [week_num(f) for f in npz_files]
    print(f'weeks {weeks[0]}-{weeks[-1]} ({len(weeks)} files)')

    class_names, num_classes = load_class_names(args.dataset_root)
    focal = args.focal_class
    focal_name = class_names.get(focal, f'class_{focal}')
    print(f'focal class {focal} = {focal_name}')

    # ── focal class across all weeks, + background from all weeks ──────────────
    focal_embs, focal_weeks = [], []
    bg_pool = {}                       # class -> list of per-week sampled arrays
    for f in npz_files:
        wn = week_num(f)
        emb, lab = load_npz(f)

        fe = sample_class(emb, lab, focal, args.max_focal_per_week, rng)
        if len(fe):
            focal_embs.append(fe); focal_weeks.extend([wn] * len(fe))

        # spread the background budget across weeks so it reflects all weeks
        per_week_bg = max(1, args.max_bg_per_class // len(npz_files) + 1)
        for c in np.unique(lab):
            if c == focal:
                continue
            be = sample_class(emb, lab, c, per_week_bg, rng)
            bg_pool.setdefault(c, []).append(be)

    focal_emb = np.concatenate(focal_embs)
    focal_weeks = np.array(focal_weeks)
    bg_emb = np.concatenate([np.concatenate(v) for v in bg_pool.values()])
    print(f'focal points: {len(focal_emb)}   background points: {len(bg_emb)} '
          f'({len(bg_pool)} classes)')

    # ── PCA fit on the data from ALL weeks (focal + background union) ──────────
    all_emb = np.concatenate([bg_emb, focal_emb])
    print(f'fitting PCA(2) on {len(all_emb)} points (dim {all_emb.shape[1]})...')
    pca = PCA(2, random_state=args.seed).fit(all_emb)
    bg_xy = pca.transform(bg_emb)
    focal_xy = pca.transform(focal_emb)
    var = pca.explained_variance_ratio_ * 100

    # weekly centroids (in PCA space) for the movement track
    uw = np.array(sorted(np.unique(focal_weeks)))
    cent = np.array([focal_xy[focal_weeks == w].mean(0) for w in uw])

    # ── plot (dark theme) ─────────────────────────────────────────────────────
    BG = '#0f0f14'
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    ax.scatter(bg_xy[:, 0], bg_xy[:, 1], s=3, c='#c8c8d0', alpha=0.30,
               linewidths=0, zorder=1, rasterized=True)

    wmin, wmax = uw.min(), uw.max()
    norm = plt.Normalize(wmin, wmax)
    sc = ax.scatter(focal_xy[:, 0], focal_xy[:, 1], s=14,
                    c=focal_weeks, cmap='plasma', norm=norm,
                    alpha=0.85, linewidths=0.2, edgecolors='white', zorder=3,
                    rasterized=True)

    # faint centroid-movement track over the coloured cloud
    seg = np.stack([cent[:-1], cent[1:]], axis=1)
    lc = LineCollection(seg, cmap='plasma', norm=norm, array=uw[:-1],
                        linewidths=2.0, alpha=0.9, zorder=4)
    ax.add_collection(lc)
    ax.scatter(*cent[0], marker='o', s=120, facecolor='none', edgecolor='white',
               linewidths=1.8, zorder=5)
    ax.scatter(*cent[-1], marker='s', s=120, facecolor='none', edgecolor='white',
               linewidths=1.8, zorder=5)

    # robust framing: clip axes to the bulk so a few outliers don't zoom us out.
    # base the window on the focal cloud (the subject) plus the dense background.
    # frame on the bulk of ALL points (background + focal), dropping outliers.
    frame = np.vstack([bg_xy, focal_xy])
    xlo, xhi = np.percentile(frame[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(frame[:, 1], [0.5, 99.5])
    px = 0.05 * (xhi - xlo + 1e-9); py = 0.05 * (yhi - ylo + 1e-9)
    ax.set_xlim(xlo - px, xhi + px); ax.set_ylim(ylo - py, yhi + py)

    ax.set_xlabel(f'PC1 ({var[0]:.1f}% var)', color='white')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% var)', color='white')
    ax.set_title(f'Latent space: {focal_name} drift over time  (PCA fit on all weeks)\n'
                 f'background = all other classes (grey)  |  '
                 f'coloured = {focal_name}, weeks {wmin}–{wmax}  |  '
                 f'○ first week  □ last week',
                 color='white', fontsize=12)
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('#444')

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('week', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#c8c8d0',
               markersize=6, label='other classes'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              facecolor='#1a1a22', edgecolor='#444', labelcolor='white')

    m = re.search(r'week[_-]?(\d+)', inference_dir.name, re.I)
    inf_tag = f'w{m.group(1)}' if m else 'wNA'
    if args.min_week is not None or args.max_week is not None:
        inf_tag += f'_{weeks[0]}to{weeks[-1]}'
    out = Path(args.output) if args.output else \
        Path(args.out_dir) / f'pca_temporal_{inf_tag}_{focal:03d}_{focal_name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
