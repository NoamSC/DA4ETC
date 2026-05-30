#!/usr/bin/env python
"""
Shared-fit t-SNE temporal drift for the top-N apps.

t-SNE has no out-of-sample transform and every fit is a different layout, so we
fit it ONCE on (background = all other classes) + (the N focal apps across all
weeks), cache that 2-D embedding, and then render one plot per focal app from the
SAME map. Result: N mutually consistent figures for the price of a single t-SNE.

Each rendered plot (dark theme, à la figs/domain_gap_tsne.png):
  - background = everything except the highlighted app (grey)
  - highlighted app = its samples coloured by week (plasma) + a centroid track
All N plots share identical axes, so positions are directly comparable.

The fit is cached under results/tsne_cache/ so re-rendering (e.g. tweaking styling)
skips the expensive step. Pass --refit to recompute.

Usage:
    python scripts/viz/plot_tsne_temporal_multi.py --top 10
    python scripts/viz/plot_tsne_temporal_multi.py \
        --inference_dir results/inference/week_16_inference --min_week 16 --top 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config, data_utils, ...

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from plot_density_drift import load_all, class_names, rank_classes, inference_week_tag


# ---------------------------------------------------------------------------
# Sampling + shared t-SNE fit (cached)
# ---------------------------------------------------------------------------

def build_or_load_embedding(weeks, embs, labs, focal, cache_path,
                            max_bg_per_class, max_focal_per_week, perplexity,
                            seed, refit):
    """Return (xy, app, week_of) where app[i] is the class id of focal point i or
    -1 for background, and week_of[i] is its week (or -1). Cached to cache_path."""
    if cache_path.exists() and not refit:
        d = np.load(cache_path)
        if list(d['focal']) == list(focal):
            print(f'loaded cached t-SNE embedding: {cache_path}')
            return d['xy'], d['app'], d['week']
        print('cache focal set differs — refitting')

    rng = np.random.RandomState(seed)
    focal_set = set(focal)
    per_week_bg = max(1, max_bg_per_class // len(weeks) + 1)

    chunks, app_ids, week_ids = [], [], []
    for wn, emb, lab in zip(weeks, embs, labs):
        for c in focal:                                   # focal apps: dense per week
            idx = np.where(lab == c)[0]
            if len(idx) > max_focal_per_week:
                idx = rng.choice(idx, max_focal_per_week, replace=False)
            if len(idx):
                chunks.append(emb[idx]); app_ids.append(np.full(len(idx), c)); week_ids.append(np.full(len(idx), wn))
        for c in np.unique(lab):                          # background: a few per class/week
            if c in focal_set:
                continue
            idx = np.where(lab == c)[0]
            if len(idx) > per_week_bg:
                idx = rng.choice(idx, per_week_bg, replace=False)
            if len(idx):
                chunks.append(emb[idx]); app_ids.append(np.full(len(idx), -1)); week_ids.append(np.full(len(idx), -1))

    X = np.concatenate(chunks)
    app = np.concatenate(app_ids)
    week = np.concatenate(week_ids)
    n_focal = int((app >= 0).sum())
    print(f'fitting t-SNE once on {len(X)} points ({n_focal} focal across {len(focal)} apps, '
          f'{len(X) - n_focal} background), perplexity {perplexity}...')

    xy = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, init='pca',
              random_state=seed, n_jobs=-1, verbose=1).fit_transform(X)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, xy=xy, app=app, week=week, focal=np.array(focal))
    print(f'saved t-SNE cache -> {cache_path}')
    return xy, app, week


# ---------------------------------------------------------------------------
# Per-app rendering (from the shared embedding)
# ---------------------------------------------------------------------------

def render_app(xy, app, week, c, name, xlim, ylim, out_path):
    BG = '#0f0f14'
    is_focal = app == c
    fxy, fwk = xy[is_focal], week[is_focal]
    uw = np.array(sorted(np.unique(fwk)))
    cent = np.array([fxy[fwk == w].mean(0) for w in uw])

    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # everything except this app -> grey context
    ax.scatter(xy[~is_focal, 0], xy[~is_focal, 1], s=3, c='#c8c8d0', alpha=0.28,
               linewidths=0, zorder=1, rasterized=True)

    norm = plt.Normalize(uw.min(), uw.max())
    sc = ax.scatter(fxy[:, 0], fxy[:, 1], s=14, c=fwk, cmap='plasma', norm=norm,
                    alpha=0.85, linewidths=0.2, edgecolors='white', zorder=3, rasterized=True)

    seg = np.stack([cent[:-1], cent[1:]], axis=1)
    ax.add_collection(LineCollection(seg, cmap='plasma', norm=norm, array=uw[:-1],
                                     linewidths=2.0, alpha=0.9, zorder=4))
    ax.scatter(*cent[0], marker='o', s=120, facecolor='none', edgecolor='white', linewidths=1.8, zorder=5)
    ax.scatter(*cent[-1], marker='s', s=120, facecolor='none', edgecolor='white', linewidths=1.8, zorder=5)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('t-SNE 1', color='white'); ax.set_ylabel('t-SNE 2', color='white')
    ax.set_title(f'Latent space: {name} drift over time  (shared t-SNE over top apps)\n'
                 f'grey = all other points  |  coloured = {name}, weeks {uw.min()}–{uw.max()}  |  '
                 '○ first week  □ last week', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('#444')
    cbar = fig.colorbar(sc, ax=ax, pad=0.01); cbar.set_label('week', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#c8c8d0',
                              markersize=6, label='other points')],
              loc='upper right', fontsize=8, facecolor='#1a1a22', edgecolor='#444', labelcolor='white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  saved -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--out_dir', default='figs/drift')
    ap.add_argument('--cache_dir', default='results/tsne_cache')
    ap.add_argument('--classes', type=int, nargs='+', default=None,
                    help='explicit focal class ids (overrides --top)')
    ap.add_argument('--top', type=int, default=10, help='number of top-drift apps')
    ap.add_argument('--max_bg_per_class', type=int, default=25)
    ap.add_argument('--max_focal_per_week', type=int, default=80)
    ap.add_argument('--perplexity', type=float, default=40.0)
    ap.add_argument('--min_week', type=int, default=None)
    ap.add_argument('--max_week', type=int, default=None)
    ap.add_argument('--refit', action='store_true', help='recompute the t-SNE fit')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    weeks, embs, labs = load_all(args.inference_dir, args.min_week, args.max_week)
    print(f'weeks {weeks[0]}-{weeks[-1]} ({len(weeks)} files)')
    idx2name = class_names(args.dataset_root)

    if args.classes:
        focal = list(args.classes)
    else:
        focal = [c for c, _ in rank_classes(weeks, embs, labs)[:args.top]]
    names = {c: idx2name.get(c, f'class{c}') for c in focal}
    print('focal apps:', [(c, names[c]) for c in focal])

    tag = inference_week_tag(args.inference_dir)
    if args.min_week is not None or args.max_week is not None:
        tag += f'_{weeks[0]}to{weeks[-1]}'
    cache_path = Path(args.cache_dir) / f'tsne_{tag}_top{len(focal)}.npz'

    xy, app, week = build_or_load_embedding(
        weeks, embs, labs, focal, cache_path,
        args.max_bg_per_class, args.max_focal_per_week, args.perplexity, args.seed, args.refit)

    # shared axis limits (same frame for every app, so plots are comparable)
    xlo, xhi = np.percentile(xy[:, 0], [0.5, 99.5]); ylo, yhi = np.percentile(xy[:, 1], [0.5, 99.5])
    px, py = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    xlim, ylim = (xlo - px, xhi + px), (ylo - py, yhi + py)

    out_dir = Path(args.out_dir)
    for c in focal:
        render_app(xy, app, week, c, names[c], xlim, ylim,
                   out_dir / f'tsne_temporal_{tag}_{c:03d}_{names[c]}.png')
    print(f'done: {len(focal)} plots in {out_dir}')


if __name__ == '__main__':
    main()
