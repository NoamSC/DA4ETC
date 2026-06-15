#!/usr/bin/env python
"""
Shared-fit t-SNE temporal drift for the top-N apps (+ optional by-class map).

t-SNE has no out-of-sample transform and every fit is a different layout, so we
fit it ONCE on a single sample and render every figure from the SAME 2-D map.
That single sample contains:
  - the N focal apps, dense across all weeks   (for the per-app drift tracks)
  - every class at one highlight week, dense    (for the by-class background map)
  - a sparse cloud of every class/week          (grey context)
Because all figures come from one embedding, the point cloud is IDENTICAL across
them and only the colouring changes — the per-app plots and the by-class map are
directly overlay-comparable.

Each per-app plot (dark theme, à la figs/domain_gap_tsne.png):
  - background = everything except the highlighted app (grey)
  - highlighted app = its samples coloured by week (plasma) + a centroid track
The by-class map (when --highlight_week is given):
  - background = every other week (grey)
  - highlight week = its samples, one distinct colour per class

The fit is cached under results/tsne_cache/ so re-rendering (e.g. tweaking styling)
skips the expensive step. Pass --refit to recompute.

Usage:
    python scripts/viz/plot_tsne_temporal_multi.py --top 10
    python scripts/viz/plot_tsne_temporal_multi.py \
        --inference_dir results/inference/week_16_inference --min_week 16 \
        --classes 19 49 56 57 97 98 101 102 140 167 --highlight_week 16
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from plot_density_drift import load_all, class_names, rank_classes, inference_week_tag


# ---------------------------------------------------------------------------
# Sampling + shared t-SNE fit (cached)
# ---------------------------------------------------------------------------

def build_or_load_embedding(weeks, embs, labs, focal, cache_path,
                            max_bg_per_class, max_focal_per_week, perplexity,
                            seed, refit, highlight_week=None, max_highlight_per_class=60):
    """Return (xy, cls, week) where, for EVERY point i, cls[i] is its class id and
    week[i] is its (real) week. A point can serve multiple figures: focal-app
    plots read the points whose cls == focal; the by-class map reads the points
    whose week == highlight_week. Cached to cache_path."""
    hw = -1 if highlight_week is None else highlight_week
    if cache_path.exists() and not refit:
        d = np.load(cache_path)
        if list(d['focal']) == list(focal) and int(d['highlight_week']) == hw:
            print(f'loaded cached t-SNE embedding: {cache_path}')
            return d['xy'], d['cls'], d['week']
        print('cache focal set / highlight week differs — refitting')

    rng = np.random.RandomState(seed)
    focal_set = set(focal)
    per_week_bg = max(1, max_bg_per_class // len(weeks) + 1)

    # One pass: for each (week, class) take a single sample whose size is the max
    # budget across all roles that point plays. No double-sampling, so the focal
    # plots and the by-class map share the exact same underlying points.
    chunks, cls_ids, week_ids = [], [], []
    for wn, emb, lab in zip(weeks, embs, labs):
        for c in np.unique(lab):
            budget = per_week_bg
            if c in focal_set:
                budget = max(budget, max_focal_per_week)
            if highlight_week is not None and wn == highlight_week:
                budget = max(budget, max_highlight_per_class)
            idx = np.where(lab == c)[0]
            if len(idx) > budget:
                idx = rng.choice(idx, budget, replace=False)
            if len(idx):
                chunks.append(emb[idx]); cls_ids.append(np.full(len(idx), c)); week_ids.append(np.full(len(idx), wn))

    X = np.concatenate(chunks)
    cls = np.concatenate(cls_ids)
    week = np.concatenate(week_ids)
    n_focal = int(np.isin(cls, list(focal_set)).sum())
    print(f'fitting t-SNE once on {len(X)} points ({n_focal} focal-class, '
          f'{len(X) - n_focal} other; highlight_week={hw}), perplexity {perplexity}...')

    xy = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, init='pca',
              random_state=seed, n_jobs=-1, verbose=1).fit_transform(X)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, xy=xy, cls=cls, week=week, focal=np.array(focal), highlight_week=hw)
    print(f'saved t-SNE cache -> {cache_path}')
    return xy, cls, week


# ---------------------------------------------------------------------------
# Per-app rendering (from the shared embedding)
# ---------------------------------------------------------------------------

def render_app(xy, cls, week, c, name, xlim, ylim, out_path, time_label='week',
               bg_frac=1.0):
    BG = '#0f0f14'
    is_focal = cls == c
    fxy, fwk = xy[is_focal], week[is_focal]
    uw = np.array(sorted(np.unique(fwk)))
    cent = np.array([fxy[fwk == w].mean(0) for w in uw])

    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # everything except this app -> grey context (optionally thinned out so a
    # huge full-data background doesn't bury the focal track)
    bgxy = xy[~is_focal]
    if bg_frac < 1.0:
        sel = np.random.default_rng(0).random(len(bgxy)) < bg_frac
        bgxy = bgxy[sel]
    ax.scatter(bgxy[:, 0], bgxy[:, 1], s=3, c='#c8c8d0', alpha=0.28,
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
                 f'grey = all other points  |  coloured = {name}, {time_label}s '
                 f'{uw.min()}–{uw.max()}  |  '
                 f'○ first {time_label}  □ last {time_label}', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('#444')
    cbar = fig.colorbar(sc, ax=ax, pad=0.01); cbar.set_label(time_label, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
    ax.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor='#c8c8d0',
                              markersize=6, label='other points')],
              loc='upper right', fontsize=8, facecolor='#1a1a22', edgecolor='#444', labelcolor='white')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  saved -> {out_path}')


# ---------------------------------------------------------------------------
# By-class "highlight week over the all-weeks background" map (same embedding)
# ---------------------------------------------------------------------------

def render_background(xy, cls, week, highlight_week, xlim, ylim, out_path,
                      time_label='week', bg_frac=1.0):
    BG = '#0f0f14'
    is_hi = week == highlight_week
    hi_cls = cls[is_hi]
    uniq = np.array(sorted(np.unique(hi_cls)))
    palette = cm.gist_ncar(np.linspace(0.02, 0.98, len(uniq)))   # one distinct colour per class
    color_of = {c: palette[i] for i, c in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # all other weeks -> grey context (drawn first, underneath; optionally thinned)
    bgxy = xy[~is_hi]
    if bg_frac < 1.0:
        sel = np.random.default_rng(0).random(len(bgxy)) < bg_frac
        bgxy = bgxy[sel]
    ax.scatter(bgxy[:, 0], bgxy[:, 1], s=3, c='#c8c8d0', alpha=0.28,
               linewidths=0, zorder=1, rasterized=True)

    # highlight week -> one colour per class, on top of the grey
    hxy = xy[is_hi]
    cols = np.array([color_of[c] for c in hi_cls])
    ax.scatter(hxy[:, 0], hxy[:, 1], s=12, c=cols, alpha=0.9,
               linewidths=0.15, edgecolors='#0f0f14', zorder=3, rasterized=True)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('t-SNE 1', color='white'); ax.set_ylabel('t-SNE 2', color='white')
    ax.set_title(f'Latent space: {time_label} {highlight_week} classes over the '
                 f'all-{time_label}s background  (shared t-SNE over top apps)\n'
                 f'grey = all other {time_label}s  |  coloured = {time_label} {highlight_week}, '
                 f'one colour per class ({len(uniq)} classes)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for s in ax.spines.values():
        s.set_color('#444')

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
    ap.add_argument('--max_highlight_per_class', type=int, default=60)
    ap.add_argument('--highlight_week', type=int, default=None,
                    help='if given, also render a by-class map of this week from the SAME fit')
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

    if args.highlight_week is not None and args.highlight_week not in weeks:
        raise SystemExit(f'highlight_week {args.highlight_week} not in available weeks {weeks}')

    tag = inference_week_tag(args.inference_dir)
    if args.min_week is not None or args.max_week is not None:
        tag += f'_{weeks[0]}to{weeks[-1]}'
    hi_suffix = '' if args.highlight_week is None else f'_hi{args.highlight_week}'
    cache_path = Path(args.cache_dir) / f'tsne_{tag}_top{len(focal)}{hi_suffix}.npz'

    xy, cls, week = build_or_load_embedding(
        weeks, embs, labs, focal, cache_path,
        args.max_bg_per_class, args.max_focal_per_week, args.perplexity, args.seed, args.refit,
        highlight_week=args.highlight_week, max_highlight_per_class=args.max_highlight_per_class)

    # shared axis limits (same frame for every figure, so plots are comparable)
    xlo, xhi = np.percentile(xy[:, 0], [0.5, 99.5]); ylo, yhi = np.percentile(xy[:, 1], [0.5, 99.5])
    px, py = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    xlim, ylim = (xlo - px, xhi + px), (ylo - py, yhi + py)

    out_dir = Path(args.out_dir)
    for c in focal:
        render_app(xy, cls, week, c, names[c], xlim, ylim,
                   out_dir / f'tsne_temporal_{tag}_{c:03d}_{names[c]}.png')
    print(f'done: {len(focal)} per-app plots in {out_dir}')

    if args.highlight_week is not None:
        render_background(xy, cls, week, args.highlight_week, xlim, ylim,
                          out_dir / f'tsne_background_{tag}_week{args.highlight_week}_byclass.png')
        print(f'done: by-class map for week {args.highlight_week}')


if __name__ == '__main__':
    main()
