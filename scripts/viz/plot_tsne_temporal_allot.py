#!/usr/bin/env python
"""
Fig-1-style t-SNE teleportation panels for the ALLOT dataset.

Same rendering as plot_tsne_temporal_multi.py — per-app panels coloured by
time with a centroid track, plus a by-class background map — but over the
Allot windowed timeline (48 chronological ~2% windows; classes are private
numeric app IDs, closed world, 111 classes, labelled "app-<id>"), and with a
different fit policy: the t-SNE is fitted ONCE on ALL embeddings of every
window via openTSNE (no per-class sampling budgets), so the layout is unbiased
by the focal-app choice and the cached fit supports rendering ANY class later
(--classes ...) without a refit.

Focal apps are auto-ranked by the same drift-travel score used for CESNET
(range of the per-window centroid projected on the early→late axis, divided by
the within-class spread); the threshold on per-window embedding counts adapts
downward until enough classes qualify (Allot windows carry ~4.3k embeddings
over 111 classes, so the CESNET default of 20/week is often too strict).

Usage:
    python scripts/viz/plot_tsne_temporal_allot.py \
        --inference_dir exps/allot_multimodal/early_eq/inference --top 6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling viz modules

import argparse
import json

import numpy as np

from plot_density_drift import rank_classes
from plot_tsne_temporal_multi import render_app, render_background


def fit_tsne_all(wins, embs, labs, cache_path, perplexity, n_iter, seed, refit):
    """One t-SNE over ALL embeddings of every window — no per-class budgets, so
    the layout is unbiased by any focal-app choice and the cache supports
    rendering ANY class later without a refit.

    Uses openTSNE (FFT-accelerated FIt-SNE) so the full ~200k-point fit with a
    long optimisation is tractable; KL divergence is logged per 50 iterations
    (verbose) so convergence is checkable in the run log.
    """
    if cache_path.exists() and not refit:
        d = np.load(cache_path)
        print(f'loaded cached FULL t-SNE embedding: {cache_path} '
              f'({len(d["xy"])} points)')
        return d['xy'], d['cls'], d['week']

    X = np.concatenate(embs)
    cls = np.concatenate(labs)
    win = np.concatenate([np.full(len(l), w) for w, l in zip(wins, labs)])
    print(f'fitting t-SNE on ALL {len(X)} embeddings '
          f'(perplexity {perplexity}, {n_iter} iterations after early '
          f'exaggeration) ...')
    from openTSNE import TSNE
    emb = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
               initialization='pca', random_state=seed, n_jobs=-1,
               verbose=True).fit(X)
    # log a concrete convergence number (the per-iter verbose trace can get
    # swallowed by the slurm wrapper; this always lands in the .out log)
    print(f'  final KL divergence: {float(emb.kl_divergence):.4f}')
    xy = np.asarray(emb, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, xy=xy, cls=cls, week=win)
    print(f'saved FULL t-SNE cache -> {cache_path}')
    return xy, cls, win


def load_all_windows(inference_dir):
    """(window_nums, [embeddings], [labels-aligned-to-embeddings]) over the
    chronological window_*.npz files of an Allot inference run."""
    fs = sorted(Path(inference_dir).glob('window_*.npz'))
    wins, embs, labs = [], [], []
    for f in fs:
        d = np.load(f)
        wn = int(d['window_index']) if 'window_index' in d.files \
            else int(f.stem.split('_')[1])
        wins.append(wn)
        embs.append(d['embeddings'].astype(np.float32))
        labs.append(d['true_labels'][d['embedding_indices']].astype(np.int64))
    return wins, embs, labs


def allot_class_names(inference_dir):
    """Class index -> 'app-<original Allot id>' from the experiment's
    label_mapping.json (sits next to the inference dir)."""
    f = Path(inference_dir).resolve().parent / 'label_mapping.json'
    try:
        m = json.loads(f.read_text())
        return {v: f'app-{k}' for k, v in m.items()}
    except Exception as e:
        print(f'  (label mapping unavailable: {e})')
        return {}


def rank_classes_adaptive(wins, embs, labs, want=15):
    """rank_classes() with a per-window count threshold that backs off until
    enough classes are rankable (Allot windows are small)."""
    scores = []
    for thr in (20, 12, 8, 5, 3):
        scores = rank_classes(wins, embs, labs, min_per_week=thr)
        if len(scores) >= want:
            print(f'ranked {len(scores)} classes with >= {thr} embeddings '
                  f'in every window')
            return scores
    print(f'ranked only {len(scores)} classes (smallest threshold)')
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir',
                    default='exps/allot_multimodal/early_eq/inference')
    ap.add_argument('--out_dir', default='figs/allot_tsne')
    ap.add_argument('--cache_dir', default='results/tsne_cache')
    ap.add_argument('--classes', type=int, nargs='+', default=None,
                    help='explicit focal class indices (skip auto-ranking)')
    ap.add_argument('--top', type=int, default=12,
                    help='number of top-drift apps to render')
    ap.add_argument('--highlight_window', type=int, default=None,
                    help='by-class map for this window (default: first window)')
    ap.add_argument('--perplexity', type=float, default=40.0)
    ap.add_argument('--n_iter', type=int, default=3000,
                    help='t-SNE iterations after early exaggeration')
    ap.add_argument('--bg_frac', type=float, default=0.3,
                    help='fraction of grey background points to draw (rendering '
                         'only; thins the full-data haze, no effect on the fit)')
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    wins, embs, labs = load_all_windows(args.inference_dir)
    if not wins:
        raise SystemExit(f'no window_*.npz under {args.inference_dir}')
    print(f'windows {wins[0]}-{wins[-1]} ({len(wins)} files, '
          f'{sum(len(l) for l in labs)} embeddings)')
    idx2name = allot_class_names(args.inference_dir)

    if args.classes:
        focal = list(args.classes)
    else:
        scores = rank_classes_adaptive(wins, embs, labs)
        print('top drift-travel classes:')
        for c, s in scores[:max(args.top, 15)]:
            print(f'  {c:3d}  {idx2name.get(c, f"class{c}"):>10s}  travel={s:.2f}')
        focal = [c for c, _ in scores[:args.top]]
    names = {c: idx2name.get(c, f'class{c}') for c in focal}
    print('focal apps:', [(c, names[c]) for c in focal])

    hw = args.highlight_window if args.highlight_window is not None else wins[0]
    if hw not in wins:
        raise SystemExit(f'highlight_window {hw} not in available windows {wins}')

    slice_tag = Path(args.inference_dir).resolve().parent.name   # e.g. early_eq
    # cache is focal-agnostic: it holds the layout of EVERY embedding, so any
    # class can be rendered later (--classes ...) without refitting
    cache_path = (Path(args.cache_dir) /
                  f'tsne_allot_{slice_tag}_full_p{int(args.perplexity)}'
                  f'_i{args.n_iter}.npz')

    xy, cls, win = fit_tsne_all(wins, embs, labs, cache_path,
                                args.perplexity, args.n_iter, args.seed,
                                args.refit)

    # shared axis limits (same frame for every figure, so plots are comparable)
    xlo, xhi = np.percentile(xy[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(xy[:, 1], [0.5, 99.5])
    px, py = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
    xlim, ylim = (xlo - px, xhi + px), (ylo - py, yhi + py)

    out_dir = Path(args.out_dir)
    for c in focal:
        render_app(xy, cls, win, c, names[c], xlim, ylim,
                   out_dir / f'tsne_allot_{slice_tag}_{c:03d}_{names[c]}.png',
                   time_label='window', bg_frac=args.bg_frac)
    print(f'done: {len(focal)} per-app plots in {out_dir}')

    render_background(xy, cls, win, hw, xlim, ylim,
                      out_dir / f'tsne_allot_{slice_tag}_background_win{hw}_byclass.png',
                      time_label='window', bg_frac=args.bg_frac)
    print(f'done: by-class map for window {hw}')


if __name__ == '__main__':
    main()
