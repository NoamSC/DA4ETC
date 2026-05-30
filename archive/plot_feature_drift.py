#!/usr/bin/env python
"""
Feature-space drift analysis — original embedding space, no dimensionality reduction tricks.

Three arguments against/for discrete vs. continuous drift:

1. Global distance from reference week over time
   - Centroid L2 distance (first moment)
   - Sliced Wasserstein distance (full distribution, 100 random projections)

2. Week-over-week delta
   - Consecutive centroid distance: flat = continuous, spikes = discrete jumps

3. Per-class centroid drift heatmap (classes × weeks)
   - Which classes are responsible for large jumps?

Also overlays model accuracy for direct comparison.

Usage:
    python plot_feature_drift.py
    python plot_feature_drift.py --inference_dir figs/week_1_inference --output figs/feature_drift.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sliced_wasserstein(X, Y, n_projections=100, seed=0):
    """Sliced Wasserstein distance: average 1-D Wasserstein over random projections."""
    rng = np.random.RandomState(seed)
    directions = rng.randn(n_projections, X.shape[1])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    total = 0.0
    for v in directions:
        total += wasserstein_distance(X @ v, Y @ v)
    return total / n_projections


def centroid_l2(X, Y):
    return float(np.linalg.norm(X.mean(0) - Y.mean(0)))


def per_class_centroid_dist(emb_ref, labels_ref, emb_t, labels_t):
    """Return dict class -> L2(centroid_ref, centroid_t). Only for shared classes."""
    classes = set(np.unique(labels_ref)) & set(np.unique(labels_t))
    result = {}
    for c in classes:
        mu_ref = emb_ref[labels_ref == c].mean(0)
        mu_t   = emb_t[labels_t == c].mean(0)
        result[c] = float(np.linalg.norm(mu_ref - mu_t))
    return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def week_num(path):
    return int(re.search(r'(\d+)$', Path(path).stem).group(1))


def load_npz(path, subsample=4000, seed=0):
    data = np.load(path)
    emb, labels = data['embeddings'], data['true_labels']
    if subsample and len(emb) > subsample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(emb), subsample, replace=False)
        emb, labels = emb[idx], labels[idx]
    acc = float((data['true_labels'] == data['pred_labels']).mean())
    return emb, labels, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir', default='figs/week_1_inference')
    parser.add_argument('--dataset_root',  default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--output',        default='figs/feature_drift.png')
    parser.add_argument('--subsample',     type=int, default=4000,
                        help='Samples per week for Wasserstein (default 4000)')
    parser.add_argument('--n_proj',        type=int, default=100,
                        help='Sliced Wasserstein projections (default 100)')
    parser.add_argument('--top_classes',   type=int, default=30,
                        help='Classes to show in heatmap (sorted by max drift)')
    parser.add_argument('--seed',          type=int, default=0)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    npz_files = sorted(inference_dir.glob('WEEK-2022-*.npz'), key=lambda p: week_num(p))
    weeks = [week_num(p) for p in npz_files]
    print(f"Found {len(weeks)} weeks: {weeks[0]}–{weeks[-1]}")

    # load class names for heatmap labels
    try:
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from train_per_week_cesnet import load_label_mapping
        mapping, _ = load_label_mapping(Path(args.dataset_root))
        idx2name = {v: k for k, v in mapping.items()}
    except Exception:
        idx2name = {}

    # load reference week (week 0 = first available)
    print("Loading reference week...")
    ref_emb, ref_labels, ref_acc = load_npz(npz_files[0], subsample=args.subsample, seed=args.seed)
    print(f"  Reference: week {weeks[0]}, {len(ref_emb)} samples, acc={ref_acc:.3f}")

    # ------------------------------------------------------------------ #
    # Compute metrics for every week                                       #
    # ------------------------------------------------------------------ #
    centroid_dists  = []   # from reference week
    wasserstein_dists = []
    consecutive_dists = []  # week[t] vs week[t-1]
    accuracies      = [ref_acc]
    class_drift     = {}   # class -> list of dists (indexed by weeks[1:])

    prev_emb = ref_emb

    for i, (path, wn) in enumerate(zip(npz_files, weeks)):
        emb, labels, acc = load_npz(path, subsample=args.subsample, seed=args.seed)
        accuracies.append(acc) if i > 0 else None

        cd = centroid_l2(ref_emb, emb)
        centroid_dists.append(cd)

        print(f"  week {wn:02d}: centroid_dist={cd:.4f}", end='', flush=True)

        wd = sliced_wasserstein(ref_emb, emb, n_projections=args.n_proj, seed=args.seed)
        wasserstein_dists.append(wd)
        print(f"  swd={wd:.4f}", end='')

        consec = centroid_l2(prev_emb, emb)
        consecutive_dists.append(consec)
        print(f"  consec={consec:.4f}  acc={acc:.3f}")

        # per-class centroid drift from reference
        pc = per_class_centroid_dist(ref_emb, ref_labels, emb, labels)
        for c, d in pc.items():
            class_drift.setdefault(c, [None] * len(weeks))
            class_drift[c][i] = d

        prev_emb = emb

    centroid_dists     = np.array(centroid_dists)
    wasserstein_dists  = np.array(wasserstein_dists)
    consecutive_dists  = np.array(consecutive_dists)
    accuracies         = np.array(accuracies)

    # normalise for overlay
    wd_norm = wasserstein_dists / wasserstein_dists.max()
    cd_norm = centroid_dists / centroid_dists.max()

    # ------------------------------------------------------------------ #
    # Build per-class drift matrix                                         #
    # ------------------------------------------------------------------ #
    all_classes = sorted(class_drift.keys())
    drift_matrix = np.array([
        [class_drift[c][i] if class_drift[c][i] is not None else np.nan
         for i in range(len(weeks))]
        for c in all_classes
    ])  # shape: (n_classes, n_weeks)

    # sort by max drift, take top K
    max_drift = np.nanmax(drift_matrix, axis=1)
    top_idx   = np.argsort(max_drift)[::-1][:args.top_classes]
    top_classes   = [all_classes[i] for i in top_idx]
    top_matrix    = drift_matrix[top_idx]
    top_labels    = [idx2name.get(c, str(c)) for c in top_classes]

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(4, 1, figsize=(14, 18),
                             gridspec_kw={'height_ratios': [2, 1.5, 1.5, 3]})
    fig.suptitle('Feature-Space Drift Over Time\n'
                 '(week-1 trained model, original 600-dim embedding space)',
                 fontsize=13, fontweight='bold', y=0.99)

    xs = np.array(weeks)

    # ── Panel 1: cumulative distance from reference ─────────────────── #
    ax = axes[0]
    ax2 = ax.twinx()
    l1, = ax.plot(xs, centroid_dists,  'o-', color='steelblue', lw=2, ms=4,
                  label='Centroid L2 distance')
    l2, = ax.plot(xs, wasserstein_dists, 's--', color='darkorange', lw=2, ms=4,
                  label='Sliced Wasserstein (100 proj)')
    l3, = ax2.plot(xs, accuracies * 100, 'v:', color='mediumseagreen', lw=1.5, ms=4, alpha=0.8,
                   label='Model accuracy (%)')
    ax.set_ylabel('Distance from week-0 distribution')
    ax2.set_ylabel('Accuracy (%)', color='mediumseagreen')
    ax2.tick_params(axis='y', colors='mediumseagreen')
    ax.set_title('Cumulative drift from reference week (week 0)\n'
                 'Smooth curve → continuous drift | Step-function shape → discrete jumps',
                 fontsize=10)
    ax.legend(handles=[l1, l2, l3], fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs[::4])

    # ── Panel 2: week-over-week delta ────────────────────────────────── #
    ax = axes[1]
    ax.bar(xs, consecutive_dists, color='steelblue', alpha=0.7, width=0.7)
    ax.axhline(consecutive_dists.mean(), color='red', lw=1.5, ls='--',
               label=f'Mean Δ = {consecutive_dists.mean():.3f}')
    ax.axhline(consecutive_dists.mean() + 2 * consecutive_dists.std(),
               color='red', lw=1, ls=':', label='Mean ± 2σ')
    # annotate the spikes
    threshold = consecutive_dists.mean() + 2 * consecutive_dists.std()
    for i, (wn, d) in enumerate(zip(xs, consecutive_dists)):
        if d > threshold:
            ax.annotate(f'W{wn}', (wn, d), textcoords='offset points',
                        xytext=(0, 4), ha='center', fontsize=7, color='red')
    ax.set_ylabel('Centroid L2 distance\nto previous week')
    ax.set_title('Week-over-week centroid delta — spikes indicate abrupt distribution shifts',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(xs[::4])

    # ── Panel 3: consecutive vs cumulative ratio ─────────────────────── #
    # If drift is purely continuous, week-over-week delta should be ~constant
    # and cumulative distance ≈ sum of deltas.
    # We show the cumulative sum of consecutive deltas vs actual cumulative distance.
    ax = axes[2]
    cumsum_consec = np.cumsum(consecutive_dists)
    ax.plot(xs, cumsum_consec, 'o-', color='steelblue', lw=2, ms=4,
            label='Cumulative sum of consecutive deltas')
    ax.plot(xs, centroid_dists, 's--', color='darkorange', lw=2, ms=4,
            label='Direct distance from week 0')
    ax.fill_between(xs, cumsum_consec, centroid_dists, alpha=0.15, color='purple',
                    label='Gap (triangle inequality slack)')
    ax.set_ylabel('L2 distance')
    ax.set_title('Cumulative consecutive deltas vs. direct distance from week 0\n'
                 'Large gap → path is non-monotone (oscillations or reversals)',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs[::4])

    # ── Panel 4: per-class centroid drift heatmap ────────────────────── #
    ax = axes[3]
    im = ax.imshow(top_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=np.nanpercentile(top_matrix, 95))
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels([str(w) for w in weeks], fontsize=6, rotation=90)
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=7)
    ax.set_xlabel('Week')
    ax.set_title(f'Per-class centroid L2 drift from week 0 (top {args.top_classes} by max drift)\n'
                 'Horizontal bands = class-specific jumps | Vertical bands = global event',
                 fontsize=10)
    plt.colorbar(im, ax=ax, label='Centroid L2 distance from week-0 centroid', pad=0.01)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out_path}")

    # ------------------------------------------------------------------ #
    # Print summary stats                                                  #
    # ------------------------------------------------------------------ #
    print("\n=== Summary ===")
    print(f"Centroid distance range: {centroid_dists.min():.4f} – {centroid_dists.max():.4f}")
    print(f"Sliced Wasserstein range: {wasserstein_dists.min():.4f} – {wasserstein_dists.max():.4f}")
    print(f"Consec delta: mean={consecutive_dists.mean():.4f}  std={consecutive_dists.std():.4f}  "
          f"max={consecutive_dists.max():.4f} (week {weeks[consecutive_dists.argmax()]})")
    spike_weeks = [weeks[i] for i, d in enumerate(consecutive_dists)
                   if d > consecutive_dists.mean() + 2 * consecutive_dists.std()]
    print(f"Spike weeks (>mean+2σ): {spike_weeks}")


if __name__ == '__main__':
    main()
