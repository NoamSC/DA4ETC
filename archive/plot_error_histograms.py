#!/usr/bin/env python
"""Plot per-class error rate histograms for forward train->test week combinations."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

cms_dir = Path('figs/cms')
weeks = [10, 20, 30, 40, 50]

# Forward pairs only: train_week <= test_week
pairs = [(tw_train, tw_test) for tw_train in weeks for tw_test in weeks if tw_train <= tw_test]

# Load confusion matrices
cms = {}
for tw_train, tw_test in pairs:
    path = cms_dir / f'cm_{tw_train}_to_{tw_test}.npy'
    assert path.exists(), f"Missing {path}"
    cms[(tw_train, tw_test)] = np.load(path)

# Compute per-class error rate and sample counts
error_rates = {}
sample_counts = {}
for key, cm in cms.items():
    row_sums = cm.sum(axis=1)
    correct = np.diag(cm)
    mask = row_sums > 0
    err = np.where(mask, 1.0 - correct / row_sums, np.nan)
    error_rates[key] = err
    sample_counts[key] = row_sums

# ---- Plot 1: Absolute error rate histograms (upper-triangular 5x5 grid) ----
for weighted, suffix, color in [(False, '', None), (True, '_weighted', 'steelblue')]:
    fig, axes = plt.subplots(len(weeks), len(weeks), figsize=(4 * len(weeks), 3.5 * len(weeks)),
                             sharey=True, sharex=True)

    for i, tw_train in enumerate(weeks):
        for j, tw_test in enumerate(weeks):
            ax = axes[i, j]
            if tw_train > tw_test:
                ax.set_visible(False)
                continue

            err = error_rates[(tw_train, tw_test)]
            counts = sample_counts[(tw_train, tw_test)].astype(float)
            valid_mask = ~np.isnan(err)
            valid_err = err[valid_mask]
            valid_counts = counts[valid_mask]

            kwargs = dict(bins=20, range=(0, 1), edgecolor='black', alpha=0.7)
            if weighted:
                mean_val = np.average(valid_err, weights=valid_counts)
                kwargs['weights'] = valid_counts
                kwargs['color'] = color
            else:
                mean_val = np.nanmean(err)

            ax.hist(valid_err, **kwargs)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2,
                       label=f'{mean_val:.2f}')
            ax.legend(fontsize=7, loc='upper left')

            if i == 0:
                ax.set_title(f'Test {tw_test}', fontsize=11, fontweight='bold')
            if j == 0 or (j > 0 and tw_train > weeks[j - 1]):
                ax.set_ylabel(f'Train {tw_train}', fontsize=11, fontweight='bold')

    weight_label = ' (sample-weighted)' if weighted else ''
    fig.suptitle(f'Per-class Error Rate{weight_label}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = cms_dir / f'error_rate_histograms{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")

# ---- Plot 2: Error rate CHANGE histograms (diagonal baseline) ----
for weighted, suffix, color in [(False, '', 'coral'), (True, '_weighted', 'steelblue')]:
    fig, axes = plt.subplots(len(weeks), len(weeks), figsize=(4 * len(weeks), 3.5 * len(weeks)),
                             sharey=True, sharex=True)

    for i, tw_train in enumerate(weeks):
        baseline = error_rates[(tw_train, tw_train)]
        baseline_counts = sample_counts[(tw_train, tw_train)].astype(float)
        for j, tw_test in enumerate(weeks):
            ax = axes[i, j]
            if tw_train > tw_test:
                ax.set_visible(False)
                continue

            diff = error_rates[(tw_train, tw_test)] - baseline
            valid_mask = ~np.isnan(diff)
            valid_diff = diff[valid_mask]
            valid_counts = baseline_counts[valid_mask]

            kwargs = dict(bins=30, edgecolor='black', alpha=0.7, color=color)
            if weighted:
                mean_val = np.average(valid_diff, weights=valid_counts)
                kwargs['weights'] = valid_counts
            else:
                mean_val = np.nanmean(diff)

            ax.hist(valid_diff, **kwargs)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.6)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.2,
                       label=f'{mean_val:+.3f}')
            ax.legend(fontsize=7, loc='upper left')

            if i == 0:
                ax.set_title(f'Test {tw_test}', fontsize=11, fontweight='bold')
            if j == 0 or (j > 0 and tw_train > weeks[j - 1]):
                ax.set_ylabel(f'Train {tw_train}', fontsize=11, fontweight='bold')

    weight_label = ' (sample-weighted)' if weighted else ''
    fig.suptitle(f'Per-class Error Rate Change vs Diagonal{weight_label}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = cms_dir / f'error_rate_change_histograms{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")
