#!/usr/bin/env python
"""
Script to run temporal generalization evaluation.
Equivalent to running the temporal_generalization_analysis.ipynb notebook up to the evaluation step.
"""

# --- repo path bootstrap (added during refactor: keeps flat cross-imports working) ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---


import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from temporal_generalization import (
    evaluate_temporal_generalization,
    results_to_dataframe,
    save_results,
    load_results,
)
from train_per_week_cesnet import load_label_mapping


def main():
    parser = argparse.ArgumentParser(description='Run temporal generalization evaluation')
    parser.add_argument('--experiment_dir', type=str,
                        default='exps/cesnet_multimodal_each_week_train_v01',
                        help='Path to experiment directory with trained models')
    parser.add_argument('--dataset_root', type=str,
                        default='/home/anatbr/dataset/CESNET-TLS-Year22_v2',
                        help='Path to CESNET dataset root')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to save results (default: <experiment_dir>/temporal_generalization_results.json)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Resolution for data loading')
    parser.add_argument('--data_sample_frac', type=float, default=0.1,
                        help='Fraction of each weeks data to use for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth',
                        help='Name of checkpoint file to load')
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation even if results exist')
    parser.add_argument('--num_jobs', type=int, default=1,
                        help='Total number of parallel jobs (for array splitting)')
    parser.add_argument('--job_id', type=int, default=0,
                        help='This job index 0..num_jobs-1 (maps to SLURM_ARRAY_TASK_ID)')
    args = parser.parse_args()

    # Set paths
    experiment_dir = Path(args.experiment_dir)
    dataset_root = Path(args.dataset_root)
    results_path = Path(args.results_path) if args.results_path else experiment_dir / 'temporal_generalization_results.json'

    # Compute test week indices for this job
    num_test_weeks = 53  # weeks 0-52
    test_week_indices = None
    if args.num_jobs > 1:
        all_indices = list(range(num_test_weeks))
        chunks = np.array_split(all_indices, args.num_jobs)
        test_week_indices = list(chunks[args.job_id])

    print(f"Experiment directory: {experiment_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Results path: {results_path}")
    print(f"Device: {args.device}")
    print(f"Data sample fraction: {args.data_sample_frac}")
    if test_week_indices is not None:
        print(f"Job {args.job_id}/{args.num_jobs}: test weeks {test_week_indices[0]}-{test_week_indices[-1]}")
    print()

    # Load label mapping
    print("Loading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"Number of classes: {num_classes}")
    print()

    # Check if results already exist
    if results_path.exists() and not args.force:
        print(f"Loading cached results from {results_path}")
        results = load_results(results_path)
        print(f"Loaded {len(results)} evaluation results")
    else:
        print("Running temporal generalization evaluation...")
        print("This may take a while. Results will be cached for future use.\n")

        results = evaluate_temporal_generalization(
            experiment_dir=experiment_dir,
            dataset_root=dataset_root,
            label_indices_mapping=label_indices_mapping,
            num_classes=num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resolution=args.resolution,
            data_sample_frac=args.data_sample_frac,
            seed=args.seed,
            device=args.device,
            checkpoint_name=args.checkpoint_name,
            test_week_indices=test_week_indices
        )

        # Save combined results only for single-job runs
        if args.num_jobs <= 1:
            save_results(results, results_path)
            print(f"Results saved to {results_path}")

        print(f"\nCompleted {len(results)} evaluations")

    # Only plot for single-job runs (array jobs save per-week JSONs)
    if args.num_jobs <= 1:
        plot_degradation(results, experiment_dir)
        plot_temporal_timeseries(results, experiment_dir, future_only=True)


def plot_degradation(results, experiment_dir):
    df = results_to_dataframe(results)
    df['weeks_elapsed'] = df['test_week_num'] - df['train_week_num']

    # Only keep future weeks (model tested on weeks after training)
    future_df = df[df['weeks_elapsed'] > 0].copy()
    if future_df.empty:
        print("No future-week evaluations to plot.")
        return

    # Get each model's accuracy on its own training week as baseline
    same_week = df[df['weeks_elapsed'] == 0].set_index('train_week_num')['accuracy']
    future_df['baseline_accuracy'] = future_df['train_week_num'].map(same_week)
    future_df['accuracy_drop'] = future_df['baseline_accuracy'] - future_df['accuracy']

    # Aggregate by weeks_elapsed
    degradation = future_df.groupby('weeks_elapsed')['accuracy_drop'].agg(['mean', 'std', 'count']).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: absolute accuracy vs weeks elapsed
    abs_deg = future_df.groupby('weeks_elapsed')['accuracy'].agg(['mean', 'std']).reset_index()
    ax1.plot(abs_deg['weeks_elapsed'], abs_deg['mean'], 'o-', linewidth=2, markersize=4)
    ax1.fill_between(abs_deg['weeks_elapsed'],
                     abs_deg['mean'] - abs_deg['std'],
                     abs_deg['mean'] + abs_deg['std'], alpha=0.2)
    ax1.set_xlabel('Weeks After Training', fontsize=13)
    ax1.set_ylabel('Accuracy (%)', fontsize=13)
    ax1.set_title('Absolute Accuracy vs Temporal Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Right: accuracy drop vs weeks elapsed
    ax2.plot(degradation['weeks_elapsed'], degradation['mean'], 'o-', color='tab:red', linewidth=2, markersize=4)
    ax2.fill_between(degradation['weeks_elapsed'],
                     degradation['mean'] - degradation['std'],
                     degradation['mean'] + degradation['std'], alpha=0.2, color='tab:red')
    ax2.set_xlabel('Weeks After Training', fontsize=13)
    ax2.set_ylabel('Accuracy Drop (pp)', fontsize=13)
    ax2.set_title('Accuracy Degradation vs Temporal Distance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = experiment_dir / 'temporal_degradation.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Degradation plot saved to {plot_path}")


def plot_temporal_timeseries(results, experiment_dir, future_only=True):
    """
    Multi-line time-series: accuracy vs week number (0-52).
    Each colored line = model trained on a different week.
    Star = validation accuracy (train_week == test_week).
    Black line with circles = T+1 test accuracy.
    future_only: if True, only show test weeks >= training week.
    """
    df = results_to_dataframe(results)
    df['accuracy_frac'] = df['accuracy'] / 100.0

    train_weeks = sorted(df['train_week_num'].unique())
    cmap = plt.cm.turbo(np.linspace(0, 1, len(train_weeks)))
    train_week_to_color = dict(zip(train_weeks, cmap))

    fig, ax = plt.subplots(figsize=(18, 8))

    # Collect T+1 points
    t1_test_weeks = []
    t1_accuracies = []

    for tw in train_weeks:
        tw_df = df[df['train_week_num'] == tw].sort_values('test_week_num')
        if tw_df.empty:
            continue

        if future_only:
            tw_df = tw_df[tw_df['test_week_num'] >= tw]

        color = train_week_to_color[tw]
        ax.plot(tw_df['test_week_num'], tw_df['accuracy_frac'],
                '-', color=color, alpha=0.5, linewidth=1.2)

        # Star marker on validation week (train == test)
        val_row = tw_df[tw_df['test_week_num'] == tw]
        if not val_row.empty:
            ax.scatter(val_row['test_week_num'].values[0],
                       val_row['accuracy_frac'].values[0],
                       marker='*', s=200, color=color,
                       edgecolors='black', linewidths=0.8, zorder=5)

        # T+1 point
        t1_row = tw_df[tw_df['test_week_num'] == tw + 1]
        if not t1_row.empty:
            t1_test_weeks.append(t1_row['test_week_num'].values[0])
            t1_accuracies.append(t1_row['accuracy_frac'].values[0])

    # Black T+1 line
    if t1_test_weeks:
        order = np.argsort(t1_test_weeks)
        t1_test_weeks = np.array(t1_test_weeks)[order]
        t1_accuracies = np.array(t1_accuracies)[order]
        ax.plot(t1_test_weeks, t1_accuracies, 'o-', color='black',
                linewidth=2.5, markersize=5, label='Test on T+1', zorder=6)

    ax.set_xlabel('Week Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Temporal Generalization: Per-Model Performance Across Weeks\n'
                 r'$\bigstar$ = validation (same week)  |  $\bullet$ black = test on week T+1',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(-0.5, 52.5)
    ax.set_xticks(range(0, 53, 5))
    ax.grid(True, alpha=0.3)

    # Color bar for training week
    sm = plt.cm.ScalarMappable(cmap='turbo',
                               norm=plt.Normalize(vmin=min(train_weeks), vmax=max(train_weeks)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Training Week', fontsize=12)

    ax.legend(fontsize=12, loc='lower left')
    plt.tight_layout()
    plot_path = experiment_dir / 'temporal_timeseries.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Time-series plot saved to {plot_path}")


if __name__ == '__main__':
    main()
