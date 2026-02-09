#!/usr/bin/env python
"""
Temporal generalization with importance-weighted accuracy.

Evaluates a single trained model (e.g. week 33) on a range of future weeks,
computing accuracy reweighted to the training-week label prior so that label
shift is factored out:

    w(y) = p_train(y) / p_t(y)

    reweighted_acc = sum_i w(y_i) * 1[y_hat_i == y_i]  /  sum_i w(y_i)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

from temporal_generalization import (
    load_model_from_checkpoint,
    create_week_loader,
    get_available_weeks,
    extract_week_number,
)
from training.trainer import validate
from train_per_week_cesnet import load_label_mapping


def get_label_distribution(parquet_path, label_indices_mapping):
    """Return normalised label counts restricted to the classes in label_indices_mapping."""
    df = pd.read_parquet(parquet_path, columns=["APP"])
    df = df[df["APP"].isin(label_indices_mapping)]
    counts = df["APP"].value_counts(normalize=True)
    return counts


def compute_reweighted_metrics(all_labels, all_predictions, all_logits,
                               p_train_by_idx, num_classes):
    """Compute raw and importance-reweighted metrics.

    Metrics: accuracy, top-k accuracy (k=3,5,10), macro-averaged F1.

    Args:
        all_labels:       list[int] – true class indices
        all_predictions:  list[int] – predicted class indices
        all_logits:       Tensor [N, C] – raw logits from the model
        p_train_by_idx:   dict[int, float] – p_train(y) keyed by class index
        num_classes:      int – total number of classes

    Returns:
        dict with raw and reweighted values for every metric, plus diagnostics.
    """
    labels = np.array(all_labels)
    preds = np.array(all_predictions)
    n = len(labels)

    # ---- per-sample importance weights ----
    label_counts = Counter(all_labels)
    p_t_by_idx = {cls: count / n for cls, count in label_counts.items()}
    weights = np.array([
        p_train_by_idx.get(y, 0.0) / p_t_by_idx.get(y, 1.0)
        for y in all_labels
    ])

    # ---- JS divergence ----
    p_train_vec = np.array([p_train_by_idx.get(c, 0.0) for c in range(num_classes)])
    p_t_vec = np.array([p_t_by_idx.get(c, 0.0) for c in range(num_classes)])
    if p_train_vec.sum() > 0:
        p_train_vec /= p_train_vec.sum()
    if p_t_vec.sum() > 0:
        p_t_vec /= p_t_vec.sum()
    js_div = float(jensenshannon(p_train_vec, p_t_vec))

    # ---- accuracy (top-1) ----
    correct = (preds == labels).astype(float)
    raw_acc = 100.0 * correct.mean()
    rw_acc = 100.0 * (weights * correct).sum() / weights.sum()

    # ---- top-k accuracy ----
    topk_vals = [3, 5, 10]
    logits_t = all_logits  # already a Tensor [N, C]
    labels_t = torch.tensor(labels, dtype=torch.long)

    raw_topk, rw_topk = {}, {}
    for k in topk_vals:
        topk_preds = logits_t.topk(k, dim=1).indices          # [N, k]
        hits = topk_preds.eq(labels_t.unsqueeze(1)).any(1).numpy().astype(float)
        raw_topk[k] = 100.0 * float(hits.mean())
        rw_topk[k] = 100.0 * float((weights * hits).sum() / weights.sum())

    # ---- macro-averaged F1 ----
    classes_present = sorted(set(all_labels) | set(all_predictions))

    def _macro_f1(sample_weights):
        """Weighted macro F1 over classes present in the test set."""
        f1s = []
        for c in classes_present:
            is_true = (labels == c)
            is_pred = (preds == c)
            tp = (sample_weights * is_true * is_pred).sum()
            fp = (sample_weights * ~is_true * is_pred).sum()
            fn = (sample_weights * is_true * ~is_pred).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return 100.0 * float(np.mean(f1s))

    raw_macro_f1 = _macro_f1(np.ones(n))
    rw_macro_f1 = _macro_f1(weights)

    # ---- per-class importance weights ----
    importance_weights = {}
    for cls in set(list(label_counts.keys()) + list(p_train_by_idx.keys())):
        pt = p_t_by_idx.get(cls, 0.0)
        ptrain = p_train_by_idx.get(cls, 0.0)
        importance_weights[int(cls)] = ptrain / pt if pt > 0 else 0.0

    return {
        "accuracy": float(raw_acc),
        "reweighted_accuracy": float(rw_acc),
        **{f"top{k}_accuracy": float(raw_topk[k]) for k in topk_vals},
        **{f"reweighted_top{k}_accuracy": float(rw_topk[k]) for k in topk_vals},
        "macro_f1": float(raw_macro_f1),
        "reweighted_macro_f1": float(rw_macro_f1),
        "js_divergence": js_div,
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_max": float(weights.max()),
        "weight_min": float(weights.min()),
        "n_classes_in_test": len(label_counts),
        "n_classes_in_train": len(p_train_by_idx),
        "importance_weights": importance_weights,
        "p_train": {int(k): v for k, v in p_train_by_idx.items()},
        "p_t": {int(k): v for k, v in p_t_by_idx.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Future-only temporal generalization with importance-weighted accuracy"
    )
    parser.add_argument("--experiment_dir", type=str,
                        default="exps/cesnet_multimodal_each_week_train_v01")
    parser.add_argument("--dataset_root", type=str,
                        default="/home/anatbr/dataset/CESNET-TLS-Year22_v2")
    parser.add_argument("--train_week", type=int, required=True,
                        help="Week number the model was trained on (e.g. 33)")
    parser.add_argument("--start_week", type=int, default=None,
                        help="First evaluation week (default: same as train_week)")
    parser.add_argument("--end_week", type=int, default=52,
                        help="Last evaluation week (inclusive, default: 52)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--data_sample_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pth")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Total number of parallel jobs (for SLURM array splitting)")
    parser.add_argument("--job_id", type=int, default=0,
                        help="This job index 0..num_jobs-1")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip evaluation, just plot from existing per-week JSONs")
    args = parser.parse_args()

    if args.start_week is None:
        args.start_week = args.train_week

    experiment_dir = Path(args.experiment_dir)
    dataset_root = Path(args.dataset_root)

    if args.plot_only:
        results_subdir = experiment_dir / "temporal_generalization_reweighted"
        results = load_all_per_week_jsons(results_subdir)
        print(f"Loaded {len(results)} per-week results from {results_subdir}")
        plot_reweighted_results(results, experiment_dir)
        return

    # ---- Load label mapping ----
    print("Loading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"Number of classes: {num_classes}")

    # ---- Load the single trained model ----
    train_week_dir = experiment_dir / f"week_{args.train_week}"
    checkpoint_path = train_week_dir / "weights" / args.checkpoint_name
    config_path = train_week_dir / "config.json"
    print(f"\nLoading model from {train_week_dir}")
    model = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, args.device)

    # ---- Compute p_train(y) from training data ----
    train_parquet = dataset_root / f"WEEK-2022-{args.train_week:02d}" / "train.parquet"
    print(f"Computing p_train from {train_parquet}")
    p_train = get_label_distribution(train_parquet, label_indices_mapping)
    # Map to class-index keys
    p_train_by_idx = {
        label_indices_mapping[app]: prob
        for app, prob in p_train.items()
        if app in label_indices_mapping
    }
    print(f"  p_train covers {len(p_train_by_idx)} classes")

    # ---- Determine test weeks ----
    all_week_dirs = get_available_weeks(dataset_root)
    test_week_dirs = [
        d for d in all_week_dirs
        if args.start_week <= extract_week_number(d.name) <= args.end_week
    ]
    print(f"\nTest weeks: {test_week_dirs[0].name} to {test_week_dirs[-1].name} "
          f"({len(test_week_dirs)} weeks)")

    # Split across SLURM array jobs
    if args.num_jobs > 1:
        chunks = np.array_split(list(range(len(test_week_dirs))), args.num_jobs)
        indices = list(chunks[args.job_id])
        test_week_dirs = [test_week_dirs[i] for i in indices]
        print(f"Job {args.job_id}/{args.num_jobs}: evaluating {len(test_week_dirs)} weeks")

    # ---- Per-week results directory ----
    results_subdir = experiment_dir / "temporal_generalization_reweighted"
    results_subdir.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    results = {}

    for week_dir in test_week_dirs:
        week_name = week_dir.name
        week_num = extract_week_number(week_name)
        result_path = results_subdir / f"{week_name}.json"

        # Skip if already computed
        if result_path.exists():
            print(f"  {week_name}: cached – skipping")
            with open(result_path, "r") as f:
                cached = json.load(f)
            results[week_name] = cached
            continue

        loader = create_week_loader(
            week_dir, label_indices_mapping,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resolution=args.resolution,
            data_sample_frac=args.data_sample_frac,
            seed=args.seed,
        )
        if loader is None:
            print(f"  {week_name}: no test data – skipping")
            continue

        # Run inference (return logits for top-k)
        val_loss, val_acc, all_labels, all_predictions, all_logits, _ = validate(
            model, loader, criterion, args.device, return_features=True
        )

        # All reweighted metrics
        rw = compute_reweighted_metrics(
            all_labels, all_predictions, all_logits, p_train_by_idx, num_classes
        )

        metrics = {
            "test_week": week_name,
            "test_week_num": week_num,
            "train_week": args.train_week,
            "loss": val_loss,
            "total_samples": len(all_labels),
            **{k: rw[k] for k in rw if k not in ("p_train", "p_t", "importance_weights")},
            "importance_weights": rw["importance_weights"],
            "p_train": rw["p_train"],
            "p_t": rw["p_t"],
        }
        results[week_name] = metrics

        # Save per-week JSON
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  {week_name}: acc={rw['accuracy']:.2f}%  rw_acc={rw['reweighted_accuracy']:.2f}%  "
              f"top3={rw['top3_accuracy']:.1f}/{rw['reweighted_top3_accuracy']:.1f}  "
              f"F1={rw['macro_f1']:.2f}/{rw['reweighted_macro_f1']:.2f}  "
              f"JS={rw['js_divergence']:.4f}  (n={len(all_labels)})")

        del loader

    # ---- Save combined results ----
    combined_path = results_subdir / "results_combined.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {combined_path}")
    print(f"Evaluated {len(results)} weeks")

    # ---- Plot if single-job run ----
    if args.num_jobs <= 1:
        plot_reweighted_results(results, experiment_dir)


def load_all_per_week_jsons(results_dir):
    """Load all per-week JSONs from the results directory."""
    results = {}
    for p in sorted(results_dir.glob("WEEK-2022-*.json")):
        with open(p) as f:
            data = json.load(f)
        results[p.stem] = data
    return results


def plot_reweighted_results(results, experiment_dir):
    """Plot raw vs reweighted for all metrics: accuracy, top-k, macro F1."""
    if not results:
        print("No results to plot.")
        return

    weeks = []
    data = {}  # metric_name -> (raw_list, rw_list)

    # Define metrics to plot: (raw_key, rw_key, label)
    metric_defs = [
        ("accuracy", "reweighted_accuracy", "Top-1 Accuracy (%)"),
        ("top3_accuracy", "reweighted_top3_accuracy", "Top-3 Accuracy (%)"),
        ("top5_accuracy", "reweighted_top5_accuracy", "Top-5 Accuracy (%)"),
        ("top10_accuracy", "reweighted_top10_accuracy", "Top-10 Accuracy (%)"),
        ("macro_f1", "reweighted_macro_f1", "Macro F1 (%)"),
    ]

    # Collect data
    for week_name in sorted(results, key=lambda w: results[w].get("test_week_num", 0)):
        m = results[week_name]
        weeks.append(m["test_week_num"])
        for raw_key, rw_key, _ in metric_defs:
            if raw_key not in data:
                data[raw_key] = ([], [])
            data[raw_key][0].append(m.get(raw_key))
            data[raw_key][1].append(m.get(rw_key))

    weeks = np.array(weeks)

    # Filter to metrics that are present in the results
    available = [(rk, rwk, label) for rk, rwk, label in metric_defs
                 if rk in data and data[rk][0][0] is not None]

    n_metrics = len(available)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(13, 3.5 * n_metrics),
                             sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, (raw_key, _, ylabel) in zip(axes, available):
        raw = np.array(data[raw_key][0])
        rw = np.array(data[raw_key][1])
        ax.plot(weeks, raw, 'o-', label='Raw', linewidth=2, markersize=5,
                color='tab:blue')
        ax.plot(weeks, rw, 's--', label='Reweighted', linewidth=2, markersize=5,
                color='tab:orange')
        ax.fill_between(weeks, raw, rw, alpha=0.15, color='tab:orange')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        'Temporal Generalization: Raw vs Importance-Reweighted Metrics\n'
        r'$w(y)=p_{\mathrm{train}}(y)\,/\,p_t(y)$',
        fontsize=14, fontweight='bold')
    axes[-1].set_xlabel('Test week', fontsize=13)

    plt.tight_layout()
    plot_path = Path(experiment_dir) / 'temporal_reweighted_metrics.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
