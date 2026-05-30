#!/usr/bin/env python
"""
Three dual-axis plots over time (model trained on week 1):

  A — Latent Centroid Drift   (left) vs F1 (right)
  B — Class-Cond. Entropy     (left) vs F1 (right)
  C — BBSE label-shift weight (left) vs F1 (right)

All unsupervised metrics use predicted pseudo-labels (no ground truth needed).
F1 uses true labels and is shown only to validate lead time.

Usage:
    python plot_week1_temporal.py
    python plot_week1_temporal.py --classes 61 103 151 155 --output figs/abc.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


# ── helpers ───────────────────────────────────────────────────────────────────

def load_class_names(dataset_root):
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from train_per_week_cesnet import load_label_mapping
    mapping, num_classes = load_label_mapping(Path(dataset_root))
    return {v: k for k, v in mapping.items()}, num_classes


def load_weeks(inference_dir):
    files = sorted(
        Path(inference_dir).glob('WEEK-2022-*.npz'),
        key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    )
    weeks = []
    for f in files:
        wn = int(re.search(r'(\d+)$', f.stem).group(1))
        d = np.load(f)
        weeks.append((wn, d['true_labels'], d['pred_labels'], d['softmax'], d['embeddings']))
    return weeks


def pick_classes(weeks, n=5):
    """Top-2 by frequency, 1 mid, 2 rare — all present in every week."""
    all_true = np.concatenate([w[1] for w in weeks])
    counts = np.bincount(all_true, minlength=int(all_true.max()) + 1)
    always = None
    for _, true, *_ in weeks:
        s = set(np.unique(true).tolist())
        always = s if always is None else always & s
    cands = np.array(sorted(always))
    order = np.argsort(-counts[cands])
    cands = cands[order]
    picks = list(dict.fromkeys([
        int(cands[0]), int(cands[1]),
        int(cands[len(cands) // 2]),
        int(cands[-2]), int(cands[-1]),
    ]))
    return picks[:n]


# ── metric functions (pseudo-label grouped) ───────────────────────────────────

def centroid_dist_pseudo(emb, pred, cls, ref_centroid):
    """Mean L2 distance of embeddings predicted as `cls` to ref_centroid."""
    mask = pred == cls
    if mask.sum() == 0 or ref_centroid is None:
        return np.nan
    return float(np.linalg.norm(emb[mask] - ref_centroid, axis=1).mean())


def entropy_pseudo(softmax, pred, cls):
    """Mean Shannon entropy of softmax for samples predicted as `cls`."""
    mask = pred == cls
    if mask.sum() == 0:
        return np.nan
    p = np.clip(softmax[mask], 1e-12, 1.0)
    return float(-np.sum(p * np.log(p), axis=1).mean())


def class_f1(true, pred, cls):
    mask_t = true == cls
    if mask_t.sum() == 0:
        return np.nan
    tp = int(((pred == cls) & (true == cls)).sum())
    fp = int(((pred == cls) & (true != cls)).sum())
    fn = int(((pred != cls) & (true == cls)).sum())
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0


def macro_f1(true, pred, num_classes):
    return f1_score(true, pred, labels=list(range(num_classes)),
                    average='macro', zero_division=0)


# ── BBSE ──────────────────────────────────────────────────────────────────────

def build_bbse_reference(ref_true, ref_pred, ref_softmax, num_classes):
    """
    Returns:
        C_T_pinv  — pseudo-inverse of C^T  (for BBSE label-shift estimation)
        p_train   — P(Y) in reference week
        h_ref     — per-class reference entropy E[entropy | true=c] in reference week
    """
    cm = confusion_matrix(ref_true, ref_pred, labels=list(range(num_classes))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C = cm / row_sums                          # C[i,j] = P(pred=j | true=i)
    C_T_pinv = np.linalg.pinv(C.T)

    counts = np.bincount(ref_true, minlength=num_classes).astype(float)
    p_train = counts / counts.sum()

    # Per-class reference entropy (conditioned on true label, computed on reference week)
    h_ref = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = ref_true == c
        if mask.sum() > 0:
            p = np.clip(ref_softmax[mask], 1e-12, 1.0)
            h_ref[c] = float(-np.sum(p * np.log(p), axis=1).mean())

    return C_T_pinv, p_train, h_ref


def bbse_residuals(softmax, pred, num_classes, C_T_pinv, p_train, h_ref):
    """
    For each class c, compute the BBSE drift residual:
        residual(c) = actual_entropy(c) - h_ref(c)

    Where:
        actual_entropy(c)  = mean softmax entropy for samples predicted as c (pseudo-label)
        h_ref(c)           = reference entropy for class c (from training week, true labels)

    Also returns the global residual:
        actual_global - expected_global
        expected_global = sum_c p_hat(c) * h_ref(c)   (BBSE-adjusted expectation)
        actual_global   = mean entropy over all samples
    """
    # BBSE estimate of true P(Y) in this week
    counts = np.bincount(pred, minlength=num_classes).astype(float)
    q_hat = counts / counts.sum()
    p_hat = C_T_pinv @ q_hat
    p_hat = np.clip(p_hat, 0, None)
    if p_hat.sum() > 0:
        p_hat /= p_hat.sum()

    # Global expected entropy under label shift alone
    valid = ~np.isnan(h_ref)
    expected_global = float(np.dot(p_hat[valid], h_ref[valid]))

    # Global actual entropy
    p_all = np.clip(softmax, 1e-12, 1.0)
    actual_global = float(-np.sum(p_all * np.log(p_all), axis=1).mean())
    global_residual = actual_global - expected_global

    # Per-class residual: actual_entropy(c) - h_ref(c)
    per_class = {}
    for c in range(num_classes):
        mask = pred == c
        if mask.sum() == 0 or np.isnan(h_ref[c]):
            per_class[c] = np.nan
        else:
            p = np.clip(softmax[mask], 1e-12, 1.0)
            h_actual = float(-np.sum(p * np.log(p), axis=1).mean())
            per_class[c] = h_actual - h_ref[c]

    return per_class, global_residual


# ── plotting ──────────────────────────────────────────────────────────────────

def dual_axis_plot(ax_l, ax_r, week_nums, metric_series, f1_series, macro_series,
                   classes, class_names, colors,
                   metric_label, metric_lims=None):
    """
    Left axis  (solid lines): unsupervised metric per class
    Right axis (dashed lines): F1 per class + macro F1 (thick black dashed)
    """
    for color, cls in zip(colors, classes):
        name = class_names.get(cls, str(cls))
        vals = np.array(metric_series[cls], dtype=float)
        ax_l.plot(week_nums, vals, color=color, linewidth=1.8,
                  marker='o', markersize=3.5, label=name)

    for color, cls in zip(colors, classes):
        name = class_names.get(cls, str(cls))
        f1 = np.array(f1_series[cls], dtype=float)
        ax_r.plot(week_nums, f1, color=color, linewidth=1.2,
                  linestyle='--', marker='s', markersize=2.5, alpha=0.75)

    ax_r.plot(week_nums, macro_series, color='black', linewidth=2,
              linestyle='--', label='Macro F1', zorder=5)

    ax_l.set_ylabel(metric_label, fontsize=10)
    ax_r.set_ylabel('F1', fontsize=10, color='#444444')
    ax_r.set_ylim(0, 1)
    ax_r.tick_params(axis='y', labelcolor='#444444')
    if metric_lims:
        ax_l.set_ylim(*metric_lims)
    ax_l.grid(True, alpha=0.2)
    ax_l.spines['top'].set_visible(False)
    ax_r.spines['top'].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir', default='figs/week_1_inference')
    parser.add_argument('--dataset_root',  default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--classes', nargs='+', type=int, default=None)
    parser.add_argument('--reference_week', type=int, default=1)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    output_path = Path(args.output) if args.output else inference_dir / 'temporal_abc.png'

    print("Loading class names...")
    class_names, num_classes = load_class_names(args.dataset_root)

    print("Loading npz files...")
    weeks = load_weeks(inference_dir)
    print(f"  {len(weeks)} weeks: {weeks[0][0]}–{weeks[-1][0]}")

    classes = args.classes if args.classes is not None else pick_classes(weeks)
    print(f"  Classes: {[(c, class_names.get(c,'?')) for c in classes]}")

    # Reference week data
    ref = next(((t, p, s, e) for wn, t, p, s, e in weeks if wn == args.reference_week), None)
    if ref is None:
        ref = weeks[0][1:]
        print(f"  Warning: week {args.reference_week} not found, using week {weeks[0][0]}")
    ref_true, ref_pred, ref_soft, ref_emb = ref

    # Reference centroids (from true labels in reference week)
    ref_centroids = {}
    for cls in classes:
        mask = ref_true == cls
        ref_centroids[cls] = ref_emb[mask].mean(axis=0) if mask.sum() > 0 else None

    # BBSE reference
    print("Building BBSE reference...")
    C_T_pinv, p_train, h_ref = build_bbse_reference(ref_true, ref_pred, ref_soft, num_classes)

    # Per-week metrics
    week_nums        = []
    macro_f1s        = []
    global_residuals = []
    cls_f1_w         = {c: [] for c in classes}
    cls_ent_w        = {c: [] for c in classes}
    cls_dist_w       = {c: [] for c in classes}
    cls_bbse_w       = {c: [] for c in classes}

    for wn, true, pred, softmax, emb in weeks:
        week_nums.append(wn)
        macro_f1s.append(macro_f1(true, pred, num_classes))
        per_class_res, global_res = bbse_residuals(softmax, pred, num_classes,
                                                    C_T_pinv, p_train, h_ref)
        global_residuals.append(global_res)
        for cls in classes:
            cls_f1_w[cls].append(class_f1(true, pred, cls))
            cls_ent_w[cls].append(entropy_pseudo(softmax, pred, cls))
            cls_dist_w[cls].append(centroid_dist_pseudo(emb, pred, cls, ref_centroids[cls]))
            cls_bbse_w[cls].append(per_class_res[cls])

    week_nums        = np.array(week_nums)
    macro_f1s        = np.array(macro_f1s)
    global_residuals = np.array(global_residuals)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(classes)))
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)

    panels = [
        (cls_dist_w, 'A', 'Centroid Drift  (L2)'),
        (cls_ent_w,  'B', 'Class-Cond. Entropy'),
        (cls_bbse_w, 'C', 'BBSE Drift Residual\nactual entropy − BBSE-expected entropy'),
    ]

    for i, ((metric_dict, label, ylabel), ax) in enumerate(zip(panels, axes)):
        ax_r = ax.twinx()
        dual_axis_plot(
            ax, ax_r, week_nums,
            metric_dict, cls_f1_w, macro_f1s,
            classes, class_names, colors,
            metric_label=ylabel,
        )
        # Panel C: also draw the global BBSE residual
        if label == 'C':
            ax.plot(week_nums, global_residuals, color='black', linewidth=2.5,
                    linestyle='-', marker='D', markersize=4,
                    label='Global residual (actual − BBSE-expected)', zorder=6)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax.text(-0.055, 0.5, label, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='center')
        ax.set_xticks(week_nums)
        ax.tick_params(axis='x', labelsize=8, rotation=45)

    # Shared legend (metric lines only — right-axis is implicitly same colors dashed)
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_, fontsize=8, loc='upper right', ncol=2,
                   framealpha=0.9, edgecolor='#cccccc',
                   title='Solid = metric  |  Dashed = F1')

    # Macro F1 legend entry on panel C
    h2, l2 = axes[-1].get_figure().axes[-1].get_legend_handles_labels()
    axes[-1].legend(h2, l2, fontsize=8, loc='upper right',
                    framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(
        f'Drift Metrics vs F1 — model trained on week {args.reference_week}  '
        f'({len(weeks)} test weeks: {weeks[0][0]}–{weeks[-1][0]})',
        fontsize=13, fontweight='bold',
    )
    axes[-1].set_xlabel('Week Number', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved -> {output_path}")


if __name__ == '__main__':
    main()
