#!/usr/bin/env python
"""
Three paper figures from CESNET week-by-week inference results.

  Fig A — BBSE Estimation Accuracy (L1 error vs true label distribution)
  Fig B — Mirror Effect: Macro F1 (down) vs Entropy Gap (up)
  Fig C — App-level degradation grid (top-N classes by train frequency)

Usage:
    python plot_paper_figures.py [--inference_dir DIR] [--dataset_root DIR]
                                 [--reference_week W] [--grid_classes N]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
import seaborn as sns


# ── helpers ────────────────────────────────────────────────────────────────────

def load_class_names(dataset_root):
    import sys; sys.path.insert(0, '.')
    try:
        from train_per_week_cesnet import load_label_mapping
        mapping, _ = load_label_mapping(Path(dataset_root))
        return {v: k for k, v in mapping.items()}
    except Exception:
        return {}


def load_weeks(inference_dir):
    files = sorted(
        Path(inference_dir).glob('WEEK-2022-*.npz'),
        key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    )
    out = []
    for f in files:
        wn = int(re.search(r'(\d+)$', f.stem).group(1))
        d  = np.load(f)
        out.append((wn, d['true_labels'], d['pred_labels'], d['softmax']))
    return out


def regularized_bbse(confusion_matrix_T, observed_preds):
    """
    Constrained least-squares label shift estimation.

    confusion_matrix_T : C^T  (K×K), where C[i,j] = P(pred=j | true=i)
    observed_preds     : q_hat (K,), empirical predicted-class frequencies

    Solves: min ||C^T @ mu - q_hat||²  s.t.  mu >= 0, sum(mu) = 1
    This is equivalent to the user-supplied formulation with confusion_matrix=C^T.
    """
    K    = confusion_matrix_T.shape[1]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bnds = tuple((0.0, 1.0) for _ in range(K))
    res  = minimize(
        lambda mu: np.sum((np.dot(confusion_matrix_T, mu) - observed_preds) ** 2),
        np.ones(K) / K,
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options={'ftol': 1e-10, 'maxiter': 1000},
    )
    return res.x


def build_bbse(ref_true, ref_pred, ref_softmax, num_classes):
    cm = confusion_matrix(ref_true, ref_pred, labels=list(range(num_classes))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C        = cm / row_sums                   # C[i,j] = P(pred=j | true=i)
    C_T      = C.T
    C_T_pinv = np.linalg.pinv(C_T, rcond=1e-2)

    counts  = np.bincount(ref_true, minlength=num_classes).astype(float)
    p_train = counts / counts.sum()

    h_ref    = np.full(num_classes, np.nan)
    conf_ref = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = ref_true == c
        if mask.sum() > 0:
            p         = np.clip(ref_softmax[mask], 1e-12, 1.0)
            h_ref[c]    = float(-np.sum(p * np.log(p), axis=1).mean())
            conf_ref[c] = float(ref_softmax[mask].max(axis=1).mean())

    return C_T_pinv, C_T, p_train, h_ref, conf_ref


def estimate_label_dist(pred, num_classes, C_T_pinv):
    """Returns (p_bbse, q_hat) — BBSE estimate and raw predicted frequencies."""
    q_hat = np.bincount(pred, minlength=num_classes).astype(float)
    q_hat /= q_hat.sum()
    p_hat = C_T_pinv @ q_hat
    p_hat = np.clip(p_hat, 0, None)
    if p_hat.sum() > 0:
        p_hat /= p_hat.sum()
    return p_hat, q_hat


def sld_em_estimation(train_priors, test_posteriors, max_iter=100, tol=1e-6,
                      max_weight=None, alpha=1.0, lam=0.0, naive_init=False):
    """
    SLD-EM label shift estimation with optional robustness knobs.

    max_weight : clip importance ratio to this value (None = no clamp)
    alpha      : raise importance ratio to this power before applying (<1 dampens)
    lam        : regularization strength pulling estimate toward train_priors each step
    naive_init : initialize from argmax frequencies instead of train_priors
    """
    if naive_init:
        counts = np.bincount(test_posteriors.argmax(axis=1),
                             minlength=len(train_priors)).astype(float)
        current_priors = counts / counts.sum() if counts.sum() > 0 else np.copy(train_priors)
    else:
        current_priors = np.copy(train_priors)

    for _ in range(max_iter):
        old_priors = np.copy(current_priors)
        weights = current_priors / np.where(train_priors > 0, train_priors, 1e-12)
        if max_weight is not None:
            weights = np.clip(weights, 0.0, max_weight)
        if alpha != 1.0:
            weights = np.power(weights, alpha)
        weighted_posteriors = test_posteriors * weights
        row_sums = np.sum(weighted_posteriors, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_posteriors = weighted_posteriors / row_sums
        current_priors = np.mean(new_posteriors, axis=0)
        if lam > 0.0:
            current_priors = (1.0 - lam) * current_priors + lam * train_priors
        if np.linalg.norm(current_priors - old_priors, ord=1) < tol:
            break
    return current_priors


def reweighted_macro_f1(cm_raw, p_train, p_test, num_classes):
    """
    Macro F1 after reweighting test samples to match training priors.
    Scales confusion matrix rows by w_i = p_train[i] / p_test[i],
    eliminating label shift so only covariate shift remains.
    """
    weights = np.where(p_test > 0, p_train / np.where(p_test > 0, p_test, 1.0), 0.0)
    cm_w = cm_raw * weights[:, np.newaxis]
    f1s = []
    for c in range(num_classes):
        if cm_raw[c].sum() == 0:
            continue
        tp = cm_w[c, c]
        fp = cm_w[:, c].sum() - tp
        fn = cm_w[c, :].sum() - tp
        d  = 2 * tp + fp + fn
        if d > 0:
            f1s.append(2 * tp / d)
    return float(np.mean(f1s)) if f1s else 0.0


def synthetic_macro_f1(C_base_norm, p_t, num_classes):
    """
    Macro F1 with per-class behaviour frozen at C_base_norm but class priors = p_t.
    C_base_norm[i,j] = P(pred=j | true=i) from the reference week.
    Isolates the label-shift penalty alone.
    """
    cm_syn = C_base_norm * p_t[:, np.newaxis]
    f1s = []
    for c in range(num_classes):
        if p_t[c] == 0:
            continue
        tp = cm_syn[c, c]
        fp = cm_syn[:, c].sum() - tp
        fn = cm_syn[c, :].sum() - tp
        d  = 2 * tp + fp + fn
        if d > 0:
            f1s.append(2 * tp / d)
    return float(np.mean(f1s)) if f1s else 0.0


def entropy_gap(softmax, pred, num_classes, C_T_pinv, h_ref):
    """Global BBSE entropy residual: actual mean entropy − BBSE-expected entropy."""
    p_hat, q_hat = estimate_label_dist(pred, num_classes, C_T_pinv)
    valid    = ~np.isnan(h_ref)
    expected = float(np.dot(p_hat[valid], h_ref[valid]))
    p_all    = np.clip(softmax, 1e-12, 1.0)
    actual   = float(-np.sum(p_all * np.log(p_all), axis=1).mean())
    return actual - expected


def entropy_gap_raw(softmax, h_ref, p_train):
    """Entropy residual without label-shift correction: actual − train-prior-weighted expected."""
    valid    = ~np.isnan(h_ref)
    expected = float(np.dot(p_train[valid], h_ref[valid]))
    p_all    = np.clip(softmax, 1e-12, 1.0)
    actual   = float(-np.sum(p_all * np.log(p_all), axis=1).mean())
    return actual - expected


def confidence_gap(softmax, pred, num_classes, C_T_pinv, conf_ref):
    """Confidence residual: actual mean max-softmax − BBSE-expected mean max-softmax."""
    p_hat, _ = estimate_label_dist(pred, num_classes, C_T_pinv)
    valid    = ~np.isnan(conf_ref)
    expected = float(np.dot(p_hat[valid], conf_ref[valid]))
    actual   = float(softmax.max(axis=1).mean())
    return actual - expected


def confidence_gap_raw(softmax, conf_ref, p_train):
    """Confidence residual without label-shift correction: actual − train-prior-weighted expected."""
    valid    = ~np.isnan(conf_ref)
    expected = float(np.dot(p_train[valid], conf_ref[valid]))
    actual   = float(softmax.max(axis=1).mean())
    return actual - expected


def per_class_f1(true, pred, cls):
    tp = int(((pred == cls) & (true == cls)).sum())
    fp = int(((pred == cls) & (true != cls)).sum())
    fn = int(((pred != cls) & (true == cls)).sum())
    d  = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else np.nan


# ── data loading ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir',  default='figs/week_1_inference')
    parser.add_argument('--dataset_root',   default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--reference_week', type=int, default=0)
    parser.add_argument('--grid_classes',   type=int, default=9,
                        help='Number of top classes for the app grid (4 or 9)')
    parser.add_argument('--output_dir',     default='figs')
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['a', 'b', 'c', 'c2', 'd', 'e', 'f', 'g', 'h'],
                        metavar='FIG', help='Figures to skip: a b c c2 d e f g h')
    parser.add_argument('--recompute', action='store_true',
                        help='Ignore cached metrics and recompute from scratch')
    parser.add_argument('--em', action='store_true',
                        help='Run SLD-EM estimation (slow; off by default)')
    parser.add_argument('--confidence', action='store_true',
                        help='Use max-softmax confidence gap instead of entropy gap in Fig B')
    args = parser.parse_args()
    skip = set(args.skip)

    inference_dir = Path(args.inference_dir)
    output_dir    = Path(args.output_dir)

    print("Loading class names...")
    class_names = load_class_names(args.dataset_root)

    print("Loading weeks...")
    weeks = load_weeks(inference_dir)
    print(f"  {len(weeks)} weeks: {weeks[0][0]}–{weeks[-1][0]}")

    ref = next(((t, p, s) for wn, t, p, s in weeks if wn == args.reference_week), None)
    if ref is None:
        print(f"  Warning: week {args.reference_week} not found, using week {weeks[0][0]}")
        ref = weeks[0][1:]
    ref_true, ref_pred, ref_soft = ref
    num_classes = ref_soft.shape[1]

    C_T_pinv, C_T, p_train, h_ref, conf_ref = build_bbse(ref_true, ref_pred, ref_soft, num_classes)
    C_base_norm = C_T.T  # C_base_norm[i,j] = P(pred=j | true=i) from reference week

    # Top classes by training frequency for the grid figure
    train_counts  = np.bincount(ref_true, minlength=num_classes).astype(float)
    train_pct     = train_counts / train_counts.sum() * 100
    grid_classes  = np.argsort(-train_counts)[:args.grid_classes].tolist()

    # Test weeks only (exclude reference)
    test_weeks = [(wn, t, p, s) for wn, t, p, s in weeks if wn != args.reference_week]

    # ── metrics cache ─────────────────────────────────────────────────────────
    cache_path = inference_dir / f'metrics_cache_ref{args.reference_week}_grid{args.grid_classes}.npz'

    if cache_path.exists() and not args.recompute:
        print(f"Loading cached metrics from {cache_path} ...")
        cache = np.load(cache_path, allow_pickle=False)
        week_nums       = cache['week_nums']
        macro_f1s       = cache['macro_f1s']
        ent_gaps        = cache['ent_gaps']
        ent_gaps_raw     = cache['ent_gaps_raw']     if 'ent_gaps_raw'     in cache else None
        ent_gaps_oracle  = cache['ent_gaps_oracle']  if 'ent_gaps_oracle'  in cache else None
        conf_gaps        = cache['conf_gaps']        if 'conf_gaps'        in cache else None
        conf_gaps_raw    = cache['conf_gaps_raw']    if 'conf_gaps_raw'    in cache else None
        conf_gaps_oracle = cache['conf_gaps_oracle'] if 'conf_gaps_oracle' in cache else None
        l1_bbse_arr     = cache['l1_bbse_arr']
        l1_reg_arr      = cache['l1_reg_arr']
        l1_naive_arr    = cache['l1_naive_arr']
        l1_prior_arr    = cache['l1_prior_arr']
        l1_em_arr       = cache['l1_em_arr'] if 'l1_em_arr' in cache else None
        corrected_f1s   = cache['corrected_f1s']
        synthetic_f1s   = cache['synthetic_f1s']
        baseline_f1     = float(cache['baseline_f1'])
        grid_classes_arr = cache['grid_classes_arr']
        cls_f1_mat      = cache['cls_f1_mat']      # (n_grid, n_test)
        cls_phat_mat    = cache['cls_phat_mat']
        cls_ptrue_mat   = cache['cls_ptrue_mat']
        cls_bbse_err_mat = cache['cls_bbse_err_mat']  # (num_classes, n_test)
        grid_classes    = grid_classes_arr.tolist()
        cls_f1_w        = {int(c): cls_f1_mat[i].tolist()   for i, c in enumerate(grid_classes)}
        cls_phat_w      = {int(c): cls_phat_mat[i].tolist() for i, c in enumerate(grid_classes)}
        cls_ptrue_w     = {int(c): cls_ptrue_mat[i].tolist() for i, c in enumerate(grid_classes)}
        cls_bbse_err_w  = {c: cls_bbse_err_mat[c].tolist() for c in range(num_classes)}
        print(f"  Loaded {len(week_nums)} weeks from cache.")
    else:
        if args.recompute and cache_path.exists():
            print("Recompute flag set — ignoring existing cache.")
        print("Computing per-week metrics...")
        week_nums        = []
        macro_f1s        = []
        ent_gaps         = []
        ent_gaps_raw     = []
        ent_gaps_oracle  = []
        conf_gaps        = []
        conf_gaps_raw    = []
        conf_gaps_oracle = []
        l1_bbse_arr      = []
        l1_reg_arr       = []
        l1_naive_arr     = []
        l1_prior_arr     = []
        l1_em_arr        = [] if args.em else None
        cls_f1_w         = {c: [] for c in grid_classes}
        cls_phat_w       = {c: [] for c in grid_classes}
        cls_ptrue_w      = {c: [] for c in grid_classes}
        cls_bbse_err_w   = {c: [] for c in range(num_classes)}
        corrected_f1s    = []
        synthetic_f1s    = []

        for wn, true, pred, softmax in tqdm(test_weeks, desc='  weeks'):
            week_nums.append(wn)

            p_true_week = np.bincount(true, minlength=num_classes).astype(float)
            p_true_week /= p_true_week.sum()

            p_bbse, q_hat = estimate_label_dist(pred, num_classes, C_T_pinv)
            p_reg = regularized_bbse(C_T, q_hat)
            l1_bbse_arr.append(float(np.mean(np.abs(p_bbse   - p_true_week))))
            l1_reg_arr.append( float(np.mean(np.abs(p_reg    - p_true_week))))
            l1_naive_arr.append(float(np.mean(np.abs(q_hat   - p_true_week))))
            l1_prior_arr.append(float(np.mean(np.abs(p_train - p_true_week))))
            if args.em:
                p_em = sld_em_estimation(p_train, softmax)
                l1_em_arr.append(float(np.mean(np.abs(p_em - p_true_week))))

            macro_f1s.append(f1_score(true, pred, labels=list(range(num_classes)),
                                      average='macro', zero_division=0))
            ent_gaps.append(entropy_gap(softmax, pred, num_classes, C_T_pinv, h_ref))
            ent_gaps_raw.append(entropy_gap_raw(softmax, h_ref, p_train))
            ent_gaps_oracle.append(entropy_gap_raw(softmax, h_ref, p_true_week))
            conf_gaps.append(confidence_gap(softmax, pred, num_classes, C_T_pinv, conf_ref))
            conf_gaps_raw.append(confidence_gap_raw(softmax, conf_ref, p_train))
            conf_gaps_oracle.append(confidence_gap_raw(softmax, conf_ref, p_true_week))

            cm_week = confusion_matrix(true, pred, labels=list(range(num_classes))).astype(float)
            corrected_f1s.append(reweighted_macro_f1(cm_week, p_train, p_true_week, num_classes))
            synthetic_f1s.append(synthetic_macro_f1(C_base_norm, p_true_week, num_classes))

            for c in grid_classes:
                cls_f1_w[c].append(per_class_f1(true, pred, c))
                cls_phat_w[c].append(float(q_hat[c] * 100))
                cls_ptrue_w[c].append(float(p_true_week[c] * 100))

            for c in range(num_classes):
                cls_bbse_err_w[c].append(float(np.abs(p_bbse[c] - p_true_week[c])))

        week_nums       = np.array(week_nums)
        macro_f1s       = np.array(macro_f1s)
        ent_gaps        = np.array(ent_gaps)
        ent_gaps_raw    = np.array(ent_gaps_raw)
        ent_gaps_oracle = np.array(ent_gaps_oracle)
        conf_gaps       = np.array(conf_gaps)
        conf_gaps_raw   = np.array(conf_gaps_raw)
        conf_gaps_oracle = np.array(conf_gaps_oracle)
        l1_bbse_arr     = np.array(l1_bbse_arr)
        l1_reg_arr      = np.array(l1_reg_arr)
        l1_naive_arr    = np.array(l1_naive_arr)
        l1_prior_arr    = np.array(l1_prior_arr)
        l1_em_arr       = np.array(l1_em_arr) if args.em else None
        corrected_f1s   = np.array(corrected_f1s)
        synthetic_f1s   = np.array(synthetic_f1s)

        baseline_f1 = synthetic_macro_f1(C_base_norm, p_train, num_classes)

        # save cache
        grid_classes_arr = np.array(grid_classes)
        cls_f1_mat       = np.array([cls_f1_w[c]       for c in grid_classes])
        cls_phat_mat     = np.array([cls_phat_w[c]     for c in grid_classes])
        cls_ptrue_mat    = np.array([cls_ptrue_w[c]    for c in grid_classes])
        cls_bbse_err_mat = np.array([cls_bbse_err_w[c] for c in range(num_classes)])
        np.savez_compressed(
            cache_path,
            week_nums=week_nums, macro_f1s=macro_f1s,
            ent_gaps=ent_gaps, ent_gaps_raw=ent_gaps_raw, ent_gaps_oracle=ent_gaps_oracle,
            conf_gaps=conf_gaps, conf_gaps_raw=conf_gaps_raw, conf_gaps_oracle=conf_gaps_oracle,
            l1_bbse_arr=l1_bbse_arr, l1_reg_arr=l1_reg_arr,
            l1_naive_arr=l1_naive_arr, l1_prior_arr=l1_prior_arr,
            **(dict(l1_em_arr=l1_em_arr) if args.em else {}),
            corrected_f1s=corrected_f1s, synthetic_f1s=synthetic_f1s,
            baseline_f1=np.array(baseline_f1),
            grid_classes_arr=grid_classes_arr,
            cls_f1_mat=cls_f1_mat, cls_phat_mat=cls_phat_mat,
            cls_ptrue_mat=cls_ptrue_mat, cls_bbse_err_mat=cls_bbse_err_mat,
        )
        print(f"  Saved metrics cache → {cache_path}")

    print(f"  Baseline F1 (ref week, frozen C): {baseline_f1:.4f}")

    if args.confidence and conf_gaps is not None:
        gaps        = conf_gaps
        gaps_raw    = conf_gaps_raw
        gaps_oracle = conf_gaps_oracle
        gap_label   = 'Confidence gap (BBSE-corrected)'
        gap_label_r = 'Confidence gap (uncorrected)'
        gap_ylabel  = 'Confidence Gap\n(actual − expected mean max-softmax)'
    else:
        gaps        = ent_gaps
        gaps_raw    = ent_gaps_raw
        gaps_oracle = ent_gaps_oracle
        gap_label   = 'Entropy gap (BBSE-corrected)'
        gap_label_r = 'Entropy gap (uncorrected)'
        gap_ylabel  = 'Entropy Gap\n(actual − expected)'
    rho, _ = pearsonr(macro_f1s, gaps)

    # ── shared style ───────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size':   11,
        'axes.grid':   True,
        'grid.alpha':  0.22,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })
    MKW = dict(marker='o', markersize=4, linewidth=1.8)

    ref_label = f'week {args.reference_week}'
    n_test    = len(test_weeks)
    span      = f'{week_nums[0]}–{week_nums[-1]}'

    n_grid     = args.grid_classes
    n_cols     = 3 if n_grid > 4 else 2
    n_rows     = (n_grid + n_cols - 1) // n_cols
    xtick_step = max(1, len(week_nums) // 6)

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure A — BBSE Estimation Accuracy
    # ═══════════════════════════════════════════════════════════════════════════
    if 'a' in skip:
        print("Skipping Fig A")
    else:
        print("Plotting Fig A — estimation accuracy...")
        fig_a, ax = plt.subplots(figsize=(10, 4))
        ax.plot(week_nums, l1_prior_arr, color='#999999', linewidth=1.4,
                linestyle='--', marker='s', markersize=3.5,
                label='Static prior (train distribution)')
        ax.plot(week_nums, l1_naive_arr, color='#e07b39', **MKW,
                label='Uncalibrated (raw predicted frequencies)')
        ax.plot(week_nums, l1_bbse_arr,  color='#2c7bb6', **MKW,
                label='BBSE (truncated pseudo-inverse)')
        ax.plot(week_nums, l1_reg_arr,   color='#1a9641', **MKW,
                label='Regularized BBSE (constrained least-squares)')
        if l1_em_arr is not None:
            ax.plot(week_nums, l1_em_arr, color='#9b59b6', **MKW,
                    label='SLD-EM')
        ax.set_xlabel('Week Number', fontsize=12)
        ax.set_ylabel(r'$L_1$ Error (MAE)', fontsize=12)
        ax.set_title(
            f'Label Distribution Estimation Accuracy Over Time\n'
            f'(reference: {ref_label},  {n_test} test weeks: {span})',
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks(week_nums[::max(1, len(week_nums) // 12)])
        ax.legend(fontsize=10, framealpha=0.9)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        out_a = output_dir / 'fig_estimation_accuracy.png'
        fig_a.savefig(out_a, dpi=200, bbox_inches='tight')
        plt.close(fig_a)
        print(f"  Saved → {out_a}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure B — Mirror Effect
    # ═══════════════════════════════════════════════════════════════════════════
    if 'b' in skip:
        print("Skipping Fig B")
    else:
        print("Plotting Fig B — mirror effect...")
        rho_raw = pearsonr(macro_f1s, gaps_raw)[0] if gaps_raw is not None else float('nan')
        fig_b, ax_l = plt.subplots(figsize=(10, 4))
        ax_r = ax_l.twinx()
        ax_r.spines['top'].set_visible(False)
        l1, = ax_l.plot(week_nums, macro_f1s, color='#2c7bb6', **MKW,
                        label='Macro F1 (true labels)')
        l2, = ax_r.plot(week_nums, gaps, color='#d7191c',
                        marker='s', markersize=4, linewidth=1.8,
                        label=f'{gap_label}  (ρ = {rho:.3f})')
        legend_handles = [l1, l2]
        if gaps_raw is not None:
            l3, = ax_r.plot(week_nums, gaps_raw, color='#e08020',
                            marker='^', markersize=4, linewidth=1.8, linestyle='--',
                            label=f'{gap_label_r}  (ρ = {rho_raw:.3f})')
            legend_handles.append(l3)
        if gaps_oracle is not None:
            rho_oracle = pearsonr(macro_f1s, gaps_oracle)[0]
            l4, = ax_r.plot(week_nums, gaps_oracle, color='#27ae60',
                            marker='D', markersize=4, linewidth=1.8, linestyle=':',
                            label=f'Perfect correction (true prior from the labels)  (ρ = {rho_oracle:.3f})')
            legend_handles.append(l4)
        ax_l.set_xlabel('Week Number', fontsize=12)
        ax_l.set_ylabel('Macro F1', fontsize=12, color='#2c7bb6')
        ax_r.set_ylabel(gap_ylabel, fontsize=12, color='#d7191c')
        ax_l.tick_params(axis='y', labelcolor='#2c7bb6')
        ax_r.tick_params(axis='y', labelcolor='#d7191c')
        ax_l.set_xticks(week_nums[::max(1, len(week_nums) // 12)])
        ax_l.set_title(
            'Macro F1 vs Entropy Gap Over Time — The Mirror Effect\n'
            f'(reference: {ref_label},  {n_test} test weeks: {span})',
            fontsize=12, fontweight='bold'
        )
        ax_l.legend(handles=legend_handles, fontsize=10, loc='center right', framealpha=0.9)
        plt.tight_layout()
        out_b = output_dir / 'fig_mirror_effect.png'
        fig_b.savefig(out_b, dpi=200, bbox_inches='tight')
        plt.close(fig_b)
        print(f"  Saved → {out_b}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure C — App-level degradation grid (F1 + class %)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'c' in skip:
        print("Skipping Fig C")
    else:
        print("Plotting Fig C — app degradation grid...")
        fig_c, axes = plt.subplots(n_rows, n_cols,
                                   figsize=(n_cols * 4.5, n_rows * 3.2),
                                   squeeze=False)
        for idx, cls in enumerate(grid_classes):
            row, col = divmod(idx, n_cols)
            ax       = axes[row][col]
            ax_p     = ax.twinx()
            ax_p.spines['top'].set_visible(False)
            name    = class_names.get(int(cls), f'Class {cls}')
            f1_vals = np.array(cls_f1_w[cls],   dtype=float)
            ph_vals = np.array(cls_phat_w[cls],  dtype=float)
            pt_vals = np.array(cls_ptrue_w[cls], dtype=float)
            lf1, = ax.plot(week_nums, f1_vals,
                           color='#2c7bb6', linewidth=1.8, marker='o', markersize=3.5,
                           label='True F1')
            lpt, = ax_p.plot(week_nums, pt_vals,
                             color='#888888', linewidth=1.2, linestyle='--',
                             marker=None, label='True % (test)')
            lph, = ax_p.plot(week_nums, ph_vals,
                             color='#d7191c', linewidth=1.6, linestyle='-.',
                             marker='s', markersize=3, label='Predicted %')
            ax_p.axhline(train_pct[cls], color='#d7191c', linewidth=0.9,
                         linestyle=':', alpha=0.7)
            ax.set_ylim(0, 1)
            ax_p.set_ylim(bottom=0)
            ax.set_ylabel('F1', fontsize=9, color='#2c7bb6')
            ax_p.set_ylabel('Class %', fontsize=9, color='#d7191c')
            ax.tick_params(axis='y', labelcolor='#2c7bb6', labelsize=8)
            ax_p.tick_params(axis='y', labelcolor='#d7191c', labelsize=8)
            ax.set_xticks(week_nums[::xtick_step])
            ax.tick_params(axis='x', labelsize=7, rotation=45)
            ax.grid(True, alpha=0.2)
            ax.set_title(f'{name}  (train: {train_pct[cls]:.2f}%)',
                         fontsize=10, fontweight='bold', pad=4)
            ax.legend(handles=[lf1, lph, lpt], fontsize=7.5,
                      loc='upper right', framealpha=0.85)
        for idx in range(n_grid, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)
        fig_c.suptitle(
            f'App-Level Degradation — Top {n_grid} Classes by Training Frequency\n'
            f'(reference: {ref_label},  dashed line = train %,  '
            f'{n_test} test weeks: {span})',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        out_c = output_dir / 'fig_app_degradation_grid.png'
        fig_c.savefig(out_c, dpi=200, bbox_inches='tight')
        plt.close(fig_c)
        print(f"  Saved → {out_c}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure C2 — App-level degradation grid with Pearson ρ, no F1
    # ═══════════════════════════════════════════════════════════════════════════
    if 'c2' in skip:
        print("Skipping Fig C2")
    else:
        print("Plotting Fig C2 — app degradation grid with correlation...")
        fig_c2, axes2 = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 4.5, n_rows * 3.2),
                                     squeeze=False)
        for idx, cls in enumerate(grid_classes):
            row, col = divmod(idx, n_cols)
            ax       = axes2[row][col]
            name    = class_names.get(int(cls), f'Class {cls}')
            ph_vals = np.array(cls_phat_w[cls],  dtype=float)
            pt_vals = np.array(cls_ptrue_w[cls], dtype=float)
            valid = ~(np.isnan(ph_vals) | np.isnan(pt_vals))
            rho_c = float(pearsonr(ph_vals[valid], pt_vals[valid])[0]) if valid.sum() > 1 else float('nan')
            ax.plot(week_nums, pt_vals,
                    color='#888888', linewidth=1.2, linestyle='--',
                    label='True % (test)')
            ax.plot(week_nums, ph_vals,
                    color='#d7191c', linewidth=1.6, linestyle='-.',
                    marker='s', markersize=3, label='Predicted %')
            ax.axhline(train_pct[cls], color='#d7191c', linewidth=0.9,
                       linestyle=':', alpha=0.7, label=f'Train % ({train_pct[cls]:.2f}%)')
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Class %', fontsize=9)
            ax.set_xticks(week_nums[::xtick_step])
            ax.tick_params(axis='x', labelsize=7, rotation=45)
            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f'{name}  (train: {train_pct[cls]:.2f}%,  ρ={rho_c:.2f})',
                         fontsize=10, fontweight='bold', pad=4)
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
        for idx in range(n_grid, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes2[row][col].set_visible(False)
        fig_c2.suptitle(
            f'App-Level Degradation — Top {n_grid} Classes by Training Frequency\n'
            f'(reference: {ref_label},  ρ = Pearson corr between predicted % and true %,  '
            f'{n_test} test weeks: {span})',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        out_c2 = output_dir / 'fig_app_degradation_grid_corr.png'
        fig_c2.savefig(out_c2, dpi=200, bbox_inches='tight')
        plt.close(fig_c2)
        print(f"  Saved → {out_c2}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure D — Per-class BBSE error vs train frequency (scatter)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'd' in skip:
        print("Skipping Fig D")
    else:
        print("Plotting Fig D — per-class BBSE error scatter...")
        present_classes = np.where(train_counts > 0)[0]
        mean_bbse_err   = np.array([np.mean(cls_bbse_err_w[c]) for c in present_classes])
        class_train_pct = train_pct[present_classes]
        fig_d, ax_d = plt.subplots(figsize=(9, 6))
        ax_d.scatter(class_train_pct, mean_bbse_err * 100,
                     alpha=0.55, s=30, color='#2c7bb6', edgecolors='none')
        for cls in grid_classes:
            if train_counts[cls] == 0:
                continue
            x    = train_pct[cls]
            y    = np.mean(cls_bbse_err_w[cls]) * 100
            name = class_names.get(int(cls), f'Class {cls}')
            ax_d.annotate(name, (x, y), fontsize=7.5, ha='left',
                          xytext=(4, 2), textcoords='offset points')
        ax_d.set_xlabel('Train distribution %', fontsize=12)
        ax_d.set_ylabel('Mean BBSE absolute error (×100)', fontsize=12)
        ax_d.set_title(
            f'Per-class BBSE error vs. train frequency\n'
            f'(reference: {ref_label},  {n_test} test weeks: {span})',
            fontsize=12, fontweight='bold'
        )
        ax_d.grid(True, alpha=0.22)
        ax_d.spines['top'].set_visible(False)
        ax_d.spines['right'].set_visible(False)
        plt.tight_layout()
        out_d = output_dir / 'fig_bbse_per_class_scatter.png'
        fig_d.savefig(out_d, dpi=200, bbox_inches='tight')
        plt.close(fig_d)
        print(f"  Saved → {out_d}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure E (6a) — Covariate Shift Penalty (label shift removed)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'e' in skip:
        print("Skipping Fig E")
    else:
        print("Plotting Fig E — covariate shift penalty...")
        with sns.axes_style("whitegrid"):
            fig_e, ax_e = plt.subplots(figsize=(11, 4.5))
            ax_e.plot(week_nums, macro_f1s,
                      color='#e74c3c', linewidth=2.0, marker='o', markersize=4,
                      label='Raw F1 (Combined Shifts)')
            ax_e.plot(week_nums, corrected_f1s,
                      color='#2980b9', linewidth=2.0, linestyle='--',
                      marker='s', markersize=4,
                      label='Corrected F1 (Covariate Shift Only)')
            ax_e.set_xlabel('Week Number', fontsize=13)
            ax_e.set_ylabel('Macro F1 Score', fontsize=13)
            ax_e.set_ylim(max(0, min(macro_f1s.min(), corrected_f1s.min()) - 0.05), 1.02)
            ax_e.set_xticks(week_nums[::max(1, len(week_nums) // 12)])
            ax_e.set_title(
                'Figure 6a — The Covariate Shift Penalty\n'
                f'(reference: {ref_label},  {n_test} test weeks: {span})\n'
                'Corrected line removes label shift — gap shows its contribution',
                fontsize=12, fontweight='bold'
            )
            ax_e.legend(fontsize=11, framealpha=0.9)
            plt.tight_layout()
            out_e = output_dir / 'fig_covariate_penalty.png'
            fig_e.savefig(out_e, dpi=200, bbox_inches='tight')
            plt.close(fig_e)
        print(f"  Saved → {out_e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure F (6b) — Label Shift Penalty (covariate shift removed)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'f' in skip:
        print("Skipping Fig F")
    else:
        print("Plotting Fig F — label shift penalty...")
        with sns.axes_style("whitegrid"):
            fig_f, ax_f = plt.subplots(figsize=(11, 4.5))
            ax_f.axhline(baseline_f1, color='#7f8c8d', linewidth=1.5, linestyle=':',
                         label=f'Week {args.reference_week} Baseline F1 ({baseline_f1:.3f})')
            ax_f.plot(week_nums, synthetic_f1s,
                      color='#27ae60', linewidth=2.0, marker='o', markersize=4,
                      label='Synthetic F1 (Label Shift Only)')
            ax_f.set_xlabel('Week Number', fontsize=13)
            ax_f.set_ylabel('Macro F1 Score', fontsize=13)
            y_min = min(synthetic_f1s.min(), baseline_f1) - 0.05
            ax_f.set_ylim(max(0, y_min), 1.02)
            ax_f.set_xticks(week_nums[::max(1, len(week_nums) // 12)])
            ax_f.set_title(
                'Figure 6b — The Label Shift Penalty\n'
                f'(reference: {ref_label},  frozen C_base,  {n_test} test weeks: {span})\n'
                'Per-class behaviour frozen — fluctuation driven by traffic mix alone',
                fontsize=12, fontweight='bold'
            )
            ax_f.legend(fontsize=11, framealpha=0.9)
            plt.tight_layout()
            out_f = output_dir / 'fig_label_shift_penalty.png'
            fig_f.savefig(out_f, dpi=200, bbox_inches='tight')
            plt.close(fig_f)
        print(f"  Saved → {out_f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure G — Combined Penalties (all three Macro F1 lines)
    # ═══════════════════════════════════════════════════════════════════════════
    if 'g' in skip:
        print("Skipping Fig G")
    else:
        print("Plotting Fig G — combined penalties...")
        with sns.axes_style("whitegrid"):
            fig_g, ax_g = plt.subplots(figsize=(11, 4.5))
            ax_g.axhline(baseline_f1, color='#7f8c8d', linewidth=1.5, linestyle=':',
                         label=f'Baseline F1 @ week {args.reference_week} ({baseline_f1:.3f})')
            ax_g.plot(week_nums, macro_f1s,
                      color='#e74c3c', linewidth=2.0, marker='o', markersize=4,
                      label='Raw F1 (both shifts)')
            ax_g.plot(week_nums, corrected_f1s,
                      color='#2980b9', linewidth=2.0, linestyle='--',
                      marker='s', markersize=4,
                      label='Corrected F1 (covariate shift only, label shift removed)')
            ax_g.plot(week_nums, synthetic_f1s,
                      color='#27ae60', linewidth=2.0, linestyle='-.',
                      marker='^', markersize=4,
                      label='Synthetic F1 (label shift only, covariate shift frozen)')
            y_min = min(macro_f1s.min(), corrected_f1s.min(), synthetic_f1s.min(), baseline_f1) - 0.05
            ax_g.set_ylim(max(0, y_min), 1.02)
            ax_g.set_xlabel('Week Number', fontsize=13)
            ax_g.set_ylabel('Macro F1 Score', fontsize=13)
            ax_g.set_xticks(week_nums[::max(1, len(week_nums) // 12)])
            ax_g.set_title(
                'Decomposing Performance Degradation: Covariate vs Label Shift\n'
                f'(reference: {ref_label},  {n_test} test weeks: {span})',
                fontsize=12, fontweight='bold'
            )
            ax_g.legend(fontsize=11, framealpha=0.9)
            plt.tight_layout()
            out_g = output_dir / 'fig_combined_penalties.png'
            fig_g.savefig(out_g, dpi=200, bbox_inches='tight')
            plt.close(fig_g)
        print(f"  Saved → {out_g}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure H — Estimation Accuracy under Extreme Synthetic Label Shift
    #   Weeks < 3  : real data, unchanged
    #   Weeks >= 3 : per-class prior drawn i.i.d. from LogUniform[lo, hi],
    #                rows resampled to match that prior.
    #   Two output files: 10x range [1e-1, 1e1] and 100x range [1e-2, 1e2].
    # ═══════════════════════════════════════════════════════════════════════════
    if 'h' in skip:
        print("Skipping Fig H")
    else:
        EXTREME_FROM_WN = 3
        N_RESAMPLE      = 8_000

        for range_lo, range_hi in [(1e-1, 1e1), (1e-2, 1e2)]:
            tag = f'{int(round(1/range_lo))}x'   # "10x" or "100x"
            print(f"Plotting Fig H — estimation accuracy extreme shift [{range_lo:.0e}, {range_hi:.0e}]...")
            rng_extreme = np.random.default_rng(42)   # same seed → same resampling structure

            h_week_nums = []
            h_l1_prior  = []
            h_l1_naive  = []
            h_l1_bbse   = []
            h_l1_reg    = []

            for wn, true, pred, sm in test_weeks:
                if wn < EXTREME_FROM_WN:
                    t_w, pred_w = true, pred
                    p_true_week = np.bincount(t_w, minlength=num_classes).astype(float)
                    p_true_week /= p_true_week.sum()
                else:
                    log_w   = rng_extreme.uniform(np.log(range_lo), np.log(range_hi), size=num_classes)
                    p_synth = np.exp(log_w);  p_synth /= p_synth.sum()

                    idx = []
                    for c in range(num_classes):
                        cls_idx = np.where(true == c)[0]
                        n_c     = int(round(p_synth[c] * N_RESAMPLE))
                        if n_c > 0 and len(cls_idx) > 0:
                            idx.append(rng_extreme.choice(cls_idx, size=n_c, replace=True))
                    if not idx:
                        t_w, pred_w = true, pred
                        p_true_week = np.bincount(t_w, minlength=num_classes).astype(float)
                        p_true_week /= p_true_week.sum()
                    else:
                        idx = np.concatenate(idx)
                        t_w, pred_w = true[idx], pred[idx]
                        p_true_week = p_synth   # ground truth IS the synthetic prior

                p_bbse, q_hat = estimate_label_dist(pred_w, num_classes, C_T_pinv)
                p_reg = regularized_bbse(C_T, q_hat)

                h_week_nums.append(wn)
                h_l1_prior.append(float(np.mean(np.abs(p_train - p_true_week))))
                h_l1_naive.append(float(np.mean(np.abs(q_hat   - p_true_week))))
                h_l1_bbse.append( float(np.mean(np.abs(p_bbse  - p_true_week))))
                h_l1_reg.append(  float(np.mean(np.abs(p_reg   - p_true_week))))

            h_week_nums = np.array(h_week_nums)

            with sns.axes_style("whitegrid"):
                fig_h, ax_h = plt.subplots(figsize=(11, 4.5))

                extreme_wns = h_week_nums[h_week_nums >= EXTREME_FROM_WN]
                if len(extreme_wns):
                    ax_h.axvspan(extreme_wns[0], extreme_wns[-1] + 0.5,
                                 alpha=0.06, color='#c0392b', zorder=0)
                    ax_h.axvline(extreme_wns[0], color='#888', linestyle=':', linewidth=1.2)

                ax_h.plot(h_week_nums, h_l1_prior, color='#999999', linewidth=1.4,
                          linestyle='--', marker='s', markersize=3.5,
                          label='Static prior (train distribution)')
                ax_h.plot(h_week_nums, h_l1_naive, color='#e07b39', **MKW,
                          label='Naive (raw predicted frequencies)')
                ax_h.plot(h_week_nums, h_l1_bbse,  color='#2c7bb6', **MKW,
                          label='BBSE (truncated pseudo-inverse)')
                ax_h.plot(h_week_nums, h_l1_reg,   color='#1a9641', **MKW,
                          label='Regularized BBSE (constrained least-squares)')

                ax_h.set_xlabel('Week Number', fontsize=12)
                ax_h.set_ylabel(r'$L_1$ Error (MAE)', fontsize=12)
                ax_h.set_xticks(h_week_nums[::max(1, len(h_week_nums) // 12)])
                ax_h.set_title(
                    f'Estimation Accuracy — Extreme Synthetic Label Shift from week {EXTREME_FROM_WN}\n'
                    f'(reference: {ref_label},  per-class prior ~ LogUniform[{range_lo:.0e}, {range_hi:.0e}])',
                    fontsize=12, fontweight='bold'
                )
                ax_h.legend(fontsize=11, framealpha=0.9)
                plt.tight_layout()
                out_h = output_dir / f'fig_estimation_extreme_{tag}.png'
                fig_h.savefig(out_h, dpi=200, bbox_inches='tight')
                plt.close(fig_h)
            print(f"  Saved → {out_h}")


if __name__ == '__main__':
    main()
