#!/usr/bin/env python
"""
Binary Anomaly Detection & AUROC experiment.

Turns the paper's robustness claim (Sec. V-C, Fig. 10) into a *threshold-based*
detection task instead of a bare correlation number:

    A monitor must raise an alarm iff the deployed model has GENUINELY degraded
    (covariate-shift-driven F1 collapse), using only unlabelled softmax /
    predictions, while NOT firing on benign traffic-volume volatility (a viral
    spike that changes the class mix but leaves per-class model behaviour — and
    hence true F1 — intact).

Ground truth
------------
anomaly(week) = 1  iff  the week's true covariate-driven Macro F1 < --f1_threshold.
On CESNET-TLS-Year22 (ref = Week 1) this cleanly splits weeks 1-9 (healthy,
F1 ~ 0.72-0.76) from weeks 10-52 (degraded, F1 ~ 0.34-0.57). A viral volume
spike never changes a week's covariate degradation status (the model weights
and per-class conditionals are untouched), so it is pure nuisance the detector
must be robust to. The label is therefore fixed by the source week's full-data
clean Macro F1, not recomputed per noisy window.

Two regimes
-----------
* CLEAN  : windows resampled at the natural class mix.        -> all detectors strong
* VIRAL  : one random app surges to dominate fraction f of a window (benign
           volume spike, as in Fig. 10).                      -> decisive false-alarm test

Detectors (all unsupervised, from window softmax / preds only)
--------------------------------------------------------------
* uncorrected : per-sample entropy gap with a fixed train-prior baseline (no label-shift correction)
* mfwdd       : MFWDD-style global drift severity = feature-importance-weighted
                per-channel Wasserstein distance of the window's softmax channels
                vs the reference week (a global, label-shift-sensitive trigger)
* bbse        : BBSE-corrected per-sample entropy residual (ours)

The uncorrected gap and the BBSE residual are the exact quantities plotted in
the paper's Fig. 9 (Mirror Effect); here we evaluate them as binary detectors.

Outputs
-------
* figs/fig_auroc_anomaly.png        — ROC curves (clean + viral) with matched-TPR markers
* figs/fig_auroc_score_dist.png     — score distributions explaining the false alarm
* figs/auroc_anomaly_results.json   — all AUROC / FPR@TPR numbers
* prints the headline false-alarm-reduction sentence
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'archive'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_paper_figures import sld_em_estimation, regularized_bbse
from label_shift_estimators import (fit_bcts_calibrator, calibrate,
                                    build_soft_confusion_pinv, bbse_soft_estimate)


# ── BBSE machinery (mirrors archive/plot_paper_figures.py) ──────────────────────

def load_weeks(inference_dir):
    files = sorted(Path(inference_dir).glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    out = []
    for f in files:
        wn = int(re.search(r'(\d+)$', f.stem).group(1))
        d = np.load(f)
        out.append((wn, d['true_labels'], d['pred_labels'], d['softmax']))
    return out


def build_bbse(ref_true, ref_pred, ref_softmax, num_classes):
    cm = confusion_matrix(ref_true, ref_pred, labels=list(range(num_classes))).astype(float)
    row = cm.sum(1, keepdims=True); row[row == 0] = 1
    C_T = (cm / row).T
    C_T_pinv = np.linalg.pinv(C_T, rcond=1e-2)
    counts = np.bincount(ref_true, minlength=num_classes).astype(float)
    p_train = counts / counts.sum()
    h_ref = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = ref_true == c
        if m.sum() > 0:
            p = np.clip(ref_softmax[m], 1e-12, 1.0)
            h_ref[c] = float(-np.sum(p * np.log(p), axis=1).mean())
    return C_T_pinv, C_T, p_train, h_ref


def bbse_estimate(pred, num_classes, C_T_pinv):
    q = np.bincount(pred, minlength=num_classes).astype(float); q /= q.sum()
    p = np.clip(C_T_pinv @ q, 0, None)
    if p.sum() > 0:
        p /= p.sum()
    return p


def mean_entropy(softmax):
    p = np.clip(softmax, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p), axis=1).mean())


def gap_uncorrected(softmax, h_ref, p_train, valid):
    return mean_entropy(softmax) - float(np.dot(p_train[valid], h_ref[valid]))


def gap_bbse(softmax, pred, num_classes, C_T_pinv, h_ref, valid):
    w = bbse_estimate(pred, num_classes, C_T_pinv)
    return mean_entropy(softmax) - float(np.dot(w[valid], h_ref[valid]))


def gap_from_w(softmax, w, h_ref, valid):
    """Entropy residual with an arbitrary estimated label marginal w."""
    return mean_entropy(softmax) - float(np.dot(w[valid], h_ref[valid]))


def mfwdd_severity(softmax, ref_sorted, weights, channels, m, rng):
    """
    MFWDD-style global drift severity: feature-importance-weighted mean of
    per-channel 1-Wasserstein distances between the window's softmax channels
    and the reference week's. For equal sample sizes, W1 = mean|sort(a)-sort(b)|,
    so we vectorise across channels. weights sum to 1 (MFWDD's Σ w_i = 1).
    """
    rows = rng.choice(softmax.shape[0], size=m, replace=True)
    win = np.sort(softmax[np.ix_(rows, channels)], axis=0)   # (m, nch)
    w1 = np.abs(win - ref_sorted).mean(axis=0)               # (nch,)
    return float(np.dot(weights, w1))


# ── window synthesis ────────────────────────────────────────────────────────────

def clean_window(n_total, n_win, rng):
    return rng.choice(n_total, size=n_win, replace=True)


def viral_window(true, n_win, frac, cls, rng):
    """One app `cls` surges to fraction `frac`; remainder drawn at the natural mix."""
    base = np.where(true == cls)[0]
    n_c = int(round(frac * n_win))
    spike = rng.choice(base, size=n_c, replace=True)
    rest = rng.choice(len(true), size=n_win - n_c, replace=True)
    return np.concatenate([spike, rest])


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--reference_week', type=int, default=0)
    ap.add_argument('--f1_threshold', type=float, default=0.60)
    ap.add_argument('--n_win', type=int, default=8000)
    ap.add_argument('--clean_seeds', type=int, default=6)
    ap.add_argument('--viral_windows', type=int, default=12,
                    help='viral-spike windows per week (random app & fraction each)')
    ap.add_argument('--frac_lo', type=float, default=0.30)
    ap.add_argument('--frac_hi', type=float, default=0.85)
    ap.add_argument('--min_class_n', type=int, default=20,
                    help='min flows for an app to be a viral-spike candidate in a week')
    ap.add_argument('--mfwdd_channels', type=int, default=120)
    ap.add_argument('--mfwdd_m', type=int, default=3000)
    ap.add_argument('--tpr_targets', type=float, nargs='+', default=[0.90, 0.95])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='figs')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.clean_seeds, args.viral_windows = 2, 4

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print('Loading weeks ...')
    weeks = load_weeks(args.inference_dir)
    ref = next((w for w in weeks if w[0] == args.reference_week), weeks[0])
    _, ref_true, ref_pred, ref_soft = ref
    K = ref_soft.shape[1]
    C_T_pinv, C_T, p_train, h_ref = build_bbse(ref_true, ref_pred, ref_soft, K)
    valid = ~np.isnan(h_ref)

    # estimator variants: RLLS / SLD-EM share the same residual, only w changes
    print('Fitting BCTS calibrator + soft confusion on the reference week ...')
    bcts = fit_bcts_calibrator(ref_soft, ref_true, K, seed=args.seed)
    Cs_T_pinv = build_soft_confusion_pinv(ref_soft, ref_true, K)

    # MFWDD reference: top-k classes by train freq, Σw=1, pre-sorted channels
    chan = np.argsort(-p_train)[:args.mfwdd_channels]
    chan = chan[p_train[chan] > 0]
    w_mfwdd = p_train[chan] / p_train[chan].sum()
    rsub = rng.choice(len(ref_soft), size=min(args.mfwdd_m, len(ref_soft)), replace=False)
    ref_sorted = np.sort(ref_soft[np.ix_(rsub, chan)], axis=0)
    m_mfwdd = ref_sorted.shape[0]

    test_weeks = [w for w in weeks if w[0] != args.reference_week]

    # binary ground truth from each week's full-data clean Macro F1
    week_f1 = {wn: f1_score(t, p, labels=list(range(K)), average='macro', zero_division=0)
               for wn, t, p, _ in test_weeks}
    is_anom = {wn: int(week_f1[wn] < args.f1_threshold) for wn in week_f1}
    n_pos = sum(is_anom.values()); n_neg = len(is_anom) - n_pos
    print(f'  weeks: {len(test_weeks)}  degraded(+): {n_pos}  healthy(-): {n_neg}'
          f'  (anomaly iff clean Macro F1 < {args.f1_threshold})')

    records = []

    def score(idx, true, pred, soft, wn, regime, win_f1=np.nan):
        p, sm = pred[idx], soft[idx]
        rec = dict(
            week=int(wn), regime=regime, label=is_anom[wn], win_f1=float(win_f1),
            uncorrected=gap_uncorrected(sm, h_ref, p_train, valid),
            bbse=gap_bbse(sm, p, K, C_T_pinv, h_ref, valid),
            mfwdd=mfwdd_severity(sm, ref_sorted, w_mfwdd, chan, m_mfwdd, rng),
        )
        # estimator variants (deterministic — placed after the rng-consuming
        # mfwdd call so the original detectors' numbers reproduce exactly)
        q = np.bincount(p, minlength=K).astype(float); q /= q.sum()
        rec['rlls'] = gap_from_w(sm, regularized_bbse(C_T, q, ftol=1e-8, maxiter=300),
                                 h_ref, valid)
        rec['em'] = gap_from_w(sm, sld_em_estimation(p_train, sm), h_ref, valid)
        rec['em_bcts'] = gap_from_w(sm, sld_em_estimation(p_train, calibrate(sm, bcts)),
                                    h_ref, valid)
        rec['bbse_soft'] = gap_from_w(sm, bbse_soft_estimate(sm, Cs_T_pinv),
                                      h_ref, valid)
        records.append(rec)

    print('Building CLEAN pool ...')
    for wn, true, pred, soft in test_weeks:
        for _ in range(args.clean_seeds):
            score(clean_window(len(true), args.n_win, rng),
                  true, pred, soft, wn, 'clean')

    print('Building VIRAL pool (benign single-app volume spikes) ...')
    spike_f1_healthy = []
    for wn, true, pred, soft in test_weeks:
        cand = [c for c in range(K) if (true == c).sum() >= args.min_class_n]
        if not cand:
            continue
        for _ in range(args.viral_windows):
            cls = int(rng.choice(cand))
            frac = float(rng.uniform(args.frac_lo, args.frac_hi))
            idx = viral_window(true, args.n_win, frac, cls, rng)
            # benign check: the spiked app's own per-class F1 in this window
            wf1 = f1_score((true[idx] == cls).astype(int),
                           (pred[idx] == cls).astype(int), zero_division=0)
            if is_anom[wn] == 0:
                spike_f1_healthy.append(wf1)
            score(idx, true, pred, soft, wn, 'viral', wf1)

    print(f'  total windows: {len(records)}')
    if spike_f1_healthy:
        print(f'  benign check — spiked-app F1 on HEALTHY viral windows: '
              f'mean={np.mean(spike_f1_healthy):.3f} (stays high ⇒ no real degradation)')

    # ── evaluation ───────────────────────────────────────────────────────────────
    detectors = [
        ('uncorrected', 'Uncorrected entropy gap',       '#e08020'),
        ('mfwdd',       'MFWDD-style global drift',       '#7b3fa0'),
        ('bbse',        'BBSE-corrected residual (ours)', '#d7191c'),
        ('bbse_soft',   'BBSE-soft-corrected residual',   '#74add1'),
        ('rlls',        'RLLS-corrected residual',        '#1a9641'),
        ('em',          'SLD-EM-corrected residual',      '#9b59b6'),
        ('em_bcts',     'MLLS+BCTS-corrected residual',   '#c0392b'),
    ]

    def pool(regime):
        r = [x for x in records if x['regime'] == regime]
        return r, np.array([x['label'] for x in r])

    def fpr_at_tpr(y, sc, t):
        fpr, tpr, _ = roc_curve(y, sc)
        ok = tpr >= t
        return float(fpr[ok][0]) if ok.any() else 1.0

    results = {'config': vars(args), 'week_clean_f1': week_f1, 'is_anom': is_anom,
               'n_pos': n_pos, 'n_neg': n_neg,
               'benign_spiked_app_f1_healthy': float(np.mean(spike_f1_healthy)) if spike_f1_healthy else None,
               'regimes': {}}

    print('\n=== AUROC ===')
    for regime in ['clean', 'viral']:
        r, y = pool(regime)
        results['regimes'][regime] = {'n': len(r), 'detectors': {}}
        print(f'\n[{regime}]  n={len(r)}  pos={int(y.sum())} neg={int((1 - y).sum())}')
        for key, label, _ in detectors:
            sc = np.array([x[key] for x in r])
            auc = roc_auc_score(y, sc)
            d = {'auroc': float(auc),
                 'fpr_at_tpr': {f'{t:.2f}': fpr_at_tpr(y, sc, t) for t in args.tpr_targets}}
            results['regimes'][regime]['detectors'][key] = d
            fs = '  '.join(f'FPR@{t:.0%}={d["fpr_at_tpr"][f"{t:.2f}"]:.3f}' for t in args.tpr_targets)
            print(f'  {label:34s} AUROC={auc:.3f}   {fs}')

    st = results['regimes']['viral']['detectors']
    t_head = max(args.tpr_targets)
    head = (
        f'Under benign viral-spike volatility, at matched {t_head:.0%} detection the '
        f'BBSE-corrected residual cuts the false-alarm rate to '
        f'{st["bbse"]["fpr_at_tpr"][f"{t_head:.2f}"]:.1%}, vs '
        f'{st["uncorrected"]["fpr_at_tpr"][f"{t_head:.2f}"]:.1%} (uncorrected entropy) and '
        f'{st["mfwdd"]["fpr_at_tpr"][f"{t_head:.2f}"]:.1%} (MFWDD-style global); '
        f'AUROC {st["bbse"]["auroc"]:.3f} vs {st["uncorrected"]["auroc"]:.3f} / '
        f'{st["mfwdd"]["auroc"]:.3f}.')
    results['headline'] = head
    print('\n=== HEADLINE ===\n' + head)

    with open(out / 'auroc_anomaly_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out / 'auroc_anomaly_results.json'}")

    # ── Fig 1: ROC curves, clean vs viral ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, regime, title in zip(
            axes, ['clean', 'viral'],
            ['(a) Clean — no volume volatility',
             '(b) Viral spikes — benign single-app volume surges']):
        r, y = pool(regime)
        ax.plot([0, 1], [0, 1], color='#bbbbbb', ls='--', lw=1)
        for key, label, color in detectors:
            sc = np.array([x[key] for x in r])
            fpr, tpr, _ = roc_curve(y, sc)
            auc = roc_auc_score(y, sc)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{label}  (AUROC={auc:.3f})')
            ok = tpr >= t_head
            if ok.any():
                ax.scatter([fpr[ok][0]], [tpr[ok][0]], color=color, s=44, zorder=5,
                           edgecolors='white', linewidths=0.8)
        ax.axhline(t_head, color='#888', lw=0.8, ls=':')
        ax.set_xlabel('False Alarm Rate (FPR)', fontsize=12)
        ax.set_ylabel('Detection Rate (TPR)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Binary Anomaly Detection: genuine covariate degradation vs '
                 'benign volume volatility\n'
                 f'(ref Week {args.reference_week}; dotted line = {t_head:.0%} matched detection; '
                 'markers = operating point)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / 'fig_auroc_anomaly.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out / 'fig_auroc_anomaly.png'}")

    # ── Fig 2: score distributions explaining the false alarm ────────────────────
    groups = [('Healthy\nclean', 'clean', 0, '#2c7bb6'),
              ('Healthy\nviral spike', 'viral', 0, '#e08020'),
              ('Degraded\n(genuine)', 'viral', 1, '#d7191c')]
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5),
                               gridspec_kw={'wspace': 0.25})
    for ax, key, label in zip(axes2, ['uncorrected', 'bbse'],
                              ['Uncorrected entropy gap', 'BBSE-corrected residual (ours)']):
        data, colors, ticks = [], [], []
        for i, (g, regime, lab, color) in enumerate(groups):
            vals = [x[key] for x in records if x['regime'] == regime and x['label'] == lab]
            data.append(vals); colors.append(color); ticks.append(g)
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        for i, vals in enumerate(data):
            jit = np.random.default_rng(i).normal(0, 0.05, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jit, vals, s=6, color=colors[i],
                       alpha=0.35, zorder=3)
        # threshold that catches 95% of genuine degradation for THIS detector
        r, y = pool('viral')
        sc = np.array([x[key] for x in r])
        fpr, tpr, thr = roc_curve(y, sc)
        ok = tpr >= t_head
        thr95 = thr[ok][0] if ok.any() else thr[-1]
        ax.axhline(thr95, color='#444', ls='--', lw=1.3,
                   label=f'{t_head:.0%}-detection threshold')
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(ticks, fontsize=10)
        ax.set_ylabel('Detector score', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig2.suptitle('Why the uncorrected gap false-alarms under benign viral spikes',
                  fontsize=13, fontweight='bold', y=0.99)
    fig2.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.97, wspace=0.22)
    fig2.savefig(out / 'fig_auroc_score_dist.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved → {out / 'fig_auroc_score_dist.png'}")


if __name__ == '__main__':
    main()
