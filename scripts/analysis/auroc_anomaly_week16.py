#!/usr/bin/env python
"""
Week-16 viral-spike binary anomaly-detection / AUROC experiment.

Thin wrapper around scripts/analysis/auroc_anomaly_detection.py that moves the
asymmetric single-app viral-spike false-alarm test onto the **Week-16 healthy
reference** (Modality-2 clean-regime reference week), with a strict
forward-only (future) evaluation set (weeks 17-52), and aggregates over
>=5 seeds (mean +/- std), so the abstract / Sec VI can report tracking rho,
FAR@95 and AUROC all on a SINGLE reference week.

Why a wrapper (not editing the original): other agents touch the analysis
scripts; this reuses every detector + injection routine verbatim and only:
  * points the loader at the vanilla frozen-source Week-16 inference outputs
    (results/inference/cesnet-tls-year22/cnn/train_week_16/),
  * restricts the test set to forward weeks 17-52 (no past/reference leakage),
  * loops the stochastic injection over seeds {1,2,3,4,42} and reports
    mean +/- std AUROC and FAR@95 (= FPR at 95% detection) per detector.

Ground truth, regimes, detectors and scoring are IDENTICAL to the Week-1 run;
only the reference week + seed aggregation differ, so the Week-16 table is
directly format-comparable to figs/week1_start/auroc_anomaly_results.json.

NOTE: inference uses the shared 10%-sample protocol (frac=0.1); never compare
these AUROC magnitudes to any full-data number.

Outputs (versioned, ref-16 suffix):
  figs/fig_auroc_anomaly_ref16.png            -- mean ROC curves (clean + viral)
  figs/auroc_anomaly_results_ref16.json       -- per-seed + aggregated numbers
  prints a markdown AUROC / FAR@95 table (mean +/- std over seeds)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'archive'))
sys.path.insert(0, str(HERE))

# Reuse every detector / injection routine verbatim from the canonical script.
from auroc_anomaly_detection import (
    load_weeks, build_bbse, bbse_estimate, mean_entropy,
    gap_uncorrected, gap_bbse, gap_from_w, mfwdd_severity,
    clean_window, viral_window,
)
from plot_paper_figures import sld_em_estimation, regularized_bbse
from label_shift_estimators import (fit_bcts_calibrator, calibrate,
                                    build_soft_confusion_pinv, bbse_soft_estimate)


DETECTORS = [
    ('uncorrected', 'Uncorrected entropy gap',       '#e08020'),
    ('mfwdd',       'MFWDD-style global drift',       '#7b3fa0'),
    ('bbse',        'BBSE-corrected residual',        '#d7191c'),
    ('bbse_soft',   'BBSE-soft-corrected residual',   '#74add1'),
    ('rlls',        'RLLS-corrected residual',        '#1a9641'),
    ('em',          'SLD-EM-corrected residual',      '#9b59b6'),
    ('em_bcts',     'MLLS+BCTS-corrected residual',   '#c0392b'),
]


def fpr_at_tpr(y, sc, t):
    fpr, tpr, _ = roc_curve(y, sc)
    ok = tpr >= t
    return float(fpr[ok][0]) if ok.any() else 1.0


def run_one_seed(seed, weeks, ref, test_weeks, is_anom, K, args,
                 C_T_pinv, C_T, p_train, h_ref, valid,
                 bcts, Cs_T_pinv, chan, w_mfwdd, ref_sorted, m_mfwdd):
    """Run the full clean+viral pool for a single seed; return per-regime
    detector AUROC + FAR@tpr and the (y, scores) arrays for ROC averaging."""
    rng = np.random.default_rng(seed)
    records = []

    def score(idx, true, pred, soft, wn, regime, win_f1=np.nan):
        p, sm = pred[idx], soft[idx]
        rec = dict(
            week=int(wn), regime=regime, label=is_anom[wn], win_f1=float(win_f1),
            uncorrected=gap_uncorrected(sm, h_ref, p_train, valid),
            bbse=gap_bbse(sm, p, K, C_T_pinv, h_ref, valid),
            mfwdd=mfwdd_severity(sm, ref_sorted, w_mfwdd, chan, m_mfwdd, rng),
        )
        q = np.bincount(p, minlength=K).astype(float); q /= q.sum()
        rec['rlls'] = gap_from_w(sm, regularized_bbse(C_T, q, ftol=1e-8, maxiter=300),
                                 h_ref, valid)
        rec['em'] = gap_from_w(sm, sld_em_estimation(p_train, sm), h_ref, valid)
        rec['em_bcts'] = gap_from_w(sm, sld_em_estimation(p_train, calibrate(sm, bcts)),
                                    h_ref, valid)
        rec['bbse_soft'] = gap_from_w(sm, bbse_soft_estimate(sm, Cs_T_pinv),
                                      h_ref, valid)
        records.append(rec)

    for wn, true, pred, soft in test_weeks:
        for _ in range(args.clean_seeds):
            score(clean_window(len(true), args.n_win, rng), true, pred, soft, wn, 'clean')

    spike_f1_healthy = []
    for wn, true, pred, soft in test_weeks:
        cand = [c for c in range(K) if (true == c).sum() >= args.min_class_n]
        if not cand:
            continue
        for _ in range(args.viral_windows):
            cls = int(rng.choice(cand))
            frac = float(rng.uniform(args.frac_lo, args.frac_hi))
            idx = viral_window(true, args.n_win, frac, cls, rng)
            wf1 = f1_score((true[idx] == cls).astype(int),
                           (pred[idx] == cls).astype(int), zero_division=0)
            if is_anom[wn] == 0:
                spike_f1_healthy.append(wf1)
            score(idx, true, pred, soft, wn, 'viral', wf1)

    out = {'spiked_app_f1_healthy': float(np.mean(spike_f1_healthy)) if spike_f1_healthy else None}
    rocs = {}
    for regime in ['clean', 'viral']:
        r = [x for x in records if x['regime'] == regime]
        y = np.array([x['label'] for x in r])
        out[regime] = {'n': len(r), 'pos': int(y.sum()), 'neg': int((1 - y).sum()),
                       'detectors': {}}
        rocs[regime] = {}
        for key, _, _ in DETECTORS:
            sc = np.array([x[key] for x in r])
            out[regime]['detectors'][key] = {
                'auroc': float(roc_auc_score(y, sc)),
                'fpr_at_tpr': {f'{t:.2f}': fpr_at_tpr(y, sc, t) for t in args.tpr_targets},
            }
            rocs[regime][key] = (y, sc)
    return out, rocs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir',
                    default='results/inference/cesnet-tls-year22/cnn/train_week_16')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--forward_only', action='store_true', default=True,
                    help='restrict test set to weeks strictly after the reference '
                         '(future-only frozen-forward eval); on by default')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 42])
    ap.add_argument('--f1_threshold', type=float, default=0.60)
    ap.add_argument('--n_win', type=int, default=8000)
    ap.add_argument('--clean_seeds', type=int, default=6)
    ap.add_argument('--viral_windows', type=int, default=12)
    ap.add_argument('--frac_lo', type=float, default=0.30)
    ap.add_argument('--frac_hi', type=float, default=0.85)
    ap.add_argument('--min_class_n', type=int, default=20)
    ap.add_argument('--mfwdd_channels', type=int, default=120)
    ap.add_argument('--mfwdd_m', type=int, default=3000)
    ap.add_argument('--tpr_targets', type=float, nargs='+', default=[0.90, 0.95])
    ap.add_argument('--output_dir', default='figs')
    ap.add_argument('--tag', default='ref16')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.clean_seeds, args.viral_windows, args.seeds = 2, 4, [1, 2]

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    print(f'Loading weeks from {args.inference_dir} ...')
    weeks = load_weeks(args.inference_dir)
    ref = next((w for w in weeks if w[0] == args.reference_week), weeks[0])
    _, ref_true, ref_pred, ref_soft = ref
    K = ref_soft.shape[1]
    C_T_pinv, C_T, p_train, h_ref = build_bbse(ref_true, ref_pred, ref_soft, K)
    valid = ~np.isnan(h_ref)

    # estimator-variant machinery (deterministic in the reference; fit once).
    print('Fitting BCTS calibrator + soft confusion on the Week-16 reference ...')
    bcts = fit_bcts_calibrator(ref_soft, ref_true, K, seed=42)
    Cs_T_pinv = build_soft_confusion_pinv(ref_soft, ref_true, K)

    # MFWDD reference channels (top-k by train freq, sum w = 1) — fix the rng so
    # the reference set is identical across seeds (only injection varies).
    ref_rng = np.random.default_rng(0)
    chan = np.argsort(-p_train)[:args.mfwdd_channels]
    chan = chan[p_train[chan] > 0]
    w_mfwdd = p_train[chan] / p_train[chan].sum()
    rsub = ref_rng.choice(len(ref_soft), size=min(args.mfwdd_m, len(ref_soft)), replace=False)
    ref_sorted = np.sort(ref_soft[np.ix_(rsub, chan)], axis=0)
    m_mfwdd = ref_sorted.shape[0]

    # forward-only frozen-forward eval: strictly future weeks (no leakage of the
    # reference or any past week into the monitored set).
    def keep(wn):
        if wn == args.reference_week:
            return False
        if args.forward_only and wn <= args.reference_week:
            return False
        return True
    test_weeks = [w for w in weeks if keep(w[0])]

    week_f1 = {wn: f1_score(t, p, labels=list(range(K)), average='macro', zero_division=0)
               for wn, t, p, _ in test_weeks}
    is_anom = {wn: int(week_f1[wn] < args.f1_threshold) for wn in week_f1}
    n_pos = sum(is_anom.values()); n_neg = len(is_anom) - n_pos
    print(f'  test weeks (forward {min(week_f1)}-{max(week_f1)}): {len(test_weeks)}  '
          f'degraded(+): {n_pos}  healthy(-): {n_neg}  '
          f'(anomaly iff clean Macro F1 < {args.f1_threshold})')

    # ── per-seed runs ────────────────────────────────────────────────────────
    per_seed = {}
    roc_store = {'clean': {k: [] for k, _, _ in DETECTORS},
                 'viral': {k: [] for k, _, _ in DETECTORS}}
    for s in args.seeds:
        print(f'Running seed {s} ...')
        res, rocs = run_one_seed(
            s, weeks, ref, test_weeks, is_anom, K, args,
            C_T_pinv, C_T, p_train, h_ref, valid,
            bcts, Cs_T_pinv, chan, w_mfwdd, ref_sorted, m_mfwdd)
        per_seed[s] = res
        for regime in ['clean', 'viral']:
            for k, _, _ in DETECTORS:
                roc_store[regime][k].append(rocs[regime][k])

    # ── aggregate (mean +/- std over seeds) ──────────────────────────────────
    def agg(regime, key, field, sub=None):
        vals = []
        for s in args.seeds:
            d = per_seed[s][regime]['detectors'][key]
            vals.append(d[field][sub] if sub else d[field])
        return float(np.mean(vals)), float(np.std(vals))

    t_head = max(args.tpr_targets)
    th = f'{t_head:.2f}'
    summary = {'config': vars(args),
               'reference_week': args.reference_week,
               'week_clean_f1': week_f1, 'is_anom': is_anom,
               'n_pos': n_pos, 'n_neg': n_neg,
               'seeds': args.seeds,
               'spiked_app_f1_healthy_mean':
                   float(np.mean([per_seed[s]['spiked_app_f1_healthy']
                                  for s in args.seeds
                                  if per_seed[s]['spiked_app_f1_healthy'] is not None]))
                   if any(per_seed[s]['spiked_app_f1_healthy'] is not None for s in args.seeds) else None,
               'regimes': {}, 'per_seed': per_seed}
    for regime in ['clean', 'viral']:
        summary['regimes'][regime] = {'detectors': {}}
        for k, _, _ in DETECTORS:
            am, as_ = agg(regime, k, 'auroc')
            f90m, f90s = agg(regime, k, 'fpr_at_tpr', '0.90')
            f95m, f95s = agg(regime, k, 'fpr_at_tpr', th if t_head == 0.95 else '0.95')
            summary['regimes'][regime]['detectors'][k] = {
                'auroc_mean': am, 'auroc_std': as_,
                'far90_mean': f90m, 'far90_std': f90s,
                'far95_mean': f95m, 'far95_std': f95s}

    json_path = out / f'auroc_anomaly_results_{args.tag}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved -> {json_path}')

    # ── markdown table ───────────────────────────────────────────────────────
    def fmt(m, s):
        return f'{m:.3f} ± {s:.3f}'
    md = []
    md.append(f'## Week-16 viral-spike AUROC / FAR@95 (forward weeks 17-52, '
              f'{len(args.seeds)} seeds {args.seeds})\n')
    md.append(f'Reference: WEEK-2022-16 (vanilla frozen source). '
              f'f1_threshold={args.f1_threshold} -> {n_pos} degraded(+) / {n_neg} healthy(-). '
              f'10%-sample protocol.\n')
    for regime in ['clean', 'viral']:
        md.append(f'\n### {regime.upper()} regime\n')
        md.append('| Detector | AUROC (mean±std) | FAR@90 (mean±std) | FAR@95 (mean±std) |')
        md.append('|---|---|---|---|')
        dets = summary['regimes'][regime]['detectors']
        for k, label, _ in DETECTORS:
            d = dets[k]
            md.append(f'| {label} | {fmt(d["auroc_mean"], d["auroc_std"])} '
                      f'| {fmt(d["far90_mean"], d["far90_std"])} '
                      f'| {fmt(d["far95_mean"], d["far95_std"])} |')
    md_text = '\n'.join(md)
    md_path = out / f'auroc_anomaly_table_{args.tag}.md'
    md_path.write_text(md_text + '\n')
    print('\n' + md_text)
    print(f'\nSaved -> {md_path}')

    # ── headline ─────────────────────────────────────────────────────────────
    v = summary['regimes']['viral']['detectors']
    head = (
        f'[Week-16] Under benign viral-spike volatility, at matched 95% detection '
        f'the BBSE-corrected residual cuts FAR@95 to '
        f'{v["bbse"]["far95_mean"]:.1%} (±{v["bbse"]["far95_std"]:.1%}), vs '
        f'{v["uncorrected"]["far95_mean"]:.1%} (uncorrected) and '
        f'{v["mfwdd"]["far95_mean"]:.1%} (MFWDD-style); AUROC '
        f'{v["bbse"]["auroc_mean"]:.3f} vs {v["uncorrected"]["auroc_mean"]:.3f} / '
        f'{v["mfwdd"]["auroc_mean"]:.3f}.')
    print('\n=== HEADLINE ===\n' + head)

    # ── mean ROC figure ──────────────────────────────────────────────────────
    grid = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, regime, title in zip(
            axes, ['clean', 'viral'],
            ['(a) Clean — no volume volatility',
             '(b) Viral spikes — benign single-app volume surges']):
        ax.plot([0, 1], [0, 1], color='#bbbbbb', ls='--', lw=1)
        for k, label, color in DETECTORS:
            tprs = []
            for (y, sc) in roc_store[regime][k]:
                fpr, tpr, _ = roc_curve(y, sc)
                tprs.append(np.interp(grid, fpr, tpr))
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[0] = 0.0
            am = summary['regimes'][regime]['detectors'][k]['auroc_mean']
            ax.plot(grid, mean_tpr, color=color, lw=2.2,
                    label=f'{label}  (AUROC={am:.3f})')
        ax.axhline(t_head, color='#888', lw=0.8, ls=':')
        ax.set_xlabel('False Alarm Rate (FPR)', fontsize=12)
        ax.set_ylabel('Detection Rate (TPR)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9.5, loc='lower right', framealpha=0.92)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Binary Anomaly Detection (Week-16 reference, forward 17-52): '
                 'genuine covariate degradation vs benign volume volatility\n'
                 f'(mean ROC over seeds {args.seeds}; dotted = {t_head:.0%} detection)',
                 fontsize=12.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = out / f'fig_auroc_anomaly_{args.tag}.png'
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {fig_path}')


if __name__ == '__main__':
    main()
