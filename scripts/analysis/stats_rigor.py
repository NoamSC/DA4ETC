#!/usr/bin/env python
"""
Statistical rigor for the anomaly-monitor headline metrics
==========================================================

Adds two things the headline AUROC / FAR@95 (false-alarm rate at 95% TPR)
numbers were missing:

  1. BOOTSTRAP 95% CONFIDENCE INTERVALS on AUROC and FAR@95 for every
     detector, in both the clean and the decisive viral-spike regime.
  2. A MACRO-F1 ANOMALY-THRESHOLD SENSITIVITY SWEEP (cutoff 0.55 / 0.60 / 0.65)
     showing the headline numbers are not threshold-cherry-picked.

It is a *read-and-rescore* wrapper: it imports the scoring machinery from
``auroc_anomaly_detection`` (window synthesis, BBSE residual, MFWDD severity,
estimator variants) and never edits that script.  The window scores it
bootstraps over are produced exactly the way the headline figure produces
them — but here we replay them across several master seeds so the bootstrap
can resample over the natural unit (week x seed clusters of windows).

Reference-week rule
-------------------
The headline anomaly metrics live on WEEK-16 (the clean / Modality-2 reference).
Default ``--inference_dir`` is the frozen-source Week-16 *vanilla* (no TTA)
per-week stream; ``--reference_week 16``.  Pass ``--forward`` to restrict the
test set to future-only weeks (>16), the strict matched-protocol evaluation;
without it the full 52-week stream is used (the monitor's deployed setting).
Both are reported and clearly labelled.  Use ``--inference_dir
results/inference_auditfix/week_1_vanilla_bs64 --reference_week 1`` ONLY to
reproduce the Week-1 sensor-event validation numbers (flagged as such).

Resampling unit
---------------
The headline detector is evaluated on synthetic *windows*.  In the viral
regime every window is one injection (random app x surge fraction) drawn on a
test week; in the clean regime every window is a natural-mix resample.  Windows
from the same (week, seed) are correlated, so the bootstrap resamples whole
(week, seed) CLUSTERS with replacement (a cluster / block bootstrap), not
individual windows -- this is the honest unit and gives wider, truthful CIs.
Multiple master seeds (default 1,2,3,4,42) are replayed so seed variance is
folded into the interval.

Outputs (versioned)
-------------------
  results/stats_rigor/stats_rigor_<tag>.json   point + CI + sweep + config
  figs/fig_threshold_sensitivity.png           AUROC & FAR@95 vs F1 cutoff
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'archive'))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

import auroc_anomaly_detection as A  # scoring machinery (imported, never edited)
from plot_paper_figures import sld_em_estimation, regularized_bbse
from label_shift_estimators import (fit_bcts_calibrator, calibrate,
                                    build_soft_confusion_pinv, bbse_soft_estimate)


HEADLINE_DETECTORS = [
    ('bbse',        'BBSE-corrected residual (ours)'),
    ('uncorrected', 'Uncorrected entropy gap'),
    ('mfwdd',       'MFWDD-style global drift'),
]
VARIANT_DETECTORS = [
    ('rlls',        'RLLS-corrected residual'),
    ('em',          'SLD-EM-corrected residual'),
    ('em_bcts',     'MLLS+BCTS-corrected residual'),
    ('bbse_soft',   'BBSE-soft-corrected residual'),
]
HEADLINE = ['bbse', 'uncorrected', 'mfwdd']  # ours / uncorrected / MFWDD


# ── window-score generation (replays the headline scoring per master seed) ──────

def build_window_records(weeks, ref_week, exclude_weeks, args, master_seed,
                         with_variants=False):
    """Reproduce auroc_anomaly_detection's per-window scoring for one seed.

    Returns a list of record dicts, each tagging (week, regime, seed) plus the
    raw detector scores.  Labels (anomaly y/n) are NOT baked in here -- they are
    assigned later per F1 threshold so the sweep can relabel without rescoring.
    """
    rng = np.random.default_rng(master_seed)
    ref = next((w for w in weeks if w[0] == ref_week), weeks[0])
    _, ref_true, ref_pred, ref_soft = ref
    K = ref_soft.shape[1]

    C_T_pinv, C_T, p_train, h_ref = A.build_bbse(ref_true, ref_pred, ref_soft, K)
    valid = ~np.isnan(h_ref)
    if with_variants:
        bcts = fit_bcts_calibrator(ref_soft, ref_true, K, seed=master_seed)
        Cs_T_pinv = build_soft_confusion_pinv(ref_soft, ref_true, K)

    chan = np.argsort(-p_train)[:args.mfwdd_channels]
    chan = chan[p_train[chan] > 0]
    w_mfwdd = p_train[chan] / p_train[chan].sum()
    rsub = rng.choice(len(ref_soft), size=min(args.mfwdd_m, len(ref_soft)), replace=False)
    ref_sorted = np.sort(ref_soft[np.ix_(rsub, chan)], axis=0)
    m_mfwdd = ref_sorted.shape[0]

    if args.forward:
        test_weeks = [w for w in weeks if w[0] > ref_week and w[0] not in exclude_weeks]
    else:
        test_weeks = [w for w in weeks
                      if w[0] != ref_week and w[0] not in exclude_weeks]

    records = []

    def score(idx, pred, soft, wn, regime):
        p, sm = pred[idx], soft[idx]
        rec = dict(
            week=int(wn), regime=regime, seed=int(master_seed),
            uncorrected=A.gap_uncorrected(sm, h_ref, p_train, valid),
            bbse=A.gap_bbse(sm, p, K, C_T_pinv, h_ref, valid),
            mfwdd=A.mfwdd_severity(sm, ref_sorted, w_mfwdd, chan, m_mfwdd, rng),
        )
        if with_variants:
            q = np.bincount(p, minlength=K).astype(float); q /= q.sum()
            rec['rlls'] = A.gap_from_w(sm, regularized_bbse(C_T, q, ftol=1e-8, maxiter=300),
                                       h_ref, valid)
            rec['em'] = A.gap_from_w(sm, sld_em_estimation(p_train, sm), h_ref, valid)
            rec['em_bcts'] = A.gap_from_w(sm, sld_em_estimation(p_train, calibrate(sm, bcts)),
                                          h_ref, valid)
            rec['bbse_soft'] = A.gap_from_w(sm, bbse_soft_estimate(sm, Cs_T_pinv),
                                            h_ref, valid)
        records.append(rec)

    for wn, true, pred, soft in test_weeks:
        for _ in range(args.clean_seeds):
            score(A.clean_window(len(true), args.n_win, rng), pred, soft, wn, 'clean')

    for wn, true, pred, soft in test_weeks:
        cand = [c for c in range(K) if (true == c).sum() >= args.min_class_n]
        if not cand:
            continue
        for _ in range(args.viral_windows):
            cls = int(rng.choice(cand))
            frac = float(rng.uniform(args.frac_lo, args.frac_hi))
            idx = A.viral_window(true, args.n_win, frac, cls, rng)
            score(idx, pred, soft, wn, 'viral')

    week_f1 = {wn: float(f1_score(t, p, labels=list(range(K)),
                                  average='macro', zero_division=0))
               for wn, t, p, _ in test_weeks}
    return records, week_f1


# ── metrics + cluster bootstrap ─────────────────────────────────────────────────

def far_at_tpr(y, sc, t):
    """False-alarm rate (FPR) at the first operating point with TPR >= t."""
    fpr, tpr, _ = roc_curve(y, sc)
    ok = tpr >= t
    return float(fpr[ok][0]) if ok.any() else 1.0


def point_metrics(recs, key, labels, tpr=0.95):
    y = labels
    sc = np.array([r[key] for r in recs])
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan'), float('nan')
    return float(roc_auc_score(y, sc)), far_at_tpr(y, sc, tpr)


def cluster_bootstrap(recs, key, labels, clusters, tpr=0.95,
                      n_boot=2000, seed=12345):
    """Percentile bootstrap resampling whole week-clusters (seeds nested).

    clusters : int array, same length as recs, cluster id per record.
    Returns dict with point, lo, hi for auroc and far.
    """
    rng = np.random.default_rng(seed)
    sc = np.array([r[key] for r in recs])
    y = labels
    uniq = np.unique(clusters)
    # precompute row indices per cluster
    rows_of = {c: np.where(clusters == c)[0] for c in uniq}
    aurocs, fars = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_of[c] for c in pick])
        yb, sb = y[idx], sc[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aurocs.append(roc_auc_score(yb, sb))
        fars.append(far_at_tpr(yb, sb, tpr))
    pa, pf = point_metrics(recs, key, labels, tpr)

    def ci(arr):
        if not arr:
            return (float('nan'), float('nan'))
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    alo, ahi = ci(aurocs)
    flo, fhi = ci(fars)
    return dict(auroc=pa, auroc_ci=[alo, ahi],
                far95=pf, far95_ci=[flo, fhi],
                n_boot_used=len(aurocs))


def label_for_threshold(recs, week_f1, f1_thr):
    return np.array([int(week_f1[r['week']] < f1_thr) for r in recs])


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir',
                    default='results/inference_auditfix/week_16_vanilla_bs64',
                    help='frozen-source vanilla per-week stream (default Week-16)')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--exclude_weeks', type=int, nargs='+', default=[],
                    help='weeks to drop from the test set entirely')
    ap.add_argument('--forward', action='store_true',
                    help='future-only eval: keep only weeks > reference_week '
                         '(strict matched protocol). Default uses the full stream.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 42])
    ap.add_argument('--f1_thresholds', type=float, nargs='+',
                    default=[0.55, 0.60, 0.65])
    ap.add_argument('--headline_f1', type=float, default=0.60,
                    help='F1 cutoff used for the CI table (default 0.60)')
    ap.add_argument('--regime', default='viral', choices=['viral', 'clean'],
                    help='regime for the CI table (viral = decisive false-alarm test)')
    ap.add_argument('--tpr', type=float, default=0.95)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--boot_seed', type=int, default=12345)
    # window-synthesis knobs mirror auroc_anomaly_detection defaults
    ap.add_argument('--n_win', type=int, default=8000)
    ap.add_argument('--clean_seeds', type=int, default=6)
    ap.add_argument('--viral_windows', type=int, default=12)
    ap.add_argument('--frac_lo', type=float, default=0.30)
    ap.add_argument('--frac_hi', type=float, default=0.85)
    ap.add_argument('--min_class_n', type=int, default=20)
    ap.add_argument('--mfwdd_channels', type=int, default=120)
    ap.add_argument('--mfwdd_m', type=int, default=3000)
    ap.add_argument('--variants', action='store_true',
                    help='also score the 4 estimator-variant residuals '
                         '(rlls/em/em_bcts/bbse_soft); slow (~10x). Off by default.')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out_json_dir', default='results/stats_rigor')
    ap.add_argument('--out_fig', default='figs/fig_threshold_sensitivity.png')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds = [1, 42]; args.clean_seeds, args.viral_windows = 2, 4
        args.n_boot = 300

    tag = args.tag or (f'ref{args.reference_week}'
                       f'{"_fwd" if args.forward else "_full"}')
    out_dir = Path(args.out_json_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)

    ref_flag = ('WEEK-16 (clean / Modality-2 reference)' if args.reference_week == 16
                else f'WEEK-{args.reference_week} '
                     '(NOTE: not the clean reference -- Week-1 only validates the '
                     'Week-10 sensor event)')
    print(f'Reference: {ref_flag}')
    print(f'Inference dir: {args.inference_dir}')
    print(f'Eval set: {"forward-only (>ref)" if args.forward else "full stream"}'
          f'   seeds={args.seeds}')

    weeks = A.load_weeks(args.inference_dir)
    if not weeks:
        sys.exit(f'No WEEK-2022-*.npz under {args.inference_dir}')

    # ── replay window scoring across master seeds ─────────────────────────────
    detector_list = HEADLINE_DETECTORS + (VARIANT_DETECTORS if args.variants else [])
    all_recs = []
    week_f1 = None
    for s in args.seeds:
        print(f'  scoring windows @ seed {s} ...')
        recs, wf1 = build_window_records(weeks, args.reference_week,
                                         args.exclude_weeks, args, s,
                                         with_variants=args.variants)
        all_recs.extend(recs)
        week_f1 = wf1  # identical across seeds (full-data per-week F1)
    print(f'  total windows: {len(all_recs)}  '
          f'(clean {sum(r["regime"]=="clean" for r in all_recs)}, '
          f'viral {sum(r["regime"]=="viral" for r in all_recs)})')

    # ── CI table on the chosen regime + headline F1 threshold ─────────────────
    regime_recs = [r for r in all_recs if r['regime'] == args.regime]
    y_head = label_for_threshold(regime_recs, week_f1, args.headline_f1)
    # cluster id = week ONLY: the anomaly label and per-week F1 are keyed on
    # week and are seed-invariant, so all seeds of a week are the SAME unit
    # (seeds fold in as nested windows). Clustering by (week,seed) would
    # pseudo-replicate and overstate precision. Resample whole week-clusters.
    cl_keys = sorted({r['week'] for r in regime_recs})
    cl_map = {k: i for i, k in enumerate(cl_keys)}
    clusters = np.array([cl_map[r['week']] for r in regime_recs])

    n_pos = int(y_head.sum()); n_neg = len(y_head) - n_pos
    print(f'\nCI table regime={args.regime}  F1<{args.headline_f1}: '
          f'pos={n_pos} neg={n_neg}  clusters={len(cl_keys)}')

    ci_table = {}
    for key, label in detector_list:
        ci_table[key] = cluster_bootstrap(
            regime_recs, key, y_head, clusters, tpr=args.tpr,
            n_boot=args.n_boot, seed=args.boot_seed)
        c = ci_table[key]
        print(f'  {label:34s} AUROC={c["auroc"]:.3f} '
              f'[{c["auroc_ci"][0]:.3f},{c["auroc_ci"][1]:.3f}]   '
              f'FAR@{args.tpr:.0%}={c["far95"]:.3f} '
              f'[{c["far95_ci"][0]:.3f},{c["far95_ci"][1]:.3f}]')

    # ── threshold sensitivity sweep ───────────────────────────────────────────
    print('\nThreshold sensitivity sweep (regime=%s):' % args.regime)
    sweep = {}
    for thr in args.f1_thresholds:
        y = label_for_threshold(regime_recs, week_f1, thr)
        np_, nn = int(y.sum()), len(y) - int(y.sum())
        row = {'n_pos': np_, 'n_neg': nn, 'detectors': {}}
        for key, _ in detector_list:
            au, fa = point_metrics(regime_recs, key, y, tpr=args.tpr)
            row['detectors'][key] = {'auroc': au, 'far95': fa}
        sweep[f'{thr:.2f}'] = row
        hd = row['detectors']['bbse']
        print(f'  F1<{thr:.2f}  (pos={np_} neg={nn}):  '
              f'ours AUROC={hd["auroc"]:.3f} FAR@{args.tpr:.0%}={hd["far95"]:.3f}  | '
              f'uncorr AUROC={row["detectors"]["uncorrected"]["auroc"]:.3f} '
              f'FAR={row["detectors"]["uncorrected"]["far95"]:.3f}  | '
              f'mfwdd AUROC={row["detectors"]["mfwdd"]["auroc"]:.3f} '
              f'FAR={row["detectors"]["mfwdd"]["far95"]:.3f}')

    # ── persist ───────────────────────────────────────────────────────────────
    result = dict(
        config=dict(vars(args), tag=tag),
        bootstrap=dict(unit='week cluster (seeds nested)',
                       n_resamples=args.n_boot, ci='95% percentile',
                       rng_seed=args.boot_seed, n_clusters=len(cl_keys),
                       seeds=args.seeds),
        reference_note=ref_flag,
        sample_caveat='frozen-source vanilla inference, data_sample_frac=0.1 '
                      '(do not cross-compare to full-data numbers)',
        week_clean_f1=week_f1,
        ci_table=ci_table,
        threshold_sweep=sweep,
        n_windows=len(all_recs),
        headline=_headline_sentence(ci_table, args))
    json_path = out_dir / f'stats_rigor_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved -> {json_path}')

    _plot_sensitivity(sweep, args, tag)
    print(f'Saved -> {args.out_fig}')
    print('\n=== HEADLINE ===\n' + result['headline'])


def _headline_sentence(ci_table, args):
    b, u, m = ci_table['bbse'], ci_table['uncorrected'], ci_table['mfwdd']
    return (
        f'[{("Week-%d" % args.reference_week)}, '
        f'{"forward-only" if args.forward else "full-stream"}, '
        f'{args.regime} regime, F1<{args.headline_f1}] '
        f'BBSE residual (ours): AUROC {b["auroc"]:.3f} '
        f'[{b["auroc_ci"][0]:.3f},{b["auroc_ci"][1]:.3f}], '
        f'FAR@{args.tpr:.0%} {b["far95"]:.3f} '
        f'[{b["far95_ci"][0]:.3f},{b["far95_ci"][1]:.3f}].  '
        f'Uncorrected: AUROC {u["auroc"]:.3f} '
        f'[{u["auroc_ci"][0]:.3f},{u["auroc_ci"][1]:.3f}], '
        f'FAR {u["far95"]:.3f} [{u["far95_ci"][0]:.3f},{u["far95_ci"][1]:.3f}].  '
        f'MFWDD: AUROC {m["auroc"]:.3f} '
        f'[{m["auroc_ci"][0]:.3f},{m["auroc_ci"][1]:.3f}], '
        f'FAR {m["far95"]:.3f} [{m["far95_ci"][0]:.3f},{m["far95_ci"][1]:.3f}].  '
        f'Bootstrap: {args.n_boot} resamples over week clusters (seeds nested), '
        f'percentile 95% CI, rng seed {args.boot_seed}.')


def _plot_sensitivity(sweep, args, tag):
    thrs = sorted(sweep.keys(), key=float)
    x = [float(t) for t in thrs]
    col = {'bbse': '#d7191c', 'uncorrected': '#e08020', 'mfwdd': '#7b3fa0'}
    lab = {'bbse': 'BBSE residual (ours)', 'uncorrected': 'Uncorrected gap',
           'mfwdd': 'MFWDD-style global'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key in HEADLINE:
        au = [sweep[t]['detectors'][key]['auroc'] for t in thrs]
        fa = [sweep[t]['detectors'][key]['far95'] for t in thrs]
        axes[0].plot(x, au, marker='o', lw=2.2, color=col[key], label=lab[key])
        axes[1].plot(x, fa, marker='o', lw=2.2, color=col[key], label=lab[key])
    axes[0].set_ylabel('AUROC'); axes[0].set_ylim(0.3, 1.02)
    axes[1].set_ylabel(f'FAR@{args.tpr:.0%} (false-alarm rate)')
    for ax in axes:
        ax.set_xlabel('Macro-F1 anomaly cutoff')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    npos = [sweep[t]['n_pos'] for t in thrs]
    nneg = [sweep[t]['n_neg'] for t in thrs]
    for ax in axes:
        for xi, p, n in zip(x, npos, nneg):
            ax.annotate(f'+{p}/-{n}', (xi, ax.get_ylim()[0]),
                        fontsize=7, ha='center', va='bottom', color='#666')
    fig.suptitle(
        f'Anomaly-threshold sensitivity -- Week-{args.reference_week} '
        f'{"forward-only" if args.forward else "full-stream"}, {args.regime} regime '
        f'(+pos/-neg windows annotated)',
        fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out_fig, dpi=190, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
