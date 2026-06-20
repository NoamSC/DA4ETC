#!/usr/bin/env python
"""
Natural benign-volume-surge SCAN (Step 1 only is trustworthy).

!! DEPRECATED MONITOR STEPS — read this first !!
-------------------------------------------------
Steps 2/3 of this script score each app's surge with the GLOBAL, population-wide
monitor gaps (gap_uncorrected / gap_bbse from auroc_anomaly_detection.py). Those
are aggregate over the WHOLE window and drift across weeks for reasons unrelated
to the surging app (other classes degrade over time), while a single app's surge
is <1% of the population. A global gap therefore CANNOT isolate one app's surge,
and the old claim "the BBSE residual ignores the benign surge while the
uncorrected gap reacts" was NOT supported by the data (both global gaps drift
together for population reasons). That framing was conceptually wrong.

The correct, per-class analysis lives in natural_falsealarm_perclass.py, which
uses the PER-CLASS BBSE-corrected entropy residual from
precision_isolation_ablation.py. Use that script for the monitor result.

Only STEP 1 (the surge SCAN that produces scan_candidates.json) is still used;
it just identifies benign-surge vs degrading class-weeks and makes no monitor
claim.

Original motivation (retained for the scan):
A benign surge = a single application's traffic VOLUME (its share of P(Y)) jumps
sharply between weeks while its true class-conditional behaviour P(X|Y) is intact
(per-class recall stays high, the app is not degrading).

Protocol
--------
* Source: frozen Week-16 CNN (clean-regime / Modality-2 reference week).
  Inference dir: results/inference/cesnet-tls-year22/cnn/train_week_16/
  These are the full-data (not 10%-sampled) frozen forward-eval outputs.
* Reference week for the monitor's BBSE machinery + entropy baseline = Week 16.
* Forward-only evaluation: we scan weeks 17..52 (future of the source).
  Week 10 is in the past of a Week-16 source and is excluded anyway; we never
  use it as a surge example (it is a documented sensor artifact, not benign).

Step 1 - SCAN.  For every class, find weeks where its per-week sample share jumps
>= --min_ratio x over its Week-16 baseline share while its per-class recall stays
>= --min_recall (no genuine degradation) and the absolute volume is non-trivial.
Classes that surge *while* their recall collapses (e.g. apple-location) are real
degradation, not benign surges, and are reported separately / excluded.

Step 2 - MONITOR.  For the cleanest natural surge we run the actual monitor on the
real per-week data (no synthetic injection): the BBSE-corrected residual
(gap_bbse) and the uncorrected residual (gap_uncorrected) from
auroc_anomaly_detection.py, week by week, over a window bracketing the surge. A
benign surge should leave the BBSE residual flat (tracking only the slow true
covariate trend) while the uncorrected residual jumps at the surge week.

Outputs (versioned)
-------------------
* results/analysis/natural_falsealarm/scan_candidates.json   - all surge candidates
* results/analysis/natural_falsealarm/monitor_trace.json      - per-week monitor scores
* figs/natural_falsealarm/fig_natural_surge_<app>.png         - surge vs monitor plot
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
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

# Reuse the *exact* monitor functions used in the paper's AUROC experiment.
from auroc_anomaly_detection import (build_bbse, gap_uncorrected, gap_bbse,
                                     mean_entropy)


def load_weeks(inference_dir):
    files = sorted(Path(inference_dir).glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    out = {}
    for f in files:
        wn = int(re.search(r'(\d+)$', f.stem).group(1))
        d = np.load(f)
        out[wn] = dict(true=d['true_labels'], pred=d['pred_labels'],
                       soft=d['softmax'])
    return out


def per_class_recall(true, pred, c):
    m = true == c
    return float((pred[m] == c).mean()) if m.sum() > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir',
                    default='results/inference/cesnet-tls-year22/cnn/train_week_16')
    ap.add_argument('--dataset_root',
                    default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--scan_lo', type=int, default=17)   # forward-only
    ap.add_argument('--scan_hi', type=int, default=52)
    ap.add_argument('--min_ratio', type=float, default=3.0,
                    help='min surge: week share / baseline (week-16) share')
    ap.add_argument('--min_recall', type=float, default=0.80,
                    help='per-class recall must stay >= this (no degradation)')
    ap.add_argument('--min_abs_share', type=float, default=0.004,
                    help='min absolute week share so the surge is operationally real')
    ap.add_argument('--min_class_n', type=int, default=400,
                    help='min flows of the class in the surge week')
    ap.add_argument('--inject_nwin', type=int, default=20000,
                    help='window size for the controlled real-flow injection (STEP 3)')
    ap.add_argument('--inject_seeds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--results_dir', default='results/analysis/natural_falsealarm')
    ap.add_argument('--fig_dir', default='figs/natural_falsealarm')
    ap.add_argument('--legacy', action='store_true', default=False,
                    help='OPT-IN: also run the DEPRECATED Step-2 (global monitor '
                         'trace) and Step-3 (real-flow injection) analyses. These '
                         'are conceptually wrong (population-wide global gaps '
                         'cannot isolate one app\'s surge) and are cited NOWHERE '
                         'in the paper; the corrected per-class result lives in '
                         'natural_falsealarm_perclass.py. Default: Step-1 scan only.')
    args = ap.parse_args()

    res_dir = Path(args.results_dir); res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    # label names
    lm = json.load(open(Path(args.dataset_root) / 'label_mapping.json'))
    inv = {v: k for k, v in lm.items()}

    print('Loading frozen Week-16-source inference for all weeks ...')
    weeks = load_weeks(args.inference_dir)
    ref = weeks[args.reference_week]
    K = ref['soft'].shape[1]
    print(f'  {len(weeks)} weeks, {K} classes, ref=Week {args.reference_week} '
          f'(n={len(ref["true"])})')

    # ---- per-week per-class shares + recalls + macro-F1 ----
    base_share = np.bincount(ref['true'], minlength=K).astype(float)
    base_share /= base_share.sum()

    scan_weeks = [w for w in range(args.scan_lo, args.scan_hi + 1) if w in weeks]
    share = {w: np.bincount(weeks[w]['true'], minlength=K).astype(float)
                / len(weeks[w]['true']) for w in weeks}
    macro_f1 = {w: float(f1_score(weeks[w]['true'], weeks[w]['pred'],
                labels=list(range(K)), average='macro', zero_division=0))
                for w in weeks}

    # ---- STEP 1: scan for benign single-app volume surges ----
    benign, degrading = [], []
    for c in range(K):
        b = base_share[c]
        if b < 1e-5:
            continue
        for w in scan_weeks:
            s = share[w][c]
            ratio = s / b
            n_c = int((weeks[w]['true'] == c).sum())
            if ratio < args.min_ratio or s < args.min_abs_share or n_c < args.min_class_n:
                continue
            rec = per_class_recall(weeks[w]['true'], weeks[w]['pred'], c)
            entry = dict(cls=c, app=inv.get(c, str(c)), week=w,
                         base_share=float(b), week_share=float(s),
                         ratio=float(ratio), recall=float(rec), n_class=n_c,
                         week_macro_f1=macro_f1[w])
            (benign if rec >= args.min_recall else degrading).append(entry)

    benign.sort(key=lambda e: -e['ratio'])
    degrading.sort(key=lambda e: -e['ratio'])

    print(f'\n=== STEP 1: surge scan (forward weeks {args.scan_lo}-{args.scan_hi}) ===')
    print(f'  thresholds: ratio>={args.min_ratio}x  recall>={args.min_recall}  '
          f'abs_share>={args.min_abs_share}  n>={args.min_class_n}')
    print(f'  BENIGN surges (recall stays high): {len(benign)}')
    for e in benign[:15]:
        print(f"    {e['app']:20s} (c{e['cls']}) w{e['week']:2d}  "
              f"x{e['ratio']:.1f}  share {e['base_share']:.4f}->{e['week_share']:.4f}  "
              f"recall={e['recall']:.3f}  n={e['n_class']}")
    print(f'  DEGRADING surges (recall collapses -> excluded): {len(degrading)}')
    for e in degrading[:8]:
        print(f"    {e['app']:20s} (c{e['cls']}) w{e['week']:2d}  "
              f"x{e['ratio']:.1f}  recall={e['recall']:.3f}  (EXCLUDED)")

    json.dump(dict(config=vars(args), benign=benign, degrading=degrading,
                   base_share={int(c): float(base_share[c]) for c in range(K)},
                   macro_f1=macro_f1),
              open(res_dir / 'scan_candidates.json', 'w'), indent=2)
    print(f"  saved -> {res_dir / 'scan_candidates.json'}")

    if not benign:
        print('\n!! NO clean natural benign surge found at these thresholds. '
              'Report synthetic-only limitation.')
        return

    if not args.legacy:
        print('\n[Step-1 scan complete] Skipping DEPRECATED Step-2/Step-3 monitor '
              'analyses (pass --legacy to run them). The corrected per-class result '
              'lives in natural_falsealarm_perclass.py.')
        return

    # ---- STEP 2: run the monitor on the cleanest benign surge ----
    # pick the surge with the highest recall among the top-ratio ones (most
    # unambiguously benign); break ties toward a locally-flat macro-F1 plateau.
    top = sorted(benign, key=lambda e: (-e['recall'], -e['ratio']))[0]
    cls, surge_w = top['cls'], top['week']
    app = top['app']
    print(f"\n=== STEP 2: monitor on benign surge  app='{app}' (c{cls}) "
          f"surge week {surge_w} ===")

    # build the monitor's BBSE machinery from the reference week
    C_T_pinv, C_T, p_train, h_ref = build_bbse(ref['true'], ref['pred'],
                                               ref['soft'], K)
    valid = ~np.isnan(h_ref)

    # run the REAL per-week data through the monitor across a bracket window
    lo = max(args.reference_week, surge_w - 6)
    hi = min(args.scan_hi, surge_w + 6)
    trace = []
    for w in range(lo, hi + 1):
        if w not in weeks:
            continue
        wk = weeks[w]
        unc = gap_uncorrected(wk['soft'], h_ref, p_train, valid)
        bb = gap_bbse(wk['soft'], wk['pred'], K, C_T_pinv, h_ref, valid)
        trace.append(dict(week=w,
                          app_share=float(share[w][cls]),
                          app_recall=per_class_recall(wk['true'], wk['pred'], cls),
                          macro_f1=macro_f1[w],
                          uncorrected=float(unc), bbse=float(bb),
                          mean_entropy=mean_entropy(wk['soft'])))
    json.dump(dict(app=app, cls=int(cls), surge_week=int(surge_w),
                   reference_week=args.reference_week, trace=trace),
              open(res_dir / 'monitor_trace.json', 'w'), indent=2)

    # quantify the false-alarm behaviour at the surge step
    tw = {t['week']: t for t in trace}
    prev_w = surge_w - 1 if (surge_w - 1) in tw else min(tw)
    d_share = tw[surge_w]['app_share'] / tw[prev_w]['app_share']
    d_unc = tw[surge_w]['uncorrected'] - tw[prev_w]['uncorrected']
    d_bbse = tw[surge_w]['bbse'] - tw[prev_w]['bbse']
    d_f1 = tw[surge_w]['macro_f1'] - tw[prev_w]['macro_f1']
    print(f"  surge step w{prev_w}->w{surge_w}: volume x{d_share:.2f}  "
          f"(true macro-F1 change {d_f1:+.3f})")
    print(f"    uncorrected residual change: {d_unc:+.4f}")
    print(f"    BBSE-corrected residual change: {d_bbse:+.4f}  (ours)")
    # variability across the whole bracket (how 'flat' is each detector?)
    unc_arr = np.array([t['uncorrected'] for t in trace])
    bb_arr = np.array([t['bbse'] for t in trace])
    print(f"  across bracket w{lo}-{hi}: std(uncorrected)={unc_arr.std():.4f}  "
          f"std(bbse)={bb_arr.std():.4f}  (lower = fewer spurious moves)")

    # ---- STEP 3: controlled REAL-FLOW surge injection ----------------------------
    # A single app's *natural* peak share here is tiny (<1% of a week), so on the
    # raw whole-week trace the surge perturbs the monitor only mildly. To expose
    # the false-alarm differential while staying real (no synthetic softmax), we
    # take the surging app's OWN real, non-degraded flows from the reference week
    # and resample windows where its share is swept from its baseline up to and
    # beyond its observed natural peak. Because these are the app's genuine flows
    # at its genuine (high) recall, any monitor rise is a pure false alarm.
    # We pick, among the benign candidates, the one that is the strongest entropy
    # nuisance (largest |H_class - H_ref_baseline|): the app most able to fool an
    # uncorrected entropy detector.
    overall_H = mean_entropy(ref['soft'])
    benign_classes = sorted({e['cls'] for e in benign})
    def class_H(c):
        m = ref['true'] == c
        return mean_entropy(ref['soft'][m]) if m.sum() > 0 else float('nan')
    nuis = sorted(benign_classes, key=lambda c: -abs(class_H(c) - overall_H))
    inj_cls = nuis[0]
    inj_app = inv.get(inj_cls, str(inj_cls))
    inj_peak = max(e['week_share'] for e in benign if e['cls'] == inj_cls)
    inj_base = base_share[inj_cls]
    print(f"\n=== STEP 3: controlled real-flow injection  app='{inj_app}' (c{inj_cls}) ===")
    print(f"  class entropy {class_H(inj_cls):.4f} vs week mean {overall_H:.4f}; "
          f"natural share {inj_base:.4f} -> peak {inj_peak:.4f}")
    rng = np.random.default_rng(args.seed)
    idx_c = np.where(ref['true'] == inj_cls)[0]
    idx_all = np.arange(len(ref['true']))
    n_win = args.inject_nwin
    # sweep: baseline, natural peak, and amplified shares (label-shift severity)
    fracs = sorted(set([float(inj_base), float(inj_peak), 0.05, 0.10, 0.20, 0.40]))
    inj = []
    for frac in fracs:
        u_s, b_s = [], []
        for s in range(args.inject_seeds):
            r = np.random.default_rng(1000 * s + 7)
            nc = int(round(frac * n_win))
            spike = r.choice(idx_c, size=nc, replace=True)
            rest = r.choice(idx_all, size=n_win - nc, replace=True)
            ix = np.concatenate([spike, rest])
            sm, pr = ref['soft'][ix], ref['pred'][ix]
            u_s.append(gap_uncorrected(sm, h_ref, p_train, valid))
            b_s.append(gap_bbse(sm, pr, K, C_T_pinv, h_ref, valid))
        inj.append(dict(frac=float(frac),
                        uncorrected_mean=float(np.mean(u_s)), uncorrected_std=float(np.std(u_s)),
                        bbse_mean=float(np.mean(b_s)), bbse_std=float(np.std(b_s))))
    # false-alarm magnitude = drift of the residual away from its baseline-share value
    u0 = inj[0]['uncorrected_mean']; b0 = inj[0]['bbse_mean']
    for r in inj:
        r['uncorrected_falsealarm'] = float(r['uncorrected_mean'] - u0)
        r['bbse_falsealarm'] = float(r['bbse_mean'] - b0)
    print("  frac     uncorrected(FA)      bbse(FA)         [residual drift from baseline share]")
    for r in inj:
        print(f"   {r['frac']:.3f}   {r['uncorrected_mean']:+.4f} ({r['uncorrected_falsealarm']:+.4f})   "
              f"{r['bbse_mean']:+.4f} ({r['bbse_falsealarm']:+.4f})")
    # headline false-alarm reduction at the largest swept share
    big = inj[-1]
    if abs(big['uncorrected_falsealarm']) > 1e-6:
        fa_red = 1.0 - abs(big['bbse_falsealarm']) / abs(big['uncorrected_falsealarm'])
        print(f"  at {big['frac']:.0%} share: |FA| uncorrected={abs(big['uncorrected_falsealarm']):.4f}  "
              f"bbse={abs(big['bbse_falsealarm']):.4f}  -> BBSE cuts the false-alarm drift by {fa_red:.0%}")
    json.dump(dict(app=inj_app, cls=int(inj_cls), n_win=n_win,
                   class_entropy=class_H(inj_cls), week_mean_entropy=overall_H,
                   natural_base_share=float(inj_base), natural_peak_share=float(inj_peak),
                   sweep=inj),
              open(res_dir / 'injection_sweep.json', 'w'), indent=2)
    print(f"  saved -> {res_dir / 'injection_sweep.json'}")

    # ---- figure: surge vs monitor ----
    ws = [t['week'] for t in trace]
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.5, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axC = fig.add_subplot(gs[:, 1])

    # (a) the benign surge: app volume share + its recall
    axA.bar(ws, [t['app_share'] * 100 for t in trace], color='#3b7dd8',
            alpha=0.8, label=f'{app} volume share (%)')
    axA.axvline(surge_w, color='#d7191c', ls='--', lw=1.4, alpha=0.7)
    axA.set_ylabel(f'{app} share (%)', color='#3b7dd8', fontsize=11)
    axA.tick_params(axis='y', labelcolor='#3b7dd8')
    axR = axA.twinx()
    axR.plot(ws, [t['app_recall'] for t in trace], 'o-', color='#1a9641',
             lw=2, label=f'{app} per-class recall')
    axR.axhline(args.min_recall, color='#1a9641', ls=':', lw=1, alpha=0.5)
    axR.set_ylabel('per-class recall', color='#1a9641', fontsize=11)
    axR.set_ylim(0, 1.02)
    axR.tick_params(axis='y', labelcolor='#1a9641')
    axA.set_title(f'(a) Benign volume surge: {app} volume jumps x{top["ratio"]:.1f} '
                  f'at week {surge_w}, recall stays high (no degradation)',
                  fontsize=11, fontweight='bold')
    axA.grid(True, axis='y', alpha=0.2)

    # (b) monitor scores
    axB.plot(ws, [t['uncorrected'] for t in trace], 's-', color='#e08020',
             lw=2.2, label='Uncorrected entropy gap')
    axB.plot(ws, [t['bbse'] for t in trace], 'o-', color='#d7191c',
             lw=2.4, label='BBSE-corrected residual (ours)')
    axB.axvline(surge_w, color='#d7191c', ls='--', lw=1.4, alpha=0.7,
                label=f'surge week {surge_w}')
    axBt = axB.twinx()
    axBt.plot(ws, [t['macro_f1'] for t in trace], '--', color='#555',
              lw=1.4, alpha=0.7, label='true Macro-F1 (right axis)')
    axBt.set_ylabel('true Macro-F1', color='#555', fontsize=10)
    axBt.tick_params(axis='y', labelcolor='#555')
    axB.set_xlabel('week (Week-16 frozen source, forward eval)', fontsize=11)
    axB.set_ylabel('monitor residual', fontsize=11)
    axB.set_title('(b) GLOBAL monitor gaps over the bracket (DEPRECATED: these '
                  'are population-wide and cannot isolate one app\'s surge; '
                  'see natural_falsealarm_perclass.py)',
                  fontsize=10, fontweight='bold')
    axB.grid(True, alpha=0.2)
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBt.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left', framealpha=0.9)
    for ax in (axA, axB):
        ax.spines['top'].set_visible(False)

    # (c) controlled real-flow injection sweep: false-alarm drift vs surge share
    fr = np.array([r['frac'] for r in inj]) * 100
    axC.errorbar(fr, [r['uncorrected_falsealarm'] for r in inj],
                 yerr=[r['uncorrected_std'] for r in inj], marker='s', color='#e08020',
                 lw=2.2, capsize=3, label='Uncorrected entropy gap')
    axC.errorbar(fr, [r['bbse_falsealarm'] for r in inj],
                 yerr=[r['bbse_std'] for r in inj], marker='o', color='#d7191c',
                 lw=2.4, capsize=3, label='BBSE-corrected residual (ours)')
    axC.axhline(0, color='#888', lw=1, ls=':')
    axC.axvline(inj_peak * 100, color='#3b7dd8', ls='--', lw=1.3, alpha=0.8,
                label=f'natural peak share ({inj_peak*100:.1f}%)')
    axC.set_xlabel(f'{inj_app} injected share of window (%)', fontsize=10.5)
    axC.set_ylabel('false-alarm drift of residual\n(vs baseline share)', fontsize=10.5)
    axC.set_title(f'(c) Controlled real-flow surge ({inj_app}):\nBBSE residual stays '
                  'flat as benign share grows', fontsize=10.5, fontweight='bold')
    axC.legend(fontsize=9, loc='upper left', framealpha=0.9)
    axC.grid(True, alpha=0.25)
    axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    fig.suptitle('Natural benign volume surge SCAN (DEPRECATED monitor panels) '
                 '-- use natural_falsealarm_perclass.py for the per-class result\n'
                 f'(frozen Week-16 source; ref=Week {args.reference_week}; '
                 'forward-only weeks)', fontsize=11.5, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = fig_dir / f'fig_natural_surge_{app}.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out_png}")
    print(f"  saved -> {res_dir / 'monitor_trace.json'}")


if __name__ == '__main__':
    main()
