#!/usr/bin/env python
"""
Detection AUROC vs injected label-shift severity (Ours vs Naive vs MFWDD)
=========================================================================

Companion to the isolation contamination stage (isolation_estimator_variants
--contamination), which evaluates only severities {10, 100, 1000} and plots
one ROC panel per severity.  This script sweeps a DENSE severity grid and
condenses the result into a single line figure:

    x  injected label-shift severity s  (per-class prior weights drawn
       LogUniform[1/s, s], then normalized; s=1 -> uniform prior)
    y  pooled detection AUROC (positives = truly degraded class-weeks,
       negatives = healthy classes of clean weeks under the injected shift)

Detectors: Ours (BBSE-corrected entropy residual), Naive (uncorrected) and
MFWDD (feature-weighted embedding drift; disable with --no_mfwdd).

Protocol is identical to the contamination / robustness stages of
precision_isolation_ablation: positives and ground truth are reused
bit-identically from the existing ablation/variant caches where available;
the injection resamples rows (and, for MFWDD, embedding rows) grouped by
TRUE class, so P(X|Y) is exactly invariant (pure label shift).  Embedding
resampling uses a dedicated RNG stream so the ours/naive draws are
bit-identical with and without MFWDD.

Usage
  python scripts/analysis/labelshift_severity_sweep.py \
      --inference_dir results/inference/week_16_inference \
      --reference_week 16 --output_dir figs/isolation_w16

Outputs (under --output_dir)
  panels/fig_isolation_auroc_vs_labelshift.png   the sweep figure
  labelshift_sweep_cache.npz                     per-(sev,week,trial) scores
  labelshift_sweep_mfwdd_positives.npz           MFWDD per-(class,week) scores
                                                 (only if not in base cache)
  labelshift_sweep_results.json                  AUROCs + bootstrap CIs
"""

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precision_isolation_ablation import (
    load_week, entropy_rows, build_reference, estimate_label_dist,
    resample_label_shift, load_feature_importance, wasserstein1d_norm,
    DELTA_DEG, EPS_HEALTHY, N_MIN, N_MIN_EMB)
from isolation_estimator_variants import (corrected_residual, naive_residual,
                                          build_estimators, EST_NAMES)

SEVERITIES_DEFAULT = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
N_BOOT = 500
EMB_RNG_OFFSET = 7919   # embedding draws on their own stream -> ours/naive
                        # draws are bit-identical with and without MFWDD


print = partial(print, flush=True)


def load_ground_truth(out_dir):
    """Ground truth + Ours/Naive (+ MFWDD if present) per-(class,week) scores
    from the existing ablation cache (isolation_scores_cache.npz) or, failing
    that, the estimator-variants cache.  Returns dict or None."""
    base = out_dir / 'isolation_scores_cache.npz'
    var = out_dir / 'estimator_variants_cache.npz'
    if base.exists():
        z = np.load(base)
        keys = dict(R_corr='R_corr', R_naive='R_naive')
        src = base
    elif var.exists():
        z = np.load(var)
        if 'R_corr_base' not in z.files:
            return None
        keys = dict(R_corr='R_corr_base', R_naive='R_naive')
        src = var
    else:
        return None
    print(f"Ground truth / baseline scores from {src}")
    out = dict(week_nums=z['week_nums'],
               R_corr=z[keys['R_corr']], R_naive=z[keys['R_naive']],
               F1_true=z['F1_true'], Ntrue=z['Ntrue'], f1_ref=z['f1_ref'])
    if 'S_mfwdd' in z.files:
        out['S_mfwdd'] = z['S_mfwdd']
    return out


def mfwdd_scores(emb, pred_emb, ref, feat_imp, num_classes):
    """Per-class MFWDD score — identical to the MFWDD block of
    score_arrays() in precision_isolation_ablation."""
    s = np.full(num_classes, np.nan)
    for c in range(num_classes):
        if c not in ref['ref_region_emb']:
            continue
        tmask = pred_emb == c
        if tmask.sum() < N_MIN_EMB:
            continue
        test_e = emb[tmask]
        ref_e = ref['ref_region_emb'][c]
        wimp = feat_imp[c]
        top = np.argsort(-wimp)[:120]
        wsum = wimp[top].sum()
        acc = 0.0
        for i in top:
            acc += wimp[i] * wasserstein1d_norm(ref_e[:, i], test_e[:, i])
        s[c] = acc / wsum if wsum > 0 else acc
    return s


def pooled_auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--reference_week', type=int, default=1)
    ap.add_argument('--output_dir', default='figs/isolation')
    ap.add_argument('--checkpoint', default=None,
                    help='source-model checkpoint for MFWDD feature importance '
                         '(default: exps/cesnet_multimodal_each_week_train_v01/'
                         'week_<ref>/weights/best_model.pth)')
    ap.add_argument('--severities', type=float, nargs='+',
                    default=list(SEVERITIES_DEFAULT))
    ap.add_argument('--n_trials', type=int, default=8,
                    help='shift draws per (clean week, severity)')
    ap.add_argument('--n_resample', type=int, default=12000)
    ap.add_argument('--n_emb_resample', type=int, default=6000)
    ap.add_argument('--no_mfwdd', action='store_true')
    ap.add_argument('--variant_estimators', action='store_true',
                    help='also sweep the label-shift estimator variants '
                         '(bbse_soft/rlls/em/em_bcts) from '
                         'isolation_estimator_variants')
    ap.add_argument('--clean_weeks', type=int, nargs=2, default=None,
                    metavar=('LO', 'HI'),
                    help='inclusive clean-week range '
                         '(default: 2-9 for ref 1, 13-20 for ref 16)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--suffix', default='',
                    help='appended to cache/json/figure filenames, e.g. _fwd '
                         'for a forward-only clean-week variant')
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    severities = [float(s) for s in args.severities]
    dets = ['ours', 'naive'] + ([] if args.no_mfwdd else ['mfwdd'])
    if args.variant_estimators:
        dets += EST_NAMES
    ckpt = args.checkpoint or (f'exps/cesnet_multimodal_each_week_train_v01/'
                               f'week_{args.reference_week}/weights/best_model.pth')

    gt = load_ground_truth(out_dir)
    if gt is None:
        raise SystemExit(f"No ablation/variant cache in {out_dir} — run "
                         f"precision_isolation_ablation.py or "
                         f"isolation_estimator_variants.py first.")
    week_nums, f1_ref = gt['week_nums'], gt['f1_ref']
    F1_true, Ntrue = gt['F1_true'], gt['Ntrue']

    # ground truth — same definitions as the ablation / variant scripts
    drop = f1_ref[None, :] - F1_true
    present = (Ntrue >= N_MIN) & np.isfinite(f1_ref)[None, :] & (f1_ref[None, :] > 0)
    gt_deg = present & (drop > DELTA_DEG)
    gt_healthy = present & (drop <= EPS_HEALTHY)
    eval_mask = gt_deg | gt_healthy
    print(f"Ground truth: {gt_deg.sum()} degraded / {gt_healthy.sum()} healthy "
          f"class-weeks over {len(week_nums)} weeks")

    inference_dir = Path(args.inference_dir)
    files = sorted(inference_dir.glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    file_of = {int(re.search(r'(\d+)$', p.stem).group(1)): p for p in files}
    wk_index = {int(w): i for i, w in enumerate(week_nums)}

    # reference stats + feature importance, built once on first use
    _ctx = {}
    def get_ctx():
        if 'ref' not in _ctx:
            ref_file = file_of[args.reference_week]
            print(f"Building reference statistics from {ref_file.name} ...")
            ref_true, ref_pred, ref_soft, ref_emb, ref_eidx = load_week(ref_file)
            nc = ref_soft.shape[1]
            ref = build_reference(ref_true, ref_pred, ref_soft, ref_emb,
                                  ref_eidx, nc)
            feat_imp = (None if args.no_mfwdd else
                        load_feature_importance(ckpt, nc, ref_emb.shape[1]))
            est = (build_estimators(ref_soft, ref_true, nc, args.seed)
                   if args.variant_estimators else None)
            _ctx.update(ref=ref, num_classes=nc, feat_imp=feat_imp,
                        estimators=est)
        return _ctx['ref'], _ctx['feat_imp'], _ctx['num_classes']

    # ── per-(class,week) scores on the REAL weeks (positives + clean AUROC) ──
    scores_clean = dict(ours=gt['R_corr'], naive=gt['R_naive'])
    if not args.no_mfwdd:
        if 'S_mfwdd' in gt:
            print("MFWDD week scores from base ablation cache")
            scores_clean['mfwdd'] = gt['S_mfwdd']
        else:
            pos_cache = out_dir / 'labelshift_sweep_mfwdd_positives.npz'
            if pos_cache.exists() and not args.recompute:
                zz = np.load(pos_cache)
                assert np.array_equal(zz['week_nums'], week_nums)
                scores_clean['mfwdd'] = zz['S_mfwdd']
                print(f"MFWDD week scores from {pos_cache}")
            else:
                ref, feat_imp, num_classes = get_ctx()
                print(f"Computing MFWDD scores for {len(week_nums)} weeks ...")
                S = []
                for wn in week_nums:
                    true, pred, soft, emb, eidx = load_week(file_of[int(wn)])
                    S.append(mfwdd_scores(emb, pred[eidx], ref, feat_imp,
                                          num_classes))
                    print(f"  week {int(wn):2d}: {np.isfinite(S[-1]).sum()} "
                          f"classes scored", flush=True)
                scores_clean['mfwdd'] = np.array(S)
                np.savez_compressed(pos_cache, week_nums=week_nums,
                                    S_mfwdd=scores_clean['mfwdd'])
                print(f"Saved MFWDD week scores -> {pos_cache}")

    if args.variant_estimators:
        var_cache = out_dir / 'estimator_variants_cache.npz'
        if not var_cache.exists():
            raise SystemExit(f"--variant_estimators needs {var_cache} — run "
                             f"isolation_estimator_variants.py first.")
        vz = np.load(var_cache)
        assert np.array_equal(vz['week_nums'], week_nums), \
            'estimator-variants cache week order differs from ground truth'
        for n in EST_NAMES:
            scores_clean[n] = vz[f'R_{n}']
        print(f"Estimator-variant week scores from {var_cache}")

    pos, auroc_clean = {}, {}
    for d in dets:
        sc = scores_clean[d]
        pos[d] = sc[gt_deg & np.isfinite(sc)]
        m = eval_mask & np.isfinite(sc)
        auroc_clean[d] = pooled_auroc(sc[m & gt_deg], sc[m & gt_healthy])
        print(f"  clean {d:5s}: AUROC={auroc_clean[d]:.3f}  "
              f"(pos={len(pos[d])} reused for the sweep)")

    if args.clean_weeks:
        lo, hi = args.clean_weeks
    elif args.reference_week == 16:
        lo, hi = 13, 20
    else:
        lo, hi = 2, 9
    clean_wns = [int(w) for w in week_nums if lo <= w <= hi]
    print(f"Clean weeks for injection: {clean_wns}; severities {severities}; "
          f"{args.n_trials} trials each")

    # ── negatives: healthy classes under injected shift, per (sev,week,trial) ─
    cache = out_dir / f'labelshift_sweep_cache{args.suffix}.npz'
    sev_key = lambda s: str(s).replace('.', 'p')
    cache_ok = False
    if cache.exists() and not args.recompute:
        zz = np.load(cache)
        cache_ok = (np.allclose(zz['severities'], severities)
                    and all(f'S_{d}_{sev_key(s)}' in zz.files
                            for d in dets for s in severities))
        if not cache_ok:
            print(f"Cache {cache} lacks some detectors/severities — recomputing")
    if cache_ok:
        print(f"Loading cached sweep scores from {cache}")
        cells = {d: {s: [a for a in np.split(zz[f'S_{d}_{sev_key(s)}'],
                                             np.cumsum(zz[f'L_{d}_{sev_key(s)}'])[:-1])
                         if len(zz[f'L_{d}_{sev_key(s)}'])]
                     for s in severities} for d in dets}
        tv_mean = {s: float(v) for s, v in zip(severities, zz['tv_mean'])}
    else:
        ref, feat_imp, num_classes = get_ctx()
        rng = np.random.default_rng(args.seed)
        rng_emb = np.random.default_rng(args.seed + EMB_RNG_OFFSET)
        cells = {d: {s: [] for s in severities} for d in dets}
        tv_acc = {s: [] for s in severities}
        for wn in clean_wns:
            i = wk_index[wn]
            healthy = ((Ntrue[i] >= N_MIN) & np.isfinite(f1_ref)
                       & ((f1_ref - F1_true[i]) <= EPS_HEALTHY))
            true, pred, soft, emb, eidx = load_week(file_of[wn])
            H_all = entropy_rows(soft)
            rows_by_true = [np.where(true == c)[0] for c in range(num_classes)]
            if not args.no_mfwdd:
                emb_true = true[eidx]
                emb_rows_by_true = [np.where(emb_true == c)[0]
                                    for c in range(num_classes)]
                pred_emb_all = pred[eidx]
            p_nat = np.bincount(true, minlength=num_classes).astype(float)
            p_nat /= p_nat.sum()
            for sev in severities:
                for _ in range(args.n_trials):
                    logw = rng.uniform(np.log(1.0 / sev), np.log(sev),
                                       size=num_classes)
                    p_synth = np.exp(logw); p_synth /= p_synth.sum()
                    tv_acc[sev].append(0.5 * np.abs(p_synth - p_nat).sum())
                    ix = resample_label_shift(rows_by_true, p_synth,
                                              args.n_resample, rng)
                    if len(ix) == 0:
                        continue
                    pred_rs, H = pred[ix], H_all[ix]
                    p_bbse, _ = estimate_label_dist(pred_rs, num_classes,
                                                    ref['C_T_pinv'])
                    sc = {'ours': corrected_residual(pred_rs, H, p_bbse, ref,
                                                     num_classes),
                          'naive': naive_residual(pred_rs, H, ref, num_classes)}
                    if not args.no_mfwdd:
                        eix = resample_label_shift(emb_rows_by_true, p_synth,
                                                   args.n_emb_resample, rng_emb)
                        sc['mfwdd'] = (mfwdd_scores(emb[eix], pred_emb_all[eix],
                                                    ref, feat_imp, num_classes)
                                       if len(eix) else
                                       np.full(num_classes, np.nan))
                    if args.variant_estimators:
                        soft_rs = soft[ix]
                        for name in EST_NAMES:
                            try:
                                p_est = _ctx['estimators'][name](pred_rs, soft_rs)
                                sc[name] = corrected_residual(pred_rs, H, p_est,
                                                              ref, num_classes)
                            except Exception as e:
                                print(f"    wk {wn} sev {sev} {name} FAILED: "
                                      f"{type(e).__name__}: {e}")
                                sc[name] = np.full(num_classes, np.nan)
                    for d in dets:
                        v = sc[d][healthy]
                        cells[d][sev].append(v[np.isfinite(v)])
            print(f"  clean week {wn}: done ({int(healthy.sum())} healthy classes)",
                  flush=True)
        tv_mean = {s: float(np.mean(tv_acc[s])) for s in severities}
        np.savez_compressed(
            cache, severities=np.array(severities),
            tv_mean=np.array([tv_mean[s] for s in severities]),
            **{f'S_{d}_{sev_key(s)}': (np.concatenate(cells[d][s])
                                       if cells[d][s] else np.array([]))
               for d in dets for s in severities},
            **{f'L_{d}_{sev_key(s)}': np.array([len(a) for a in cells[d][s]],
                                               dtype=int)
               for d in dets for s in severities})
        print(f"Saved sweep cache -> {cache}")

    # ── pooled AUROC per severity + bootstrap CI over (week,trial) cells ─────
    boot_rng = np.random.default_rng(args.seed + 1)
    results = {d: dict(auroc_clean=auroc_clean[d], severities={}) for d in dets}
    for d in dets:
        for s in severities:
            cell_list = [a for a in cells[d][s] if len(a)]
            neg = np.concatenate(cell_list) if cell_list else np.array([])
            au = pooled_auroc(pos[d], neg)
            boots = []
            if len(cell_list) > 1:
                for _ in range(N_BOOT):
                    pick = boot_rng.integers(0, len(cell_list), len(cell_list))
                    nb = np.concatenate([cell_list[j] for j in pick])
                    boots.append(pooled_auroc(pos[d], nb))
            lo_ci, hi_ci = (np.percentile(boots, [2.5, 97.5]).tolist()
                            if boots else (float('nan'), float('nan')))
            results[d]['severities'][str(s)] = dict(
                auroc=au, ci_lo=lo_ci, ci_hi=hi_ci,
                n_neg=int(len(neg)), n_cells=len(cell_list),
                tv_from_natural=tv_mean[s])
            print(f"  {d:5s} sev {s:6g}: AUROC={au:.3f} "
                  f"[{lo_ci:.3f},{hi_ci:.3f}]  (n_neg={len(neg)})")

    with open(out_dir / f'labelshift_sweep_results{args.suffix}.json', 'w') as f:
        json.dump(dict(config=dict(reference_week=args.reference_week,
                                   inference_dir=str(inference_dir),
                                   checkpoint=(None if args.no_mfwdd else ckpt),
                                   clean_weeks=clean_wns,
                                   severities=severities,
                                   n_trials=args.n_trials,
                                   n_resample=args.n_resample,
                                   n_emb_resample=args.n_emb_resample,
                                   n_boot=N_BOOT, seed=args.seed),
                       results=results), f, indent=2)
    print(f"Saved results -> "
          f"{out_dir / f'labelshift_sweep_results{args.suffix}.json'}")

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False})
    COL = dict(ours='#2c7bb6', naive='#e07b39', mfwdd='#1a9850',
               bbse_soft='#74add1', rlls='#41ab5d', em='#d73027',
               em_bcts='#a50026')
    LAB = dict(ours='Ours: BBSE-corrected residual',
               naive='Naive (uncorrected)',
               mfwdd='MFWDD (feature-weighted embedding drift)',
               bbse_soft='BBSE (soft confusion)',
               rlls='RLLS',
               em='SLD-EM / MLLS',
               em_bcts='MLLS + BCTS calibration')
    MRK = dict(ours='o', naive='s', mfwdd='^', bbse_soft='D', rlls='v',
               em='P', em_bcts='X')
    STY = dict(bbse_soft='--', em_bcts='--')

    many = len(dets) > 3   # estimator panel: clean AUROC in legend, no anchors
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.array(severities)
    for d in dets:
        rows = [results[d]['severities'][str(s)] for s in severities]
        au = np.array([r['auroc'] for r in rows])
        lo_b = np.array([r['ci_lo'] for r in rows])
        hi_b = np.array([r['ci_hi'] for r in rows])
        lbl = (f"{LAB[d]}  (no-shift {auroc_clean[d]:.3f})" if many
               else LAB[d])
        ax.plot(xs, au, color=COL[d], marker=MRK[d], ms=4 if many else 5,
                lw=1.8 if many else 2.0, ls=STY.get(d, '-'), label=lbl)
        ok = np.isfinite(lo_b)
        ax.fill_between(xs[ok], lo_b[ok], hi_b[ok], color=COL[d], alpha=0.15,
                        lw=0)
        if not many:
            ax.axhline(auroc_clean[d], color=COL[d], ls='--', lw=1.2, alpha=0.65)
            below = au[-1] >= auroc_clean[d]  # dodge the curve at the right edge
            ax.annotate(f"no injected shift ({auroc_clean[d]:.3f})",
                        xy=(xs[-1], auroc_clean[d]), ha='right',
                        xytext=(-3, -11 if below else 4),
                        textcoords='offset points',
                        fontsize=7.5, color=COL[d], alpha=0.9)
    ax.axhline(0.5, color='#999', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1\n(uniform)', '10', '100', '1000'])
    ax.set_xlabel('Injected label-shift severity $s$ '
                  '(per-class prior weights $\\sim$ LogUniform$[1/s,\\,s]$)')
    ax.set_ylabel('Detection AUROC')
    ax.set_title('Per-class degradation detection vs injected label shift\n'
                 f'(Week-{args.reference_week} reference; negatives = healthy '
                 f'classes of clean weeks {clean_wns[0]}–{clean_wns[-1]})',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5 if many else 8.5, loc='lower left')
    panel_dir = out_dir / 'panels'; panel_dir.mkdir(exist_ok=True)
    out_fig = panel_dir / f'fig_isolation_auroc_vs_labelshift{args.suffix}.png'
    fig.savefig(out_fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out_fig}")


if __name__ == '__main__':
    main()
