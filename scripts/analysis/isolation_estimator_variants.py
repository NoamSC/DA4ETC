#!/usr/bin/env python
"""
Label-Shift Estimator Variants for the Per-Class Isolation Detector
===================================================================

Extends panel A of the precision-isolation ablation (Sec. VI): swaps the
label-shift estimator inside the corrected entropy residual

    r_corr(c) = obs_H(R_c) - E[H | estimated composition of R_c]

while keeping everything else (reference stats, residual construction,
ground truth) identical.  All estimators are classical label-shift
correctors -- they assume P(X|Y) is invariant and estimate only the target
label marginal P(Y):

  bbse_pinv  hard-prediction BBSE, truncated pseudo-inverse  (paper's "Ours";
             reproduced from / checked against the base ablation cache)
  bbse_soft  BBSE with the soft (expected) confusion matrix   [Lipton'18]
             (solved with the same truncated pseudo-inverse as the paper's
             hard BBSE — abstention's unregularized solve is singular at
             180 classes)
  rlls       regularized least-squares label shift            [Azizzadenesheli'19]
  em         SLD-EM / MLLS                                    [Saerens'02]
  em_bcts    MLLS + bias-corrected temperature scaling        [Alexandari'20]

bbse_soft / rlls / em / em_bcts use the tested reference implementations from
the `abstention` library (Kundaje lab, authors of Alexandari et al. 2020).

If <output_dir>/isolation_scores_cache.npz (written by
precision_isolation_ablation.py) exists, the ground truth and the Ours/Naive
curves are taken from it bit-identically, and the recomputed BBSE residual is
checked against it.  Otherwise (e.g. the Week-16 source, where the full
ablation was never run) the script computes ground truth and both baselines
itself with the same definitions.

Usage
  python scripts/analysis/isolation_estimator_variants.py                # Week-1 source
  python scripts/analysis/isolation_estimator_variants.py \
      --inference_dir results/inference/week_16_inference \
      --reference_week 16 --output_dir figs/isolation_w16               # Week-16 source

Outputs (under --output_dir)
  panels/fig_isolation_A_roc_estimators.png   panel-A-style ROC, one curve/estimator
  estimator_variants_cache.npz                per (class,week) residuals (+ gt if computed)
  estimator_variants_metrics.json             AUROCs
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
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precision_isolation_ablation import (
    load_week, entropy_rows, build_reference, estimate_label_dist,
    per_class_f1, resample_label_shift,
    N_MIN_REGION, DELTA_DEG, EPS_HEALTHY, N_MIN)

N_VALID_FIT = 100_000   # ref-week subsample used to fit the abstention adapters
SOFT_CLIP   = 1e-7      # clip before log for calibration / EM stability

EST_NAMES = ['bbse_soft', 'rlls', 'em', 'em_bcts']


# ── estimators ───────────────────────────────────────────────────────────────
def build_estimators(ref_soft, ref_true, num_classes, seed):
    """Returns dict name -> fn(pred, soft) -> estimated target P(Y).

    The abstention adapters share a fixed validation subsample of the
    reference week (softmax + one-hot labels); BCTS is fitted once here.
    """
    from abstention.calibration import TempScaling
    from abstention.label_shift import (EMImbalanceAdapter,
                                        RLLSImbalanceAdapter)

    rng = np.random.default_rng(seed)
    n = min(N_VALID_FIT, len(ref_true))
    sub = rng.choice(len(ref_true), size=n, replace=False)
    v_soft = np.clip(ref_soft[sub].astype(np.float64), SOFT_CLIP, 1.0)
    v_soft /= v_soft.sum(axis=1, keepdims=True)
    v_1hot = np.eye(num_classes, dtype=np.float64)[ref_true[sub]]
    p_src = v_1hot.mean(axis=0)          # empirical source prior of the subsample

    print(f"  Fitting BCTS calibrator on {n} reference samples ...")
    bcts = TempScaling(bias_positions='all')(
        valid_preacts=v_soft, valid_labels=v_1hot, posterior_supplied=True)
    v_soft_cal = bcts(v_soft)

    # soft BBSE: soft confusion C_s[k,c] = E[softmax_c | true k], inverted with
    # the same truncated pinv (rcond=1e-2) the paper uses for hard BBSE
    counts = v_1hot.sum(axis=0)
    C_soft = (v_1hot.T @ v_soft) / np.clip(counts[:, None], 1.0, None)
    Cs_T_pinv = np.linalg.pinv(C_soft.T, rcond=1e-2)

    def bbse_soft_fn(pred, soft):
        q = soft.astype(np.float64).mean(axis=0)
        p = np.clip(Cs_T_pinv @ q, 0, None)
        s = p.sum()
        return p / s if s > 0 else np.full(num_classes, 1.0 / num_classes)

    # (adapter, valid_probs, calib, mask_support).  mask_support restricts the
    # adapter to classes present in the fit subsample (p_src > 0).  abstention's
    # EM divides the per-iteration posterior by the source prior when
    # estimate_priors_from_valid_labels=True, so a class absent from the 100k
    # subsample (p_src == 0; e.g. class 128 here) gives 0/0 -> all-NaN
    # multipliers -> the silent uniform fallback below, collapsing em/em_bcts to
    # a constant 1/num_classes for every window.  Those classes have no P(X|Y)
    # reference and are unestimable anyway, so we drop them from the EM solve and
    # scatter the result back.  RLLS is a regularized least-squares solve with no
    # such division, so it keeps the full support unchanged.
    keep = p_src > 0
    adapters = {
        'rlls':      (RLLSImbalanceAdapter(),           v_soft,     None, False),
        'em':        (EMImbalanceAdapter(estimate_priors_from_valid_labels=True),
                      v_soft,     None, True),
        'em_bcts':   (EMImbalanceAdapter(estimate_priors_from_valid_labels=True),
                      v_soft_cal, bcts, True),
    }

    def make_fn(adapter, valid_probs, calib, mask_support):
        sup = keep if mask_support else np.ones(num_classes, dtype=bool)
        vp = valid_probs[:, sup]
        vp = vp / vp.sum(axis=1, keepdims=True)
        v1h = v_1hot[:, sup]
        psrc = p_src[sup]

        def fn(pred, soft):
            t = np.clip(soft.astype(np.float64), SOFT_CLIP, 1.0)
            t /= t.sum(axis=1, keepdims=True)
            if calib is not None:
                t = calib(t)
            tk = t[:, sup]
            tk = tk / tk.sum(axis=1, keepdims=True)
            f = adapter(valid_labels=v1h,
                        tofit_initial_posterior_probs=tk,
                        valid_posterior_probs=vp)
            w = np.array(f.multipliers, dtype=np.float64).ravel()
            p = np.zeros(num_classes, dtype=np.float64)
            p[sup] = np.clip(w * psrc, 0, None)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                print(f"    WARNING: estimator returned non-finite/degenerate "
                      f"multipliers; falling back to uniform")
                return np.full(num_classes, 1.0 / num_classes)
            return p / s
        return fn

    fns = {name: make_fn(*spec) for name, spec in adapters.items()}
    fns['bbse_soft'] = bbse_soft_fn
    return fns


def corrected_residual(pred, H, p_est, ref, num_classes):
    """Identical to score_arrays() in precision_isolation_ablation, with the
    BBSE estimate replaced by an arbitrary p_est."""
    C, cell_H, ref_region_H = ref['C'], ref['cell_H'], ref['ref_region_H']
    r = np.full(num_classes, np.nan)
    for c in range(num_classes):
        rmask = pred == c
        if rmask.sum() < N_MIN_REGION or np.isnan(ref_region_H[c]):
            continue
        obs = H[rmask].mean()
        mass = p_est * C[:, c]
        valid = (mass > 0) & ~np.isnan(cell_H[:, c])
        if valid.sum() > 0 and mass[valid].sum() > 0:
            w = mass[valid] / mass[valid].sum()
            r[c] = obs - float(np.dot(w, cell_H[valid, c]))
        else:
            r[c] = obs - ref_region_H[c]
    return r


def naive_residual(pred, H, ref, num_classes):
    """Naive (uncorrected) residual: obs entropy minus frozen reference
    region entropy — same scorability gates as score_arrays()."""
    ref_region_H = ref['ref_region_H']
    r = np.full(num_classes, np.nan)
    for c in range(num_classes):
        rmask = pred == c
        if rmask.sum() < N_MIN_REGION or np.isnan(ref_region_H[c]):
            continue
        r[c] = H[rmask].mean() - ref_region_H[c]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--reference_week', type=int, default=1)
    ap.add_argument('--output_dir', default='figs/isolation')
    ap.add_argument('--max_weeks', type=int, default=None, help='debug: limit #test weeks')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contamination', action='store_true',
                    help='also run the injected-label-shift contamination stage')
    ap.add_argument('--clean_weeks', type=int, nargs=2, default=None,
                    metavar=('LO', 'HI'),
                    help='inclusive clean-week range for contamination '
                         '(default: 2-9 for ref 1, 13-20 for ref 16)')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_cache = out_dir / 'isolation_scores_cache.npz'
    var_cache = out_dir / 'estimator_variants_cache.npz'
    have_base = base_cache.exists()

    inference_dir = Path(args.inference_dir)
    files = sorted(inference_dir.glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    wn_of = lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    file_of = {wn_of(p): p for p in files}

    _ctx = {}
    def get_ref_estimators():
        """Reference stats + fitted estimators, built once on first use."""
        if 'ref' not in _ctx:
            ref_file = file_of[args.reference_week]
            print(f"Reference week: {args.reference_week} ({ref_file.name})")
            ref_true, ref_pred, ref_soft, ref_emb, ref_eidx = load_week(ref_file)
            nc = ref_soft.shape[1]
            print("Building reference statistics ...")
            ref = build_reference(ref_true, ref_pred, ref_soft, ref_emb,
                                  ref_eidx, nc)
            _ctx.update(ref=ref, num_classes=nc,
                        estimators=build_estimators(ref_soft, ref_true, nc,
                                                    args.seed))
        return _ctx['ref'], _ctx['estimators'], _ctx['num_classes']

    # ── obtain per-(class,week) scores + ground-truth ingredients ────────────
    if var_cache.exists() and not args.recompute:
        print(f"Loading cached variant scores from {var_cache}")
        zz = np.load(var_cache)
        week_nums = zz['week_nums']
        R_est = {n: zz[f'R_{n}'] for n in EST_NAMES}
        if have_base:
            z = np.load(base_cache)
            assert np.array_equal(z['week_nums'], week_nums)
            R_corr_base, R_naive = z['R_corr'], z['R_naive']
            F1_true, Ntrue, f1_ref = z['F1_true'], z['Ntrue'], z['f1_ref']
        else:
            R_corr_base, R_naive = zz['R_corr_base'], zz['R_naive']
            F1_true, Ntrue, f1_ref = zz['F1_true'], zz['Ntrue'], zz['f1_ref']
    else:
        if have_base:
            z = np.load(base_cache)
            week_nums = z['week_nums']
            R_corr_base, R_naive = z['R_corr'], z['R_naive']
            F1_true, Ntrue, f1_ref = z['F1_true'], z['Ntrue'], z['f1_ref']
            print(f"Base ablation cache found ({base_cache}) — reusing its "
                  f"ground truth and Ours/Naive scores")
        else:
            week_nums = np.array([wn_of(p) for p in files
                                  if wn_of(p) != args.reference_week])
            print(f"No base ablation cache in {out_dir} — computing ground "
                  f"truth and baselines from scratch")
        if args.max_weeks:
            week_nums = week_nums[:args.max_weeks]

        ref, estimators, num_classes = get_ref_estimators()
        R_est = {n: [] for n in EST_NAMES}
        base_rows = {k: [] for k in ('R_corr_base', 'R_naive', 'F1_true', 'Ntrue')}
        for i, wn in enumerate(week_nums):
            true, pred, soft, emb, eidx = load_week(file_of[wn])
            H = entropy_rows(soft)
            p_bbse, _ = estimate_label_dist(pred, num_classes, ref['C_T_pinv'])
            r_bbse = corrected_residual(pred, H, p_bbse, ref, num_classes)
            if have_base:
                # sanity: repo-BBSE residual must reproduce the cached "Ours" row
                if not np.allclose(np.nan_to_num(r_bbse),
                                   np.nan_to_num(R_corr_base[i]), atol=1e-6):
                    raise RuntimeError(f'week {wn}: recomputed BBSE residual does '
                                       f'not match cache — ref/eval mismatch')
            else:
                base_rows['R_corr_base'].append(r_bbse)
                base_rows['R_naive'].append(naive_residual(pred, H, ref, num_classes))
                base_rows['F1_true'].append(per_class_f1(true, pred, num_classes))
                base_rows['Ntrue'].append(np.bincount(true, minlength=num_classes))
            msg = []
            for name in EST_NAMES:
                try:
                    p_est = estimators[name](pred, soft)
                    r = corrected_residual(pred, H, p_est, ref, num_classes)
                except Exception as e:
                    print(f"    week {wn} {name} FAILED: {type(e).__name__}: {e}")
                    r = np.full(num_classes, np.nan)
                R_est[name].append(r)
                msg.append(f"{name} {np.isfinite(r).sum()}")
            print(f"  week {wn:2d}: scored  ({', '.join(msg)} classes)", flush=True)

        R_est = {n: np.array(v) for n, v in R_est.items()}
        extra = {}
        if not have_base:
            R_corr_base = np.array(base_rows['R_corr_base'])
            R_naive = np.array(base_rows['R_naive'])
            F1_true = np.array(base_rows['F1_true'])
            Ntrue = np.array(base_rows['Ntrue'])
            f1_ref = ref['f1_ref']
            extra = dict(R_corr_base=R_corr_base, R_naive=R_naive,
                         F1_true=F1_true, Ntrue=Ntrue, f1_ref=f1_ref)
        np.savez_compressed(var_cache, week_nums=week_nums,
                            **{f'R_{n}': R_est[n] for n in EST_NAMES}, **extra)
        print(f"Saved variant scores cache -> {var_cache}")

    # ── ground truth (same definitions as precision_isolation_ablation) ─────
    drop = f1_ref[None, :] - F1_true
    present = (Ntrue >= N_MIN) & np.isfinite(f1_ref)[None, :] & (f1_ref[None, :] > 0)
    gt_deg = present & (drop > DELTA_DEG)
    gt_healthy = present & (drop <= EPS_HEALTHY)
    eval_mask = gt_deg | gt_healthy
    print(f"Ground truth: {gt_deg.sum()} degraded, {gt_healthy.sum()} healthy "
          f"class-weeks over {len(week_nums)} weeks")

    # ── pooled detection AUROC (same protocol as panel A) ────────────────────
    curves = [('ours',  R_corr_base, 'BBSE, trunc. pinv'),
              ('bbse_soft', R_est['bbse_soft'], 'BBSE (soft confusion, trunc. pinv)'),
              ('rlls',  R_est['rlls'],  'RLLS'),
              ('em',    R_est['em'],    'SLD-EM / MLLS'),
              ('em_bcts', R_est['em_bcts'], 'MLLS + BCTS calibration'),
              ('naive', R_naive, 'Naive (uncorrected)')]

    results = {}
    for name, sc, lab in curves:
        m = eval_mask & np.isfinite(sc)
        y = gt_deg[m].astype(int)
        au = roc_auc_score(y, sc[m]) if 0 < y.sum() < m.sum() else float('nan')
        results[name] = dict(auroc=float(au), n_eval=int(m.sum()), n_pos=int(y.sum()))
        print(f"  {name:10s}: AUROC={au:.3f}  (n={m.sum()}, pos={y.sum()})")

    # ── figure: panel-A clone with estimator-variant curves ─────────────────
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False})
    COL = dict(ours='#2c7bb6', bbse_soft='#74add1', rlls='#1a9850',
               em='#d73027', em_bcts='#a50026', naive='#e07b39')
    STY = dict(ours='-', bbse_soft='--', rlls='-', em='-', em_bcts='--', naive=':')

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, sc, lab in curves:
        m = eval_mask & np.isfinite(sc)
        y = gt_deg[m].astype(int)
        if y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, sc[m])
        ax.plot(fpr, tpr, color=COL[name], ls=STY[name], lw=2.0,
                label=f"{lab}  (AUROC={results[name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color='#999', ls='--', lw=1)
    ax.set_xlabel('False positive rate (healthy classes)')
    ax.set_ylabel('True positive rate (degraded classes)')
    ax.set_title('Per-class degradation detection — label-shift estimator variants\n'
                 f'(Week-{args.reference_week} reference, pooled over all class-weeks)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    panel_dir = out_dir / 'panels'; panel_dir.mkdir(exist_ok=True)
    out_fig = panel_dir / 'fig_isolation_A_roc_estimators.png'
    fig.savefig(out_fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out_fig}")

    with open(out_dir / 'estimator_variants_metrics.json', 'w') as f:
        json.dump(dict(config=dict(reference_week=args.reference_week,
                                   inference_dir=str(inference_dir),
                                   n_valid_fit=N_VALID_FIT, seed=args.seed,
                                   n_test_weeks=int(len(week_nums))),
                       detection=results), f, indent=2)
    print(f"Saved metrics -> {out_dir / 'estimator_variants_metrics.json'}")

    if not args.contamination:
        return

    # ════════ contamination stage: injected PURE label shift ════════════════
    # Same protocol as robustness_label_shift() in precision_isolation_ablation:
    # on clean weeks, tilt the label prior in log-space (LogUniform 1/sev..sev)
    # and resample rows grouped by TRUE class -> P(X|Y) exactly invariant.
    # Negatives = healthy classes under shift; positives = real degraded scores.
    severities = (10, 100, 1000)
    n_trials, n_resample = 4, 12000
    if args.clean_weeks:
        lo, hi = args.clean_weeks
    elif args.reference_week == 16:
        lo, hi = 13, 20      # paper's healthy regime for the week-16 reference
    else:
        lo, hi = 2, 9        # pre-artifact clean weeks (week-1 reference)
    clean_wns = [int(w) for w in week_nums if lo <= w <= hi]
    print(f"\nContamination stage: clean weeks {clean_wns}, severities {severities}")

    contam_cache = out_dir / 'estimator_variants_contamination_cache.npz'
    det_names = [c[0] for c in curves]
    if contam_cache.exists() and not args.recompute:
        print(f"Loading cached contamination scores from {contam_cache}")
        zz = np.load(contam_cache)
        shift_scores = {d: {s: zz[f'S_{d}_{s}'] for s in severities}
                        for d in det_names}
    else:
        ref, estimators, num_classes = get_ref_estimators()
        rng = np.random.default_rng(args.seed)
        wk_index = {int(w): i for i, w in enumerate(week_nums)}
        shift_scores = {d: {s: [] for s in severities} for d in det_names}
        for wn in clean_wns:
            i = wk_index[wn]
            healthy = ((Ntrue[i] >= N_MIN) & np.isfinite(f1_ref)
                       & ((f1_ref - F1_true[i]) <= EPS_HEALTHY))
            true, pred, soft, emb, eidx = load_week(file_of[wn])
            rows_by_true = [np.where(true == c)[0] for c in range(num_classes)]
            for sev in severities:
                for _ in range(n_trials):
                    logw = rng.uniform(np.log(1.0 / sev), np.log(float(sev)),
                                       size=num_classes)
                    p_synth = np.exp(logw); p_synth /= p_synth.sum()
                    ix = resample_label_shift(rows_by_true, p_synth, n_resample, rng)
                    if len(ix) == 0:
                        continue
                    pred_rs, soft_rs = pred[ix], soft[ix]
                    H = entropy_rows(soft_rs)
                    p_bbse, _ = estimate_label_dist(pred_rs, num_classes,
                                                    ref['C_T_pinv'])
                    sc = {'ours': corrected_residual(pred_rs, H, p_bbse, ref,
                                                     num_classes),
                          'naive': naive_residual(pred_rs, H, ref, num_classes)}
                    for name in EST_NAMES:
                        try:
                            p_est = estimators[name](pred_rs, soft_rs)
                            sc[name] = corrected_residual(pred_rs, H, p_est,
                                                          ref, num_classes)
                        except Exception as e:
                            print(f"    wk {wn} sev {sev} {name} FAILED: "
                                  f"{type(e).__name__}: {e}")
                            sc[name] = np.full(num_classes, np.nan)
                    for d in det_names:
                        s = sc[d][healthy]
                        shift_scores[d][sev].append(s[np.isfinite(s)])
            print(f"  clean week {wn}: done ({int(healthy.sum())} healthy classes)",
                  flush=True)
        shift_scores = {d: {s: (np.concatenate(v) if v else np.array([]))
                            for s, v in shift_scores[d].items()}
                        for d in det_names}
        np.savez_compressed(contam_cache,
                            **{f'S_{d}_{s}': shift_scores[d][s]
                               for d in det_names for s in severities})
        print(f"Saved contamination cache -> {contam_cache}")

    contam_results = {}
    for name, sc, lab in curves:
        pos = sc[gt_deg & np.isfinite(sc)]
        row = {'auroc_clean': results[name]['auroc']}
        for sev in severities:
            neg = shift_scores[name][sev]
            if len(pos) and len(neg):
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                row[f'auroc_{sev}x'] = float(roc_auc_score(y, np.r_[pos, neg]))
            else:
                row[f'auroc_{sev}x'] = float('nan')
        contam_results[name] = row
        print(f"  {name:10s}: clean={row['auroc_clean']:.3f}  "
              + "  ".join(f"{s}x={row[f'auroc_{s}x']:.3f}" for s in severities))

    # ROC figures (panel-A style, contaminated negatives), one per severity;
    # the unsuffixed filename keeps pointing at the worst severity.
    worst = severities[-1]
    for sev in severities:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for name, sc, lab in curves:
            pos = sc[gt_deg & np.isfinite(sc)]
            neg = shift_scores[name][sev]
            if len(pos) == 0 or len(neg) == 0:
                continue
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            fpr, tpr, _ = roc_curve(y, np.r_[pos, neg])
            ax.plot(fpr, tpr, color=COL[name], ls=STY[name], lw=2.0,
                    label=f"{lab}  (AUROC={contam_results[name][f'auroc_{sev}x']:.3f})")
        ax.plot([0, 1], [0, 1], color='#999', ls='--', lw=1)
        ax.set_xlabel(f'False positive rate (healthy classes under {sev}× label shift)')
        ax.set_ylabel('True positive rate (degraded classes)')
        ax.set_title('Per-class degradation detection under EXTREME label shift\n'
                     f'(Week-{args.reference_week} reference; negatives = clean weeks '
                     f'{lo}–{hi} under {sev}× injected shift)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=7.5, loc='lower right')
        outs = [panel_dir / f'fig_isolation_A_roc_estimators_labelshift_{sev}x.png']
        if sev == worst:
            outs.append(panel_dir / 'fig_isolation_A_roc_estimators_labelshift.png')
        for o in outs:
            fig.savefig(o, dpi=180, bbox_inches='tight')
            print(f"Saved figure -> {o}")
        plt.close(fig)

    with open(out_dir / 'estimator_variants_contamination.json', 'w') as f:
        json.dump(dict(config=dict(reference_week=args.reference_week,
                                   clean_weeks=clean_wns,
                                   severities=list(severities),
                                   n_trials=n_trials, n_resample=n_resample,
                                   seed=args.seed),
                       auroc=contam_results), f, indent=2)
    print(f"Saved contamination metrics -> "
          f"{out_dir / 'estimator_variants_contamination.json'}")


if __name__ == '__main__':
    main()
