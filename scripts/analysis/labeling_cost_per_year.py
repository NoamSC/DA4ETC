#!/usr/bin/env python
"""
Labeling-cost-per-year comparison ("the big table of the things that matter")
=============================================================================

On CESNET-TLS-Year22 (Week-16 reference, forward weeks 17-52), we ask the single
operationally honest question: *to keep the 180-class catalogue healthy over one
year, how many target labels does each maintenance strategy spend?*

We reuse the EXACT per-(class,week) panel of the Sec. VI isolation ablation
(figs/isolation_w16/isolation_scores_cache.npz): per-class true F1, reference F1,
sample counts, and the three detectors' scores (ours = BBSE-corrected entropy
residual; naive = uncorrected; MFWDD = feature-weighted embedding drift).

Ground truth (identical to the isolation ablation):
  degraded(c,t)  iff  f1_ref(c) - f1_true(c,t) > DELTA_DEG  and  Ntrue >= N_MIN
  healthy(c,t)   iff  f1_ref(c) - f1_true(c,t) <= EPS_HEALTHY
A *degradation episode* for class c = a maximal run of consecutive evaluable weeks
in which c is degraded.  A repair (prototype recompute) at the episode onset fixes
the class for the rest of that episode, so EACH EPISODE costs K labels exactly once
-- persistence is free.  False alarms recur weekly (a healthy class wrongly flagged
must be checked every week it trips), so they are counted per class-week.

Strategies compared (all normalised to a 52-week year):
  * UDA / TTA (DANN, TENT, CoTTA, AdaBN): 0 target labels (and <=0 accuracy).
  * Periodic full retrain: R retrains/yr x FULL_RETRAIN labels.
  * Global-detector-triggered retrain (e.g. MFWDD): a global drift detector
    cannot reliably isolate WHICH class broke (per-class AUROC ~0.72 vs ours
    ~0.88; at ours' recall it would false-flag ~70% of healthy classes, and its
    per-class recall caps at ~0.62 because the embedding-drift statistic is
    undefined on small regions).  So its only safe response to an alarm is a
    full catalogue retrain.  We therefore charge it the SAME labels as a periodic
    full retrain at whatever cadence the operator picks -- the point is the
    PER-ALARM cost (a full retrain) vs ours (K labels on one class), not how
    often each fires (on a continuously drifting stream both fire constantly).
  * Ours (BBSE monitor + few-shot repair): K labels per detected episode plus
    K labels per false alarm.  recall and bleed read at ours' operating point.

Outputs:  results/labeling_cost/labeling_cost_per_year.json  +  a printed table.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')

DELTA_DEG   = 0.15
EPS_HEALTHY = 0.05
N_MIN       = 30
WEEKS_YEAR  = 52
# Measured per-class F1 AFTER few-shot repair of the detected teleported classes
# (joint few-shot repair, K=50; FINAL_CLAIMS / results/repair/joint_repair_v01):
# docker-registry(49) 0.86, eset-edtd(57) 0.88, skype(140) 0.43 (NCM, conservative).
# We compute ours' mean-F1 by substituting these INTO THE SAME softmax isolation
# panel the other strategies use (not by adding the NCM-180-space +0.019 lift, which
# lives on a different metric scale). This credits only the measured teleporters -> a
# LOWER BOUND, while the labeling budget counts the FULL monitor+false-alarm cost.
REPAIRED_F1 = {49: 0.86, 57: 0.88, 140: 0.43}
# Full-catalogue retrain label budget (Week-16 train.parquet, ~46k/class x 180).
PER_CLASS_RETRAIN = 46_000
N_CLASSES         = 180
FULL_RETRAIN      = PER_CLASS_RETRAIN * N_CLASSES        # ~8.26M


def episodes_per_class(deg, present):
    """List of episode lengths per class = maximal runs of degraded evaluable weeks."""
    n_ep = 0
    classes_hit = 0
    for c in range(deg.shape[1]):
        idx = np.where(present[:, c])[0]
        if len(idx) == 0:
            continue
        seq = deg[idx, c].astype(int)
        runs, run = [], 0
        for v in seq:
            if v:
                run += 1
            elif run > 0:
                runs.append(run); run = 0
        if run > 0:
            runs.append(run)
        n_ep += len(runs)
        if runs:
            classes_hit += 1
    return n_ep, classes_hit


def decay_curve(F1_true, f1_ref, present):
    """macro-F1 as a function of age = weeks since (re)training. age 0 = reference."""
    ref = float(np.nanmean(f1_ref[np.isfinite(f1_ref) & (f1_ref > 0)]))
    ages, macro = [0], [ref]
    for t in range(F1_true.shape[0]):
        ages.append(t + 1)
        macro.append(float(np.nanmean(F1_true[t][present[t]])))
    return np.array(ages), np.array(macro)


def mean_f1_at_cadence(ages, macro, period_wk, horizon):
    """mean macro-F1 over `horizon` deployed weeks if retrained every `period_wk` weeks.
    Week w has age (w % period_wk)+1, so the first deployed week after a retrain is
    age 1 (never age 0, the reference week, which is not a deployed week).
    IDEALISED: every retrain recovers to ~peak and re-decays along the same week-16
    forward-drift shape -- optimistic for the retrain strategies."""
    amax = ages.max()
    return float(np.mean([np.interp(min((w % period_wk) + 1, amax), ages, macro)
                          for w in range(horizon)]))


def ours_panel_meanf1(F1_true, drop, present, repaired):
    """ours' mean macro-F1 in the SAME softmax panel space: substitute the measured
    post-repair F1 for each flagged class at its degraded weeks, then re-average."""
    F = F1_true.copy()
    for c, rf in repaired.items():
        for t in range(F.shape[0]):
            if present[t, c] and np.isfinite(drop[t, c]) and drop[t, c] > DELTA_DEG:
                F[t, c] = max(F[t, c], rf)
    wk = [np.nanmean(F[t][present[t]]) for t in range(F.shape[0]) if present[t].sum() > 0]
    return float(np.mean(wk))


def mfwdd_retrain_cadence(S_mfwdd, present, yr, n_sigmas=(0.5, 1.0, 1.5, 2.0)):
    """MFWDD as a GLOBAL weekly drift detector: per-week mean drift over evaluable
    classes; a retrain is triggered on each rising crossing of a healthy baseline
    (calmest 6 forward weeks + n_sigma). The count is THRESHOLD-DEPENDENT, so we
    return the full range across n_sigmas (report as 'order 5-10x/yr', not a crisp
    point) plus the n_sigma=1 central value."""
    g = []
    for t in range(S_mfwdd.shape[0]):
        m = present[t] & np.isfinite(S_mfwdd[t])
        g.append(float(np.mean(S_mfwdd[t][m])) if m.sum() else np.nan)
    g = np.array(g)
    base = np.sort(g[np.isfinite(g)])[:6]
    out = {}
    for ns in n_sigmas:
        thr = base.mean() + ns * base.std()
        fired = g > thr
        edges = int(np.sum(fired[1:] & ~fired[:-1])) + (1 if fired[0] else 0)
        out[ns] = edges * yr
    return out, base.mean() + 1.0 * base.std()


def threshold_at_recall(scores, gt_pos_mask, present, target_recall):
    """Threshold achieving recall closest to target over evaluable degraded cells.
    Recall is measured exactly as (flagged & degraded)/degraded with strict '>'."""
    pos = scores[gt_pos_mask & present]
    pos = pos[np.isfinite(pos)]
    n_pos = int(gt_pos_mask.sum())
    if len(pos) == 0 or n_pos == 0:
        return np.inf, 0.0
    cand = np.unique(pos)
    best_thr, best_err = np.inf, 1e9
    for thr in cand:
        rec = float((pos > thr).sum() / n_pos)
        err = abs(rec - target_recall)
        if err < best_err:
            best_err, best_thr = err, thr
    rec = float((pos > best_thr).sum() / n_pos)
    return float(best_thr), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='figs/isolation_w16/isolation_scores_cache.npz')
    ap.add_argument('--metrics', default='figs/isolation_w16/isolation_metrics_fwd.json',
                    help="canonical operating points (recall / natural_bleed) from the "
                         "Sec. VI isolation ablation -- used verbatim for consistency")
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--target_recall', type=float, default=0.78,
                    help="shared detection recall for the MFWDD firing analysis")
    ap.add_argument('--K', type=int, nargs='+', default=[10, 50],
                    help="few-shot labels per flagged class")
    ap.add_argument('--periodic', type=int, nargs='+', default=[1, 4, 12],
                    help="reference periodic-retrain schedules per year")
    ap.add_argument('--output_dir', default='results/labeling_cost')
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    wn = z['week_nums']
    fwd = wn > args.reference_week
    wnf = wn[fwd]
    F1t = z['F1_true'][fwd]
    Nt  = z['Ntrue'][fwd]
    f1ref = z['f1_ref']
    R_corr  = z['R_corr'][fwd]     # ours
    S_mfwdd = z['S_mfwdd'][fwd]    # MFWDD
    n_fwd = len(wnf)
    yr = WEEKS_YEAR / n_fwd        # per-year scale factor

    # present mask identical to the isolation ablation (Ntrue>=N_MIN & f1_ref>0)
    present = (Nt >= N_MIN) & np.isfinite(f1ref)[None, :] & (f1ref[None, :] > 0)
    drop = f1ref[None, :] - F1t
    deg     = present & np.isfinite(drop) & (drop > DELTA_DEG)
    healthy = present & np.isfinite(drop) & (drop <= EPS_HEALTHY)

    n_episodes, n_classes_hit = episodes_per_class(deg, present)
    deg_cw = int(deg.sum())
    heal_cw = int(healthy.sum())

    # ── ours: CANONICAL operating point from the isolation ablation ───────────
    # recall (over degraded cells) and natural_bleed (clean-calibrated per-class
    # false-alarm rate on genuinely healthy classes) are taken verbatim so the
    # labeling table is consistent with the published Sec. VI numbers.
    op = json.load(open(args.metrics))['operating_point']
    recall_ours = float(op['ours']['recall'])
    bleed_ours  = float(op['ours']['natural_bleed'])         # clean-week FA rate (4.9%)
    bleed_ours_ls = float(op['ours']['labelshift_bleed'])    # FA under benign label shift

    # repairs = detected episodes (persistence free); false alarms = per class-week.
    detected_episodes  = recall_ours * n_episodes
    fa_classweeks_ours = bleed_ours * heal_cw
    fa_classweeks_ls   = bleed_ours_ls * heal_cw             # stressed-regime FA

    # ── MFWDD diagnostics (justify "must retrain", do NOT manufacture a retrain count).
    # Its per-class recall caps at ~0.62 (statistic undefined on small regions) and at
    # the most permissive useful threshold it false-flags the majority of healthy
    # classes -- so it cannot target a repair and its only safe response is a full
    # retrain.  We therefore charge MFWDD the periodic-retrain budget (cadence chosen
    # by the operator), NOT a per-week strawman.
    thr_mf, recall_mf = threshold_at_recall(S_mfwdd, deg, present, args.target_recall)
    flagged_mf = (S_mfwdd > thr_mf) & present
    bleed_mf  = float((flagged_mf & healthy).sum() / max(healthy.sum(), 1))
    # MFWDD as a global weekly drift detector -> retrains/yr (THRESHOLD-DEPENDENT range)
    mf_range, mf_thr = mfwdd_retrain_cadence(S_mfwdd, present, yr)
    mf_retrains_yr = mf_range[1.0]                    # central (n_sigma=1) value
    mf_period_wk = max(1, int(round(WEEKS_YEAR / mf_retrains_yr)))

    # ── mean macro-F1 over the year delivered by each strategy (softmax panel) ─
    ages, macro = decay_curve(F1t, f1ref, present)
    f1_frozen = mean_f1_at_cadence(ages, macro, period_wk=n_fwd + 1, horizon=n_fwd)
    f1_ours   = ours_panel_meanf1(F1t, drop, present, REPAIRED_F1)   # same-space substitution
    f1_mfwdd  = mean_f1_at_cadence(ages, macro, mf_period_wk, n_fwd)
    f1_periodic = {r: mean_f1_at_cadence(ages, macro, max(1, round(WEEKS_YEAR / r)), n_fwd)
                   for r in args.periodic}

    out = {
        'config': dict(reference_week=args.reference_week, forward_weeks=int(n_fwd),
                       weeks=[int(w) for w in wnf], year_scale=yr,
                       target_recall=args.target_recall, DELTA_DEG=DELTA_DEG,
                       EPS_HEALTHY=EPS_HEALTHY, N_MIN=N_MIN,
                       per_class_retrain=PER_CLASS_RETRAIN, full_retrain=FULL_RETRAIN),
        'ground_truth': dict(degraded_classweeks=deg_cw, healthy_classweeks=heal_cw,
                             episodes_horizon=n_episodes, classes_degraded=n_classes_hit,
                             episodes_per_year=round(n_episodes * yr, 1),
                             classes_degraded_frac=round(n_classes_hit / N_CLASSES, 3)),
        'ours': dict(recall=recall_ours, bleed_natural=bleed_ours,
                     bleed_labelshift=bleed_ours_ls,
                     detected_episodes_per_year=round(detected_episodes * yr, 1),
                     false_alarms_per_year=round(fa_classweeks_ours * yr, 1),
                     false_alarms_per_year_labelshift=round(fa_classweeks_ls * yr, 1)),
        'mfwdd': dict(recall_cap=recall_mf, bleed_at_recall_cap=bleed_mf,
                      retrains_per_year_central=round(mf_retrains_yr, 1),
                      retrains_per_year_range={str(k): round(v, 1) for k, v in mf_range.items()},
                      retrain_period_wk=mf_period_wk, global_threshold=mf_thr,
                      note='global drift detector: cannot isolate -> full retrain per alarm; '
                           'cadence is threshold-dependent (order 5-10x/yr)'),
        'mean_macro_f1_year': dict(
            uda_tta=round(f1_frozen, 3),          # <=0 gain; TENT/CoTTA lower
            ours=round(f1_ours, 3),               # same-panel substitution of repaired teleporters
            ours_lift=round(f1_ours - f1_frozen, 4),
            mfwdd_retrain_idealised=round(f1_mfwdd, 3),
            periodic_idealised={r: round(v, 3) for r, v in f1_periodic.items()},
            frozen=round(f1_frozen, 3), retrain_ceiling=round(float(macro[0]), 3),
            note='frozen/ours measured on the softmax isolation panel; retrain rows are '
                 'IDEALISED (recover to ~peak each retrain, week-16 decay shape)'),
        'labels_per_year': {},
    }

    L = out['labels_per_year']
    L['uda_tta'] = 0
    L['mfwdd_retrain'] = int(round(mf_retrains_yr) * FULL_RETRAIN)
    for r in args.periodic:
        L[f'periodic_retrain_x{r}'] = int(r * FULL_RETRAIN)
    for K in args.K:
        actions    = detected_episodes + fa_classweeks_ours       # repairs + natural FA
        actions_ls = detected_episodes + fa_classweeks_ls         # repairs + label-shift FA
        # upper bound: charge K every degraded class-week (no persistence credit)
        ub = (deg_cw + fa_classweeks_ours)
        L[f'ours_K{K}'] = int(round((actions * yr) * K))
        L[f'ours_K{K}_labelshift_bleed'] = int(round((actions_ls * yr) * K))
        L[f'ours_K{K}_upperbound_per_classweek'] = int(round((ub * yr) * K))
        L[f'ours_K{K}_repairs_only'] = int(round((detected_episodes * yr) * K))
        L[f'ours_K{K}_fa_only'] = int(round((fa_classweeks_ours * yr) * K))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / 'labeling_cost_per_year.json', 'w') as f:
        json.dump(out, f, indent=2)

    # ── pretty print ──────────────────────────────────────────────────────────
    print(f"\n=== CESNET-TLS-Year22  ref-wk{args.reference_week}  forward {n_fwd} wk "
          f"-> /yr x{yr:.2f} ===")
    print(f"Ground truth: {deg_cw} degraded class-weeks, {heal_cw} healthy; "
          f"{n_episodes} degradation episodes across {n_classes_hit}/{N_CLASSES} classes "
          f"(~{n_episodes*yr:.0f} episodes/yr)")
    print(f"Ours  @recall {recall_ours:.2f}: bleed {bleed_ours:.3f} (LS {bleed_ours_ls:.3f}) "
          f"-> {detected_episodes*yr:.0f} repairs/yr + {fa_classweeks_ours*yr:.0f} FA/yr")
    rng = '/'.join(f'{mf_range[s]:.0f}' for s in (0.5, 1.0, 1.5, 2.0))
    print(f"MFWDD recall caps at {recall_mf:.2f} (bleed {bleed_mf:.3f}); global drift "
          f"detector triggers {rng} retrains/yr at n_sigma=0.5/1/1.5/2 (order 5-10x/yr)")
    print(f"Mean macro-F1/yr (panel): frozen/UDA {f1_frozen:.3f} | ours {f1_ours:.3f} "
          f"(+{f1_ours-f1_frozen:.3f}) | MFWDD-retrain~{f1_mfwdd:.3f}(ideal) | ceiling {macro[0]:.3f}")
    print(f"\nFull retrain = {FULL_RETRAIN:,} labels (180 x {PER_CLASS_RETRAIN:,})\n")
    hdr = f"{'strategy':34s}{'labels/yr':>16s}{'x retrain':>12s}{'meanF1':>9s}"
    print(hdr); print('-' * len(hdr))
    one = FULL_RETRAIN
    f1map = {'uda_tta': f1_frozen, 'mfwdd_retrain': f1_mfwdd}
    for r, v in f1_periodic.items():
        f1map[f'periodic_retrain_x{r}'] = v
    for K in args.K:
        f1map[f'ours_K{K}'] = f1_ours
    for k, v in L.items():
        if k.endswith('_repairs_only') or k.endswith('_fa_only') \
           or k.endswith('_labelshift_bleed') or k.endswith('_upperbound_per_classweek'):
            continue
        f1s = f"{f1map[k]:.3f}" if k in f1map else "  -  "
        print(f"{k:34s}{v:>16,}{v/one:>11.3%}{f1s:>9s}")
    print()


if __name__ == '__main__':
    main()
