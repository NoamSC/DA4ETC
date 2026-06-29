#!/usr/bin/env python
"""
EXPERIMENT A' — ALL-FLAGGED joint few-shot REPAIR -> GLOBAL macro-F1 over the
forward YEAR (paper "The Illusion of Continuous Drift").

Why this experiment
-------------------
joint_repair_global_f1.py repairs only the THREE hand-picked teleporters
(49 docker-registry, 57 eset-edtd, 140 skype) on a single eval week, crediting
"ours" a tiny +0.009 panel-F1. That undersells (and cherry-picks) the loop.

The HONEST full-loop question: over the whole forward year, if we repair EVERY
class the label-free monitor flags on each week -- INCLUDING the monitor's false
alarms (healthy classes wrongly flagged, which the loop would also try to repair
and pay any poaching cost for) -- what GLOBAL macro-F1 (all 180 classes) does the
diagnose-and-repair loop deliver, and does repairing dozens of classes at once
compound stable-class damage?

Protocol (apples-to-apples with joint_repair_global_f1.py; reuses its helpers)
-----------------------------------------------------------------------------
  * Frozen Week-16 source; forward-only eval (weeks 17..52); 10%-subsample
    saved embeddings (results/inference_auditfix/week_16_vanilla_bs64).
  * Monitor panel: figs/isolation_w16/isolation_scores_cache.npz
      R_corr[w,c] = BBSE-corrected entropy-residual monitor score (ref week 16),
      Ntrue[w,c]  = full-data per-(week,class) support, F1_true, f1_ref.
  * FLAGGED set on eval week w = { c : R_corr[w,c] > THR (=0.21595, the monitor's
    calibrated operating point) AND Ntrue[w,c] >= N_MIN (=30, class present) }.
    This includes false alarms BY DESIGN -- the loop does not know ground truth.
  * REPAIRABLE = flagged class with a post-drift support pool of >= MIN_PER_CLASS
    (=40) embedded shots drawn from EARLIER degraded weeks (R_corr>THR, w'<w),
    disjoint from the eval week (no leakage). Classes that have vanished / lack
    post-drift support cannot be repaired -- counted and reported.
  * Repair = replace each repairable class's frozen Week-16 NCM prototype with the
    mean of K (=50) post-drift shots (NCM single-prototype), ALL flagged classes
    swapped SIMULTANEOUSLY, then a single global nearest-prototype decision. We
    also report a "knn_joint" variant where the canonical multi-modal class
    (skype, 140) uses the k-NN set-distance corrector (mirroring the 3-class
    script), the rest NCM.
  * 5 seeds for the support draw. BEFORE (source-only frozen NCM) and AFTER are
    scored on the SAME full week-w embedded set in the SAME frozen embedding
    space -> a true apples-to-apples delta.

Reported per week and averaged over the forward year:
  * global macro-F1 (180) BEFORE vs AFTER, ΔF1
  * macro-F1 over the STABLE (never-flagged-all-year) classes BEFORE vs AFTER
    = CUMULATIVE poaching check
  * per-repaired-class post-repair recall + F1
  * #flagged / #repairable / #unrepairable(no support) / #false-alarm
    (false alarm = class actually healthy: f1_ref - F1_true <= EPS_HEALTHY=0.05)

CAVEAT: 10%-subsample embeddings; frozen Week-16 NCM prototypes for the in-space
        before/after; macro-F1 over 180 classes scores empty/low-support classes
        as 0 (same convention BEFORE & AFTER, so every Δ is apples-to-apples).
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/analysis"))

import few_shot_repair_loop as F          # data loading, prototypes, ncm, macro_f1
import few_shot_repair_knn as KNN         # set-distance helpers

OUT_RES = os.path.join(ROOT, "results/repair/joint_repair_allflagged_v01")
PANEL = os.path.join(ROOT, "figs/isolation_w16/isolation_scores_cache.npz")

THR = 0.21595234348064105     # monitor operating point (isolation_metrics_fwd.json)
N_MIN = 30                    # class-present gate (config n_min)
EPS_HEALTHY = 0.05            # false-alarm gate (config eps_healthy)
K = 50
MIN_PER_CLASS = 40
SEEDS = [1, 2, 3, 4, 42]
REF_WEEK = 16
KNN_CLASSES = {140}           # canonical multi-modal class -> k-NN corrector

_EMB_CACHE = {}


def load_week(week):
    """Memoized week-embedding loader (avoids tens of thousands of redundant
    np.load calls when building per-class support pools across the year)."""
    if week not in _EMB_CACHE:
        _EMB_CACHE[week] = F.load_week_embeddings(week)
    return _EMB_CACHE[week]


def macro_f1_all(true, pred, classes):
    return F.macro_f1(true, pred, classes)


def per_class_recall_f1(true, pred, c):
    tp = int(np.sum((pred == c) & (true == c)))
    fp = int(np.sum((pred == c) & (true != c)))
    fn = int(np.sum((pred != c) & (true == c)))
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    if not (tp + fp) or not (tp + fn) or (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return rec, f1


def support_pool_degraded(c, eval_week, flagged_weeks_for_c):
    """Embedded class-c samples from EARLIER degraded weeks (R_corr>THR) before
    eval_week. Disjoint from eval_week (no leakage). This is the generic,
    drift-week-free analogue of the 3-class script's post-drift pool: it draws
    only from weeks where the class was itself flagged, so it excludes the stale
    pre-drift cloud (which would pull the new prototype back to the source)."""
    pool = []
    for w in flagged_weeks_for_c:
        if w < eval_week:
            emb, lab = load_week(w)
            pool.append(emb[lab == c])
    return np.concatenate(pool, axis=0) if pool else np.zeros((0, 600))


def knn_joint_predict(emb, protos, valid, flagged, knn_shots, ncm_swapped):
    """Global nearest-class decision. Stable pool = valid protos NOT flagged.
    Flagged classes repaired by NCM contribute their NEW (swapped) prototype to
    the pool; flagged classes in KNN_CLASSES compete via k-NN set-distance
    (k=min(3,K)) at tau=0 (win on raw distance vs nearest stable prototype)."""
    n = emb.shape[0]
    stable_valid = valid.copy()
    for c in flagged:
        stable_valid[c] = False
    sidx = np.where(stable_valid)[0]
    Pst = protos[sidx]
    d_stable = KNN._sq_dist_to_set(emb, Pst)
    best_stable_d = d_stable.min(axis=1)
    pred = sidx[d_stable.argmin(axis=1)]

    # NCM-repaired flagged classes: add their swapped prototype as a competitor
    ncm_flag = [c for c in flagged if c not in KNN_CLASSES]
    if ncm_flag:
        Pncm = np.stack([ncm_swapped[c] for c in ncm_flag])
        d_ncm = KNN._sq_dist_to_set(emb, Pncm)
        best_ncm_j = d_ncm.argmin(axis=1)
        best_ncm_d = d_ncm[np.arange(n), best_ncm_j]
        best_ncm_c = np.array(ncm_flag)[best_ncm_j]
        win = best_ncm_d <= best_stable_d
        pred[win] = best_ncm_c[win]
        best_stable_d = np.minimum(best_stable_d, best_ncm_d)

    # k-NN flagged classes
    knn_flag = [c for c in flagged if c in KNN_CLASSES]
    for c in knn_flag:
        shots = knn_shots[c]
        kk = min(3, len(shots))
        d_c = KNN.dc_knn(emb, shots, kk)
        win = d_c <= best_stable_d
        pred[win] = c
        best_stable_d = np.minimum(best_stable_d, d_c)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 forward weeks, 1 seed")
    args = ap.parse_args()

    os.makedirs(OUT_RES, exist_ok=True)
    seeds = [SEEDS[0]] if args.quick else SEEDS

    names = sorted(json.load(open(F.LABEL_MAP)).keys())
    n_classes = len(names)

    # --- monitor panel (ref week 16) ---
    P = np.load(PANEL)
    wn = P["week_nums"]
    R = P["R_corr"]
    Nt = P["Ntrue"]
    F1t = P["F1_true"]
    f1ref = P["f1_ref"]
    row_of = {int(wn[i]): i for i in range(len(wn))}

    # --- frozen Week-16 source NCM prototypes ---
    src_emb, src_lab = load_week(REF_WEEK)
    src_protos, src_counts = F.class_prototypes(src_emb, src_lab, n_classes)
    src_valid = src_counts > 0
    all_classes = np.arange(n_classes)

    fwd_weeks = sorted(int(w) for w in wn if w > REF_WEEK and int(w) in row_of)
    if args.quick:
        fwd_weeks = fwd_weeks[1:3]   # two populated weeks (18, 19)

    def flagged_on(w):
        r = row_of[w]
        return [c for c in range(n_classes)
                if R[r, c] > THR and Nt[r, c] >= N_MIN]

    # never-flagged-all-year STABLE set (cumulative poaching check), valid in src
    all_fwd_rows = [row_of[w] for w in sorted(int(x) for x in wn if x > REF_WEEK)]
    stable_classes = np.array(
        [c for c in range(n_classes) if src_valid[c]
         and all(not (R[r, c] > THR and Nt[r, c] >= N_MIN) for r in all_fwd_rows)])

    # precompute, per class, the forward weeks where it is flagged (for the pool)
    flagged_weeks = {c: [w for w in fwd_weeks_all()
                         if R[row_of[w], c] > THR and Nt[row_of[w], c] >= N_MIN]
                     for c in range(n_classes)}

    per_week = {}
    glob_before, glob_after_ncm, glob_after_knn = [], [], []
    stab_before, stab_after_ncm, stab_after_knn = [], [], []
    repaired_class_records = {}   # cname -> list of per-week dicts

    for w in fwd_weeks:
        flagged = flagged_on(w)
        emb_e, lab_e = load_week(w)

        # repairability: post-drift support pool from EARLIER flagged weeks
        repairable, unrepairable = [], []
        pools = {}
        for c in flagged:
            pool = support_pool_degraded(c, w, flagged_weeks[c])
            if pool.shape[0] >= MIN_PER_CLASS:
                repairable.append(c)
                pools[c] = pool
            else:
                unrepairable.append(c)

        # false alarms among flagged (class actually healthy per panel)
        false_alarms = [c for c in flagged
                        if (f1ref[c] - F1t[row_of[w], c]) <= EPS_HEALTHY]

        # ---- BEFORE: source-only global / stable macro-F1 (frozen NCM) --------
        base_pred = F.ncm_predict(emb_e, src_protos, src_valid)
        b_glob = macro_f1_all(lab_e, base_pred, all_classes)
        b_stab = macro_f1_all(lab_e, base_pred, stable_classes)

        # ---- AFTER (ncm_joint) + (knn_joint) over 5 seeds ---------------------
        ncm_g, ncm_s, knn_g, knn_s = [], [], [], []
        rec_ncm = {c: [] for c in repairable}
        f1_ncm = {c: [] for c in repairable}
        rec_knn = {c: [] for c in repairable}
        f1_knn = {c: [] for c in repairable}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rep = src_protos.copy()
            rv = src_valid.copy()
            swapped = {}
            knn_shots = {}
            for c in repairable:
                pool = pools[c]
                idx = rng.choice(len(pool), size=min(K, len(pool)), replace=False)
                shots = pool[idx]
                swapped[c] = shots.mean(axis=0)
                rep[c] = swapped[c]
                rv[c] = True
                knn_shots[c] = shots
            # ncm_joint
            pred_ncm = F.ncm_predict(emb_e, rep, rv)
            ncm_g.append(macro_f1_all(lab_e, pred_ncm, all_classes))
            ncm_s.append(macro_f1_all(lab_e, pred_ncm, stable_classes))
            # knn_joint
            pred_knn = knn_joint_predict(emb_e, src_protos, src_valid,
                                         repairable, knn_shots, swapped)
            knn_g.append(macro_f1_all(lab_e, pred_knn, all_classes))
            knn_s.append(macro_f1_all(lab_e, pred_knn, stable_classes))
            for c in repairable:
                r1, fa = per_class_recall_f1(lab_e, pred_ncm, c)
                r2, fb = per_class_recall_f1(lab_e, pred_knn, c)
                rec_ncm[c].append(r1); f1_ncm[c].append(fa)
                rec_knn[c].append(r2); f1_knn[c].append(fb)

        def ms(x):
            return dict(mean=float(np.nanmean(x)), std=float(np.nanstd(x)))

        glob_before.append(b_glob)
        glob_after_ncm.append(float(np.nanmean(ncm_g)))
        glob_after_knn.append(float(np.nanmean(knn_g)))
        stab_before.append(b_stab)
        stab_after_ncm.append(float(np.nanmean(ncm_s)))
        stab_after_knn.append(float(np.nanmean(knn_s)))

        rep_classes = {}
        for c in repairable:
            cn = names[c]
            rec = per_class_recall_f1(lab_e, base_pred, c)
            entry = dict(
                class_idx=int(c), name=cn,
                n_eval=int((lab_e == c).sum()),
                n_support_pool=int(pools[c].shape[0]),
                is_false_alarm=bool(c in false_alarms),
                method="knn" if c in KNN_CLASSES else "ncm",
                before_recall=float(rec[0]), before_f1=float(rec[1]),
                after_ncm_recall=ms(rec_ncm[c]), after_ncm_f1=ms(f1_ncm[c]),
                after_knn_recall=ms(rec_knn[c]), after_knn_f1=ms(f1_knn[c]),
            )
            rep_classes[str(c)] = entry
            repaired_class_records.setdefault(cn, []).append(
                dict(week=w, **{k: entry[k] for k in
                                ("after_ncm_recall", "after_ncm_f1",
                                 "before_recall", "is_false_alarm")}))

        per_week[str(w)] = dict(
            week=w,
            n_flagged=len(flagged),
            n_repairable=len(repairable),
            n_unrepairable=len(unrepairable),
            n_false_alarm=len(false_alarms),
            flagged_classes=[names[c] for c in flagged],
            unrepairable_classes=[names[c] for c in unrepairable],
            false_alarm_classes=[names[c] for c in false_alarms],
            n_eval_subsample=int(len(lab_e)),
            before={"global_macroF1_180": b_glob, "stable_macroF1": b_stab},
            after_ncm_joint={
                "global_macroF1_180": ms(ncm_g),
                "stable_macroF1": ms(ncm_s),
                "delta_global": float(np.nanmean(ncm_g) - b_glob),
                "delta_stable": float(np.nanmean(ncm_s) - b_stab)},
            after_knn_joint={
                "global_macroF1_180": ms(knn_g),
                "stable_macroF1": ms(knn_s),
                "delta_global": float(np.nanmean(knn_g) - b_glob),
                "delta_stable": float(np.nanmean(knn_s) - b_stab)},
            repaired_classes=rep_classes,
        )
        print(f"[week {w:>2}] flagged={len(flagged):>3} repairable={len(repairable):>3} "
              f"unrepairable={len(unrepairable):>3} false_alarm={len(false_alarms):>3} | "
              f"globalF1 {b_glob:.4f} -> ncm {np.nanmean(ncm_g):.4f} "
              f"(Δ{np.nanmean(ncm_g)-b_glob:+.4f}) knn {np.nanmean(knn_g):.4f} "
              f"(Δ{np.nanmean(knn_g)-b_glob:+.4f}) | stable {b_stab:.4f}->"
              f"{np.nanmean(ncm_s):.4f}")

    def yr(x):
        return dict(mean=float(np.nanmean(x)), std=float(np.nanstd(x)),
                    n_weeks=len(x))

    summary = {
        "experiment": "A_prime_all_flagged_joint_repair_global_macroF1_forward_year",
        "source_week": REF_WEEK,
        "forward_weeks": fwd_weeks,
        "threshold": THR, "n_min": N_MIN, "eps_healthy": EPS_HEALTHY,
        "K": K, "min_per_class": MIN_PER_CLASS, "seeds": seeds,
        "knn_classes": sorted(KNN_CLASSES),
        "n_classes": n_classes,
        "n_stable_neverflagged_classes": int(len(stable_classes)),
        "stable_classes": [names[c] for c in stable_classes],
        "caveat": ("10%-subsample embeddings; frozen Week-16 NCM prototypes; "
                   "macro-F1 over 180 classes (empty/low-support scored 0, same "
                   "convention before & after); support pool = earlier flagged "
                   "weeks (post-drift), disjoint from eval week."),
        "year_mean": {
            "global_macroF1_180_before": yr(glob_before),
            "global_macroF1_180_after_ncm_joint": yr(glob_after_ncm),
            "global_macroF1_180_after_knn_joint": yr(glob_after_knn),
            "delta_global_ncm_joint": float(np.nanmean(glob_after_ncm)
                                            - np.nanmean(glob_before)),
            "delta_global_knn_joint": float(np.nanmean(glob_after_knn)
                                            - np.nanmean(glob_before)),
            "stable_macroF1_before": yr(stab_before),
            "stable_macroF1_after_ncm_joint": yr(stab_after_ncm),
            "stable_macroF1_after_knn_joint": yr(stab_after_knn),
            "delta_stable_ncm_joint": float(np.nanmean(stab_after_ncm)
                                            - np.nanmean(stab_before)),
            "delta_stable_knn_joint": float(np.nanmean(stab_after_knn)
                                            - np.nanmean(stab_before)),
            "mean_flagged_per_week": float(np.mean(
                [per_week[k]["n_flagged"] for k in per_week])),
            "mean_repairable_per_week": float(np.mean(
                [per_week[k]["n_repairable"] for k in per_week])),
            "mean_unrepairable_per_week": float(np.mean(
                [per_week[k]["n_unrepairable"] for k in per_week])),
            "mean_false_alarm_per_week": float(np.mean(
                [per_week[k]["n_false_alarm"] for k in per_week])),
        },
        "per_week": per_week,
        "repaired_class_history": repaired_class_records,
    }

    out = os.path.join(OUT_RES, "metrics.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    ym = summary["year_mean"]
    print("\n" + "=" * 80)
    print("ALL-FLAGGED JOINT REPAIR -> GLOBAL macro-F1, forward YEAR "
          f"(weeks {fwd_weeks[0]}..{fwd_weeks[-1]}, {len(fwd_weeks)} weeks)")
    print("=" * 80)
    print(f"mean flagged/week     = {ym['mean_flagged_per_week']:.1f}")
    print(f"mean repairable/week  = {ym['mean_repairable_per_week']:.1f}")
    print(f"mean unrepairable/wk  = {ym['mean_unrepairable_per_week']:.1f}")
    print(f"mean false-alarm/week = {ym['mean_false_alarm_per_week']:.1f}")
    print(f"\nGLOBAL macro-F1 (180), year mean:")
    print(f"  BEFORE (source-only)   = {ym['global_macroF1_180_before']['mean']:.4f}")
    print(f"  AFTER  ncm_joint       = {ym['global_macroF1_180_after_ncm_joint']['mean']:.4f}"
          f"  (Δ {ym['delta_global_ncm_joint']:+.4f})")
    print(f"  AFTER  knn_joint       = {ym['global_macroF1_180_after_knn_joint']['mean']:.4f}"
          f"  (Δ {ym['delta_global_knn_joint']:+.4f})")
    print(f"\nSTABLE macro-F1 ({len(stable_classes)} never-flagged), year mean (poaching check):")
    print(f"  BEFORE = {ym['stable_macroF1_before']['mean']:.4f}  "
          f"AFTER ncm = {ym['stable_macroF1_after_ncm_joint']['mean']:.4f}  "
          f"(Δ {ym['delta_stable_ncm_joint']:+.4f})")
    print("\nwrote", out)


def fwd_weeks_all():
    P = np.load(PANEL)
    wn = P["week_nums"]
    return sorted(int(w) for w in wn if w > REF_WEEK)


if __name__ == "__main__":
    main()
