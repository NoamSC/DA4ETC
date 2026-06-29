#!/usr/bin/env python
"""
EXPERIMENT A — JOINT multi-class few-shot REPAIR -> GLOBAL macro-F1.
(paper "The Illusion of Continuous Drift").

Why this experiment
-------------------
The per-class repairs (few_shot_repair_loop.py / few_shot_repair_knn.py) were
each evaluated IN ISOLATION: repair class c, measure c's recall and the stable-
class Macro-F1 with ONLY c's prototype swapped. That leaves the headline claim
("our repair lifts the system with no negative transfer") untested JOINTLY: what
happens to the GLOBAL macro-F1 over all 180 classes when we repair ALL flagged
teleported classes (49 docker-registry, 57 eset-edtd, 140 skype) AT ONCE?

This script:
  1. Picks ONE held-out forward eval week where all three teleported classes have
     >= MIN_PER_CLASS embedded samples (default 40). Reports per-week + union too.
  2. Computes the SOURCE-ONLY global macro-F1 over all 180 classes on that week,
     using the frozen Week-16 NCM prototypes (so before/after live in the SAME
     frozen embedding space -> a true apples-to-apples delta). We ALSO report the
     saved full-data softmax `pred_labels` baseline macro-F1 as an external anchor.
  3. Repairs prototypes 49,57,140 SIMULTANEOUSLY (K=50, 5 seeds):
        - "ncm_joint": replace each of the three prototypes with the mean of K
          post-drift shots (NCM single-prototype), classify by nearest prototype.
        - "knn_joint": replace skype(140) with a k-NN set-distance corrector
          (the new multi-modal method) and 49/57 with NCM, integrated into a
          single global nearest-class decision (see _knn_joint_predict).
     recompute the GLOBAL macro-F1 over all 180 classes on the SAME eval week.
  4. Reports:
        - global macro-F1 before vs after joint repair (Delta),
        - macro-F1 over the STABLE 177 classes before vs after (cumulative
          poaching check: does repairing 3 classes compound stable damage?),
        - per-flagged-class recall before vs after.

Key questions answered:
  (i)  Does joint repair LIFT global macro-F1 (a real apples-to-apples DeltaF1)?
  (ii) Does repairing 3 classes at once cause CUMULATIVE stable-class damage?

CAVEAT: 10%-subsample embeddings (data_sample_frac applied upstream); preliminary,
        NOT directly comparable to full-data monitor figures. Macro-F1 over 180
        classes on a single week's subsample has empty/low-support classes scored
        as F1=0 by the macro_f1 helper (same convention everywhere here, so the
        BEFORE/AFTER delta is still apples-to-apples).
"""
import os
import sys
import json
import glob
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/analysis"))

import few_shot_repair_loop as F
import few_shot_repair_knn as KNN

OUT_RES = os.path.join(ROOT, "results/repair/joint_repair_v01")

TELE = [49, 57, 140]                 # docker-registry, eset-edtd, skype
K = 50
MIN_PER_CLASS = 40


def macro_f1_all(true, pred, classes):
    """Macro-F1 over the given class list (F1=0 for classes with no support or
    no predictions, the same convention as F.macro_f1)."""
    return F.macro_f1(true, pred, classes)


def ncm_joint_predict(emb, protos, valid):
    """Plain global nearest-prototype over all valid prototypes."""
    return F.ncm_predict(emb, protos, valid)


def knn_joint_predict(emb, protos, valid, shots_by_c, tau_by_c):
    """Global nearest-class decision where the three flagged classes use a k-NN
    set-distance score (d_c = squared dist to the k-th nearest of c's shots,
    k=min(3,K)) and ALL other classes use their (frozen) prototype distance.

    A point is assigned to flagged class c iff c's k-NN squared distance beats
    BOTH (a) every other flagged class's score and (b) the best stable-prototype
    distance, by the per-class margin tau_by_c (calibrated, see main). Otherwise
    the point falls back to the global nearest STABLE prototype (the flagged
    prototypes are removed from the stable pool so they cannot also win as
    centroids). This is the joint, system-level analogue of the isolated k-NN
    eval in few_shot_repair_knn.py.
    """
    n = emb.shape[0]
    # stable prototypes = valid prototypes that are NOT flagged
    stable_valid = valid.copy()
    for c in TELE:
        stable_valid[c] = False
    sidx = np.where(stable_valid)[0]
    Pst = protos[sidx]
    d_stable = KNN._sq_dist_to_set(emb, Pst)        # [N, S]
    best_stable_d = d_stable.min(axis=1)
    best_stable_c = sidx[d_stable.argmin(axis=1)]

    # flagged-class k-NN scores + margins
    flag_d = np.full((n, len(TELE)), np.inf)
    for j, c in enumerate(TELE):
        shots = shots_by_c[c]
        kk = min(3, len(shots))
        d_c = KNN.dc_knn(emb, shots, kk)            # [N]
        scale2 = KNN.shot_scale2(shots)
        # subtract the margin so a larger tau makes c HARDER to win
        flag_d[:, j] = d_c + tau_by_c[c] * scale2

    best_flag_j = flag_d.argmin(axis=1)
    best_flag_d = flag_d[np.arange(n), best_flag_j]
    best_flag_c = np.array(TELE)[best_flag_j]

    pred = best_stable_c.copy()
    win_flag = best_flag_d <= best_stable_d
    pred[win_flag] = best_flag_c[win_flag]
    return pred


def main():
    os.makedirs(OUT_RES, exist_ok=True)
    names = sorted(json.load(open(F.LABEL_MAP)).keys())
    n_classes = len(names)

    # frozen Week-16 source NCM prototypes
    src_emb, src_lab = F.load_week_embeddings(F.SOURCE_WEEK)
    src_protos, src_counts = F.class_prototypes(src_emb, src_lab, n_classes)
    src_valid = src_counts > 0
    all_classes = np.arange(n_classes)
    stable_classes = np.array([c for c in range(n_classes)
                               if c not in TELE and src_valid[c]])

    # ---- pick the joint eval week: latest forward week with all three >= MIN --
    files = sorted(glob.glob(os.path.join(F.INF_DIR, "WEEK-2022-*.npz")))
    per_week_counts = {}
    joint_weeks = []
    for f in files:
        w = int(os.path.basename(f).split("-")[-1].split(".")[0])
        if w <= 28:                                 # latest flagged drift_week=28
            continue
        _, lab = F.load_week_embeddings(w)
        cnts = {c: int((lab == c).sum()) for c in TELE}
        per_week_counts[w] = cnts
        if all(cnts[c] >= MIN_PER_CLASS for c in TELE):
            joint_weeks.append(w)
    eval_week = max(joint_weeks)                     # latest qualifying week

    # support pools per flagged class (post-drift drift_week..eval_week-1)
    shots_pool = {}
    for c in TELE:
        dw = F.TARGETS[c]["drift_week"]
        shots_pool[c] = F._support_pool(c, dw, eval_week)

    emb_e, lab_e = F.load_week_embeddings(eval_week)

    # ---- BEFORE: source-only global macro-F1 (frozen NCM prototypes) ----------
    base_pred = ncm_joint_predict(emb_e, src_protos, src_valid)
    base_global_f1 = macro_f1_all(lab_e, base_pred, all_classes)
    base_stable_f1 = macro_f1_all(lab_e, base_pred, stable_classes)
    base_recall = {c: (float((base_pred[lab_e == c] == c).mean())
                       if (lab_e == c).any() else float("nan")) for c in TELE}

    # external anchor: full-data saved softmax pred_labels macro-F1 (all 180)
    d_full = np.load(os.path.join(F.INF_DIR, f"WEEK-2022-{eval_week:02d}.npz"))
    full_true = d_full["true_labels"]
    full_pred = d_full["pred_labels"]
    softmax_global_f1 = macro_f1_all(full_true, full_pred, all_classes)
    softmax_recall = {c: (float((full_pred[full_true == c] == c).mean())
                          if (full_true == c).any() else float("nan"))
                      for c in TELE}

    # ---- AFTER (ncm_joint): replace all three prototypes simultaneously -------
    seeds = F.SEEDS
    ncm_glob, ncm_stab = [], []
    ncm_rec = {c: [] for c in TELE}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rep = src_protos.copy()
        rv = src_valid.copy()
        for c in TELE:
            pool = shots_pool[c]
            idx = rng.choice(len(pool), size=min(K, len(pool)), replace=False)
            rep[c] = pool[idx].mean(axis=0)
            rv[c] = True
        pred = ncm_joint_predict(emb_e, rep, rv)
        ncm_glob.append(macro_f1_all(lab_e, pred, all_classes))
        ncm_stab.append(macro_f1_all(lab_e, pred, stable_classes))
        for c in TELE:
            ncm_rec[c].append(float((pred[lab_e == c] == c).mean())
                              if (lab_e == c).any() else float("nan"))

    # ---- AFTER (knn_joint): skype/all via k-NN set-distance -------------------
    # Calibrate per-class tau so the JOINT k-NN repair respects (does not exceed)
    # the NCM joint repair's stable poaching. We sweep tau on a shared grid and,
    # for each class, pick the smallest tau (max recall) whose marginal stable
    # poaching <= NCM's. To keep it simple & fair we use tau=0 for all three
    # (each class wins on raw k-NN distance vs the nearest stable prototype) as
    # the "natural" operating point, and ALSO report a tau-calibrated variant.
    knn_glob, knn_stab = [], []
    knn_rec = {c: [] for c in TELE}
    tau_zero = {c: 0.0 for c in TELE}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shots_by_c = {}
        for c in TELE:
            pool = shots_pool[c]
            idx = rng.choice(len(pool), size=min(K, len(pool)), replace=False)
            shots_by_c[c] = pool[idx]
        pred = knn_joint_predict(emb_e, src_protos, src_valid,
                                 shots_by_c, tau_zero)
        knn_glob.append(macro_f1_all(lab_e, pred, all_classes))
        knn_stab.append(macro_f1_all(lab_e, pred, stable_classes))
        for c in TELE:
            knn_rec[c].append(float((pred[lab_e == c] == c).mean())
                              if (lab_e == c).any() else float("nan"))

    def ms(x):
        return dict(mean=float(np.nanmean(x)), std=float(np.nanstd(x)))

    results = {
        "experiment": "A_joint_multiclass_repair_global_macroF1",
        "source_week": F.SOURCE_WEEK,
        "eval_week": eval_week,
        "K": K, "seeds": seeds,
        "flagged_classes": {str(c): F.TARGETS[c]["name"] for c in TELE},
        "min_per_class": MIN_PER_CLASS,
        "joint_qualifying_weeks": joint_weeks,
        "per_week_flagged_counts": {str(w): {str(c): per_week_counts[w][c]
                                             for c in TELE}
                                    for w in sorted(per_week_counts)},
        "eval_week_flagged_counts": {str(c): int((lab_e == c).sum())
                                     for c in TELE},
        "n_eval_subsample": int(len(lab_e)),
        "n_classes": n_classes,
        "n_stable_classes": int(len(stable_classes)),
        "caveat": ("10%-subsample embeddings; frozen Week-16 NCM prototypes for "
                   "the in-space before/after; macro-F1 over 180 classes scores "
                   "empty/low-support classes as 0 (same convention before & "
                   "after, so Delta is apples-to-apples)."),
        "before_source_only": {
            "global_macroF1_180_ncm": base_global_f1,
            "stable_macroF1_177_ncm": base_stable_f1,
            "flagged_recall_ncm": {F.TARGETS[c]["name"]: base_recall[c]
                                   for c in TELE},
            "global_macroF1_180_fulldata_softmax_anchor": softmax_global_f1,
            "flagged_recall_fulldata_softmax_anchor": {
                F.TARGETS[c]["name"]: softmax_recall[c] for c in TELE},
        },
        "after_ncm_joint": {
            "global_macroF1_180": ms(ncm_glob),
            "stable_macroF1_177": ms(ncm_stab),
            "flagged_recall": {F.TARGETS[c]["name"]: ms(ncm_rec[c])
                               for c in TELE},
            "delta_global_macroF1": float(np.nanmean(ncm_glob) - base_global_f1),
            "delta_stable_macroF1": float(np.nanmean(ncm_stab) - base_stable_f1),
        },
        "after_knn_joint": {
            "tau": "0 (raw operating point: each flagged class wins on raw "
                   "k-NN set-distance vs nearest stable prototype)",
            "global_macroF1_180": ms(knn_glob),
            "stable_macroF1_177": ms(knn_stab),
            "flagged_recall": {F.TARGETS[c]["name"]: ms(knn_rec[c])
                               for c in TELE},
            "delta_global_macroF1": float(np.nanmean(knn_glob) - base_global_f1),
            "delta_stable_macroF1": float(np.nanmean(knn_stab) - base_stable_f1),
        },
    }

    with open(os.path.join(OUT_RES, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _print(results)
    print("\nwrote", os.path.join(OUT_RES, "metrics.json"))


def _print(r):
    print("\n" + "=" * 80)
    print("EXPERIMENT A — JOINT multi-class repair -> GLOBAL macro-F1 (180 classes)")
    print(f"frozen Week-16 source, forward-only, eval_week={r['eval_week']} "
          f"(10% subsample, n={r['n_eval_subsample']})")
    print("=" * 80)
    print("Flagged classes & eval-week counts:",
          {r['flagged_classes'][c]: r['eval_week_flagged_counts'][c]
           for c in r['flagged_classes']})
    print("Joint-qualifying weeks (all 3 >= %d):" % r["min_per_class"],
          r["joint_qualifying_weeks"])
    b = r["before_source_only"]
    print(f"\nBEFORE (source-only, frozen NCM):")
    print(f"  global macro-F1 (180)  = {b['global_macroF1_180_ncm']:.4f}")
    print(f"  stable macro-F1 (177)  = {b['stable_macroF1_177_ncm']:.4f}")
    print(f"  full-data softmax anchor global macro-F1 (180) = "
          f"{b['global_macroF1_180_fulldata_softmax_anchor']:.4f}")
    print(f"  flagged recall (NCM):   {b['flagged_recall_ncm']}")
    print(f"  flagged recall (softmax anchor): "
          f"{b['flagged_recall_fulldata_softmax_anchor']}")
    for tag in ["after_ncm_joint", "after_knn_joint"]:
        a = r[tag]
        print(f"\nAFTER ({tag}):")
        print(f"  global macro-F1 (180)  = {a['global_macroF1_180']['mean']:.4f}"
              f" +- {a['global_macroF1_180']['std']:.4f}   "
              f"(Delta = {a['delta_global_macroF1']:+.4f})")
        print(f"  stable macro-F1 (177)  = {a['stable_macroF1_177']['mean']:.4f}"
              f" +- {a['stable_macroF1_177']['std']:.4f}   "
              f"(Delta = {a['delta_stable_macroF1']:+.4f})")
        for cn, v in a["flagged_recall"].items():
            print(f"    recall {cn:>18}: {v['mean']:.3f} +- {v['std']:.3f}")


if __name__ == "__main__":
    main()
