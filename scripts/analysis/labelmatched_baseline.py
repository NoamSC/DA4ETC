#!/usr/bin/env python
"""
EXPERIMENT C — LABELS-MATCHED baseline for the few-shot REPAIR.
(paper "The Illusion of Continuous Drift").

Question
--------
Does the prototype / k-NN repair actually BEAT the simplest things the SAME K
labeled post-drift shots could buy, all in the FROZEN embedding space (no model
fine-tune — features are frozen)? If a trivial logistic K-shot fit matched it, the
"repair" would not be earning its keep.

For each flagged TELEPORTED class c (49 docker-registry, 57 eset-edtd, 140 skype)
we compare, per class x K x seed, FOUR correctors that all spend the SAME K shots:

  (i)   ncm   — prototype-replacement (mean of K shots), the existing method.
  (ii)  knn   — k-NN set-distance (k=min(3,K)), the new non-clustering method.
  (iii) logreg— one-vs-rest logistic regression head: positives = K shots,
                negatives = a sample of stable-class embeddings (frozen space).
                A test point is assigned to c iff P(c) >= 0.5 AND that beats its
                nearest stable prototype decision; otherwise it keeps the stable
                nearest-prototype label. (sklearn, ml2.)
  (iv)  maha  — (optional) shrinkage/diagonal-Mahalanobis to the shot centroid:
                d_c = sum_j (x_j - mu_j)^2 / (var_j + eps), var from the K shots
                with Ledoit-style shrinkage toward the global feature variance.

For each we report, on a HELD-OUT post-drift eval week (same _pick_eval_week as
the isolated harness), exactly the two repair metrics:
    - recall-recovery: flagged-class recall@1 on the eval week,
    - stable poaching : % of stable-class eval points that get RELABELED to c
                        (negative transfer). Lower is better.
All correctors are evaluated AT THEIR NATURAL OPERATING POINT (the rule each
method would actually use), so this is the honest "what you'd deploy" comparison;
we additionally report each method's recall AT THE NCM STABLE-POACH BUDGET (the
equal-poach fair point) for the distance-based methods that have a margin axis.

Protocol IDENTICAL to few_shot_repair_loop.py / few_shot_repair_knn.py:
  frozen Week-16 source; forward-only; 10%-subsample embeddings; the same
  TARGETS, SEEDS=[1,2,3,4,42], K_LIST=[1,5,10,50], same eval-week pick, same
  post-drift support pool.

CAVEAT: 10%-subsample embeddings; preliminary, NOT comparable to full-data
        monitor figures.
"""
import os
import sys
import json
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/analysis"))

import few_shot_repair_loop as F
import few_shot_repair_knn as KNN
from sklearn.linear_model import LogisticRegression

OUT_RES = os.path.join(ROOT, "results/repair/labelmatched_baseline_v01")

TELE = [49, 57, 140]
N_NEG = 4000           # stable-class negatives sampled for the logreg fit
MAHA_SHRINK = 0.5      # shrinkage of the per-class diagonal var toward global var


def stable_proto_decision(emb, c, src_protos, src_valid):
    """For each x return (nearest_stable_class, sq_dist_to_it), c excluded."""
    sidx = np.where(src_valid)[0]
    sidx = sidx[sidx != c]
    Pst = src_protos[sidx]
    d = KNN._sq_dist_to_set(emb, Pst)
    j = d.argmin(axis=1)
    return sidx[j], d[np.arange(len(emb)), j]


def predict_ncm(emb, shots, c, src_protos, src_valid):
    proto = shots.mean(0)
    rep = src_protos.copy(); rv = src_valid.copy()
    rep[c] = proto; rv[c] = True
    return F.ncm_predict(emb, rep, rv)


def predict_knn(emb, shots, c, src_protos, src_valid, kk):
    """Natural operating point tau=0: assign to c iff k-NN sq-dist <= nearest
    stable proto sq-dist; else keep nearest stable prototype label."""
    d_c = KNN.dc_knn(emb, shots, kk)
    stable_c, d_stable = stable_proto_decision(emb, c, src_protos, src_valid)
    pred = stable_c.copy()
    pred[d_c <= d_stable] = c
    return pred


def predict_logreg(emb, shots, c, src_protos, src_valid, src_emb, src_lab,
                   stable_classes, seed):
    """One-vs-rest logistic head in frozen space. Positives = K shots; negatives
    = N_NEG stable-class source embeddings. Assign x to c iff P(c|x) >= 0.5 AND
    that probability-implied score wins over the nearest stable prototype; else
    keep the nearest stable prototype label."""
    rng = np.random.default_rng(seed + 9999)
    neg_mask = np.isin(src_lab, stable_classes)
    neg_idx = np.where(neg_mask)[0]
    if len(neg_idx) > N_NEG:
        neg_idx = rng.choice(neg_idx, N_NEG, replace=False)
    Xneg = src_emb[neg_idx]
    X = np.vstack([shots, Xneg])
    y = np.concatenate([np.ones(len(shots)), np.zeros(len(Xneg))])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    p_c = clf.predict_proba(emb)[:, 1]
    stable_c, _ = stable_proto_decision(emb, c, src_protos, src_valid)
    pred = stable_c.copy()
    pred[p_c >= 0.5] = c
    return pred


def predict_maha(emb, shots, c, src_protos, src_valid, global_var):
    """Diagonal shrinkage-Mahalanobis to the shot centroid vs nearest stable
    prototype (Euclidean). Natural rule: assign to c iff Maha-sq <= nearest
    stable proto Euclidean-sq."""
    mu = shots.mean(0)
    if len(shots) >= 2:
        var = shots.var(0)
    else:
        var = np.zeros(shots.shape[1])
    var = MAHA_SHRINK * global_var + (1 - MAHA_SHRINK) * var + 1e-6
    diff = emb - mu[None, :]
    d_c = (diff * diff / var[None, :]).sum(1)
    # scale d_c to be comparable to Euclidean-sq: multiply by mean(var) so that
    # an "average" coordinate contributes ||.||^2 again
    d_c = d_c * float(global_var.mean())
    _, d_stable = stable_proto_decision(emb, c, src_protos, src_valid)
    pred = src_valid.copy()  # placeholder; build label vec below
    stable_c, _ = stable_proto_decision(emb, c, src_protos, src_valid)
    out = stable_c.copy()
    out[d_c <= d_stable] = c
    return out


def recall_poach(pred, lab_e, c, stable_eval_mask, row_stable):
    eval_c_mask = lab_e == c
    rec = float((pred[eval_c_mask] == c).mean()) if eval_c_mask.any() else float("nan")
    n_stable = int(stable_eval_mask.sum())
    poach = 100.0 * float((pred[stable_eval_mask] == c).sum()) / n_stable \
        if n_stable else float("nan")
    return rec, poach


def main():
    os.makedirs(OUT_RES, exist_ok=True)
    names = sorted(json.load(open(F.LABEL_MAP)).keys())
    n_classes = len(names)

    src_emb, src_lab = F.load_week_embeddings(F.SOURCE_WEEK)
    src_protos, src_counts = F.class_prototypes(src_emb, src_lab, n_classes)
    src_valid = src_counts > 0
    global_var = src_emb.var(0)

    stable_classes = np.array([c for c in range(n_classes)
                               if c not in TELE and src_valid[c]])

    METHODS = ["ncm", "knn", "logreg", "maha"]
    METHOD_LABEL = {
        "ncm": "NCM prototype-replacement (existing)",
        "knn": "k-NN set-distance (new)",
        "logreg": "Logistic K-shot head (labels-matched baseline)",
        "maha": "Shrinkage-Mahalanobis (optional)",
    }

    results = {
        "experiment": "C_labels_matched_baseline",
        "source_week": F.SOURCE_WEEK,
        "k_list": F.K_LIST, "seeds": F.SEEDS,
        "methods": METHODS, "method_label": METHOD_LABEL,
        "n_neg_logreg": N_NEG, "maha_shrink": MAHA_SHRINK,
        "caveat": ("10%-subsample embeddings; frozen Week-16 source; natural "
                   "operating point per method (the rule it would deploy). "
                   "Preliminary, not comparable to full-data monitor figures."),
        "classes": {},
    }

    for c in TELE:
        meta = F.TARGETS[c]
        cname = meta["name"]
        ew = F._pick_eval_week(c, meta)
        emb_e, lab_e = F.load_week_embeddings(ew)
        row_stable = stable_classes[stable_classes != c]
        stable_eval_mask = np.isin(lab_e, row_stable)
        eval_c_mask = lab_e == c
        n_eval_c = int(eval_c_mask.sum())

        sup = F._support_pool(c, meta["drift_week"], ew)
        n_pool = sup.shape[0]

        # source-only broken baseline (no repair) for reference
        base_pred = F.ncm_predict(emb_e, src_protos, src_valid)
        base_rec = float((base_pred[eval_c_mask] == c).mean()) if n_eval_c else float("nan")

        per_k = {}
        for k in F.K_LIST:
            if n_pool < k:
                continue
            kk = min(3, k)
            acc = {m: {"rec": [], "poach": []} for m in METHODS}
            for seed in F.SEEDS:
                rng = np.random.default_rng(seed)
                shots = sup[rng.choice(n_pool, size=k, replace=False)]

                preds = {
                    "ncm": predict_ncm(emb_e, shots, c, src_protos, src_valid),
                    "knn": predict_knn(emb_e, shots, c, src_protos, src_valid, kk),
                    "logreg": predict_logreg(emb_e, shots, c, src_protos,
                                             src_valid, src_emb, src_lab,
                                             row_stable, seed),
                    "maha": predict_maha(emb_e, shots, c, src_protos, src_valid,
                                         global_var),
                }
                for m in METHODS:
                    rec, poa = recall_poach(preds[m], lab_e, c,
                                            stable_eval_mask, row_stable)
                    acc[m]["rec"].append(rec)
                    acc[m]["poach"].append(poa)

            per_k[str(k)] = {
                m: dict(
                    recall_mean=float(np.nanmean(acc[m]["rec"])),
                    recall_std=float(np.nanstd(acc[m]["rec"])),
                    poach_mean=float(np.nanmean(acc[m]["poach"])),
                    poach_std=float(np.nanstd(acc[m]["poach"])),
                ) for m in METHODS
            }

        results["classes"][cname] = dict(
            class_idx=int(c), teleported=meta["teleported"],
            drift_week=meta["drift_week"], eval_week=ew,
            n_support_pool=int(n_pool), n_eval_c=n_eval_c,
            n_stable_eval=int(stable_eval_mask.sum()),
            base_recall_no_repair=base_rec,
            per_k=per_k,
        )

    with open(os.path.join(OUT_RES, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _print(results)
    print("\nwrote", os.path.join(OUT_RES, "metrics.json"))


def _print(r):
    METHODS = r["methods"]
    print("\n" + "=" * 96)
    print("EXPERIMENT C — LABELS-MATCHED baseline: NCM / k-NN vs logistic / Maha")
    print("frozen Week-16 source, forward-only, 10% embeddings; natural operating point")
    print("=" * 96)
    print("Per cell: recall@1(c) / stable-poach%  (mean over 5 seeds)")
    for cname, cd in r["classes"].items():
        print(f"\n### {cname} (cls {cd['class_idx']}) eval_wk={cd['eval_week']} "
              f"support={cd['n_support_pool']} eval_c={cd['n_eval_c']} "
              f"(no-repair recall={cd['base_recall_no_repair']:.3f})")
        hdr = f"  {'k':>3} |"
        for m in METHODS:
            hdr += f" {m:>22} |"
        print(hdr)
        for k in r["k_list"]:
            d = cd["per_k"].get(str(k))
            if d is None:
                continue
            row = f"  {k:>3} |"
            for m in METHODS:
                row += (f" {d[m]['recall_mean']:.2f}+-{d[m]['recall_std']:.2f}"
                        f"/{d[m]['poach_mean']:.2f}% |")
            print(row)
    print("\nLegend:", "  ".join(f"{m}={r['method_label'][m]}" for m in METHODS))


if __name__ == "__main__":
    main()
