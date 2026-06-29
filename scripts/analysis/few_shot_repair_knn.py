#!/usr/bin/env python
"""
Few-shot per-class REPAIR with a NON-PARAMETRIC, MULTI-MODAL-AWARE corrector
(paper "The Illusion of Continuous Drift").

Motivation — the single-cluster failure mode
---------------------------------------------
The baseline repair (few_shot_repair_loop.py) collapses the K post-drift shots
of a flagged class c into ONE centroid (NCM single-prototype). That assumes the
post-drift cloud of c is UNIMODAL. For a multi-modal class (skype: two app
versions / endpoints) the single mean lands BETWEEN the sub-clusters, so recall
recovers only partially. The KMeans variant (repair_multiproto_tradeoff.py) fixes
recall but reintroduces negative transfer (poaches stable classes) and needs an
ad-hoc per-class cluster count M.

This script implements two NON-CLUSTERING correctors that represent class c by
the RAW K shots themselves (no centroid, no committed cluster count):

  1. k-NN set-distance:
       score(x -> c) = distance from x to its k-th nearest of c's K shots.
       (k=1 == nearest-shot.) Assign x to c if this distance beats x's distance
       to the nearest STABLE-class prototype, by a margin tau.

  2. Kernel density (Parzen / RBF):
       score(x -> c) = max_i exp(-||x - shot_i||^2 / (2 sigma^2))
       sigma = median pairwise shot distance (median heuristic). Convert to a
       pseudo-distance d = -2 sigma^2 log(score) and compare to the nearest
       stable prototype's squared distance with margin tau.

Both are the M=K limit done softly/robustly without committing to a cluster
count: every shot (or the local density) defines its own neighborhood, so a
multi-modal post-drift class is covered.

Poaching control (calibrated margin tau)
-----------------------------------------
A point x is assigned to c only if c's score beats the nearest stable prototype
by a margin. We sweep tau to trace a recovery-vs-poaching curve, exactly like the
multiproto script's axis. We also report the operating point of tau calibrated so
the repair-induced stable poaching matches the NCM baseline's budget (apples to
apples).

Protocol (IDENTICAL to few_shot_repair_loop.py — reuses its helpers):
  * Frozen Week-16 source; forward-only eval; 10%-subsample saved embeddings.
  * Same TARGETS, SEEDS=[1,2,3,4,42], K_LIST=[1,5,10,50], same eval-week pick,
    same support pool (post-drift weeks drift_week..eval_week-1).

Outputs:
  results/repair/few_shot_knn_v01/metrics.json
  figs/repair/fig_recovery_vs_poaching_knn.png

CAVEAT: 10%-subsample embeddings; preliminary, NOT comparable to full-data
        monitor figures. Same caveat as the baseline harness.
"""
import os
import sys
import json
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/analysis"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reuse the existing harness verbatim (same data loading, prototypes, eval-week
# selection, support pool, metrics, TARGETS, SEEDS, K_LIST).
import few_shot_repair_loop as F

OUT_RES = os.environ.get(
    "REPAIR_KNN_OUT_RES", os.path.join(ROOT, "results/repair/few_shot_knn_v01"))
OUT_FIG = os.environ.get(
    "REPAIR_KNN_OUT_FIG", os.path.join(ROOT, "figs/repair"))
SOURCE_WEEK = F.SOURCE_WEEK

# Margin sweep (multiplier on the per-class distance scale). 0 == "c wins on raw
# score"; positive tau makes c HARDER to win (suppresses poaching); negative tau
# makes c easier to win (raises recall, more poaching).
TAU_GRID = np.linspace(-0.6, 1.2, 31)


# ----------------------------------------------------------------------------
# Distance helpers (all in the frozen 600-d mlp_shared space, Euclidean).
# ----------------------------------------------------------------------------
def _sq_dist_to_set(emb, S):
    """[N,D] x [M,D] -> [N,M] squared Euclidean distances."""
    # ||x-s||^2 = ||x||^2 - 2 x.s + ||s||^2
    xx = (emb * emb).sum(1)[:, None]
    ss = (S * S).sum(1)[None, :]
    return xx - 2.0 * emb @ S.T + ss


def nearest_stable_sqdist(emb, c, src_protos, src_valid):
    """For each x, squared distance to its nearest STABLE prototype (c excluded)."""
    sidx = np.where(src_valid)[0]
    sidx = sidx[sidx != c]
    Pst = src_protos[sidx]
    return _sq_dist_to_set(emb, Pst).min(axis=1)


# ----------------------------------------------------------------------------
# The three c-scoring rules. Each returns d_c [N]: a (squared) distance from x to
# class c. Smaller => more c-like. Assignment to c happens when
#     d_c <= d_stable - tau * scale     (scale = per-class median shot spread^2)
# so a single tau axis is comparable across methods.
# ----------------------------------------------------------------------------
def dc_ncm(emb, shots):
    """Baseline: single centroid of the shots."""
    proto = shots.mean(0, keepdims=True)
    return _sq_dist_to_set(emb, proto)[:, 0]


def dc_kmeans(emb, shots, M, seed):
    """KMeans M sub-prototypes; d_c = min over sub-centroids."""
    if M == 1 or len(shots) <= M:
        sub = shots.mean(0, keepdims=True)
    else:
        sub = KMeans(M, n_init=10, random_state=seed).fit(shots).cluster_centers_
    return _sq_dist_to_set(emb, sub).min(axis=1)


def dc_knn(emb, shots, k):
    """k-NN set-distance: squared distance to the k-th nearest shot."""
    d2 = _sq_dist_to_set(emb, shots)            # [N, K]
    k = min(k, d2.shape[1])
    if k == 1:
        return d2.min(axis=1)
    # k-th smallest (0-indexed k-1) via partial sort
    part = np.partition(d2, k - 1, axis=1)[:, :k]
    return part[:, -1]


def dc_parzen(emb, shots, sigma2):
    """Parzen RBF density -> pseudo squared-distance.
       score = (1/K) sum_i exp(-||x-s_i||^2 / (2 sigma^2))   (MEAN kernel = proper
       Parzen density); d_c = -2 sigma^2 log(score).
       NOTE: the earlier SUM kernel (no 1/K) biased d_c downward by 2 sigma^2 log(K)
       vs the single-prototype d_stable it is compared against, so c won for ~every
       point (99% poaching). Normalising by K removes that constant inflation; d_c
       then -> nearest-shot distance for isolated points and is *smaller* in dense
       (multi-shot) regions -> density-aware, unlike the hard k-NN nearest."""
    d2 = _sq_dist_to_set(emb, shots)            # [N, K]
    K = d2.shape[1]
    # numerically stable log-MEAN-exp of (-d2 / 2sigma2)
    a = -d2 / (2.0 * sigma2)
    amax = a.max(axis=1, keepdims=True)
    logmean = amax[:, 0] + np.log(np.exp(a - amax).sum(axis=1)) - np.log(K)
    return -2.0 * sigma2 * logmean              # pseudo squared distance


def shot_scale2(shots):
    """Per-class squared length scale = (median pairwise shot distance)^2.
       Used as the tau (margin) scale so the margin axis is comparable across
       classes/methods. Robust to a single shot."""
    if len(shots) < 2:
        return 1.0
    d2 = _sq_dist_to_set(shots, shots)
    iu = np.triu_indices(len(shots), k=1)
    med = float(np.median(np.sqrt(np.maximum(d2[iu], 0.0))))
    return max(med, 1e-6) ** 2


def local_scale2(shots):
    """Local Parzen bandwidth^2 = (median NEAREST-NEIGHBOUR shot distance)^2.
       The global median PAIRWISE distance over-smooths multi-modal shots: with the
       mean kernel the density then dilutes across modes and even true class-c points
       lose. The nearest-neighbour scale keeps the kernel LOCAL, so each mode's shots
       dominate the density near that mode (the whole point of a multi-modal repair)."""
    if len(shots) < 2:
        return 1.0
    d2 = _sq_dist_to_set(shots, shots)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.maximum(d2.min(axis=1), 0.0))   # nearest-neighbour dist per shot
    med = float(np.median(nn))
    return max(med, 1e-6) ** 2


# ----------------------------------------------------------------------------
# Evaluate one (method, k, seed): returns recall@1 on eval-c and % stable poached
# over the tau grid.
# ----------------------------------------------------------------------------
def eval_method(method, shots, emb_e, eval_c_mask, stable_eval_mask, lab_e,
                row_stable, c, src_protos, src_valid, k_for_knn, seed):
    scale2 = shot_scale2(shots)
    if method == "ncm":
        d_c = dc_ncm(emb_e, shots)
    elif method.startswith("kmeans"):
        M = int(method.split("_")[1])
        d_c = dc_kmeans(emb_e, shots, M, seed)
    elif method == "knn":
        d_c = dc_knn(emb_e, shots, k_for_knn)
    elif method == "parzen":
        d_c = dc_parzen(emb_e, shots, local_scale2(shots))
    else:
        raise ValueError(method)

    d_stable = nearest_stable_sqdist(emb_e, c, src_protos, src_valid)

    n_eval_c = int(eval_c_mask.sum())
    n_stable = int(stable_eval_mask.sum())
    recs, poaches = [], []
    for tau in TAU_GRID:
        thr = d_stable - tau * scale2          # assign to c iff d_c <= thr
        win_c = d_c <= thr
        rec = float(win_c[eval_c_mask].mean()) if n_eval_c else float("nan")
        poach = 100.0 * float(win_c[stable_eval_mask].sum()) / n_stable \
            if n_stable else float("nan")
        recs.append(rec)
        poaches.append(poach)
    return np.array(recs), np.array(poaches)


def interp_recall_at_poach(recs, poaches, target_poach):
    """Recall at the operating point where poaching == target_poach (the NCM
       budget). poaches is monotone-decreasing in tau (recs too). We find, among
       tau points with poach <= target_poach, the MAX recall (best recovery that
       still respects the budget)."""
    ok = poaches <= target_poach + 1e-9
    if not ok.any():
        return float("nan")
    return float(np.nanmax(recs[ok]))


def main():
    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)

    names = sorted(json.load(open(F.LABEL_MAP)).keys())
    n_classes = len(names)

    src_emb, src_lab = F.load_week_embeddings(SOURCE_WEEK)
    src_protos, src_counts = F.class_prototypes(src_emb, src_lab, n_classes)
    src_valid = src_counts > 0

    teleported = {c for c, m in F.TARGETS.items() if m["teleported"]}
    stable_classes = np.array(
        [c for c in range(n_classes) if c not in teleported and src_valid[c]])

    METHODS = ["ncm", "kmeans_4", "knn", "parzen"]
    METHOD_LABEL = {"ncm": "NCM single-proto (baseline)",
                    "kmeans_4": "KMeans M=4 (clustering)",
                    "knn": "k-NN set-distance (new)",
                    "parzen": "Parzen RBF density (new)"}

    results = {
        "source_week": SOURCE_WEEK,
        "k_list": F.K_LIST, "seeds": F.SEEDS,
        "methods": METHODS, "method_label": METHOD_LABEL,
        "tau_grid": TAU_GRID.tolist(),
        "knn_k_rule": "k = min(3, K)",
        "parzen_sigma_rule": "sigma = median pairwise shot distance (median heuristic)",
        "caveat": ("10%-subsample embeddings; preliminary, NOT comparable to "
                   "full-data monitor figures."),
        "classes": {},
    }

    for c, meta in F.TARGETS.items():
        cname = meta["name"]
        ew = F._pick_eval_week(c, meta)
        emb_e, lab_e = F.load_week_embeddings(ew)
        row_stable = stable_classes[stable_classes != c]
        eval_c_mask = lab_e == c
        stable_eval_mask = np.isin(lab_e, row_stable)
        n_eval_c = int(eval_c_mask.sum())
        n_stable = int(stable_eval_mask.sum())

        sup_emb = F._support_pool(c, meta["drift_week"], ew)
        n_pool = sup_emb.shape[0]

        per_k = {}
        for k in F.K_LIST:
            if n_pool < k:
                continue
            knn_k = min(3, k)
            # collect per-method curves over seeds
            curves = {m: {"rec": [], "poach": []} for m in METHODS}
            for seed in F.SEEDS:
                rng = np.random.default_rng(seed)
                shots = sup_emb[rng.choice(n_pool, size=k, replace=False)]
                for m in METHODS:
                    rec, poa = eval_method(
                        m, shots, emb_e, eval_c_mask, stable_eval_mask, lab_e,
                        row_stable, c, src_protos, src_valid, knn_k, seed)
                    curves[m]["rec"].append(rec)
                    curves[m]["poach"].append(poa)

            # NCM budget at its "natural" operating point tau=0 (c wins on raw
            # nearest-prototype distance, the baseline's actual rule). Mean over
            # seeds. This is the poaching budget every method must respect.
            tau0 = int(np.argmin(np.abs(TAU_GRID - 0.0)))
            ncm_rec_seedmean = np.mean([r[tau0] for r in curves["ncm"]["rec"]])
            ncm_poach_budget = float(
                np.mean([p[tau0] for p in curves["ncm"]["poach"]]))

            method_rows = {}
            for m in METHODS:
                recs = np.array(curves[m]["rec"])      # [S, T]
                poas = np.array(curves[m]["poach"])    # [S, T]
                rec_mean = recs.mean(0)
                rec_std = recs.std(0)
                poa_mean = poas.mean(0)
                # operating point at the NCM budget, per seed then aggregate
                op_recs = [interp_recall_at_poach(recs[s], poas[s],
                                                  ncm_poach_budget)
                           for s in range(len(F.SEEDS))]
                # natural raw operating point tau=0 (no margin)
                raw_rec = recs[:, tau0]
                raw_poa = poas[:, tau0]
                method_rows[m] = dict(
                    rec_curve_mean=rec_mean.tolist(),
                    rec_curve_std=rec_std.tolist(),
                    poach_curve_mean=poa_mean.tolist(),
                    recall_at_ncm_budget_mean=float(np.nanmean(op_recs)),
                    recall_at_ncm_budget_std=float(np.nanstd(op_recs)),
                    raw_recall_mean=float(np.mean(raw_rec)),
                    raw_recall_std=float(np.std(raw_rec)),
                    raw_poach_mean=float(np.mean(raw_poa)),
                    raw_poach_std=float(np.std(raw_poa)),
                )

            per_k[str(k)] = dict(
                ncm_poach_budget_pct=ncm_poach_budget,
                ncm_raw_recall_mean=float(ncm_rec_seedmean),
                methods=method_rows,
            )

        results["classes"][cname] = dict(
            class_idx=int(c), teleported=meta["teleported"],
            drift_week=meta["drift_week"], eval_week=ew,
            n_support_pool=int(n_pool), n_eval_c=n_eval_c,
            n_stable_eval=n_stable,
            per_k=per_k,
        )

    with open(os.path.join(OUT_RES, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _print_table(results)
    _plot_recovery_vs_poach(results)
    print("\nwrote", os.path.join(OUT_RES, "metrics.json"))


# ----------------------------------------------------------------------------
def _print_table(res):
    METHODS = res["methods"]
    lbl = res["method_label"]
    print("\n" + "=" * 92)
    print("FEW-SHOT NON-CLUSTERING REPAIR (k-NN / Parzen) vs NCM / KMeans")
    print("frozen Week-16 source, forward-only, 10% embeddings")
    print("=" * 92)
    print("\nRECALL@1 at the NCM poaching budget (fair, equal-poach operating "
          "point); mean+-std over seeds")
    for cname, cd in res["classes"].items():
        tag = "TELEPORTED" if cd["teleported"] else "control(not teleported)"
        mm = "  <-- MULTI-MODAL HEADLINE" if cname == "skype" else ""
        print(f"\n### {cname} (cls {cd['class_idx']}, {tag}){mm}")
        print(f"  eval_wk={cd['eval_week']} support_pool={cd['n_support_pool']} "
              f"eval_c={cd['n_eval_c']}")
        hdr = f"  {'k':>3} | {'budget%':>7} |"
        for m in METHODS:
            hdr += f" {m:>10} |"
        print(hdr)
        for k in res["k_list"]:
            d = cd["per_k"].get(str(k))
            if d is None:
                continue
            row = f"  {k:>3} | {d['ncm_poach_budget_pct']:>6.2f}% |"
            for m in METHODS:
                v = d["methods"][m]["recall_at_ncm_budget_mean"]
                s = d["methods"][m]["recall_at_ncm_budget_std"]
                row += f" {v:.2f}+-{s:.2f} |"
            print(row)
    print("\nLegend:", "  ".join(f"{m}={lbl[m]}" for m in METHODS))
    print("\nRAW operating point (tau=0, each method's natural rule): "
          "recall / stable-poach%")
    for cname, cd in res["classes"].items():
        print(f"\n### {cname}")
        for k in res["k_list"]:
            d = cd["per_k"].get(str(k))
            if d is None:
                continue
            print(f"  k={k:>3}: " + "  ".join(
                f"{m}={d['methods'][m]['raw_recall_mean']:.2f}/"
                f"{d['methods'][m]['raw_poach_mean']:.2f}%" for m in METHODS))


def _plot_recovery_vs_poach(res):
    """Recovery-vs-poaching curves (recall on y, stable-poach% on x) for the
       teleported classes, K=50, one panel per class + skype highlighted."""
    METHODS = res["methods"]
    lbl = res["method_label"]
    colors = {"ncm": "#7f7f7f", "kmeans_4": "#2ca02c",
              "knn": "#1f77b4", "parzen": "#d62728"}
    tele = [(n, d) for n, d in res["classes"].items() if d["teleported"]]
    ncols = len(tele)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.8), sharey=True)
    if ncols == 1:
        axes = [axes]
    K_SHOW = 50
    for ax, (cname, cd) in zip(axes, tele):
        d = cd["per_k"].get(str(K_SHOW)) or cd["per_k"].get(
            str(max(int(x) for x in cd["per_k"])))
        kused = K_SHOW if str(K_SHOW) in cd["per_k"] else \
            max(int(x) for x in cd["per_k"])
        budget = d["ncm_poach_budget_pct"]
        for m in METHODS:
            rec = np.array(d["methods"][m]["rec_curve_mean"])
            poa = np.array(d["methods"][m]["poach_curve_mean"])
            ax.plot(poa, rec, "-o", ms=3, lw=1.8, color=colors[m],
                    label=lbl[m], alpha=0.9)
        ax.axvline(budget, color="black", ls=":", lw=1.4,
                   label=f"NCM budget ({budget:.2f}%)")
        ttl = cname + ("  [MULTI-MODAL]" if cname == "skype" else "")
        ax.set_title(f"{ttl}  (K={kused})", fontsize=10,
                     fontweight="bold" if cname == "skype" else "normal")
        ax.set_xlabel("stable data poached (%)  -> negative transfer")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("flagged-class recall@1")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Non-clustering repair (k-NN / Parzen) vs NCM / KMeans: "
                 "recovery vs poaching\n(frozen Week-16 source, forward-only; "
                 "left/up = better)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_FIG, "fig_recovery_vs_poaching_knn.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
