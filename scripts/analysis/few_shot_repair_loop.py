#!/usr/bin/env python
"""
Preliminary few-shot REPAIR LOOP — turning the monitor into a closed
diagnose-and-fix system (paper "The Illusion of Continuous Drift").

Story:
  * The (label-free) monitor flags a degraded class c (a teleportation:
    per-class recall cliff, frozen Week-16 source, forward-only eval).
  * The repair uses a FEW LABELS on ONLY class c. We recompute c's
    NEAREST-CLASS-MEAN (NCM) prototype in the FROZEN feature space from a
    k-shot support set drawn from the POST-DRIFT distribution, and leave the
    backbone and every other class prototype UNTOUCHED. Classify by nearest
    prototype.

Design / protocol (matched to the benchmark):
  * Frozen Week-16 source model; frozen forward (future-only) eval.
  * Reuse the saved 10%-subsample embeddings under
    results/inference_auditfix/week_16_vanilla_bs64/ (NO recompute):
      true_labels, pred_labels (full data); embeddings + embedding_indices
      (10% subsample, embedded labels = true_labels[embedding_indices]).
  * k in {1,5,10,50}; >=5 seeds (1,2,3,4,42) for the support-set DRAW.
  * Two first-class metrics:
      (A) RECOVERY: flagged-class recall@1 (NCM) on a HELD-OUT post-drift eval
          set, vs k.
      (B) STABILITY: Macro-F1 over the STABLE classes (must stay flat, Δ<eps)
          — the negative-transfer guard. Repairing c must not poach stable
          classes.
  * Assumption validation: that the teleported class lands in a SEPARABLE,
    near-empty region of the frozen feature space. We measure, on the
    post-drift support distribution of c:
      - 1-NN purity of c's post-drift cloud against stable-class reference
        prototypes (fraction whose nearest *stable* prototype is far / the
        new c-prototype wins),
      - margin = d(x, nearest stable proto) - d(x, repaired c proto),
      - separation = ||repaired c proto - nearest stable proto|| vs the
        median inter-prototype distance.

CAVEATS (honest):
  * 10%-sample embeddings: NCM numbers are NOT comparable to full-data monitor
    figures. This is a PRELIMINARY proof-of-concept.
  * Label-cost note: the MONITOR is label-free; the REPAIR needs a few labels
    on ONE class only (k shots of class c, drawn post-drift).

Outputs (versioned):
  results/repair/few_shot_repair_w16_v01/metrics.json
  figs/repair/fig_recovery_vs_stability_w16.png
  + a per-class markdown table printed to stdout.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INF_DIR = os.path.join(ROOT, "results/inference_auditfix/week_16_vanilla_bs64")
LABEL_MAP = "/home/anatbr/dataset/CESNET-TLS-Year22_v2/label_mapping.json"
OUT_RES = os.path.join(ROOT, "results/repair/few_shot_repair_w16_v01")
OUT_FIG = os.path.join(ROOT, "figs/repair")
SOURCE_WEEK = 16

# CoTTA forward negative transfer (week-16 source) for the contrast line.
# UDA_BENCHMARK_STATUS.md: CoTTA(bs64) Macro-F1 delta vs source-only is the
# headline negative-transfer magnitude (~ -0.06 mean-acc; bs256 ~ -0.014).
COTTA_FWD_NEG_TRANSFER = -0.060

# Teleported / anchored target classes (frozen Week-16 source, forward).
# drift_week = first forward week where the cliff is in effect (post-drift).
# eset-edtd REQUIRED. microsoft-defender kept as a NON-teleported control.
TARGETS = {
    57:  dict(name="eset-edtd",          drift_week=18, teleported=True),
    49:  dict(name="docker-registry",    drift_week=28, teleported=True),
    140: dict(name="skype",              drift_week=28, teleported=True),
    98:  dict(name="microsoft-defender", drift_week=22, teleported=False),
}
K_LIST = [1, 5, 10, 50]
SEEDS = [1, 2, 3, 4, 42]


def load_week_embeddings(week):
    """Return (emb [n,600], labels [n]) for the 10% embedded subsample."""
    f = os.path.join(INF_DIR, f"WEEK-2022-{week:02d}.npz")
    d = np.load(f)
    emb = d["embeddings"].astype(np.float64)
    lab = d["true_labels"][d["embedding_indices"]].astype(np.int64)
    return emb, lab


def class_prototypes(emb, lab, n_classes):
    """Mean embedding per class; NaN rows for absent classes."""
    protos = np.full((n_classes, emb.shape[1]), np.nan)
    counts = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        m = lab == c
        counts[c] = int(m.sum())
        if counts[c] > 0:
            protos[c] = emb[m].mean(axis=0)
    return protos, counts


def ncm_predict(emb, protos, valid_mask):
    """Nearest-prototype (Euclidean) prediction among valid prototypes."""
    valid_idx = np.where(valid_mask)[0]
    P = protos[valid_idx]                       # [V, D]
    # ||x - p||^2 = ||x||^2 - 2 x.p + ||p||^2  (drop ||x||^2, constant per row)
    pp = (P * P).sum(axis=1)                     # [V]
    d2 = -2.0 * emb @ P.T + pp[None, :]          # [N, V]
    pred_local = d2.argmin(axis=1)
    return valid_idx[pred_local]


def macro_f1(true, pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_week", type=int, default=None,
                    help="post-drift week to evaluate on (default: far week 44)")
    args = ap.parse_args()

    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)

    names = sorted(json.load(open(LABEL_MAP)).keys())
    n_classes = len(names)

    # ---- Frozen source prototypes from Week-16 embeddings -----------------
    src_emb, src_lab = load_week_embeddings(SOURCE_WEEK)
    src_protos, src_counts = class_prototypes(src_emb, src_lab, n_classes)
    src_valid = src_counts > 0

    # Stable classes: every class that is NOT a teleported target and has a
    # source prototype. (microsoft-defender stays stable; it is not teleported.)
    teleported = {c for c, m in TARGETS.items() if m["teleported"]}
    stable_classes = np.array(
        [c for c in range(n_classes) if c not in teleported and src_valid[c]]
    )

    eval_week = args.eval_week
    median_proto_dist = _median_inter_proto(src_protos, src_valid)

    results = {
        "source_week": SOURCE_WEEK,
        "k_list": K_LIST, "seeds": SEEDS,
        "cotta_fwd_neg_transfer_macroF1": COTTA_FWD_NEG_TRANSFER,
        "median_inter_prototype_dist": median_proto_dist,
        "label_cost_note": ("Monitor is label-free; repair needs k labels on "
                            "ONE class only (the flagged class)."),
        "caveat": ("10%-subsample embeddings; preliminary, NOT comparable to "
                   "full-data monitor figures."),
        "classes": {},
    }

    for c, meta in TARGETS.items():
        cname = meta["name"]
        # pick eval week: a late post-drift week with enough class-c samples
        ew = eval_week if eval_week is not None else _pick_eval_week(c, meta)
        emb_e, lab_e = load_week_embeddings(ew)

        # Exclude the currently-repaired class c from the stability set for this
        # row: otherwise the guard is self-referential for a class that is itself
        # in stable_classes (e.g. microsoft-defender, c=98, a non-teleported
        # control). The negative-transfer guard must measure the effect on the
        # OTHER stable classes only.
        row_stable = stable_classes[stable_classes != c]

        # Source-only NCM baseline on this eval week (the "broken" state)
        base_pred = ncm_predict(emb_e, src_protos, src_valid)
        base_recall_c = float((base_pred[lab_e == c] == c).mean()) \
            if (lab_e == c).any() else float("nan")
        # source-only stable Macro-F1 on this eval week
        stable_eval_mask = np.isin(lab_e, row_stable)
        base_stable_f1 = macro_f1(lab_e[stable_eval_mask],
                                  base_pred[stable_eval_mask], row_stable)

        # support pool: post-drift class-c embeddings from drift_week..ew-1
        sup_emb = _support_pool(c, meta["drift_week"], ew)
        n_pool = sup_emb.shape[0]

        # held-out eval-c indices (do not overlap the support week range, since
        # support comes from earlier post-drift weeks; eval week is later/disjoint)
        eval_c_mask = lab_e == c
        n_eval_c = int(eval_c_mask.sum())

        per_k = {}
        for k in K_LIST:
            if n_pool < k:
                continue
            rec_seeds, stab_seeds = [], []
            purity_seeds, margin_seeds, sep_seeds = [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_pool, size=k, replace=False)
                new_proto = sup_emb[idx].mean(axis=0)

                rep_protos = src_protos.copy()
                rep_protos[c] = new_proto
                rep_valid = src_valid.copy()
                rep_valid[c] = True

                pred = ncm_predict(emb_e, rep_protos, rep_valid)
                rec_seeds.append(float((pred[eval_c_mask] == c).mean())
                                 if n_eval_c else float("nan"))
                stab_seeds.append(
                    macro_f1(lab_e[stable_eval_mask],
                             pred[stable_eval_mask], row_stable))

                # --- assumption validation (separability of post-drift cloud)
                pv, mv, sv = _separability(
                    emb_e[eval_c_mask], new_proto, c,
                    src_protos, src_valid)
                purity_seeds.append(pv)
                margin_seeds.append(mv)
                sep_seeds.append(sv)

            per_k[str(k)] = dict(
                recall_mean=float(np.nanmean(rec_seeds)),
                recall_std=float(np.nanstd(rec_seeds)),
                stable_f1_mean=float(np.mean(stab_seeds)),
                stable_f1_std=float(np.std(stab_seeds)),
                purity_mean=float(np.mean(purity_seeds)),
                margin_mean=float(np.mean(margin_seeds)),
                sep_to_nearest_stable_mean=float(np.mean(sep_seeds)),
            )

        results["classes"][cname] = dict(
            class_idx=int(c), teleported=meta["teleported"],
            drift_week=meta["drift_week"], eval_week=ew,
            n_support_pool=int(n_pool), n_eval_c=n_eval_c,
            base_recall_c=base_recall_c,
            base_stable_macroF1=base_stable_f1,
            per_k=per_k,
        )

    with open(os.path.join(OUT_RES, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    _plot(results)
    print("\nwrote", os.path.join(OUT_RES, "metrics.json"))


def _median_inter_proto(protos, valid):
    idx = np.where(valid)[0]
    P = protos[idx]
    # subsample for speed if many
    rng = np.random.default_rng(0)
    if len(P) > 120:
        P = P[rng.choice(len(P), 120, replace=False)]
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(P), k=1)
    return float(np.median(d[iu]))


def _pick_eval_week(c, meta):
    """Latest forward week with >=40 embedded class-c samples, > drift_week."""
    files = sorted(glob.glob(os.path.join(INF_DIR, "WEEK-2022-*.npz")))
    best = None
    for f in files:
        w = int(os.path.basename(f).split("-")[-1].split(".")[0])
        if w <= meta["drift_week"]:
            continue
        _, lab = load_week_embeddings(w)
        if int((lab == c).sum()) >= 40 and w >= 44:
            best = w  # take a far, well-populated week
    if best is not None:
        return best
    # fallback: any post-drift week with enough samples
    for f in reversed(files):
        w = int(os.path.basename(f).split("-")[-1].split(".")[0])
        if w <= meta["drift_week"]:
            continue
        _, lab = load_week_embeddings(w)
        if int((lab == c).sum()) >= 30:
            return w
    return meta["drift_week"] + 1


def _support_pool(c, drift_week, eval_week):
    """All embedded class-c samples from drift_week..eval_week-1 (post-drift,
    disjoint from the eval week)."""
    files = sorted(glob.glob(os.path.join(INF_DIR, "WEEK-2022-*.npz")))
    pool = []
    for f in files:
        w = int(os.path.basename(f).split("-")[-1].split(".")[0])
        if drift_week <= w < eval_week:
            emb, lab = load_week_embeddings(w)
            pool.append(emb[lab == c])
    return np.concatenate(pool, axis=0) if pool else np.zeros((0, 600))


def _separability(emb_c, new_proto, c, src_protos, src_valid):
    """On the post-drift class-c cloud emb_c:
       purity  = frac of points whose nearest prototype (new c-proto vs all
                 stable protos) is the new c-proto,
       margin  = mean( d(x, nearest stable proto) - d(x, new c proto) ),
       sep     = ||new c proto - nearest stable proto||.
    """
    stable_idx = np.where(src_valid)[0]
    stable_idx = stable_idx[stable_idx != c]
    Pst = src_protos[stable_idx]                       # [S, D]
    d_st = np.sqrt(((emb_c[:, None, :] - Pst[None, :, :]) ** 2).sum(-1))  # [N,S]
    d_st_min = d_st.min(axis=1)
    d_new = np.sqrt(((emb_c - new_proto[None, :]) ** 2).sum(-1))           # [N]
    purity = float((d_new < d_st_min).mean())
    margin = float((d_st_min - d_new).mean())
    sep = float(np.min(np.sqrt(((new_proto[None, :] - Pst) ** 2).sum(-1))))
    return purity, margin, sep


def _print_table(res):
    print("\n" + "=" * 78)
    print("FEW-SHOT PROTOTYPE REPAIR — recovery vs stability (frozen W16, forward)")
    print("=" * 78)
    for cname, cd in res["classes"].items():
        tag = "TELEPORTED" if cd["teleported"] else "stable-control(not teleported)"
        print(f"\n### {cname} (cls {cd['class_idx']}, {tag})")
        print(f"  drift_wk={cd['drift_week']}  eval_wk={cd['eval_week']}  "
              f"support_pool={cd['n_support_pool']}  eval_c={cd['n_eval_c']}")
        print(f"  BROKEN baseline (source proto): recall@1(c)={cd['base_recall_c']:.3f}  "
              f"stable Macro-F1={cd['base_stable_macroF1']:.3f}")
        print(f"  {'k':>4} | {'recall@1(c)':>16} | {'stable Macro-F1':>18} | "
              f"{'purity':>7} {'margin':>8} {'sep':>7}")
        for k in res["k_list"]:
            d = cd["per_k"].get(str(k))
            if d is None:
                continue
            print(f"  {k:>4} | {d['recall_mean']:.3f}±{d['recall_std']:.3f}    | "
                  f"{d['stable_f1_mean']:.3f}±{d['stable_f1_std']:.3f}     | "
                  f"{d['purity_mean']:.3f} {d['margin_mean']:>8.3f} "
                  f"{d['sep_to_nearest_stable_mean']:>7.3f}")
    print(f"\n  (median inter-prototype distance = "
          f"{res['median_inter_prototype_dist']:.3f})")
    print(f"  CoTTA forward negative transfer (Macro-F1 contrast) = "
          f"{res['cotta_fwd_neg_transfer_macroF1']:+.3f}")
    print(f"  LABEL COST: {res['label_cost_note']}")


def _plot(res):
    tele = [(n, d) for n, d in res["classes"].items() if d["teleported"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax2 = ax.twinx()
    cmap = plt.cm.tab10
    ks = res["k_list"]
    for i, (cname, cd) in enumerate(tele):
        rec = [cd["per_k"].get(str(k), {}).get("recall_mean", np.nan) for k in ks]
        rstd = [cd["per_k"].get(str(k), {}).get("recall_std", 0) for k in ks]
        stab = [cd["per_k"].get(str(k), {}).get("stable_f1_mean", np.nan) for k in ks]
        col = cmap(i)
        ax.errorbar(ks, rec, yerr=rstd, marker="o", lw=2, color=col,
                    label=f"recall@1 {cname}")
        ax.scatter([ks[0]], [cd["base_recall_c"]], marker="x", s=70, color=col,
                   zorder=5)
        ax2.plot(ks, stab, marker="s", ls="--", lw=1.4, color=col, alpha=0.6,
                 label=f"stable Macro-F1 {cname}")

    # stable baseline (source-only) band + CoTTA negative-transfer contrast
    base_stab = np.mean([cd["base_stable_macroF1"] for _, cd in tele])
    ax2.axhline(base_stab, color="gray", ls=":", lw=1.3,
                label="stable Macro-F1 (no repair)")
    ax2.axhline(base_stab + res["cotta_fwd_neg_transfer_macroF1"],
                color="red", ls=":", lw=1.6,
                label=f"CoTTA fwd neg-transfer ({res['cotta_fwd_neg_transfer_macroF1']:+.2f})")

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (shots of the flagged class, post-drift)")
    ax.set_ylabel("flagged-class recall@1 (NCM)")
    ax2.set_ylabel("stable-class Macro-F1 (negative-transfer guard)")
    ax.set_ylim(-0.03, 1.03)
    ax2.set_ylim(min(0.0, base_stab + res["cotta_fwd_neg_transfer_macroF1"] - 0.05),
                 1.0)
    ax.set_title("Few-shot prototype repair: recovery vs stability\n"
                 "(frozen Week-16 source, forward-only; x = broken source-only)")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=7.5, loc="center right", ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_FIG, "fig_recovery_vs_stability_w16.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
