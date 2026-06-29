#!/usr/bin/env python
"""
Few-shot per-class REPAIR on the proprietary closed-world dataset (multimodal
model) — the Allot analogue of the TLS Week-16 repair experiment.

This is a THIN WRAPPER. It reuses, verbatim and without editing, the two TLS
repair harnesses:
    few_shot_repair_loop.py   (NCM single-prototype recovery-vs-stability loop)
    few_shot_repair_knn.py    (k-NN / Parzen non-clustering correctors + NCM/KMeans)
by monkeypatching the data-loading helpers / config of `few_shot_repair_loop`
(imported there as `F`) to read the Allot windowed inference instead of the
CESNET-TLS weekly inference. The repair *method* code is identical, so results
are directly comparable in protocol to the TLS run.

ANONYMIZATION: this dataset is private/closed-world. No paper-facing artifact
produced here uses its internal name — target classes are referred to only by
synthetic indices ("cls<idx>"), and the result dir / figures carry the neutral
tag. Any paper text must call it "a third, proprietary closed-world dataset
(anonymized for review)".

PROTOCOL (mirrors the TLS run, adapted to the Allot timeline)
-------------------------------------------------------------
* Frozen source = the `early` (or `quarter`) training slice; forward-only eval
  over the 48-window Allot timeline. Embeddings are the saved ~1% subsample
  (embedding_indices into true_labels), 600-d mlp_shared space — same schema as
  the TLS inference.
* Healthy reference window = the model's ref_win from the BBSE monitor
  (early=1). Source prototypes are computed there.
* "Degraded" windows = the monitor's is_anomaly==1 set (the natural Sep-2024 ->
  Feb/Mar-2025 domain gap). drift_window = first degraded window.
* Target (flagged / teleported) classes = degraded classes that REMAIN PRESENT
  post-drift with a clear per-class F1 cliff (recall cliff, post-drift cloud
  exists -> repairable). Classes that vanish entirely post-drift (no support,
  no eval samples) are NOT repair targets. A stable high-F1 class is kept as a
  non-teleported control.
* k in {1,5,10,50}; seeds {1,2,3,4,42}; support pool = degraded windows
  [drift_window, eval_window); eval window = latest degraded window with enough
  class-c samples (disjoint from the support pool).

Outputs:
    results/repair/few_shot_repair_allot_v01/metrics.json      (NCM loop)
    results/repair/few_shot_repair_allot_v01/metrics_knn.json  (k-NN/Parzen)
    figs/repair/fig_recovery_vs_stability_allot.png
    figs/repair/fig_recovery_vs_poaching_allot.png

CAVEAT: ~1% subsample embeddings; preliminary, NOT comparable to full-data
monitor figures (same caveat as the TLS harness).
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/analysis"))

import few_shot_repair_loop as F
import few_shot_repair_knn as KNN


# ---------------------------------------------------------------------------
# Allot config (per slice). Targets chosen from the per-class F1 drop analysis
# (degraded vs healthy windows) — degraded-but-present classes only.
# ---------------------------------------------------------------------------
SLICE_TARGETS = {
    "early_eq": {
        # cls : (drift_window, teleported)
        25:  (23, True),    # headline: F1 0.82 -> 0.24, large post-drift support
        33:  (23, True),    # F1 0.94 -> 0.54, very large support
        109: (23, True),    # F1 0.70 -> 0.27
        62:  (23, True),    # F1 0.75 -> 0.36
        83:  (23, True),    # F1 0.47 -> 0.03
        17:  (23, False),   # stable control: F1 0.96 -> 0.94 (no teleportation)
    },
    "quarter_eq": {
        # populated lazily from the monitor at runtime if --slice quarter_eq
    },
}
EVAL_MIN = 15  # min embedded class-c samples for an eval window


def build(slice_name, quick=False):
    infd = os.path.join(ROOT, "exps/allot_multimodal", slice_name, "inference")
    label_map = os.path.join(ROOT, "exps/allot_multimodal", slice_name,
                             "label_mapping.json")
    monitor = json.load(open(os.path.join(
        ROOT, "results/allot_monitor_v01/metrics.json")))["models"][slice_name]
    is_anom = {int(k): int(v) for k, v in monitor["is_anomaly"].items()}
    ref_win = int(monitor["ref_win"])

    files = {int(os.path.basename(f).split("_")[1].split(".")[0]): f
             for f in glob.glob(os.path.join(infd, "window_*.npz"))}
    degraded = set(w for w in files if is_anom.get(w, 0) == 1)

    targets_cfg = SLICE_TARGETS[slice_name]
    if not targets_cfg:
        raise SystemExit(f"No target config for slice {slice_name}; add it to "
                         "SLICE_TARGETS after the per-class drop analysis.")

    # ---- patch F's data layer to read Allot windows --------------------
    def load_week_embeddings(week):
        d = np.load(files[int(week)])
        emb = d["embeddings"].astype(np.float64)
        lab = d["true_labels"][d["embedding_indices"]].astype(np.int64)
        return emb, lab

    def pick_eval_week(c, meta):
        cand = [(w, int((load_week_embeddings(w)[1] == c).sum()))
                for w in sorted(degraded) if w > meta["drift_week"]]
        ok = [w for w, n in cand if n >= EVAL_MIN]
        if ok:
            return max(ok)                       # latest, well-populated
        return max(cand, key=lambda t: t[1])[0]  # fallback: most-populated

    def support_pool(c, drift_week, eval_week):
        pool = []
        for w in sorted(degraded):
            if drift_week <= w < eval_week:
                emb, lab = load_week_embeddings(w)
                pool.append(emb[lab == c])
        return np.concatenate(pool, axis=0) if pool else np.zeros((0, 600))

    F.INF_DIR = infd
    F.LABEL_MAP = label_map
    F.SOURCE_WEEK = ref_win
    F.load_week_embeddings = load_week_embeddings
    F._pick_eval_week = pick_eval_week
    F._support_pool = support_pool

    targets = {}
    for c, (dw, tele) in targets_cfg.items():
        targets[c] = dict(name=f"cls{c}", drift_week=dw, teleported=tele)
    if quick:
        # smoke: one teleported target, small K
        c0 = next(c for c, m in targets.items() if m["teleported"])
        targets = {c0: targets[c0]}
        F.K_LIST = [1, 5]
        F.SEEDS = [1, 2]
    F.TARGETS = targets

    KNN.SOURCE_WEEK = ref_win
    return ref_win, sorted(degraded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="early_eq",
                    choices=["early_eq", "quarter_eq"])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: one target class, small K, 2 seeds")
    ap.add_argument("--ncm_only", action="store_true",
                    help="run only the NCM recovery-vs-stability loop")
    ap.add_argument("--knn_only", action="store_true",
                    help="run only the k-NN/Parzen harness")
    args = ap.parse_args()

    tag = "allot_quick" if args.quick else "allot"
    out_res = os.path.join(ROOT, "results/repair/few_shot_repair_allot_v01")
    out_fig = os.path.join(ROOT, "figs/repair")
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)

    ref_win, degraded = build(args.slice, quick=args.quick)
    print(f"[allot-repair] slice={args.slice} ref_win={ref_win} "
          f"n_degraded={len(degraded)} targets={list(F.TARGETS)}")

    # Run k-NN/Parzen FIRST (it also writes metrics.json), move it aside, THEN
    # the NCM loop, so the two harnesses don't clobber each other's metrics.json.
    if not args.ncm_only:
        KNN.OUT_RES = out_res
        KNN.OUT_FIG = out_fig
        _run_knn(out_res, out_fig, tag)

    if not args.knn_only:
        _run_ncm(out_res, out_fig, tag)

    print("\n[allot-repair] done.")


def _run_ncm(out_res, out_fig, tag):
    """Run F.main(); it writes F.OUT_RES/metrics.json + a fixed figure name.
       We override those module-level paths and rename the figure."""
    F.OUT_RES = out_res
    F.OUT_FIG = out_fig
    sys.argv = [sys.argv[0]]  # F.main parses argv (--eval_week); give it none
    F.main()
    src = os.path.join(out_fig, "fig_recovery_vs_stability_w16.png")
    dst = os.path.join(out_fig, "fig_recovery_vs_stability_allot.png")
    if os.path.exists(src):
        os.replace(src, dst)
        print("renamed ->", dst)


def _run_knn(out_res, out_fig, tag):
    """Run KNN.main(); it writes OUT_RES/metrics.json + a fixed figure name.
       Rename both so they don't clobber the NCM metrics.json."""
    KNN.OUT_RES = out_res
    KNN.OUT_FIG = out_fig
    KNN.main()
    # KNN writes metrics.json — keep it as metrics_knn.json
    src = os.path.join(out_res, "metrics.json")
    dst = os.path.join(out_res, "metrics_knn.json")
    if os.path.exists(src):
        os.replace(src, dst)
        print("renamed ->", dst)
    fsrc = os.path.join(out_fig, "fig_recovery_vs_poaching_knn.png")
    fdst = os.path.join(out_fig, "fig_recovery_vs_poaching_allot.png")
    if os.path.exists(fsrc):
        os.replace(fsrc, fdst)
        print("renamed ->", fdst)


if __name__ == "__main__":
    main()
