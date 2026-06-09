#!/usr/bin/env python
"""
Precision-Diagnostics Isolation Ablation
========================================

Validates the operational claim of the paper (Sec. VI): when a tripwire fires,
the framework can *isolate* the affected application class's logit boundary
WITHOUT bleeding into / disturbing healthy, stable domains.

We pose this as **per-class unsupervised anomaly detection** over the 52-week
CESNET-TLS-Year22 stream, using a Week-1-trained model.  Every detector scores
each (class c, week t) using only OBSERVABLE quantities (predicted labels,
softmax, embeddings — never true labels).  All detectors operate on class c's
logit decision region  R_c(t) = { x in W_t : argmax p(x) = c }.

Three detectors
---------------
  Ours  (BBSE-corrected entropy residual)
        r_corr(c) = obs_H(R_c) - E[H | BBSE-estimated composition of R_c]
  Naive (uncorrected entropy residual)
        r_naive(c) = obs_H(R_c) - ref_H(R_c)          (composition frozen at ref)
  MFWDD (feature-weighted drift on the model's 600-d embedding)
        S(c) = sum_i w_c[i] * W1_norm( ref_R_c[:,i], test_R_c[:,i] )
        with per-class feature importance w_c = |classifier.weight[c]| (normalised)

Ours and Naive differ ONLY in the label-shift correction term -> a clean
ablation of the paper's central mechanism.

Ground truth
------------
  degraded(c,t)  iff  f1_ref(c) - f1_true(c,t) > DELTA_DEG  and class present.
We also compute the label-shift-reweighted (covariate-only) per-class F1 so we
can separate genuine covariate breaks from benign label-shift artifacts.

Outputs (under --output_dir)
  fig_precision_isolation_ablation.png   multi-panel figure
  isolation_metrics.json                 AUROC/AUPRC/bleed numbers
  isolation_scores_cache.npz             per (class,week) scores + gt
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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, '.')

# ── constants ──────────────────────────────────────────────────────────────
DELTA_DEG    = 0.15    # true F1 drop (vs ref) to count a class as degraded
EPS_HEALTHY  = 0.05    # max F1 drop to count a class as healthy/stable
N_MIN        = 30      # min true samples for a class to be evaluated at a week
N_MIN_REGION = 20      # min predicted-region samples to score a detector
N_MIN_EMB    = 15      # min embeddings in a region to compute MFWDD


# ── helpers ────────────────────────────────────────────────────────────────
def load_class_names(dataset_root):
    try:
        from data_utils.cesnet_labels import load_label_mapping
        mapping, _ = load_label_mapping(Path(dataset_root))
        return {v: k for k, v in mapping.items()}
    except Exception as e:
        print(f"  (class names unavailable: {e})")
        return {}


def load_feature_importance(ckpt_path, num_classes, emb_dim):
    """Per-class feature importance from the linear classifier head.

    w_c[i] = |classifier.weight[c, i]| normalised to sum 1 over i.
    Returns (num_classes, emb_dim).  Falls back to uniform on failure.
    """
    try:
        import torch
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        W = None
        for k, v in sd.items():
            if k.endswith('classifier.weight') and tuple(v.shape) == (num_classes, emb_dim):
                W = v.detach().cpu().numpy()
                break
        if W is None:
            raise RuntimeError('classifier.weight not found')
        imp = np.abs(W)
        imp /= imp.sum(axis=1, keepdims=True).clip(1e-12)
        print(f"  Loaded per-class feature importance from {Path(ckpt_path).name}")
        return imp
    except Exception as e:
        print(f"  (feature importance fallback to uniform: {e})")
        return np.full((num_classes, emb_dim), 1.0 / emb_dim)


def entropy_rows(softmax):
    p = np.clip(softmax, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def load_week(path):
    d = np.load(path)
    return (d['true_labels'], d['pred_labels'], d['softmax'],
            d['embeddings'], d['embedding_indices'])


def estimate_label_dist(pred, num_classes, C_T_pinv):
    q = np.bincount(pred, minlength=num_classes).astype(float)
    q /= q.sum()
    p = C_T_pinv @ q
    p = np.clip(p, 0, None)
    if p.sum() > 0:
        p /= p.sum()
    return p, q


def per_class_f1(true, pred, num_classes):
    """Vectorised per-class F1.  NaN where class has no true samples."""
    f1 = np.full(num_classes, np.nan)
    for c in range(num_classes):
        tmask = true == c
        if tmask.sum() == 0:
            continue
        pmask = pred == c
        tp = np.sum(tmask & pmask)
        fp = np.sum(~tmask & pmask)
        fn = np.sum(tmask & ~pmask)
        d = 2 * tp + fp + fn
        f1[c] = (2 * tp / d) if d > 0 else 0.0
    return f1


def wasserstein1d_norm(a, b):
    """Normalised 1-D Wasserstein distance between two samples (per feature).

    Distance is scaled by the pooled IQR so it is comparable across features
    (the paper uses a *normalised* Wasserstein distance for the global case).
    """
    a = np.sort(a); b = np.sort(b)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    qs = np.linspace(0, 1, 64)
    qa = np.quantile(a, qs); qb = np.quantile(b, qs)
    w = np.mean(np.abs(qa - qb))
    pooled = np.concatenate([a, b])
    scale = np.subtract(*np.percentile(pooled, [75, 25]))
    return float(w / scale) if scale > 1e-9 else 0.0


# ── reference-week statistics ────────────────────────────────────────────────
def build_reference(ref_true, ref_pred, ref_soft, ref_emb, ref_eidx, num_classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ref_true, ref_pred,
                          labels=list(range(num_classes))).astype(float)
    row = cm.sum(axis=1, keepdims=True); row[row == 0] = 1
    C = cm / row                       # C[k,c] = P(pred=c | true=k)
    C_T_pinv = np.linalg.pinv(C.T, rcond=1e-2)

    counts = np.bincount(ref_true, minlength=num_classes).astype(float)
    p_train = counts / counts.sum()

    H = entropy_rows(ref_soft)

    # region-level reference entropy  ref_H(R_c)  and per-cell entropy h[k,c]
    ref_region_H = np.full(num_classes, np.nan)
    cell_H       = np.full((num_classes, num_classes), np.nan)  # [k, c]
    for c in range(num_classes):
        rmask = ref_pred == c
        if rmask.sum() > 0:
            ref_region_H[c] = H[rmask].mean()
    # per-cell mean entropy (only populated cells)
    order = np.argsort(ref_pred, kind='stable')
    # straightforward loop over populated (k,c) pairs
    pairs = np.unique(np.stack([ref_true, ref_pred], axis=1), axis=0)
    for k, c in pairs:
        m = (ref_true == k) & (ref_pred == c)
        cell_H[k, c] = H[m].mean()

    # reference embeddings grouped by predicted region
    ref_pred_emb = ref_pred[ref_eidx]
    ref_region_emb = {}
    for c in range(num_classes):
        m = ref_pred_emb == c
        if m.sum() >= N_MIN_EMB:
            ref_region_emb[c] = ref_emb[m]

    f1_ref = per_class_f1(ref_true, ref_pred, num_classes)
    return dict(C=C, C_T_pinv=C_T_pinv, p_train=p_train,
                ref_region_H=ref_region_H, cell_H=cell_H,
                ref_region_emb=ref_region_emb, f1_ref=f1_ref)


def score_arrays(pred, soft, emb, pred_emb, ref, feat_imp, num_classes):
    """Per-class detector scores from raw arrays (NaN where not scorable).

    pred, soft : full-window predicted labels and softmax.
    emb        : embedding rows (a subsample);  pred_emb : their predicted labels.
    """
    C, C_T_pinv = ref['C'], ref['C_T_pinv']
    ref_region_H, cell_H = ref['ref_region_H'], ref['cell_H']
    ref_region_emb = ref['ref_region_emb']

    H = entropy_rows(soft)
    p_bbse, _ = estimate_label_dist(pred, num_classes, C_T_pinv)

    r_naive = np.full(num_classes, np.nan)
    r_corr  = np.full(num_classes, np.nan)
    s_mfwdd = np.full(num_classes, np.nan)

    for c in range(num_classes):
        rmask = pred == c
        n = rmask.sum()
        if n < N_MIN_REGION or np.isnan(ref_region_H[c]):
            pass
        else:
            obs = H[rmask].mean()
            # naive: expected entropy frozen at reference region composition
            r_naive[c] = obs - ref_region_H[c]
            # corrected: expected entropy under BBSE-estimated composition of R_c
            # mass of true-k routed to region c  ~  pi_hat[k] * C[k,c]
            mass = p_bbse * C[:, c]
            valid = (mass > 0) & ~np.isnan(cell_H[:, c])
            if valid.sum() > 0 and mass[valid].sum() > 0:
                w = mass[valid] / mass[valid].sum()
                exp_corr = float(np.dot(w, cell_H[valid, c]))
                r_corr[c] = obs - exp_corr
            else:
                r_corr[c] = obs - ref_region_H[c]

        # MFWDD feature-weighted embedding drift on region c
        if c in ref_region_emb:
            tmask = pred_emb == c
            if tmask.sum() >= N_MIN_EMB:
                test_e = emb[tmask]
                ref_e = ref_region_emb[c]
                wimp = feat_imp[c]
                # only spend on the top-weighted features (cheap + faithful)
                top = np.argsort(-wimp)[:120]
                wsum = wimp[top].sum()
                acc = 0.0
                for i in top:
                    acc += wimp[i] * wasserstein1d_norm(ref_e[:, i], test_e[:, i])
                s_mfwdd[c] = acc / wsum if wsum > 0 else acc

    return r_naive, r_corr, s_mfwdd, p_bbse


def score_week(true, pred, soft, emb, eidx, ref, feat_imp, num_classes):
    """Wrapper used for the real per-week pass."""
    return score_arrays(pred, soft, emb, pred[eidx], ref, feat_imp, num_classes)


def resample_label_shift(labels_by_true, p_synth, n_total, rng):
    """Indices that resample rows (grouped by their true class) to prior p_synth.

    labels_by_true : list-of-arrays, labels_by_true[c] = row indices with true==c.
    Pure label shift: rows keep their original features/softmax, only the class
    MIX changes -> P(X|Y) is exactly invariant.
    """
    idx = []
    for c, rows in enumerate(labels_by_true):
        if len(rows) == 0:
            continue
        n_c = int(round(p_synth[c] * n_total))
        if n_c > 0:
            idx.append(rng.choice(rows, size=n_c, replace=True))
    return np.concatenate(idx) if idx else np.arange(0)


def robustness_label_shift(clean_files, wn_of, ref, feat_imp, num_classes,
                           f1_ref, severities=(10, 100, 1000),
                           n_trials=4, n_resample=12000, n_emb_resample=6000,
                           fpr_cal=0.01, seed=42):
    """Inject PURE synthetic label shift on clean weeks and measure false alarms.

    Returns dict: per-detector clean-calibrated threshold, and per-severity
    'bleed' = mean fraction of healthy classes falsely flagged.
    """
    rng = np.random.default_rng(seed)
    dets = ['ours', 'naive', 'mfwdd']
    KEY = {'ours': 'r_corr', 'naive': 'r_naive', 'mfwdd': 's_mfwdd'}

    # 1) calibrate thresholds on the *unperturbed* clean weeks (healthy classes)
    clean_healthy_scores = {d: [] for d in dets}
    clean_cache = []   # reuse loaded weeks
    for p in clean_files:
        wn = wn_of(p)
        true, pred, soft, emb, eidx = load_week(p)
        rn, rc, sm, _ = score_arrays(pred, soft, emb, pred[eidx], ref, feat_imp, num_classes)
        nt = np.bincount(true, minlength=num_classes)
        f1t = per_class_f1(true, pred, num_classes)
        healthy = (nt >= N_MIN) & np.isfinite(f1_ref) & ((f1_ref - f1t) <= EPS_HEALTHY)
        scores = {'r_corr': rc, 'r_naive': rn, 's_mfwdd': sm}
        for d in dets:
            s = scores[KEY[d]][healthy]
            clean_healthy_scores[d].append(s[np.isfinite(s)])
        clean_cache.append((wn, true, pred, soft, emb, eidx, healthy))
    thr = {}
    for d in dets:
        alls = np.concatenate(clean_healthy_scores[d])
        thr[d] = float(np.quantile(alls, 1 - fpr_cal))

    # 2) inject label shift; collect healthy-class scores under shift so bleed
    #    can be evaluated at ANY threshold (enables matched-recall comparison).
    bleed = {d: {s: [] for s in severities} for d in dets}            # per-trial bleed @ clean thr
    shift_scores = {d: {s: [] for s in severities} for d in dets}     # raw healthy scores
    for (wn, true, pred, soft, emb, eidx, healthy) in clean_cache:
        rows_by_true = [np.where(true == c)[0] for c in range(num_classes)]
        emb_true = true[eidx]
        emb_rows_by_true = [np.where(emb_true == c)[0] for c in range(num_classes)]
        healthy_idx = np.where(healthy)[0]
        for sev in severities:
            lo, hi = 1.0 / sev, float(sev)
            for _ in range(n_trials):
                logw = rng.uniform(np.log(lo), np.log(hi), size=num_classes)
                p_synth = np.exp(logw); p_synth /= p_synth.sum()
                ix = resample_label_shift(rows_by_true, p_synth, n_resample, rng)
                eix = resample_label_shift(emb_rows_by_true, p_synth, n_emb_resample, rng)
                if len(ix) == 0 or len(eix) == 0:
                    continue
                rn, rc, sm, _ = score_arrays(
                    pred[ix], soft[ix], emb[eix], pred[eidx][eix],
                    ref, feat_imp, num_classes)
                scores = {'r_corr': rc, 'r_naive': rn, 's_mfwdd': sm}
                for d in dets:
                    s = scores[KEY[d]][healthy_idx]
                    s = s[np.isfinite(s)]
                    if len(s):
                        bleed[d][sev].append(float(np.mean(s > thr[d])))
                        shift_scores[d][sev].append(s)
    bleed_mean = {d: {int(s): (float(np.mean(v)) if v else float('nan'))
                      for s, v in bleed[d].items()} for d in dets}
    shift_scores = {d: {int(s): (np.concatenate(v) if v else np.array([]))
                        for s, v in shift_scores[d].items()} for d in dets}
    return dict(threshold=thr, fpr_cal=fpr_cal, severities=list(severities),
                bleed=bleed_mean, shift_scores=shift_scores,
                clean_weeks=[wn_of(p) for p in clean_files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir', default='results/inference/week_1_inference')
    ap.add_argument('--dataset_root',  default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--checkpoint',
                    default='exps/cesnet_multimodal_each_week_train_v01/week_1/weights/best_model.pth')
    ap.add_argument('--reference_week', type=int, default=1)
    ap.add_argument('--output_dir', default='figs')
    ap.add_argument('--max_weeks', type=int, default=None, help='debug: limit #test weeks')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inference_dir = Path(args.inference_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / 'isolation_scores_cache.npz'

    class_names = load_class_names(args.dataset_root)

    files = sorted(inference_dir.glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    wn_of = lambda p: int(re.search(r'(\d+)$', p.stem).group(1))

    ref_file = next(p for p in files if wn_of(p) == args.reference_week)
    print(f"Reference week: {args.reference_week}  ({ref_file.name})")
    ref_true, ref_pred, ref_soft, ref_emb, ref_eidx = load_week(ref_file)
    num_classes = ref_soft.shape[1]
    emb_dim = ref_emb.shape[1]

    feat_imp = load_feature_importance(args.checkpoint, num_classes, emb_dim)
    print("Building reference statistics ...")
    ref = build_reference(ref_true, ref_pred, ref_soft, ref_emb, ref_eidx, num_classes)
    train_counts = np.bincount(ref_true, minlength=num_classes).astype(float)
    train_pct = train_counts / train_counts.sum() * 100

    test_files = [p for p in files if wn_of(p) != args.reference_week]
    if args.max_weeks:
        test_files = test_files[:args.max_weeks]

    if cache_path.exists() and not args.recompute:
        print(f"Loading cached scores from {cache_path}")
        z = np.load(cache_path, allow_pickle=False)
        week_nums = z['week_nums']
        R_naive = z['R_naive']; R_corr = z['R_corr']; S_mfwdd = z['S_mfwdd']
        F1_true = z['F1_true']; F1_corr = z['F1_corr']; Ntrue = z['Ntrue']
        f1_ref = z['f1_ref']
    else:
        week_nums = []
        R_naive, R_corr, S_mfwdd = [], [], []
        F1_true, F1_corr, Ntrue = [], [], []
        for p in test_files:
            wn = wn_of(p)
            true, pred, soft, emb, eidx = load_week(p)
            rn, rc, sm, p_bbse = score_week(true, pred, soft, emb, eidx,
                                            ref, feat_imp, num_classes)
            f1t = per_class_f1(true, pred, num_classes)
            # covariate-only (label-shift-reweighted) per-class F1
            p_true = np.bincount(true, minlength=num_classes).astype(float)
            p_true /= p_true.sum()
            from sklearn.metrics import confusion_matrix
            cmw = confusion_matrix(true, pred, labels=list(range(num_classes))).astype(float)
            wgt = np.where(p_true > 0, ref['p_train'] / np.where(p_true > 0, p_true, 1.0), 0.0)
            cmw = cmw * wgt[:, None]
            f1c = np.full(num_classes, np.nan)
            nt = np.bincount(true, minlength=num_classes)
            for c in range(num_classes):
                if nt[c] > 0:
                    tp = cmw[c, c]; fp = cmw[:, c].sum() - tp; fn = cmw[c, :].sum() - tp
                    d = 2 * tp + fp + fn
                    f1c[c] = (2 * tp / d) if d > 0 else 0.0

            week_nums.append(wn)
            R_naive.append(rn); R_corr.append(rc); S_mfwdd.append(sm)
            F1_true.append(f1t); F1_corr.append(f1c); Ntrue.append(nt)
            print(f"  week {wn:2d}: scored "
                  f"(naive {np.isfinite(rn).sum()}, corr {np.isfinite(rc).sum()}, "
                  f"mfwdd {np.isfinite(sm).sum()} classes)")

        week_nums = np.array(week_nums)
        R_naive = np.array(R_naive); R_corr = np.array(R_corr); S_mfwdd = np.array(S_mfwdd)
        F1_true = np.array(F1_true); F1_corr = np.array(F1_corr); Ntrue = np.array(Ntrue)
        f1_ref = ref['f1_ref']
        np.savez_compressed(cache_path, week_nums=week_nums,
                            R_naive=R_naive, R_corr=R_corr, S_mfwdd=S_mfwdd,
                            F1_true=F1_true, F1_corr=F1_corr, Ntrue=Ntrue,
                            f1_ref=f1_ref, train_pct=train_pct)
        print(f"Saved scores cache -> {cache_path}")

    # ── ground truth labels per (class, week) ────────────────────────────────
    # degraded if true F1 drop vs ref > DELTA_DEG, class present at ref & week.
    drop = f1_ref[None, :] - F1_true                       # (W, C)
    present = (Ntrue >= N_MIN) & np.isfinite(f1_ref)[None, :] & (f1_ref[None, :] > 0)
    gt_deg = present & (drop > DELTA_DEG)
    gt_healthy = present & (drop <= EPS_HEALTHY)
    # covariate-driven degradation (genuine structural break, not label shift)
    drop_cov = f1_ref[None, :] - F1_corr
    gt_deg_cov = gt_deg & (drop_cov > DELTA_DEG)

    print(f"\nGround truth: {gt_deg.sum()} degraded class-weeks, "
          f"{gt_healthy.sum()} healthy class-weeks, "
          f"{gt_deg_cov.sum()} covariate-driven degraded.")

    # ── detection metrics (pooled over class-weeks) ──────────────────────────
    def pooled_auc(score, label_pos, label_eval):
        m = label_eval & np.isfinite(score)
        y = label_pos[m].astype(int); s = score[m]
        if y.sum() == 0 or y.sum() == m.sum():
            return float('nan'), float('nan'), m.sum(), int(y.sum())
        return (roc_auc_score(y, s), average_precision_score(y, s),
                int(m.sum()), int(y.sum()))

    eval_mask = gt_deg | gt_healthy        # evaluate only confidently-labelled classes
    results = {}
    for name, sc in [('ours', R_corr), ('naive', R_naive), ('mfwdd', S_mfwdd)]:
        au, ap_, n, npos = pooled_auc(sc, gt_deg, eval_mask)
        results[name] = dict(auroc=au, auprc=ap_, n_eval=n, n_pos=npos)
        print(f"  {name:6s}: AUROC={au:.3f}  AUPRC={ap_:.3f}  "
              f"(n={n}, pos={npos})")

    # ── DECISIVE: pure-label-shift robustness slice ──────────────────────────
    # Calibrate each detector to a 1% FPR alarm threshold on CLEAN weeks, then
    # inject synthetic pure label shift and count healthy-class false alarms.
    clean_files = [p for p in test_files if 2 <= wn_of(p) <= 9]
    rob_cache = out_dir / 'robustness_cache.npz'
    if rob_cache.exists() and not args.recompute:
        print(f"\nLoading cached robustness from {rob_cache}")
        z = np.load(rob_cache, allow_pickle=True)
        rob = z['rob'].item()
    else:
        print(f"\nLabel-shift robustness on clean weeks {[wn_of(p) for p in clean_files]} ...")
        rob = robustness_label_shift(clean_files, wn_of, ref, feat_imp, num_classes,
                                     f1_ref, severities=(10, 100, 1000),
                                     n_trials=4, seed=args.seed)
        np.savez(rob_cache, rob=np.array(rob, dtype=object))
        print(f"Saved robustness cache -> {rob_cache}")
    thr = rob['threshold']
    for d in ['ours', 'naive', 'mfwdd']:
        print(f"  {d:6s}: clean thr={thr[d]:+.4f}  bleed@sev="
              + "  ".join(f"{s}x:{rob['bleed'][d][s]:.3f}" for s in rob['severities']))

    # ── operating point at the SAME clean-calibrated threshold ───────────────
    op = {}
    SCMAP = {'ours': R_corr, 'naive': R_naive, 'mfwdd': S_mfwdd}
    worst_sev = rob['severities'][-1]
    for name, sc in SCMAP.items():
        d = sc[gt_deg & np.isfinite(sc)]
        h = sc[gt_healthy & np.isfinite(sc)]
        recall = float(np.mean(d > thr[name])) if len(d) else float('nan')
        nat_bleed = float(np.mean(h > thr[name])) if len(h) else float('nan')
        op[name] = dict(thr=float(thr[name]), recall=recall,
                        natural_bleed=nat_bleed,
                        labelshift_bleed=rob['bleed'][name][worst_sev])
        print(f"  {name:6s}: recall(real-degraded)={recall:.3f}  "
              f"natural-bleed={nat_bleed:.3f}  labelshift-bleed={rob['bleed'][name][worst_sev]:.3f}")

    # ── MATCHED-RECALL bleed: set each detector to equal recall, compare bleed ─
    # Removes the "ours has slightly lower raw recall" confound — the decisive,
    # apples-to-apples isolation comparison.
    matched = {}
    for R0 in (0.50, 0.70, 0.85):
        row = {}
        for name, sc in SCMAP.items():
            d = sc[gt_deg & np.isfinite(sc)]
            if len(d) == 0:
                continue
            t = float(np.quantile(d, 1 - R0))          # threshold giving recall R0
            ss = rob['shift_scores'][name][worst_sev]
            bleed_ls = float(np.mean(ss > t)) if len(ss) else float('nan')
            row[name] = dict(thr=t, bleed_labelshift=bleed_ls)
        matched[f'recall_{int(R0*100)}'] = row
        print(f"  matched recall={R0:.2f}: "
              + "  ".join(f"{n} bleed={row[n]['bleed_labelshift']:.3f}" for n in row))

    # ── AUROC under label-shift contamination ────────────────────────────────
    # Can the detector still separate genuinely-degraded classes from HEALTHY
    # classes whose volume was synthetically shifted?  Positives = real degraded
    # scores; negatives = healthy-under-1000x-shift scores.  Compared to the
    # clean AUROC (healthy negatives), this exposes label-shift fragility.
    contam = {}
    for name, sc in SCMAP.items():
        pos = sc[gt_deg & np.isfinite(sc)]
        neg_clean = sc[gt_healthy & np.isfinite(sc)]
        neg_shift = rob['shift_scores'][name][worst_sev]
        def auc(p, n):
            if len(p) == 0 or len(n) == 0:
                return float('nan')
            y = np.r_[np.ones(len(p)), np.zeros(len(n))]
            s = np.r_[p, n]
            return float(roc_auc_score(y, s))
        contam[name] = dict(auroc_clean=auc(pos, neg_clean),
                            auroc_contaminated=auc(pos, neg_shift))
        print(f"  {name:6s}: AUROC clean={contam[name]['auroc_clean']:.3f}  "
              f"under-labelshift={contam[name]['auroc_contaminated']:.3f}")

    metrics = dict(config=dict(delta_deg=DELTA_DEG, eps_healthy=EPS_HEALTHY,
                               n_min=N_MIN, reference_week=args.reference_week,
                               n_test_weeks=int(len(week_nums))),
                   ground_truth=dict(degraded=int(gt_deg.sum()),
                                     healthy=int(gt_healthy.sum()),
                                     covariate_driven=int(gt_deg_cov.sum())),
                   detection=results, operating_point=op,
                   matched_recall_bleed=matched,
                   auroc_under_contamination=contam,
                   robustness_label_shift=dict(
                       clean_weeks=rob['clean_weeks'], fpr_cal=rob['fpr_cal'],
                       threshold=rob['threshold'], severities=rob['severities'],
                       bleed=rob['bleed']))

    # ════════════════════════════════════════════════════════════════════════
    # FIGURE
    # ════════════════════════════════════════════════════════════════════════
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False})
    COL = dict(ours='#2c7bb6', naive='#e07b39', mfwdd='#7b3294')
    LAB = dict(ours='Ours (BBSE-corrected)', naive='Naive (uncorrected)',
               mfwdd='MFWDD (feature-weighted drift)')

    names = ['ours', 'naive', 'mfwdd']
    sevs = rob['severities']
    R0s = sorted(matched.keys(), key=lambda k: int(k.split('_')[1]))
    xr = [int(k.split('_')[1]) / 100 for k in R0s]

    # ── each panel as a self-contained draw fn (used for combined + standalone)
    def panel_A(ax):  # ROC — per-class degradation detection
        for name, sc in [('ours', R_corr), ('naive', R_naive), ('mfwdd', S_mfwdd)]:
            m = eval_mask & np.isfinite(sc)
            y = gt_deg[m].astype(int); s = sc[m]
            if y.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=COL[name], lw=2.2,
                    label=f"{LAB[name]}  (AUROC={results[name]['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], color='#999', ls='--', lw=1)
        ax.set_xlabel('False positive rate (healthy classes)')
        ax.set_ylabel('True positive rate (degraded classes)')
        ax.set_title('A. Per-class degradation detection\n(pooled over all class-weeks)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=7.5, loc='lower right')

    def panel_B(ax):  # bleed under pure label shift vs severity
        for name in names:
            ys = [rob['bleed'][name][s] for s in sevs]
            ax.plot(sevs, ys, color=COL[name], lw=2.2, marker='o', markersize=6,
                    label=f"{LAB[name].split(' (')[0]} (recall={op[name]['recall']:.2f})")
        ax.axhline(rob['fpr_cal'], color='#c0392b', ls=':', lw=1.2,
                   label=f"clean-calibrated FPR ({rob['fpr_cal']:.0%})")
        ax.set_xscale('log')
        ax.set_xticks(sevs); ax.set_xticklabels([f'{s}×' for s in sevs])
        ax.set_xlabel('Injected label-shift severity (LogUniform range)')
        ax.set_ylabel('Bleed — healthy classes falsely flagged')
        ax.set_ylim(-0.02, 1.02)
        ax.set_title('B. False alarms under PURE label shift\n'
                     '(no class truly degraded → every flag is bleed)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=7.5, loc='upper left')

    def panel_C(ax):  # detection AUROC: clean vs label-shift contamination
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w/2, [contam[n]['auroc_clean'] for n in names], w,
               color=[COL[n] for n in names], label='Clean (healthy negatives)')
        ax.bar(x + w/2, [contam[n]['auroc_contaminated'] for n in names], w,
               color=[COL[n] for n in names], alpha=0.4, hatch='//',
               label=f'Under {sevs[-1]}× label shift')
        ax.axhline(0.5, color='#999', ls='--', lw=1)
        for i, n in enumerate(names):
            ax.annotate(f"{contam[n]['auroc_clean']:.2f}", (i - w/2, contam[n]['auroc_clean']),
                        ha='center', va='bottom', fontsize=7.5)
            ax.annotate(f"{contam[n]['auroc_contaminated']:.2f}",
                        (i + w/2, contam[n]['auroc_contaminated']),
                        ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([LAB[n].split(' (')[0] for n in names],
                                             fontsize=8, rotation=12)
        ax.set_ylabel('AUROC (degraded vs negatives)')
        ax.set_ylim(0.3, 1.0)
        ax.set_title('C. Separability collapses under label shift\n'
                     '(Ours retains it; baselines do not)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=7, loc='lower left')

    def panel_D(ax):  # matched-recall bleed — apples-to-apples isolation
        for name in names:
            ys = [matched[k][name]['bleed_labelshift'] if name in matched[k] else np.nan
                  for k in R0s]
            ax.plot(xr, ys, color=COL[name], lw=2.4, marker='o', markersize=7,
                    label=LAB[name])
            for xx, yy in zip(xr, ys):
                ax.annotate(f'{yy:.2f}', (xx, yy), fontsize=7.5, ha='center',
                            va='bottom', color=COL[name])
        ax.set_xlabel('Matched recall on genuinely-degraded classes')
        ax.set_ylabel(f'Bleed — healthy classes falsely flagged\nunder {sevs[-1]}× label shift')
        ax.set_xticks(xr)
        ax.set_ylim(-0.02, 0.75)
        ax.set_title('D. Isolation at MATCHED detection power — at equal recall, '
                     'Ours disturbs far fewer healthy classes',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8.5, loc='upper left')

    def panel_E(ax):  # summary table
        ax.axis('off')
        rows = [['', 'Ours', 'Naive', 'MFWDD']]
        rows.append(['AUROC (clean)'] + [f"{results[n]['auroc']:.3f}" for n in names])
        rows.append(['AUROC (LS-contam.)'] +
                    [f"{contam[n]['auroc_contaminated']:.2f}" for n in names])
        rows.append(['Bleed @recall .70'] +
                    [f"{matched['recall_70'][n]['bleed_labelshift']:.2f}" for n in names])
        rows.append(['Bleed @recall .85'] +
                    [f"{matched['recall_85'][n]['bleed_labelshift']:.2f}" for n in names])
        tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.6)
        for j, n in enumerate(['', 'ours', 'naive', 'mfwdd']):
            tbl[(0, j)].set_facecolor('#eeeeee'); tbl[(0, j)].set_text_props(weight='bold')
            if n in COL:
                tbl[(0, j)].set_text_props(color=COL[n], weight='bold')
        ax.set_title('E. Summary  (lower bleed = better isolation)',
                     fontweight='bold', fontsize=10, pad=2)

    panels = [('A_roc_detection', panel_A), ('B_bleed_vs_severity', panel_B),
              ('C_auroc_contamination', panel_C), ('D_matched_recall_bleed', panel_D),
              ('E_summary', panel_E)]

    # combined overview figure
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    for ax, (_, fn) in zip(
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
             fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :2]),
             fig.add_subplot(gs[1, 2])], panels):
        fn(ax)
    fig.suptitle('Precision-Diagnostics Isolation Ablation — CESNET-TLS-Year22 '
                 '(reference week 1, 52-week stream)',
                 fontsize=13, fontweight='bold', y=0.99)
    out_fig = out_dir / 'fig_precision_isolation_ablation.png'
    fig.savefig(out_fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved combined figure -> {out_fig}")

    # one standalone PNG per panel
    panel_dir = out_dir / 'panels'
    panel_dir.mkdir(exist_ok=True)
    for slug, fn in panels:
        sz = (10, 4.6) if slug.startswith('D') else (6.4, 5.2)
        f1, a1 = plt.subplots(figsize=sz)
        fn(a1)
        p = panel_dir / f'fig_isolation_{slug}.png'
        f1.savefig(p, dpi=180, bbox_inches='tight')
        plt.close(f1)
        print(f"Saved panel -> {p}")

    with open(out_dir / 'isolation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {out_dir / 'isolation_metrics.json'}")


if __name__ == '__main__':
    main()
