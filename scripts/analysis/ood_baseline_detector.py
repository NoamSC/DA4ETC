#!/usr/bin/env python
"""
Per-class OOD / energy-score degradation detector (reviewer-response baseline)
==============================================================================

Defuses the critique that the paper's open-set framing is "interpretive only"
by adding a STANDARD out-of-distribution score, applied PER CLASS, as an
alternative degradation detector alongside the BBSE-corrected entropy residual.

We add two detectors, both classic single-model OOD scores aggregated per
predicted-class region R_c(t) = { x in W_t : argmax p(x) = c }:

  * energy : E(x) = -logsumexp(logits)  (Liu et al., NeurIPS'20, energy-based OOD).
             Higher energy = more OOD.  Per-class score = mean_x in R_c E(x),
             expressed as a RESIDUAL vs the reference-week per-region mean energy
             so it is directly comparable to the entropy residuals (higher =
             more degraded).
  * msp    : Maximum-Softmax-Probability (Hendrycks & Gimpel, ICLR'17).  Lower
             max-prob = more OOD; we use the per-region mean (ref_MSP - obs_MSP)
             residual so, again, higher = more degraded.

These slot in next to the existing comparison
(scripts/analysis/precision_isolation_ablation.py and auroc_anomaly_detection.py):
same reference week (Week-16), same FORWARD eval weeks (>ref), same per-class
ground truth (true F1 drop vs reference > DELTA_DEG), same AUROC + FAR@95
metrics, and the SAME pure-label-shift robustness slice — so per-class OOD can
be compared honestly to ours / uncorrected / MFWDD.

== Logits / energy provenance ==
The saved per-week .npz holds SOFTMAX, not raw logits, so true energy
(-logsumexp(logits), which depends on the additive logit constant lost by
softmax) cannot be recovered from softmax alone.  BUT the .npz also stores the
600-d pre-classifier `embeddings` (the model's `features`), and the source
checkpoint's head is a single nn.Linear(`classifier`).  We therefore recompute
EXACT logits as  z = embeddings @ W.T + b  and obtain exact energy.  This is
verified to reproduce the saved softmax to float32 precision.  Consequence:
energy is only available on the 10% embedding SUBSAMPLE (embedding_indices);
MSP is computed on the same subsample for a like-for-like comparison (it could
use the full window, but we keep both OOD scores on the identical subsample so
their numbers are directly comparable and their region sizes match).

Outputs
  results/ood_baseline/ood_metrics<suffix>.json     AUROC / FAR@95 / robustness
  results/ood_baseline/ood_scores_cache<suffix>.npz per (class,week) scores + gt
  figs/fig_ood_baseline_comparison.png              comparison figure
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
from scipy.special import logsumexp
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve)

sys.path.insert(0, '.')

# Reuse the EXACT detector machinery / constants from the existing per-class
# isolation comparison so ours / naive / mfwdd numbers are identical to the
# shared analysis script (no re-derivation, no destructive edits there).
from scripts.analysis.precision_isolation_ablation import (  # noqa: E402
    DELTA_DEG, EPS_HEALTHY, N_MIN, N_MIN_REGION, N_MIN_EMB,
    load_class_names, load_feature_importance, entropy_rows, load_week,
    per_class_f1, build_reference, score_arrays, resample_label_shift,
)


# ── OOD score machinery ─────────────────────────────────────────────────────

def load_classifier_head(ckpt_path, num_classes, emb_dim):
    """Return (W, b) of the linear classifier head so we can recompute exact
    logits z = emb @ W.T + b from the saved 600-d embeddings.  Energy needs the
    additive logit constant that softmax discards, hence this reconstruction."""
    import torch
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    Wk = next(k for k in sd if k.endswith('classifier.weight')
              and tuple(sd[k].shape) == (num_classes, emb_dim))
    bk = Wk[:-len('weight')] + 'bias'
    W = sd[Wk].detach().cpu().numpy().astype(np.float64)
    b = sd[bk].detach().cpu().numpy().astype(np.float64)
    print(f"  Loaded classifier head {W.shape} from {Path(ckpt_path).name}")
    return W, b


def logits_from_emb(emb, W, b):
    return emb.astype(np.float64) @ W.T + b


def ood_scores_per_class(pred_emb, logits, num_classes, ref_E, ref_MSP):
    """Per-class energy / MSP RESIDUAL scores over predicted regions of the
    embedding subsample.  Higher = more degraded (matches the entropy residuals).

    energy_residual(c) = mean_x in R_c [-logsumexp(z(x))]  -  ref_E(c)
    msp_residual(c)    = ref_MSP(c)  -  mean_x in R_c [max softmax]

    NaN where region too small or no reference.
    """
    E = -logsumexp(logits, axis=1)                       # per-sample energy
    MSP = np.exp(logits - logsumexp(logits, axis=1, keepdims=True)).max(axis=1)
    s_energy = np.full(num_classes, np.nan)
    s_msp = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = pred_emb == c
        n = int(m.sum())
        if n < N_MIN_EMB:
            continue
        if not np.isnan(ref_E[c]):
            s_energy[c] = float(E[m].mean()) - ref_E[c]
        if not np.isnan(ref_MSP[c]):
            s_msp[c] = ref_MSP[c] - float(MSP[m].mean())
    return s_energy, s_msp


def build_ref_ood(ref_pred_emb, ref_logits, num_classes):
    """Reference-week per-region mean energy and mean MSP."""
    E = -logsumexp(ref_logits, axis=1)
    MSP = np.exp(ref_logits - logsumexp(ref_logits, axis=1, keepdims=True)).max(axis=1)
    ref_E = np.full(num_classes, np.nan)
    ref_MSP = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = ref_pred_emb == c
        if m.sum() >= N_MIN_EMB:
            ref_E[c] = float(E[m].mean())
            ref_MSP[c] = float(MSP[m].mean())
    return ref_E, ref_MSP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inference_dir',
                    default='results/inference_auditfix/week_16_vanilla_bs64')
    ap.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    ap.add_argument('--checkpoint',
                    default='exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth')
    ap.add_argument('--reference_week', type=int, default=16)
    ap.add_argument('--forward', action='store_true', default=True,
                    help='forward-only: evaluate only weeks AFTER the reference week')
    ap.add_argument('--all_weeks', dest='forward', action='store_false',
                    help='evaluate ALL weeks != reference (overrides --forward)')
    ap.add_argument('--clean_weeks', type=int, nargs=2, default=(17, 20),
                    metavar=('LO', 'HI'),
                    help='inclusive week range for label-shift robustness calibration')
    ap.add_argument('--far_tpr', type=float, default=0.95,
                    help='TPR at which FAR (=FPR) is reported, i.e. FAR@95')
    ap.add_argument('--robust_severity', type=int, default=1000)
    ap.add_argument('--robust_trials', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default='results/ood_baseline')
    ap.add_argument('--fig_dir', default='figs')
    ap.add_argument('--suffix', default='_w16fwd')
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f'ood_scores_cache{args.suffix}.npz'

    class_names = load_class_names(args.dataset_root)
    inference_dir = Path(args.inference_dir)
    files = sorted(inference_dir.glob('WEEK-2022-*.npz'),
                   key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1)))
    wn_of = lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    if not files:
        raise SystemExit(f"No WEEK-2022-*.npz under {inference_dir}")

    ref_file = next(p for p in files if wn_of(p) == args.reference_week)
    print(f"Reference week: {args.reference_week}  ({ref_file.name})")
    print(f"Inference dir : {inference_dir}")
    ref_true, ref_pred, ref_soft, ref_emb, ref_eidx = load_week(ref_file)
    num_classes = ref_soft.shape[1]
    emb_dim = ref_emb.shape[1]

    # entropy-residual + MFWDD reference (ours / naive / mfwdd) — verbatim machinery
    feat_imp = load_feature_importance(args.checkpoint, num_classes, emb_dim)
    print("Building reference statistics (entropy residual + MFWDD) ...")
    ref = build_reference(ref_true, ref_pred, ref_soft, ref_emb, ref_eidx, num_classes)

    # OOD reference: recompute exact logits on the reference embedding subsample
    W, b = load_classifier_head(args.checkpoint, num_classes, emb_dim)
    ref_logits = logits_from_emb(ref_emb, W, b)
    ref_pred_emb = ref_pred[ref_eidx]
    # sanity: reconstructed softmax must match the saved softmax on the subsample
    recon = np.exp(ref_logits - logsumexp(ref_logits, axis=1, keepdims=True))
    max_err = float(np.abs(recon - ref_soft[ref_eidx]).max())
    print(f"  logit-reconstruction check: max|softmax_recon - softmax_saved| = {max_err:.2e}")
    if max_err > 1e-3:
        print("  WARNING: logit reconstruction error is large; energy may be unreliable.")
    ref_E, ref_MSP = build_ref_ood(ref_pred_emb, ref_logits, num_classes)

    f1_ref = ref['f1_ref']

    if args.forward:
        test_files = [p for p in files if wn_of(p) > args.reference_week]
        print(f"Forward-only: {len(test_files)} weeks after week {args.reference_week}")
    else:
        test_files = [p for p in files if wn_of(p) != args.reference_week]

    # ── per (class, week) detector scores ───────────────────────────────────
    if cache_path.exists() and not args.recompute:
        print(f"Loading cached scores from {cache_path}")
        z = np.load(cache_path)
        week_nums = z['week_nums']
        R_corr, R_naive, S_mfwdd = z['R_corr'], z['R_naive'], z['S_mfwdd']
        S_energy, S_msp = z['S_energy'], z['S_msp']
        F1_true, Ntrue = z['F1_true'], z['Ntrue']
        f1_ref = z['f1_ref']
    else:
        week_nums = []
        R_corr, R_naive, S_mfwdd, S_energy, S_msp = [], [], [], [], []
        F1_true, Ntrue = [], []
        for p in test_files:
            wn = wn_of(p)
            true, pred, soft, emb, eidx = load_week(p)
            pred_emb = pred[eidx]
            # ours / naive / mfwdd (identical machinery to the shared script)
            rn, rc, sm, _ = score_arrays(pred, soft, emb, pred_emb,
                                         ref, feat_imp, num_classes)
            # OOD: exact logits from this week's embedding subsample
            logits = logits_from_emb(emb, W, b)
            se, sp = ood_scores_per_class(pred_emb, logits, num_classes,
                                          ref_E, ref_MSP)
            f1t = per_class_f1(true, pred, num_classes)
            nt = np.bincount(true, minlength=num_classes)
            week_nums.append(wn)
            R_naive.append(rn); R_corr.append(rc); S_mfwdd.append(sm)
            S_energy.append(se); S_msp.append(sp)
            F1_true.append(f1t); Ntrue.append(nt)
            print(f"  week {wn:2d}: ours {np.isfinite(rc).sum():3d}  "
                  f"naive {np.isfinite(rn).sum():3d}  mfwdd {np.isfinite(sm).sum():3d}  "
                  f"energy {np.isfinite(se).sum():3d}  msp {np.isfinite(sp).sum():3d} classes")
        week_nums = np.array(week_nums)
        R_naive = np.array(R_naive); R_corr = np.array(R_corr); S_mfwdd = np.array(S_mfwdd)
        S_energy = np.array(S_energy); S_msp = np.array(S_msp)
        F1_true = np.array(F1_true); Ntrue = np.array(Ntrue)
        np.savez_compressed(cache_path, week_nums=week_nums,
                            R_corr=R_corr, R_naive=R_naive, S_mfwdd=S_mfwdd,
                            S_energy=S_energy, S_msp=S_msp,
                            F1_true=F1_true, Ntrue=Ntrue, f1_ref=f1_ref)
        print(f"Saved scores cache -> {cache_path}")

    # ── ground truth (identical definition to the isolation ablation) ────────
    drop = f1_ref[None, :] - F1_true
    present = (Ntrue >= N_MIN) & np.isfinite(f1_ref)[None, :] & (f1_ref[None, :] > 0)
    gt_deg = present & (drop > DELTA_DEG)
    gt_healthy = present & (drop <= EPS_HEALTHY)
    eval_mask = gt_deg | gt_healthy
    print(f"\nGround truth: {gt_deg.sum()} degraded, {gt_healthy.sum()} healthy class-weeks "
          f"(over {len(week_nums)} forward weeks)")

    DETS = [('ours', R_corr, 'Ours (BBSE-corrected residual)', '#2c7bb6'),
            ('naive', R_naive, 'Uncorrected entropy residual', '#e07b39'),
            ('mfwdd', S_mfwdd, 'MFWDD (feature-weighted drift)', '#7b3294'),
            ('energy', S_energy, 'Per-class energy (OOD)', '#1a9641'),
            ('msp', S_msp, 'Per-class MSP (OOD)', '#d7191c')]

    def far_at_tpr(y, sc, t):
        fpr, tpr, _ = roc_curve(y, sc)
        ok = tpr >= t
        return float(fpr[ok][0]) if ok.any() else 1.0

    # ── clean detection AUROC + FAR@95 (degraded vs healthy class-weeks) ─────
    detection = {}
    print(f"\n=== Per-class degradation detection (clean) ===")
    for key, sc, lab, _ in DETS:
        m = eval_mask & np.isfinite(sc)
        y = gt_deg[m].astype(int); s = sc[m]
        if y.sum() == 0 or y.sum() == m.sum():
            detection[key] = dict(auroc=float('nan'), auprc=float('nan'),
                                  far95=float('nan'), n_eval=int(m.sum()),
                                  n_pos=int(y.sum()))
            continue
        au = roc_auc_score(y, s); apr = average_precision_score(y, s)
        far = far_at_tpr(y, s, args.far_tpr)
        detection[key] = dict(auroc=float(au), auprc=float(apr), far95=float(far),
                              n_eval=int(m.sum()), n_pos=int(y.sum()))
        print(f"  {lab:34s} AUROC={au:.3f}  AUPRC={apr:.3f}  "
              f"FAR@{args.far_tpr:.0%}={far:.3f}  (n={m.sum()}, pos={int(y.sum())})")

    # ── pure-label-shift robustness: do OOD scores false-alarm on volume? ────
    # Calibrate each detector to a clean threshold, inject PURE synthetic label
    # shift on clean forward weeks (P(X|Y) invariant), measure healthy-class
    # bleed and contaminated AUROC.  The key OOD critique: energy/MSP aggregate
    # per region, so a benign volume surge changing the region's class MIX can
    # move the mean score even though no class truly degraded.
    clean_lo, clean_hi = args.clean_weeks
    clean_files = [p for p in test_files if clean_lo <= wn_of(p) <= clean_hi]
    sev = args.robust_severity
    rng = np.random.default_rng(args.seed)
    print(f"\nLabel-shift robustness on clean weeks "
          f"{[wn_of(p) for p in clean_files]} (severity {sev}x) ...")

    # collect per-detector healthy scores under pure label shift, and clean
    # healthy scores (for calibration) — using the SAME score functions.
    def all_scores_for_idx(true, pred, soft, emb, eidx, ix, eix):
        """Recompute all 5 detector scores on a resampled window."""
        pred_emb_full = pred[eidx]
        rn, rc, sm, _ = score_arrays(pred[ix], soft[ix], emb[eix],
                                     pred_emb_full[eix], ref, feat_imp, num_classes)
        logits = logits_from_emb(emb[eix], W, b)
        se, sp = ood_scores_per_class(pred_emb_full[eix], logits, num_classes,
                                      ref_E, ref_MSP)
        return {'ours': rc, 'naive': rn, 'mfwdd': sm, 'energy': se, 'msp': sp}

    keys = [k for k, *_ in DETS]
    shift_scores = {k: [] for k in keys}     # healthy scores under label shift
    clean_scores = {k: [] for k in keys}     # healthy scores, unperturbed clean
    for p in clean_files:
        true, pred, soft, emb, eidx = load_week(p)
        f1t = per_class_f1(true, pred, num_classes)
        nt = np.bincount(true, minlength=num_classes)
        healthy = (nt >= N_MIN) & np.isfinite(f1_ref) & ((f1_ref - f1t) <= EPS_HEALTHY)
        h_idx = np.where(healthy)[0]
        # unperturbed clean scores (identity resample)
        n = len(true); ne = len(eidx)
        sc0 = all_scores_for_idx(true, pred, soft, emb, eidx,
                                 np.arange(n), np.arange(ne))
        for k in keys:
            v = sc0[k][h_idx]; clean_scores[k].append(v[np.isfinite(v)])
        # pure label-shift trials
        rows_by_true = [np.where(true == c)[0] for c in range(num_classes)]
        emb_true = true[eidx]
        emb_rows_by_true = [np.where(emb_true == c)[0] for c in range(num_classes)]
        lo, hi = 1.0 / sev, float(sev)
        for _ in range(args.robust_trials):
            logw = rng.uniform(np.log(lo), np.log(hi), size=num_classes)
            p_synth = np.exp(logw); p_synth /= p_synth.sum()
            ix = resample_label_shift(rows_by_true, p_synth, 12000, rng)
            eix = resample_label_shift(emb_rows_by_true, p_synth, 6000, rng)
            if len(ix) == 0 or len(eix) == 0:
                continue
            sc = all_scores_for_idx(true, pred, soft, emb, eidx, ix, eix)
            for k in keys:
                v = sc[k][h_idx]; shift_scores[k].append(v[np.isfinite(v)])
    clean_scores = {k: (np.concatenate(v) if v else np.array([])) for k, v in clean_scores.items()}
    shift_scores = {k: (np.concatenate(v) if v else np.array([])) for k, v in shift_scores.items()}

    # bleed at matched recall + AUROC under label-shift contamination
    SCMAP = {k: sc for k, sc, *_ in DETS}
    robustness = {'severity': sev, 'clean_weeks': [wn_of(p) for p in clean_files],
                  'matched_recall_bleed': {}, 'auroc_under_contamination': {}}
    print(f"\n=== Robustness under PURE {sev}x label shift ===")
    for R0 in (0.70, 0.85):
        row = {}
        for k in keys:
            d = SCMAP[k][gt_deg & np.isfinite(SCMAP[k])]
            if len(d) == 0:
                continue
            t = float(np.quantile(d, 1 - R0))
            ss = shift_scores[k]
            row[k] = float(np.mean(ss > t)) if len(ss) else float('nan')
        robustness['matched_recall_bleed'][f'recall_{int(R0*100)}'] = row
        print(f"  bleed @recall {R0:.2f}: "
              + "  ".join(f"{k}={row[k]:.3f}" for k in keys if k in row))
    for k in keys:
        pos = SCMAP[k][gt_deg & np.isfinite(SCMAP[k])]
        neg_clean = clean_scores[k]
        neg_shift = shift_scores[k]
        def auc(pp, nn):
            if len(pp) == 0 or len(nn) == 0:
                return float('nan')
            y = np.r_[np.ones(len(pp)), np.zeros(len(nn))]
            return float(roc_auc_score(y, np.r_[pp, nn]))
        robustness['auroc_under_contamination'][k] = dict(
            auroc_clean=auc(pos, neg_clean), auroc_contaminated=auc(pos, neg_shift))
        print(f"  {k:7s}: AUROC clean={robustness['auroc_under_contamination'][k]['auroc_clean']:.3f}  "
              f"under-labelshift={robustness['auroc_under_contamination'][k]['auroc_contaminated']:.3f}")

    # ── persist metrics ──────────────────────────────────────────────────────
    metrics = dict(
        config=dict(reference_week=args.reference_week, forward=bool(args.forward),
                    inference_dir=str(inference_dir), checkpoint=args.checkpoint,
                    delta_deg=DELTA_DEG, eps_healthy=EPS_HEALTHY, n_min=N_MIN,
                    n_min_emb=N_MIN_EMB, far_tpr=args.far_tpr,
                    n_test_weeks=int(len(week_nums)), seed=args.seed,
                    data_sample_frac=0.1, logit_recon_max_err=max_err,
                    energy_note='energy/MSP on 10% embedding subsample; logits '
                                'recomputed exactly from embeddings @ W.T + b'),
        ground_truth=dict(degraded=int(gt_deg.sum()), healthy=int(gt_healthy.sum())),
        detection=detection, robustness=robustness)
    metrics_path = out_dir / f'ood_metrics{args.suffix}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {metrics_path}")

    # ── figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # A: ROC curves (clean per-class detection)
    ax = axes[0]
    for key, sc, lab, col in DETS:
        m = eval_mask & np.isfinite(sc)
        y = gt_deg[m].astype(int); s = sc[m]
        if y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=col, lw=2.2,
                label=f"{lab.split(' (')[0]} (AUROC={detection[key]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color='#999', ls='--', lw=1)
    ax.set_xlabel('False alarm rate (healthy classes)')
    ax.set_ylabel('Detection rate (degraded classes)')
    ax.set_title('A. Per-class degradation detection\n(Week-16 source, forward weeks)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

    # B: AUROC clean vs under label-shift contamination
    ax = axes[1]
    x = np.arange(len(keys)); w = 0.38
    cl = [robustness['auroc_under_contamination'][k]['auroc_clean'] for k in keys]
    co = [robustness['auroc_under_contamination'][k]['auroc_contaminated'] for k in keys]
    cols = [d[3] for d in DETS]
    ax.bar(x - w/2, cl, w, color=cols, label='Clean')
    ax.bar(x + w/2, co, w, color=cols, alpha=0.4, hatch='//',
           label=f'Under {sev}x label shift')
    ax.axhline(0.5, color='#999', ls='--', lw=1)
    for i, (a, bb) in enumerate(zip(cl, co)):
        ax.annotate(f'{a:.2f}', (i - w/2, a), ha='center', va='bottom', fontsize=7)
        ax.annotate(f'{bb:.2f}', (i + w/2, bb), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=8, rotation=15)
    ax.set_ylabel('AUROC (degraded vs healthy)')
    ax.set_ylim(0.3, 1.02)
    ax.set_title('B. Separability under PURE label shift\n'
                 '(does the detector false-alarm on volume?)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')

    # C: summary table
    ax = axes[2]; ax.axis('off')
    rows = [['', 'AUROC', f'FAR@{int(args.far_tpr*100)}', 'AUROC\n(LS-contam)',
             'Bleed\n@R.85']]
    for k, sc, lab, col in DETS:
        rows.append([lab.split(' (')[0],
                     f"{detection[k]['auroc']:.3f}",
                     f"{detection[k]['far95']:.3f}",
                     f"{robustness['auroc_under_contamination'][k]['auroc_contaminated']:.2f}",
                     f"{robustness['matched_recall_bleed']['recall_85'].get(k, float('nan')):.2f}"])
    tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.7)
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor('#eeeeee'); tbl[(0, j)].set_text_props(weight='bold')
    for i, (k, sc, lab, col) in enumerate(DETS, start=1):
        tbl[(i, 0)].set_text_props(color=col, weight='bold')
    ax.set_title('C. Summary (higher AUROC / lower FAR & bleed = better)',
                 fontweight='bold', fontsize=10, pad=2)

    fig.suptitle('Per-class OOD baselines (energy / MSP) vs BBSE-corrected residual '
                 '— CESNET-TLS-Year22, Week-16 source, forward eval',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / 'fig_ood_baseline_comparison.png'
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")

    # ── markdown comparison table (deliverable) ──────────────────────────────
    print("\n=== COMPARISON TABLE (markdown) ===\n")
    print(f"| Detector | AUROC | FAR@{int(args.far_tpr*100)} | AUROC (LS-contam) | Bleed @R.85 |")
    print("|---|---|---|---|---|")
    for k, sc, lab, col in DETS:
        print(f"| {lab.split(' (')[0]} | {detection[k]['auroc']:.3f} | "
              f"{detection[k]['far95']:.3f} | "
              f"{robustness['auroc_under_contamination'][k]['auroc_contaminated']:.3f} | "
              f"{robustness['matched_recall_bleed']['recall_85'].get(k, float('nan')):.3f} |")


if __name__ == '__main__':
    main()
