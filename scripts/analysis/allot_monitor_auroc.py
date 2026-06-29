#!/usr/bin/env python
"""
BBSE-corrected entropy-residual MONITOR on the ALLOT dataset.

Allot analogue of scripts/analysis/auroc_anomaly_detection.py (the CESNET
Sec. V-C primary result). It evaluates the label-free degradation monitor —
the BBSE-corrected entropy residual — as a binary detector on the multimodal
model's frozen-source inference over the Allot timeline.

WHY ALLOT MATTERS HERE
----------------------
Allot is exactly where the global UDA/TTA baselines do real harm (TENT/CoTTA
cost -0.21/-0.28 accuracy on the late windows). A label-free monitor that fires
on that genuine degradation — and not on benign volume volatility — is the
strongest operational story. This script provides its AUROC / FAR@95.

GROUND TRUTH (natural, not synthetic)
-------------------------------------
The Allot timeline concatenates four non-overlapping domains:
    domain_0/1 : Sep 2024  (windows 1-20)   -> SAME period as training slices
    domain_2   : Feb 2025  (windows 23-35)  -> +5 months: genuine covariate shift
    domain_3   : Mar 2025  (windows 37-51)  -> +6 months: genuine covariate shift
The per-window Macro-F1 of the frozen source model drops sharply and cleanly at
the domain_1 -> domain_2 boundary (mF1 ~0.61-0.73 healthy -> ~0.41-0.52
degraded). We label:
    anomaly(window) = 1  iff  window's true Macro-F1 < --f1_threshold
i.e. EXACTLY the CESNET construction (anomaly = genuine covariate-driven F1
collapse), but the split here is a natural multi-month domain gap rather than the
week-10 sensor artifact. This is a cleaner, real-deployment degradation signal.

PROTOCOL DIFFERENCES vs CESNET (documented honestly)
----------------------------------------------------
1. SOFTMAX: Allot inference saved only top-5 (topk_idx / topk_prob), not the full
   K-way softmax (disk-quota decision). We reconstruct a per-sample distribution:
   the 5 top probs in their class slots, and the residual mass (1 - sum top5)
   spread UNIFORMLY over the remaining K-5 classes. Top-5 captures ~95% of mass
   on average, so the entropy is a faithful (slightly conservative) proxy. We
   report both the reconstructed-entropy monitor and a top5-only entropy as a
   robustness check.
2. REFERENCE WINDOW: the model's own healthy in-distribution window. For the
   `quarter` model that is its training window (13). For the `early` model,
   window_00 was not inferred (its first partial chunk was the training slice),
   so we use window_01 (same domain_0, hours later) as the healthy reference.
3. CLOSED-WORLD: Allot is private/closed-world; inference already drops rows whose
   appId was unseen in the training slice. K differs per model (111 / 113).

DETECTORS (all unsupervised, from window top-5 / preds only)
------------------------------------------------------------
* uncorrected : entropy gap vs a FIXED train-prior baseline (no label-shift corr.)
* mfwdd       : MFWDD-style global drift = feature-importance-weighted per-channel
                1-Wasserstein distance of the window's top-1-class one-hot channel
                histogram vs the reference (a global, label-shift-sensitive trigger)
* bbse        : BBSE-corrected per-sample entropy residual (OURS)

Outputs
-------
* results/allot_monitor_v01/metrics.json
* figs/fig_allot_monitor_auroc.png
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

REPO = Path(__file__).resolve().parents[2]


# ── top-5 -> per-sample distribution reconstruction ─────────────────────────────

def reconstruct_softmax(topk_idx, topk_prob, K):
    """top-5 in their slots; residual mass spread uniformly over the other K-5.

    Returns a dense (N, K) float64 distribution that sums to 1 per row.
    Entropy of this is a faithful proxy of the true softmax entropy because the
    tail mass (~5% on average) is small; spreading it uniformly is the maximum-
    entropy completion (a conservative, slightly-high entropy estimate).
    """
    N = topk_idx.shape[0]
    p = topk_prob.astype(np.float64)
    p = np.clip(p, 0.0, 1.0)
    top_mass = p.sum(1, keepdims=True)
    tail = np.clip(1.0 - top_mass, 0.0, 1.0)
    n_tail = max(K - topk_idx.shape[1], 1)
    per_tail = tail / n_tail                       # (N,1)
    dist = np.full((N, K), 0.0)
    dist += per_tail                               # uniform tail everywhere
    rows = np.arange(N)[:, None]
    # overwrite the top-k slots with their actual prob (replacing the tail share)
    dist[rows, topk_idx.astype(int)] = p
    # renormalise (numerical safety)
    s = dist.sum(1, keepdims=True); s[s == 0] = 1.0
    return dist / s


def entropy_rows(dist):
    p = np.clip(dist, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def top5_entropy_rows(topk_prob):
    """Entropy of the renormalised top-5 only (tail ignored) — robustness check."""
    p = topk_prob.astype(np.float64)
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# ── BBSE machinery (mirrors auroc_anomaly_detection.py) ─────────────────────────

def build_bbse(ref_true, ref_pred, ref_ent, K):
    cm = confusion_matrix(ref_true, ref_pred, labels=list(range(K))).astype(float)
    row = cm.sum(1, keepdims=True); row[row == 0] = 1
    C_T = (cm / row).T
    C_T_pinv = np.linalg.pinv(C_T, rcond=1e-2)
    counts = np.bincount(ref_true, minlength=K).astype(float)
    p_train = counts / counts.sum()
    h_ref = np.full(K, np.nan)
    for c in range(K):
        m = ref_true == c
        if m.sum() > 0:
            h_ref[c] = float(ref_ent[m].mean())
    return C_T_pinv, C_T, p_train, h_ref


def bbse_estimate(pred, K, C_T_pinv):
    q = np.bincount(pred, minlength=K).astype(float)
    if q.sum() == 0:
        return np.full(K, 1.0 / K)
    q /= q.sum()
    p = np.clip(C_T_pinv @ q, 0, None)
    if p.sum() > 0:
        p /= p.sum()
    return p


def mfwdd_severity(pred, ref_hist, weights, channels, K):
    """MFWDD-style global drift on the discrete top-1 class distribution.

    Full softmax channels are unavailable on Allot, so we use the model's output
    *class* distribution (one-hot of top-1) as the channel signal: W1 distance per
    channel between the window's class frequency and the reference's, weighted by
    train-prior importance (Sigma w = 1). For a 1-D categorical channel this W1
    reduces to |freq_win - freq_ref| on that channel.
    """
    q = np.bincount(pred, minlength=K).astype(float)
    q = q / max(q.sum(), 1.0)
    return float(np.dot(weights, np.abs(q[channels] - ref_hist[channels])))


# ── window synthesis (clean + viral spike), identical logic to CESNET ───────────

def clean_window(n_total, n_win, rng):
    return rng.choice(n_total, size=n_win, replace=True)


def viral_window(true, n_win, frac, cls, rng):
    base = np.where(true == cls)[0]
    n_c = int(round(frac * n_win))
    spike = rng.choice(base, size=n_c, replace=True)
    rest = rng.choice(len(true), size=n_win - n_c, replace=True)
    return np.concatenate([spike, rest])


# ── per-model run ───────────────────────────────────────────────────────────────

def load_window(path, K):
    z = np.load(path, allow_pickle=True)
    t = z['true_labels'].astype(int)
    p = z['pred_labels'].astype(int)
    dist = reconstruct_softmax(z['topk_idx'], z['topk_prob'], K)
    ent = entropy_rows(dist)
    ent5 = top5_entropy_rows(z['topk_prob'])
    return dict(win=int(z['window_index']), start=str(z['start_ts']),
                true=t, pred=p, ent=ent, ent5=ent5)


def run_model(slice_name, args):
    exp = REPO / 'exps' / 'allot_multimodal' / slice_name
    lm = json.load(open(exp / 'label_mapping.json'))
    K = len(lm)
    man = json.load(open(exp / 'slice_manifest.json'))
    train_win = man['window_index']
    files = sorted(glob.glob(str(exp / 'inference' / 'window_*.npz')))
    windows = {}
    for f in files:
        w = load_window(f, K)
        if w['true'].size == 0:
            continue
        windows[w['win']] = w

    # reference = training window if present, else earliest available healthy window
    ref_win = train_win if train_win in windows else min(windows)
    ref = windows[ref_win]
    C_T_pinv, C_T, p_train, h_ref = build_bbse(ref['true'], ref['pred'], ref['ent'], K)
    valid = ~np.isnan(h_ref)

    # MFWDD reference: top-k classes by train freq, Sigma w = 1
    ref_hist = np.bincount(ref['pred'], minlength=K).astype(float)
    ref_hist /= max(ref_hist.sum(), 1.0)
    chan = np.argsort(-p_train)[:args.mfwdd_channels]
    chan = chan[p_train[chan] > 0]
    w_mfwdd = p_train[chan] / p_train[chan].sum()

    base_h_train = float(np.dot(p_train[valid], h_ref[valid]))

    # ground truth: per-window true Macro-F1 (full window, not resampled)
    week_f1 = {w: f1_score(d['true'], d['pred'], labels=list(range(K)),
                           average='macro', zero_division=0)
               for w, d in windows.items()}
    is_anom = {w: int(week_f1[w] < args.f1_threshold) for w in windows}

    rng = np.random.default_rng(args.seed)
    test = {w: d for w, d in windows.items() if w != ref_win}

    records = []

    def score(idx, d, regime, win_f1=np.nan):
        ent = d['ent'][idx]; ent5 = d['ent5'][idx]; pred = d['pred'][idx]
        w_bbse = bbse_estimate(pred, K, C_T_pinv)
        rec = dict(
            win=d['win'], regime=regime, label=is_anom[d['win']], win_f1=float(win_f1),
            uncorrected=float(ent.mean() - base_h_train),
            uncorrected_top5=float(ent5.mean() - float(np.dot(p_train[valid], h_ref[valid]))),
            bbse=float(ent.mean() - float(np.dot(w_bbse[valid], h_ref[valid]))),
            mfwdd=mfwdd_severity(pred, ref_hist, w_mfwdd, chan, K),
        )
        records.append(rec)

    for w, d in test.items():
        n = d['true'].size
        for _ in range(args.clean_seeds):
            score(clean_window(n, min(args.n_win, n), rng), d, 'clean')

    spike_f1_healthy = []
    for w, d in test.items():
        cand = [c for c in range(K) if (d['true'] == c).sum() >= args.min_class_n]
        if not cand:
            continue
        for _ in range(args.viral_windows):
            cls = int(rng.choice(cand))
            frac = float(rng.uniform(args.frac_lo, args.frac_hi))
            idx = viral_window(d['true'], min(args.n_win, d['true'].size), frac, cls, rng)
            wf1 = f1_score((d['true'][idx] == cls).astype(int),
                           (d['pred'][idx] == cls).astype(int), zero_division=0)
            if is_anom[w] == 0:
                spike_f1_healthy.append(wf1)
            score(idx, d, 'viral', wf1)

    return dict(K=K, train_win=train_win, ref_win=ref_win, windows=sorted(windows),
                week_f1={int(k): float(v) for k, v in week_f1.items()},
                is_anom={int(k): int(v) for k, v in is_anom.items()},
                records=records,
                spike_f1_healthy=spike_f1_healthy)


# ── evaluation ──────────────────────────────────────────────────────────────────

DETECTORS = [
    ('uncorrected',      'Uncorrected entropy gap'),
    ('uncorrected_top5', 'Uncorrected entropy (top-5 only)'),
    ('mfwdd',            'MFWDD-style global drift'),
    ('bbse',             'BBSE-corrected residual (ours)'),
]


def fpr_at_tpr(y, sc, t):
    fpr, tpr, _ = roc_curve(y, sc)
    ok = tpr >= t
    return float(fpr[ok][0]) if ok.any() else 1.0


def evaluate(records, tpr_targets):
    out = {}
    for regime in ['clean', 'viral']:
        r = [x for x in records if x['regime'] == regime]
        y = np.array([x['label'] for x in r])
        d_out = {'n': len(r), 'pos': int(y.sum()), 'neg': int((1 - y).sum()),
                 'detectors': {}}
        if y.sum() == 0 or y.sum() == len(y):
            d_out['note'] = 'degenerate: only one class present'
            out[regime] = d_out
            continue
        for key, label in DETECTORS:
            sc = np.array([x[key] for x in r])
            d_out['detectors'][key] = {
                'auroc': float(roc_auc_score(y, sc)),
                'fpr_at_tpr': {f'{t:.2f}': fpr_at_tpr(y, sc, t) for t in tpr_targets},
            }
        out[regime] = d_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--f1_threshold', type=float, default=0.57,
                    help='anomaly iff window Macro-F1 < this (clean Sep-2024 ~0.61-0.73 '
                         'vs degraded Feb/Mar-2025 ~0.41-0.52 -> 0.57 cleanly splits)')
    ap.add_argument('--n_win', type=int, default=8000)
    ap.add_argument('--clean_seeds', type=int, default=6)
    ap.add_argument('--viral_windows', type=int, default=12)
    ap.add_argument('--frac_lo', type=float, default=0.30)
    ap.add_argument('--frac_hi', type=float, default=0.85)
    ap.add_argument('--min_class_n', type=int, default=20)
    ap.add_argument('--mfwdd_channels', type=int, default=80)
    ap.add_argument('--tpr_targets', type=float, nargs='+', default=[0.90, 0.95])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='results/allot_monitor_v01')
    ap.add_argument('--fig_dir', default='figs')
    args = ap.parse_args()

    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO / args.fig_dir; fig_dir.mkdir(parents=True, exist_ok=True)

    results = {'config': vars(args), 'protocol': {
        'ground_truth': 'anomaly iff window true Macro-F1 < f1_threshold (natural '
                        'Sep-2024 healthy vs Feb/Mar-2025 degraded domain split)',
        'softmax': 'reconstructed from saved top-5 (uniform tail completion); '
                   'top-5 captures ~95% mass on average',
        'reference_window': 'training window if inferred (quarter=13), else earliest '
                            'healthy window (early=01)',
        'closed_world': 'Allot private/closed-world; unseen appIds dropped at inference',
    }, 'models': {}}

    print('=== ALLOT BBSE-monitor AUROC ===')
    for slice_name in ['early_eq', 'quarter_eq']:
        print(f'\n--- {slice_name} ---')
        m = run_model(slice_name, args)
        n_pos = sum(m['is_anom'].values()); n_neg = len(m['is_anom']) - n_pos
        print(f"  K={m['K']}  train_win={m['train_win']}  ref_win={m['ref_win']}  "
              f"windows={len(m['windows'])}  degraded(+)={n_pos} healthy(-)={n_neg}")
        ev = evaluate(m['records'], args.tpr_targets)
        spike = m['spike_f1_healthy']
        for regime in ['clean', 'viral']:
            d = ev[regime]
            print(f"  [{regime}] n={d['n']} pos={d.get('pos')} neg={d.get('neg')}"
                  + (f"  ({d['note']})" if 'note' in d else ''))
            for key, label in DETECTORS:
                if key in d.get('detectors', {}):
                    dd = d['detectors'][key]
                    far = '  '.join(f"FAR@{int(t*100)}={dd['fpr_at_tpr'][f'{t:.2f}']:.3f}"
                                    for t in args.tpr_targets)
                    print(f"     {label:34s} AUROC={dd['auroc']:.3f}   {far}")
        results['models'][slice_name] = {
            'K': m['K'], 'train_win': m['train_win'], 'ref_win': m['ref_win'],
            'n_windows': len(m['windows']), 'n_degraded': n_pos, 'n_healthy': n_neg,
            'week_macro_f1': m['week_f1'], 'is_anomaly': m['is_anom'],
            'benign_spiked_app_f1_healthy_mean':
                float(np.mean(spike)) if spike else None,
            'regimes': ev,
        }

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir / 'metrics.json'}")

    # ── figure: ROC curves per model, clean + viral ────────────────────────────
    colors = {'uncorrected': '#e08020', 'uncorrected_top5': '#f4a261',
              'mfwdd': '#7b3fa0', 'bbse': '#d7191c'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for i, slice_name in enumerate(['early_eq', 'quarter_eq']):
        recs = []
        # rebuild records for plotting from the just-computed run is costly; reload
        # cheaply by re-running scoring is wasteful -> instead reuse via re-run:
        m = run_model(slice_name, args)
        recs = m['records']
        for j, regime in enumerate(['clean', 'viral']):
            ax = axes[i, j]
            r = [x for x in recs if x['regime'] == regime]
            y = np.array([x['label'] for x in r])
            ax.plot([0, 1], [0, 1], color='#bbbbbb', ls='--', lw=1)
            if y.sum() and y.sum() != len(y):
                for key, label in DETECTORS:
                    sc = np.array([x[key] for x in r])
                    fpr, tpr, _ = roc_curve(y, sc)
                    auc = roc_auc_score(y, sc)
                    ax.plot(fpr, tpr, color=colors[key], lw=2.0,
                            label=f'{label} (AUROC={auc:.3f})')
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('False Alarm Rate'); ax.set_ylabel('Detection Rate')
            ax.set_title(f'{slice_name}  [{regime}]', fontweight='bold')
            ax.legend(fontsize=8.5, loc='lower right')
            ax.grid(True, alpha=0.25)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('ALLOT label-free monitor: genuine multi-month covariate degradation '
                 'vs benign volume volatility', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / 'fig_allot_monitor_auroc.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {fig_dir / 'fig_allot_monitor_auroc.png'}")


if __name__ == '__main__':
    main()
