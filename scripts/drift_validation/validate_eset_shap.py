#!/usr/bin/env python
"""
Case-study SHAP feature attribution on the latent-space drift of eset-edtd
(class 57) around the frozen Week-16 recall cliff at ISO week 17, 2022.

This is the clean-regime (Week-16 forward) counterpart of validate_drift_shap.py
(which targets docker-registry). It reuses the shared extraction logic in
drift_features.py but retargets:
    * APP / CLASS_ID    -> eset-edtd / 57
    * source encoder    -> frozen Week-16 Multimodal_CESNET
    * pre/post split     -> pre = weeks 14-16 (healthy, source side),
                            post = weeks 17-19 (after the ~wk17 cliff)

Real-world anchor: ESET *Dynamic Threat Defense* (EDTD) was renamed to
*ESET LiveGuard Advanced* on 27 Apr 2022 (~ISO week 17), shipping backend /
endpoint / proxy-component changes -- a plausible encrypted-signature-shift
mechanism. The timing match (recall cliff ~wk17 <-> rename ~wk17) is a
coincidence-plus-mechanism anchor, not proof of causation.

Outputs (results/eset_drift/, figs/):
    eset_latent_jump.csv / fig_eset_edtd_latent_jump.png
    eset_encoder_shap_ranking.csv
    fig_shap_eset_edtd.png            (the headline encoder-SHAP bar, -> fig:shap)
    eset_shap_ranking.csv
    fig_eset_edtd_shap_beeswarm.png
    eset_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---- repo path bootstrap (mirror validate_drift_shap.py) -------------------
_root = Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_features as DF  # noqa: E402

# -- retarget the shared extractor to eset-edtd and the wk17 split -----------
DF.APP = 'eset-edtd'
DF.CLASS_ID = 57
PRE_WEEKS = (14, 15, 16)
POST_WEEKS = (17, 18, 19)
ALL_WEEKS = range(14, 20)


def _week_to_period(week):
    if week in PRE_WEEKS:
        return 'pre'
    if week in POST_WEEKS:
        return 'post'
    return 'other'


DF.week_to_period = _week_to_period  # so extract_week tags rows correctly

from drift_features import (extract_week, load_all, load_norm_stats,  # noqa: E402
                            FLOWSTATS_NAMES)
from temporal_generalization import load_model_from_checkpoint  # noqa: E402
from data_utils.cesnet_labels import load_label_mapping  # noqa: E402

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATASET_ROOT = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
CKPT = _root / 'exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth'
CFG = _root / 'exps/cesnet_multimodal_each_week_train_v01/week_16/config.json'
FIG_DIR = _root / 'figs'
RES_DIR = _root / 'results/eset_drift'
RES_DIR.mkdir(parents=True, exist_ok=True)

META = {'week', 'period', 'tls_ja3', 'tls_sni', 'dst_port'}

PPI_NAMES = ([f'PPI_IPT_{i}' for i in range(30)]
             + [f'PPI_DIR_{i}' for i in range(30)]
             + [f'PPI_SIZE_{i}' for i in range(30)])
INPUT_NAMES = PPI_NAMES + FLOWSTATS_NAMES          # 90 + 44 = 134


def normalize(X_ppi, X_flow):
    pm, ps, fm, fs = load_norm_stats()
    ppi = (X_ppi - pm) / (ps + 1e-8)
    flow = (X_flow - fm) / (fs + 1e-8)
    return ppi.astype(np.float32), flow.astype(np.float32)


@torch.no_grad()
def embed(model, X_ppi_n, X_flow_n, bs=512):
    outs = []
    for i in range(0, len(X_ppi_n), bs):
        ppi = torch.tensor(X_ppi_n[i:i + bs], device=DEVICE)
        flow = torch.tensor(X_flow_n[i:i + bs], device=DEVICE)
        outs.append(model.forward_features(ppi, flow).cpu().numpy())
    return np.concatenate(outs)


class EncoderDrift(nn.Module):
    """Wrap the encoder so output = L2 distance of embedding from the Week-16
    eset-edtd centroid. Input is the flat 134-d normalized vector."""
    def __init__(self, model, centroid):
        super().__init__()
        self.model = model
        self.register_buffer('centroid', torch.tensor(centroid, dtype=torch.float32))

    def forward(self, x):
        ppi = x[:, :90].reshape(-1, 3, 30)
        flow = x[:, 90:]
        feats = self.model.forward_features(ppi, flow)
        return torch.sqrt(((feats - self.centroid) ** 2).sum(1, keepdim=True) + 1e-8)


def group_ppi(values, names):
    """Collapse the 90 PPI dims into 3 channel-level features; keep flowstats."""
    s = pd.Series(values, index=names)
    grouped = {
        'PPI_IPT(seq)': s[[n for n in names if n.startswith('PPI_IPT_')]].sum(),
        'PPI_DIR(seq)': s[[n for n in names if n.startswith('PPI_DIR_')]].sum(),
        'PPI_SIZE(seq)': s[[n for n in names if n.startswith('PPI_SIZE_')]].sum(),
    }
    for n in FLOWSTATS_NAMES:
        grouped[n] = s[n]
    return pd.Series(grouped)


def main():
    print(f'Device: {DEVICE}')
    label_mapping, num_classes = load_label_mapping(DATASET_ROOT)
    model = load_model_from_checkpoint(CKPT, CFG, num_classes, DEVICE)
    model.eval()
    print(f'Loaded Week-16 encoder ({num_classes} classes) from {CKPT.name}')

    lines = ['=' * 78,
             'CASE STUDY -- SHAP attribution of eset-edtd latent drift '
             '(frozen Week-16, cliff ~wk17)',
             '=' * 78, f'device={DEVICE}\n']

    # ---- (A1) Week-16 baseline centroid + per-week latent distance ----------
    print('\nComputing Week-16 baseline embedding centroid ...')
    df16, xp16, xf16 = extract_week(16, max_flows=4000)
    ppi16, flow16 = normalize(xp16, xf16)
    emb16 = embed(model, ppi16, flow16)
    centroid = emb16.mean(0)
    print(f'  Week-16 eset-edtd flows: {len(df16)}  centroid dim={centroid.shape[0]}')

    jump_rows = []
    for w in ALL_WEEKS:
        dfx, xp, xf = extract_week(w, max_flows=4000)
        if dfx is None or len(dfx) == 0:
            continue
        ppin, flown = normalize(xp, xf)
        emb = embed(model, ppin, flown)
        dist = np.linalg.norm(emb - centroid, axis=1)
        jump_rows.append({'week': w, 'period': dfx.period.iloc[0],
                          'n': len(dfx),
                          'mean_dist_from_W16': float(dist.mean()),
                          'median_dist_from_W16': float(np.median(dist)),
                          'mean_emb_norm': float(np.linalg.norm(emb, axis=1).mean())})
    jump = pd.DataFrame(jump_rows)
    jump.to_csv(RES_DIR / 'eset_latent_jump.csv', index=False)
    pre_d = jump[jump.week <= 16].mean_dist_from_W16.mean()
    post_d = jump[jump.week >= 17].mean_dist_from_W16.mean()
    step = jump.set_index('week').mean_dist_from_W16
    boundary_jump = step[17] - step[16]
    lines.append('(A) LATENT-SPACE JUMP (mean L2 distance from Week-16 baseline)')
    lines.append('-' * 78)
    for _, r in jump.iterrows():
        lines.append(f"  W{int(r.week):02d} [{r.period:>4s}]  dist={r.mean_dist_from_W16:8.3f}")
    lines.append(f'  pre(14-16) avg = {pre_d:.3f}   post(17-19) avg = {post_d:.3f}'
                 f'   step W16->W17 = {boundary_jump:+.3f}\n')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(jump.week, jump.mean_dist_from_W16, 'o-', lw=2, color='#2c3e50')
    ax.axvspan(16.5, 19.5, color='#d62728', alpha=0.08)
    ax.axvline(16.5, color='#d62728', ls='--', lw=1.5, label='wk16->wk17 cliff')
    ax.set_xlabel('week (2022)')
    ax.set_ylabel('mean L2 distance of embedding\nfrom Week-16 eset-edtd centroid')
    ax.set_title('eset-edtd latent-space drift vs Week-16 baseline')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_eset_edtd_latent_jump.png', dpi=130)
    plt.close(fig)

    # ---- load full pre/post set (shared by A2 and B) ------------------------
    print('\nLoading pre/post eset-edtd flows ...')
    df, X_ppi, X_flow = load_all(weeks=ALL_WEEKS, max_flows=4000)
    df = df[df.period.isin(['pre', 'post'])].reset_index(drop=True)
    keep = df.index.to_numpy()
    # rebuild aligned model tensors for the kept rows
    # (load_all concatenates in ALL_WEEKS order; df row order matches)
    # Re-extract aligned arrays the same way load_all does, then mask.
    dfs, ppis, flows = [], [], []
    for w in ALL_WEEKS:
        d, xp, xf = extract_week(w, max_flows=4000)
        if d is None or len(d) == 0:
            continue
        dfs.append(d); ppis.append(xp); flows.append(xf)
    df_full = pd.concat(dfs, ignore_index=True)
    X_ppi = np.concatenate(ppis)
    X_flow = np.concatenate(flows)
    mask = df_full.period.isin(['pre', 'post']).to_numpy()
    df = df_full[mask].reset_index(drop=True)
    X_ppi = X_ppi[mask]
    X_flow = X_flow[mask]
    y = (df.period == 'post').astype(int).to_numpy()

    # ---- (A2) Encoder SHAP: attribute latent distance to model inputs -------
    print('Running GradientExplainer on the encoder (latent-drift objective) ...')
    drift_model = EncoderDrift(model, centroid).to(DEVICE).eval()
    ppi_n, flow_n = normalize(X_ppi, X_flow)
    X_flat = np.concatenate([ppi_n.reshape(len(ppi_n), -1), flow_n], axis=1)  # (N,134)
    import os
    n_bg = int(os.environ.get('ESET_SHAP_BG', '60'))
    n_ex = int(os.environ.get('ESET_SHAP_EX', '400'))
    n_samp = int(os.environ.get('ESET_SHAP_NSAMPLES', '60'))
    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(X_flat), min(n_bg, len(X_flat)), replace=False)
    ex_idx = rng.choice(len(X_flat), min(n_ex, len(X_flat)), replace=False)
    background = torch.tensor(X_flat[bg_idx], device=DEVICE)
    explain = torch.tensor(X_flat[ex_idx], device=DEVICE)
    g_expl = shap.GradientExplainer(drift_model, background)
    enc_sv = g_expl.shap_values(explain, nsamples=n_samp)
    if isinstance(enc_sv, list):
        enc_sv = enc_sv[0]
    enc_sv = np.asarray(enc_sv).reshape(len(ex_idx), -1)        # (N,134)
    enc_mean_abs = np.abs(enc_sv).mean(0)
    enc_grouped = group_ppi(enc_mean_abs, INPUT_NAMES).sort_values(ascending=False)
    enc_grouped.to_csv(RES_DIR / 'eset_encoder_shap_ranking.csv',
                       header=['mean_abs_shap'])
    lines.append('(A2) ENCODER SHAP -- top features driving distance from Week-16 baseline')
    lines.append('-' * 78)
    for name, v in enc_grouped.head(10).items():
        lines.append(f'  {name:24s} mean|SHAP|={v:.4f}')
    lines.append('')

    fig, ax = plt.subplots(figsize=(9, 6))
    top = enc_grouped.head(12)[::-1]
    ax.barh(range(len(top)), top.values, color='#c0392b')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel('mean |SHAP| on latent distance-from-Week-16')
    ax.set_title('Encoder SHAP: inputs driving eset-edtd latent drift (~wk17)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_shap_eset_edtd.png', dpi=130)
    plt.close(fig)

    # ---- (B) pre-vs-post domain-classifier SHAP (interpretable) -------------
    print('Fitting pre-vs-post domain classifier + TreeExplainer ...')
    num_cols = [c for c in df.columns if c not in META]
    Xtab = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xtr, Xte, ytr, yte = train_test_split(Xtab, y, test_size=0.3,
                                          random_state=0, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=0)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    lines.append('(B) PRE-vs-POST DOMAIN CLASSIFIER (GradientBoosting)')
    lines.append('-' * 78)
    lines.append(f'  Hold-out AUC pre-vs-post = {auc:.4f}  '
                 f'(0.5=no drift, 1.0=fully separable)\n')

    expl = shap.TreeExplainer(clf)
    Xsh = Xte.sample(min(2000, len(Xte)), random_state=0)
    sv = expl.shap_values(Xsh)
    mean_abs = np.abs(sv).mean(0)
    rank = (pd.Series(mean_abs, index=num_cols)
            .sort_values(ascending=False))
    direction = {}
    for c in num_cols:
        corr = np.corrcoef(Xsh[c], sv[:, num_cols.index(c)])[0, 1]
        direction[c] = 'higher->POST' if (corr >= 0) else 'higher->PRE'
    rank_df = pd.DataFrame({'mean_abs_shap': rank,
                            'pre_mean': [df[df.period == 'pre'][c].mean() for c in rank.index],
                            'post_mean': [df[df.period == 'post'][c].mean() for c in rank.index],
                            'direction': [direction[c] for c in rank.index]})
    rank_df.to_csv(RES_DIR / 'eset_shap_ranking.csv')
    lines.append('  Top-10 drift-inducing features (mean |SHAP|):')
    for c in rank.index[:10]:
        lines.append(f'    {c:22s} |SHAP|={rank[c]:.4f}  {direction[c]}  '
                     f'(pre={df[df.period=="pre"][c].mean():.2f} -> '
                     f'post={df[df.period=="post"][c].mean():.2f})')

    plt.figure()
    shap.summary_plot(sv, Xsh, max_display=12, show=False)
    plt.title(f'eset-edtd pre-vs-post drift SHAP (AUC={auc:.3f})', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_eset_edtd_shap_beeswarm.png', dpi=130, bbox_inches='tight')
    plt.close()

    summary = '\n'.join(lines)
    (RES_DIR / 'eset_summary.txt').write_text(summary + '\n')
    print('\n' + summary)
    print('\nSaved headline figure to', FIG_DIR / 'fig_shap_eset_edtd.png')


if __name__ == '__main__':
    main()
