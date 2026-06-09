#!/usr/bin/env python
"""
Task 2 -- SHAP feature attribution on the latent-space drift of docker-registry
(class 49) around the Week 27 -> 28, 2022 concept jump.

Two complementary, model-faithful analyses are run:

(A) LATENT-JUMP VERIFICATION + ENCODER SHAP
    The trained Week-1 Multimodal_CESNET encoder maps each flow's
    (PPI 3x30, flowstats 44) inputs to a 600-d embedding. We:
      - build the Week-1 docker-registry embedding centroid (the "baseline");
      - measure each week's mean L2 distance from that centroid -> the jump;
      - wrap the encoder so its scalar output is *distance-from-baseline* and
        use shap.GradientExplainer to attribute that latent drift back to the
        134 raw model inputs, aggregated to named features.

(B) PRE-vs-POST DOMAIN-CLASSIFIER SHAP  (the interpretable headline)
    Treat pre(W25-27)=0 / post(W28-30)=1 as a binary target over interpretable
    per-flow features, fit a gradient-boosted tree, and use shap.TreeExplainer
    to rank which network characteristics drive the model to distinguish the
    two periods. Beeswarm + bar plots are saved.

Outputs (results/docker_drift/, figs/docker_drift/):
    task2_latent_jump.csv / .png
    task2_shap_ranking.csv
    task2_encoder_shap_ranking.csv
    task2_shap_beeswarm.png / task2_shap_bar.png
    task2_encoder_shap_bar.png
    task2_summary.txt
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

# ---- repo path bootstrap (mirror run_inference.py) -------------------------
_root = Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_features import (load_all, extract_week, load_norm_stats,  # noqa: E402
                            FLOWSTATS_NAMES)
from temporal_generalization import load_model_from_checkpoint  # noqa: E402
from data_utils.cesnet_labels import load_label_mapping  # noqa: E402

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATASET_ROOT = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
CKPT = _root / 'exps/cesnet_multimodal_each_week_train_v01/week_1/weights/best_model.pth'
CFG = _root / 'exps/cesnet_multimodal_each_week_train_v01/week_1/config.json'
FIG_DIR = _root / 'figs/docker_drift'
RES_DIR = _root / 'results/docker_drift'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

META = {'week', 'period', 'tls_ja3', 'tls_sni', 'dst_port'}

# 90 PPI input dims, grouped by the 3 channels (each 30 packets long)
PPI_NAMES = ([f'PPI_IPT_{i}' for i in range(30)]
             + [f'PPI_DIR_{i}' for i in range(30)]
             + [f'PPI_SIZE_{i}' for i in range(30)])
INPUT_NAMES = PPI_NAMES + FLOWSTATS_NAMES          # 90 + 44 = 134


# ---------------------------------------------------------------------------
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
    """Wrap the encoder so output = L2 distance of embedding from the Week-1
    docker-registry centroid. Input is the flat 134-d normalized vector."""
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


# ---------------------------------------------------------------------------
def main():
    print(f'Device: {DEVICE}')
    label_mapping, num_classes = load_label_mapping(DATASET_ROOT)
    model = load_model_from_checkpoint(CKPT, CFG, num_classes, DEVICE)
    model.eval()
    print(f'Loaded Week-1 encoder ({num_classes} classes) from {CKPT.name}')

    lines = ['=' * 78,
             'TASK 2 -- SHAP attribution of docker-registry latent drift (W27->W28)',
             '=' * 78, f'device={DEVICE}\n']

    # ---- (A1) Week-1 baseline centroid + per-week latent distance -----------
    print('\nComputing Week-1 baseline embedding centroid ...')
    df1, xp1, xf1 = extract_week(1, max_flows=4000)
    ppi1, flow1 = normalize(xp1, xf1)
    emb1 = embed(model, ppi1, flow1)
    centroid = emb1.mean(0)
    print(f'  Week-1 docker-registry flows: {len(df1)}  centroid dim={centroid.shape[0]}')

    jump_rows = []
    for w in range(25, 31):
        dfx, xp, xf = extract_week(w, max_flows=4000)
        ppin, flown = normalize(xp, xf)
        emb = embed(model, ppin, flown)
        dist = np.linalg.norm(emb - centroid, axis=1)
        jump_rows.append({'week': w, 'period': dfx.period.iloc[0],
                          'n': len(dfx),
                          'mean_dist_from_W1': float(dist.mean()),
                          'median_dist_from_W1': float(np.median(dist)),
                          'mean_emb_norm': float(np.linalg.norm(emb, axis=1).mean())})
    jump = pd.DataFrame(jump_rows)
    jump.to_csv(RES_DIR / 'task2_latent_jump.csv', index=False)
    pre_d = jump[jump.week <= 27].mean_dist_from_W1.mean()
    post_d = jump[jump.week >= 28].mean_dist_from_W1.mean()
    step = jump.set_index('week').mean_dist_from_W1
    boundary_jump = step[28] - step[27]
    lines.append('(A) LATENT-SPACE JUMP (mean L2 distance from Week-1 baseline)')
    lines.append('-' * 78)
    for _, r in jump.iterrows():
        lines.append(f"  W{int(r.week):02d} [{r.period:>4s}]  dist={r.mean_dist_from_W1:8.3f}")
    lines.append(f'  pre(25-27) avg = {pre_d:.3f}   post(28-30) avg = {post_d:.3f}'
                 f'   step W27->W28 = {boundary_jump:+.3f}\n')

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(jump.week, jump.mean_dist_from_W1, 'o-', lw=2, color='#2c3e50')
    ax.axvspan(27.5, 30.5, color='#d62728', alpha=0.08)
    ax.axvline(27.5, color='#d62728', ls='--', lw=1.5, label='27->28 jump')
    ax.set_xlabel('week (2022)')
    ax.set_ylabel('mean L2 distance of embedding\nfrom Week-1 docker-registry centroid')
    ax.set_title('docker-registry latent-space drift vs Week-1 baseline')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'task2_latent_jump.png', dpi=130)
    plt.close(fig)

    # ---- load full pre/post set (shared by A2 and B) ------------------------
    print('\nLoading pre/post docker-registry flows ...')
    df, X_ppi, X_flow = load_all(weeks=range(25, 31), max_flows=4000)
    y = (df.period == 'post').astype(int).to_numpy()

    # ---- (A2) Encoder SHAP: attribute latent distance to model inputs -------
    print('Running GradientExplainer on the encoder (latent-drift objective) ...')
    drift_model = EncoderDrift(model, centroid).to(DEVICE).eval()
    ppi_n, flow_n = normalize(X_ppi, X_flow)
    X_flat = np.concatenate([ppi_n.reshape(len(ppi_n), -1), flow_n], axis=1)  # (N,134)
    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(X_flat), 200, replace=False)
    ex_idx = rng.choice(len(X_flat), 1200, replace=False)
    background = torch.tensor(X_flat[bg_idx], device=DEVICE)
    explain = torch.tensor(X_flat[ex_idx], device=DEVICE)
    g_expl = shap.GradientExplainer(drift_model, background)
    enc_sv = g_expl.shap_values(explain)
    if isinstance(enc_sv, list):
        enc_sv = enc_sv[0]
    enc_sv = np.asarray(enc_sv).reshape(len(ex_idx), -1)        # (1200,134)
    enc_mean_abs = np.abs(enc_sv).mean(0)
    enc_grouped = group_ppi(enc_mean_abs, INPUT_NAMES).sort_values(ascending=False)
    enc_grouped.to_csv(RES_DIR / 'task2_encoder_shap_ranking.csv',
                       header=['mean_abs_shap'])
    lines.append('(A2) ENCODER SHAP -- top features driving distance from Week-1 baseline')
    lines.append('-' * 78)
    for name, v in enc_grouped.head(10).items():
        lines.append(f'  {name:24s} mean|SHAP|={v:.4f}')
    lines.append('')

    fig, ax = plt.subplots(figsize=(9, 6))
    top = enc_grouped.head(12)[::-1]
    ax.barh(range(len(top)), top.values, color='#8e44ad')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel('mean |SHAP| on latent distance-from-Week-1')
    ax.set_title('Encoder SHAP: inputs driving docker-registry latent drift')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'task2_encoder_shap_bar.png', dpi=130)
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
    # signed direction: + => higher value pushes toward POST
    direction = {}
    for c in num_cols:
        corr = np.corrcoef(Xsh[c], sv[:, num_cols.index(c)])[0, 1]
        direction[c] = 'higher->POST' if corr >= 0 else 'higher->PRE'
    rank_df = pd.DataFrame({'mean_abs_shap': rank,
                            'pre_mean': [df[df.period == 'pre'][c].mean() for c in rank.index],
                            'post_mean': [df[df.period == 'post'][c].mean() for c in rank.index],
                            'direction': [direction[c] for c in rank.index]})
    rank_df.to_csv(RES_DIR / 'task2_shap_ranking.csv')
    lines.append('  Top-10 drift-inducing features (mean |SHAP|):')
    for c in rank.index[:10]:
        lines.append(f'    {c:22s} |SHAP|={rank[c]:.4f}  {direction[c]}  '
                     f'(pre={df[df.period=="pre"][c].mean():.2f} -> '
                     f'post={df[df.period=="post"][c].mean():.2f})')

    # beeswarm
    plt.figure()
    shap.summary_plot(sv, Xsh, max_display=12, show=False)
    plt.title(f'docker-registry pre-vs-post drift SHAP (AUC={auc:.3f})', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'task2_shap_beeswarm.png', dpi=130, bbox_inches='tight')
    plt.close()
    # bar
    plt.figure()
    shap.summary_plot(sv, Xsh, plot_type='bar', max_display=12, show=False)
    plt.title('Top drift-inducing features (mean |SHAP|)', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'task2_shap_bar.png', dpi=130, bbox_inches='tight')
    plt.close()

    summary = '\n'.join(lines)
    (RES_DIR / 'task2_summary.txt').write_text(summary + '\n')
    print('\n' + summary)
    print('\nSaved figures to', FIG_DIR)


if __name__ == '__main__':
    main()
