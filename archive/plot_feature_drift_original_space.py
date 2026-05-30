#!/usr/bin/env python
"""
Original-space drift analysis: discrete "teleportation" vs. continuous drift.

Metrics per class per week, relative to the reference week:
  1. Centroid L2          – raw centroid shift in embedding space
  2. Mahalanobis distance – centroid shift normalised by ref-class spread
                            D = sqrt((μ_t - μ_0)^T Σ_0^{-1} (μ_t - μ_0))
                            Step-function → teleportation; ramp → continuous.
  3. Sliced Wasserstein   – full-distribution distance (random 1D projections)
  4. Fragmentation ratio  – Tr(Σ_t) / Tr(Σ_0)
                            ≈1 → cluster moved as a unit; >>1 → it split/exploded

Outputs (all in --output_dir):
  drift_metrics_aggregate.png   – 4-panel aggregate (mean ± std across classes)
  drift_velocity.png            – cumulative drift + consecutive-week centroid Δ
  drift_mahalanobis_heatmap.png – class × week Mahalanobis heatmap
  drift_fragmentation.png       – fragmentation ratio per class over time

Cache: drift_cache_ref<N>.pkl   – skip recompute on re-run; pass --recompute to force

Usage:
    python plot_feature_drift_original_space.py
    python plot_feature_drift_original_space.py --recompute
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from tqdm import tqdm


# ── data helpers ──────────────────────────────────────────────────────────────

def week_num(path):
    return int(re.search(r'(\d+)$', Path(path).stem).group(1))


def load_embeddings_by_class(npz_path):
    d = np.load(npz_path)
    emb = d['embeddings']
    idx = d['embedding_indices']
    cls_labels = d['true_labels'][idx]
    out = {}
    for c in np.unique(cls_labels):
        out[int(c)] = emb[cls_labels == c].astype(np.float64)
    return out


def load_class_names(dataset_root):
    sys.path.insert(0, str(Path(__file__).parent))
    from train_per_week_cesnet import load_label_mapping
    mapping, _ = load_label_mapping(Path(dataset_root))
    return {v: k for k, v in mapping.items()}


# ── metric helpers ────────────────────────────────────────────────────────────

def sliced_wasserstein(X, Y, n_proj, rng):
    D = X.shape[1]
    dirs = rng.randn(n_proj, D)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in dirs]))


def regularised_inv_cov(X, reg=0.01):
    """Regularised inverse covariance for Mahalanobis.
    reg fraction of mean diagonal variance is added to the diagonal."""
    cov = np.cov(X.T)
    alpha = reg * np.trace(cov) / cov.shape[0]
    return np.linalg.inv(cov + alpha * np.eye(cov.shape[0])), float(np.trace(cov))


def mahalanobis(mu_t, mu_0, inv_cov_0):
    diff = mu_t - mu_0
    return float(np.sqrt(max(0.0, diff @ inv_cov_0 @ diff)))


# ── aggregate helpers ─────────────────────────────────────────────────────────

def agg_over_weeks(dist_dict, weeks):
    means, stds, medians = [], [], []
    for wn in weeks:
        vals = [dist_dict[c][wn] for c in dist_dict
                if wn in dist_dict[c] and np.isfinite(dist_dict[c][wn])]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            medians.append(np.median(vals))
        else:
            means.append(np.nan); stds.append(np.nan); medians.append(np.nan)
    return np.array(means), np.array(stds), np.array(medians)


def top_classes_by_mean(dist_dict, ref_wn, n):
    scores = {c: np.mean([v for wn, v in d.items()
                           if wn != ref_wn and np.isfinite(v)])
              for c, d in dist_dict.items() if d}
    return sorted(scores, key=scores.get, reverse=True)[:n]


def random_classes_with_coverage(dist_dict, n, min_cov, rng):
    """Random subset of classes whose dict has ≥ min_cov week entries."""
    candidates = [c for c, d in dist_dict.items() if len(d) >= min_cov]
    rng.shuffle(candidates)
    return candidates[:n]


# ── plot helpers ──────────────────────────────────────────────────────────────

def band_plot(ax, weeks, mean, std, color, label, ref_wn, ylabel, title):
    ax.fill_between(weeks, mean - std, mean + std, alpha=0.2, color=color)
    ax.plot(weeks, mean, color=color, lw=2.5, label=label)
    ax.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.6, label=f'Ref week {ref_wn}')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def heatmap_plot(matrix, rows, col_labels, ref_col_idx, cbar_label, title, output_path):
    h = max(8, len(rows) * 0.22)
    fig, ax = plt.subplots(figsize=(20, h))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(w) for w in col_labels], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    if ref_col_idx is not None:
        ax.axvline(ref_col_idx, color='blue', lw=2, ls='--', alpha=0.6)
    ax.set_xlabel('Week number')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ── neighbourhood plot ───────────────────────────────────────────────────────

def plot_class_neighborhood(focal_classes, anchor_classes, centroids, all_weeks,
                             ref_wn, class_names, output_path):
    """
    For each focal class: distance from its moving centroid to each fixed anchor
    centroid (anchored at ref_wn), plotted over time.

    Anchor centroids never move — only the focal class centroid changes week to week.
    A sudden drop toward an anchor = discrete migration into that class's territory.
    A gradual decrease = continuous drift.
    """
    anchor_ref = {c: centroids[c][ref_wn] for c in anchor_classes
                  if ref_wn in centroids.get(c, {})}

    n_focal = len(focal_classes)
    fig, axes = plt.subplots(n_focal, 1, figsize=(16, 4.5 * n_focal), sharex=True)
    if n_focal == 1:
        axes = [axes]

    for ax, fc in zip(axes, focal_classes):
        fname = class_names.get(fc, f'cls_{fc}')
        fc_ref = centroids.get(fc, {}).get(ref_wn)
        if fc_ref is None:
            ax.set_title(f'{fname} — no ref-week centroid'); continue

        # Sort anchors by initial distance so colour encodes proximity
        init_dists = {c: float(np.linalg.norm(fc_ref - anchor_ref[c]))
                      for c in anchor_ref if c != fc}
        sorted_anchors = sorted(init_dists, key=init_dists.get)

        cmap = plt.cm.plasma
        n_a  = len(sorted_anchors)
        for rank, ac in enumerate(sorted_anchors):
            aname = class_names.get(ac, f'cls_{ac}')
            color = cmap(0.15 + 0.7 * rank / max(n_a - 1, 1))
            wns, dists = [], []
            for wn in all_weeks:
                mu = centroids.get(fc, {}).get(wn)
                if mu is not None:
                    wns.append(wn)
                    dists.append(float(np.linalg.norm(mu - anchor_ref[ac])))
            label = aname if rank < 5 else '_nolegend_'
            ax.plot(wns, dists, lw=1.1, alpha=0.65, color=color, label=label)

        # Self-drift: distance from focal centroid to its own centroid at ref_wn
        self_wns, self_dists = [], []
        for wn in all_weeks:
            mu = centroids.get(fc, {}).get(wn)
            if mu is not None:
                self_wns.append(wn)
                self_dists.append(float(np.linalg.norm(mu - fc_ref)))
        ax.plot(self_wns, self_dists, color='red', lw=2.5, ls='--', zorder=5,
                label=f'{fname} self-drift (from week {ref_wn})')

        ax.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.5)
        ax.set_ylabel('L2 distance')
        ax.set_title(f'Focal: {fname} ({fc})', fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Week number')
    fig.suptitle(
        f'Class neighbourhood drift — {n_focal} focal classes vs. '
        f'{len(anchor_classes)} fixed anchors\n'
        f'Anchor centroids fixed at week {ref_wn}.  '
        f'Colour = proximity rank (purple=nearest, yellow=farthest).  '
        f'Red dashed = focal self-drift.',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ── PCA trajectory plot ──────────────────────────────────────────────────────

def _palette(n):
    palettes = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors = []
    for cm in palettes:
        colors.extend([cm(i / 19) for i in range(20)])
    while len(colors) < n:
        colors.extend(colors)
    return colors[:n]


def _rgba_str(rgba, alpha=1.0):
    r, g, b = [int(255 * x) for x in rgba[:3]]
    return f'rgba({r},{g},{b},{alpha})'


def compute_pca_and_ellipses(classes, centroids, all_weeks, npz_files=None,
                              ellipse_every=None, ellipse_scale=0.3,
                              ellipse_max_pct=90):
    """
    Fit 2D PCA on the union of class centroids, project all centroids, optionally
    compute per-(class, week) covariance ellipses (semi-major a, semi-minor b,
    angle in radians) from the actual embeddings, and drop ellipses whose
    semi-major axis exceeds the `ellipse_max_pct`-th percentile of all axes.
    """
    from sklearn.decomposition import PCA

    fit_points = np.array([centroids[c][wn] for c in classes
                           for wn in all_weeks if wn in centroids.get(c, {})])
    pca = PCA(n_components=2).fit(fit_points)

    proj = {}
    for c in classes:
        wns = sorted(centroids.get(c, {}).keys())
        if not wns:
            continue
        arr_2d = pca.transform(np.array([centroids[c][wn] for wn in wns]))
        proj[c] = {wn: arr_2d[i] for i, wn in enumerate(wns)}

    ellipses = {}
    if npz_files is not None and ellipse_every is not None:
        npz_by_week = {week_num(f): f for f in npz_files}
        ellipse_weeks = [wn for wn in all_weeks if wn % ellipse_every == 0]
        class_set = set(classes)
        for wn in tqdm(ellipse_weeks, desc='  computing ellipses', leave=False):
            f = npz_by_week.get(wn)
            if f is None:
                continue
            d = np.load(f)
            idx = d['embedding_indices']
            labels = d['true_labels'][idx]
            emb_full = d['embeddings']
            for c in class_set:
                mask = labels == c
                if mask.sum() < 5:
                    continue
                emb_2d = pca.transform(emb_full[mask].astype(np.float64))
                mu = emb_2d.mean(axis=0)
                cov = np.cov(emb_2d.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals = np.clip(eigvals[order], 0, None)
                eigvecs = eigvecs[:, order]
                angle_rad = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                a = float(ellipse_scale * np.sqrt(eigvals[0]))   # semi-major
                b = float(ellipse_scale * np.sqrt(eigvals[1]))   # semi-minor
                ellipses[(c, wn)] = dict(mu=mu, a=a, b=b, angle_rad=angle_rad)

        # Drop ellipses with anomalously large semi-major axis (axis-blowing outliers)
        if ellipses and 0 < ellipse_max_pct < 100:
            axes = [v['a'] for v in ellipses.values()]
            threshold = float(np.percentile(axes, ellipse_max_pct))
            dropped = sum(1 for v in ellipses.values() if v['a'] > threshold)
            ellipses = {k: v for k, v in ellipses.items() if v['a'] <= threshold}
            if dropped:
                print(f'  filtered {dropped} ellipses above {ellipse_max_pct}-th percentile '
                      f'(semi-major > {threshold:.3f})')

    return pca, proj, ellipses


def plot_pca_trajectory(classes, centroids, all_weeks, class_names, output_path,
                         npz_files=None, ellipse_every=None, ellipse_scale=0.3,
                         ellipse_max_pct=90):
    """
    Static PNG: PCA(2D) projection of class centroid trajectories.
    One colour per class. Arrows = time direction. Circle = first week, square = last.
    """
    from matplotlib.patches import Ellipse

    pca, proj, ellipses = compute_pca_and_ellipses(
        classes, centroids, all_weeks, npz_files,
        ellipse_every, ellipse_scale, ellipse_max_pct)

    fig, ax = plt.subplots(figsize=(16, 14))
    colors = _palette(len(classes))

    for ci, c in enumerate(classes):
        if c not in proj or len(proj[c]) < 2:
            continue
        wns = sorted(proj[c].keys())
        traj = np.array([proj[c][wn] for wn in wns])
        col = colors[ci]

        for i in range(len(traj) - 1):
            ax.annotate('', xy=traj[i + 1], xytext=traj[i],
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.0, alpha=0.7))

        ax.plot(traj[0, 0], traj[0, 1], 'o', color=col, markersize=7,
                markeredgecolor='black', markeredgewidth=0.6, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=col, markersize=8,
                markeredgecolor='black', markeredgewidth=0.6, zorder=5)

        cname = class_names.get(c, f'cls_{c}')
        ax.annotate(cname, traj[-1], fontsize=7, color=col, alpha=0.95,
                    fontweight='bold', xytext=(5, 3), textcoords='offset points', zorder=6)

    class_to_color = {c: colors[i] for i, c in enumerate(classes)}
    for (c, wn), e in ellipses.items():
        col = class_to_color[c]
        ell = Ellipse(xy=e['mu'], width=2 * e['a'], height=2 * e['b'],
                      angle=np.degrees(e['angle_rad']),
                      facecolor=col, edgecolor=col, alpha=0.18, lw=0.4, zorder=2)
        ax.add_patch(ell)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    title = (f'PCA(2D) centroid trajectories of {len(classes)} classes across weeks '
             f'{all_weeks[0]}–{all_weeks[-1]}\n'
             f'One colour per class.  Arrows show time direction.  '
             f'Circle = first week, square = last week.')
    if ellipse_every is not None:
        title += (f'\nFilled ellipses every {ellipse_every} weeks: '
                  f'±{ellipse_scale}σ along the in-plane principal axes '
                  f'(outliers above {ellipse_max_pct}-th percentile dropped).')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_pca_trajectory_plotly(classes, centroids, all_weeks, class_names, output_path,
                                npz_files, ellipse_every=4, ellipse_scale=0.3,
                                ellipse_max_pct=90):
    """Interactive Plotly HTML version of the trajectory + ellipse plot."""
    import plotly.graph_objects as go

    pca, proj, ellipses = compute_pca_and_ellipses(
        classes, centroids, all_weeks, npz_files,
        ellipse_every, ellipse_scale, ellipse_max_pct)

    colors = _palette(len(classes))
    class_to_color = {c: colors[i] for i, c in enumerate(classes)}

    fig = go.Figure()

    for ci, c in enumerate(classes):
        if c not in proj or len(proj[c]) < 2:
            continue
        wns = sorted(proj[c].keys())
        traj = np.array([proj[c][wn] for wn in wns])
        col = colors[ci]
        line_str = _rgba_str(col, 0.85)
        cname = class_names.get(c, f'cls_{c}')

        fig.add_trace(go.Scatter(
            x=traj[:, 0], y=traj[:, 1],
            mode='lines+markers',
            name=cname,
            line=dict(color=line_str, width=1.4),
            marker=dict(color=line_str, size=5,
                        line=dict(color='black', width=0.3)),
            text=[f'{cname}  W{wn}' for wn in wns],
            hovertemplate='<b>%{text}</b><br>PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>',
            legendgroup=cname,
        ))
        fig.add_trace(go.Scatter(
            x=[traj[0, 0]], y=[traj[0, 1]],
            mode='markers', marker=dict(color=line_str, size=10, symbol='circle',
                                         line=dict(color='black', width=0.6)),
            name=cname, legendgroup=cname, showlegend=False,
            hovertemplate=f'<b>{cname}</b> start (W{wns[0]})<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=[traj[-1, 0]], y=[traj[-1, 1]],
            mode='markers+text',
            marker=dict(color=line_str, size=11, symbol='square',
                        line=dict(color='black', width=0.6)),
            text=[cname], textposition='top right',
            textfont=dict(size=9, color=line_str),
            name=cname, legendgroup=cname, showlegend=False,
            hovertemplate=f'<b>{cname}</b> end (W{wns[-1]})<extra></extra>',
        ))

    # Ellipses as filled polygons
    t = np.linspace(0, 2 * np.pi, 48)
    for (c, wn), e in ellipses.items():
        col = class_to_color[c]
        ca, sa = np.cos(e['angle_rad']), np.sin(e['angle_rad'])
        x_loc = e['a'] * np.cos(t)
        y_loc = e['b'] * np.sin(t)
        xs = ca * x_loc - sa * y_loc + e['mu'][0]
        ys = sa * x_loc + ca * y_loc + e['mu'][1]
        cname = class_names.get(c, f'cls_{c}')
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            fillcolor=_rgba_str(col, 0.18),
            line=dict(color=_rgba_str(col, 0.5), width=0.6),
            name=cname, legendgroup=cname, showlegend=False,
            hoveron='fills',
            hovertemplate=f'<b>{cname}</b> W{wn}  (±{ellipse_scale}σ)<extra></extra>',
        ))

    fig.update_layout(
        title=(f'PCA(2D) centroid trajectories — {len(classes)} classes, '
               f'weeks {all_weeks[0]}–{all_weeks[-1]}<br>'
               f'<sub>Hover for class/week.  '
               f'Ellipses every {ellipse_every} weeks at ±{ellipse_scale}σ '
               f'(outliers above {ellipse_max_pct}-th percentile filtered).</sub>'),
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
        width=1400, height=1000,
        hovermode='closest',
        plot_bgcolor='white',
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.write_html(output_path)


def plot_pca_trajectory_gif(classes, centroids, all_weeks, class_names, output_path,
                              npz_files, ellipse_every=4, ellipse_scale=0.3,
                              ellipse_max_pct=90, frame_ms=300, dpi=80):
    """Animated GIF: one frame per week, accumulating trajectory + current-week ellipses."""
    from matplotlib.patches import Ellipse
    from PIL import Image
    import io

    pca, proj, ellipses = compute_pca_and_ellipses(
        classes, centroids, all_weeks, npz_files,
        ellipse_every, ellipse_scale, ellipse_max_pct)

    # Compute global axis limits with margin
    xs, ys = [], []
    for c in classes:
        for p in proj.get(c, {}).values():
            xs.append(p[0]); ys.append(p[1])
    for e in ellipses.values():
        xs.extend([e['mu'][0] - e['a'], e['mu'][0] + e['a']])
        ys.extend([e['mu'][1] - e['a'], e['mu'][1] + e['a']])
    pad_x = 0.05 * (max(xs) - min(xs))
    pad_y = 0.05 * (max(ys) - min(ys))
    xlim = (min(xs) - pad_x, max(xs) + pad_x)
    ylim = (min(ys) - pad_y, max(ys) + pad_y)

    colors = _palette(len(classes))
    class_to_color = {c: colors[i] for i, c in enumerate(classes)}

    pc1_pct = pca.explained_variance_ratio_[0] * 100
    pc2_pct = pca.explained_variance_ratio_[1] * 100

    frames = []
    for current_wn in tqdm(all_weeks, desc='  rendering GIF frames', leave=False):
        fig, ax = plt.subplots(figsize=(14, 11))
        for ci, c in enumerate(classes):
            wns_so_far = sorted([w for w in proj.get(c, {}) if w <= current_wn])
            if not wns_so_far:
                continue
            traj = np.array([proj[c][w] for w in wns_so_far])
            col = colors[ci]
            if len(traj) >= 2:
                ax.plot(traj[:, 0], traj[:, 1], '-', color=col, lw=1.0, alpha=0.7)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=col, markersize=5,
                    markeredgecolor='black', markeredgewidth=0.4)
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=col, markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5)
            cname = class_names.get(c, f'cls_{c}')
            ax.annotate(cname, traj[-1], fontsize=6, color=col, alpha=0.85,
                        xytext=(4, 2), textcoords='offset points')

        for c in classes:
            e = ellipses.get((c, current_wn))
            if e is None:
                continue
            col = class_to_color[c]
            ell = Ellipse(xy=e['mu'], width=2 * e['a'], height=2 * e['b'],
                          angle=np.degrees(e['angle_rad']),
                          facecolor=col, edgecolor=col, alpha=0.32, lw=0.6, zorder=3)
            ax.add_patch(ell)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(f'PC1 ({pc1_pct:.1f}% var)')
        ax.set_ylabel(f'PC2 ({pc2_pct:.1f}% var)')
        ax.set_title(
            f'PCA(2D) trajectories — week {current_wn} of {all_weeks[-1]}  '
            f'({len(classes)} classes)\n'
            f'Filled ellipses = current-week ±{ellipse_scale}σ '
            f'(outliers above {ellipse_max_pct}-th pct filtered)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='PNG', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))

    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                       duration=frame_ms, loop=0, optimize=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir',  default='figs/week_1_inference')
    parser.add_argument('--dataset_root',   default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--reference_week', type=int, default=1)
    parser.add_argument('--output_dir',     default='figs')
    parser.add_argument('--n_proj',         type=int, default=100)
    parser.add_argument('--min_samples',    type=int, default=20)
    parser.add_argument('--top_n_classes',  type=int, default=12)
    parser.add_argument('--focal_classes',  type=str, default=None,
                        help='Comma-separated class indices for neighborhood plot '
                             '(default: top 5 by mean Mahalanobis drift)')
    parser.add_argument('--anchor_classes', type=str, default=None,
                        help='Comma-separated class indices for anchors '
                             '(default: 20 nearest neighbors of focal classes at ref week)')
    parser.add_argument('--n_anchors',           type=int, default=20)
    parser.add_argument('--neighbourhood_refs',  type=str, default=None,
                        help='Comma-separated reference weeks for neighbourhood plots '
                             '(default: reference_week only, e.g. "1,15")')
    parser.add_argument('--trajectory_classes',  type=str, default=None,
                        help='Comma-separated class indices for PCA trajectory plot '
                             '(default: random N classes with high week coverage)')
    parser.add_argument('--trajectory_n',        type=int, default=60,
                        help='Number of class trajectories to draw (default: 60)')
    parser.add_argument('--trajectory_ellipse_every', type=int, default=4,
                        help='Draw ellipses every N weeks on the ellipse variant (default: 4)')
    parser.add_argument('--trajectory_ellipse_scale', type=float, default=0.3,
                        help='Ellipse half-extent in σ-units along principal axes (default: 0.3)')
    parser.add_argument('--trajectory_ellipse_max_pct', type=float, default=90.0,
                        help='Drop ellipses whose semi-major axis exceeds this percentile '
                             'of all axes (filters out scale-blowing outliers; default: 90)')
    parser.add_argument('--trajectory_gif_ms',  type=int, default=300,
                        help='Frame duration in milliseconds for GIF (default: 300)')
    parser.add_argument('--trajectory_gif_dpi', type=int, default=80,
                        help='DPI for GIF frames; lower = smaller file (default: 80)')
    parser.add_argument('--recompute',      action='store_true')
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f'drift_cache_ref{args.reference_week}.pkl'

    try:
        class_names = load_class_names(args.dataset_root)
    except Exception:
        class_names = {}

    npz_files = sorted(inference_dir.glob('WEEK-2022-*.npz'), key=week_num)
    all_weeks = [week_num(f) for f in npz_files]
    ref_file = next((f for f in npz_files if week_num(f) == args.reference_week), npz_files[0])
    ref_wn = week_num(ref_file)
    print(f"Found {len(npz_files)} weeks ({all_weeks[0]}–{all_weeks[-1]}), reference = week {ref_wn}")

    # ── load from cache or compute ────────────────────────────────────────────
    if cache_path.exists() and not args.recompute:
        print(f"Loading cached metrics from {cache_path}")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        centroid_dist = cache['centroid_dist']
        sw_dist       = cache['sw_dist']
        mahal_dist    = cache['mahal_dist']
        frag_ratio    = cache['frag_ratio']
        consec_dist   = cache['consec_dist']
        centroids     = cache.get('centroids')
        if centroids is None:
            # Old cache without centroids — reload means only (fast)
            centroids = {c: {} for c in centroid_dist}
            for f in tqdm(npz_files, desc='Reloading centroids (cache upgrade)'):
                wn = week_num(f)
                wbc = load_embeddings_by_class(f)
                for c in centroids:
                    if c in wbc and len(wbc[c]) >= args.min_samples:
                        centroids[c][wn] = wbc[c].mean(axis=0)
            cache['centroids'] = centroids
            with open(cache_path, 'wb') as fh:
                pickle.dump(cache, fh)
    else:
        ref_by_class = load_embeddings_by_class(ref_file)
        common_classes = {c for c, emb in ref_by_class.items()
                          if len(emb) >= args.min_samples}

        ref_centroids = {}
        ref_inv_covs  = {}
        ref_tr_covs   = {}
        for c in tqdm(sorted(common_classes), desc='Precomputing ref covariances'):
            X = ref_by_class[c]
            ref_centroids[c] = X.mean(axis=0)
            if len(X) >= 3:
                inv_cov, tr_cov = regularised_inv_cov(X)
                ref_inv_covs[c] = inv_cov
                ref_tr_covs[c]  = tr_cov
            else:
                ref_inv_covs[c] = None
                ref_tr_covs[c]  = None

        centroid_dist  = {c: {} for c in common_classes}
        sw_dist        = {c: {} for c in common_classes}
        mahal_dist     = {c: {} for c in common_classes}
        frag_ratio     = {c: {} for c in common_classes}
        consec_dist    = {c: {} for c in common_classes}
        centroids      = {c: {} for c in common_classes}
        prev_centroids = {}

        for f in tqdm(npz_files, desc='Computing metrics per week'):
            wn = week_num(f)
            week_by_class = load_embeddings_by_class(f)

            for c in common_classes:
                if c not in week_by_class or len(week_by_class[c]) < args.min_samples:
                    continue
                ref_emb  = ref_by_class[c]
                week_emb = week_by_class[c]
                mu_t     = week_emb.mean(axis=0)
                mu_0     = ref_centroids[c]

                centroids[c][wn]     = mu_t
                centroid_dist[c][wn] = float(np.linalg.norm(mu_t - mu_0))

                if ref_inv_covs[c] is not None:
                    mahal_dist[c][wn] = mahalanobis(mu_t, mu_0, ref_inv_covs[c])

                sw_dist[c][wn] = sliced_wasserstein(ref_emb, week_emb, args.n_proj, rng)

                if ref_tr_covs[c] and ref_tr_covs[c] > 0 and len(week_emb) >= 3:
                    frag_ratio[c][wn] = float(np.trace(np.cov(week_emb.T))) / ref_tr_covs[c]

                if c in prev_centroids:
                    consec_dist[c][wn] = float(np.linalg.norm(mu_t - prev_centroids[c]))
                prev_centroids[c] = mu_t

        cache = dict(centroid_dist=centroid_dist, sw_dist=sw_dist,
                     mahal_dist=mahal_dist, frag_ratio=frag_ratio,
                     consec_dist=consec_dist, centroids=centroids,
                     all_weeks=all_weeks, ref_wn=ref_wn)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Cache saved -> {cache_path}")

    weeks_arr = np.array(all_weeks)

    # ── aggregate over classes ────────────────────────────────────────────────
    c_mean, c_std, _   = agg_over_weeks(centroid_dist, all_weeks)
    m_mean, m_std, _   = agg_over_weeks(mahal_dist,    all_weeks)
    w_mean, w_std, _   = agg_over_weeks(sw_dist,       all_weeks)
    fr_mean, fr_std, _ = agg_over_weeks(frag_ratio,    all_weeks)

    # ── PLOT 1: 4-panel aggregate ─────────────────────────────────────────────
    nbhd_refs = ([int(x) for x in args.neighbourhood_refs.split(',')]
                 if args.neighbourhood_refs else [ref_wn])
    plot_names = (['aggregate', 'velocity', 'mahal_heatmap', 'fragmentation', 'centroid_heatmap']
                  + [f'neighbourhood_ref{r}' for r in nbhd_refs]
                  + ['pca_trajectory', 'pca_trajectory_ellipses',
                     'pca_trajectory_plotly', 'pca_trajectory_gif'])
    plots = tqdm(plot_names, desc='Plotting', unit='fig')
    plots_iter = iter(plots)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True)

    band_plot(axes[0, 0], weeks_arr, c_mean, c_std, 'steelblue',
              'Mean ± std', ref_wn,
              'Centroid L2 distance',
              'Centroid L2 distance from reference week\n(raw shift in embedding space)')

    band_plot(axes[0, 1], weeks_arr, m_mean, m_std, 'mediumpurple',
              'Mean ± std', ref_wn,
              'Mahalanobis distance',
              'Mahalanobis distance of centroid from ref distribution\n'
              r'$D = \sqrt{(\mu_t - \mu_0)^T \Sigma_0^{-1} (\mu_t - \mu_0)}$'
              '\nStep-function → teleportation; ramp → continuous drift')

    band_plot(axes[1, 0], weeks_arr, w_mean, w_std, 'darkorange',
              'Mean ± std', ref_wn,
              'Sliced Wasserstein distance',
              f'Sliced Wasserstein from reference ({args.n_proj} random projections)\n'
              '(full distribution, not just centroid)')

    band_plot(axes[1, 1], weeks_arr, fr_mean, fr_std, 'forestgreen',
              'Mean ± std', ref_wn,
              r'Fragmentation ratio $\mathrm{Tr}(\Sigma_t)/\mathrm{Tr}(\Sigma_0)$',
              r'Fragmentation ratio $\mathrm{Tr}(\Sigma_t) / \mathrm{Tr}(\Sigma_0)$'
              '\n≈1 → cluster moved as a unit;  >1 → cluster exploded / split')
    axes[1, 1].axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5)

    for ax in axes[1]:
        ax.set_xlabel('Week number')
    fig.suptitle('Feature drift analysis — original embedding space\n'
                 '(mean ± std across all classes with ≥20 samples)', fontsize=13, y=1.01)
    fig.tight_layout()
    out = output_dir / 'drift_metrics_aggregate.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    next(plots_iter)

    # ── PLOT 2: velocity — cumulative + consecutive Δ ─────────────────────────
    consec_mean, consec_std = [], []
    for i in range(1, len(all_weeks)):
        wn = all_weeks[i]
        vals = [consec_dist[c][wn] for c in consec_dist
                if wn in consec_dist[c] and np.isfinite(consec_dist[c][wn])]
        consec_mean.append(np.mean(vals) if vals else np.nan)
        consec_std.append(np.std(vals) if vals else np.nan)
    consec_mean = np.array(consec_mean)
    consec_std  = np.array(consec_std)
    mid_weeks   = weeks_arr[1:]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax = axes[0]
    ax.fill_between(weeks_arr, c_mean - c_std, c_mean + c_std, alpha=0.2, color='steelblue')
    ax.plot(weeks_arr, c_mean, color='steelblue', lw=2.5)
    ax.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.6, label=f'Ref week {ref_wn}')
    ax.set_ylabel('Centroid L2 distance from reference')
    ax.set_title('Cumulative drift from reference (top)\nvs. week-to-week centroid velocity (bottom)\n'
                 'Discrete teleportation → flat velocity with isolated spikes;  '
                 'continuous drift → monotonically increasing top panel')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(mid_weeks, consec_mean, width=0.6, color='tomato', alpha=0.8,
            label='Consecutive week centroid L2 distance')
    ax2.errorbar(mid_weeks, consec_mean, yerr=consec_std,
                 fmt='none', ecolor='darkred', elinewidth=1, capsize=3, alpha=0.5)
    ax2.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.6)
    ax2.set_xlabel('Week number')
    ax2.set_ylabel('Mean centroid Δ (consecutive weeks)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = output_dir / 'drift_velocity.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    next(plots_iter)

    # ── PLOT 3: Mahalanobis heatmap ───────────────────────────────────────────
    min_cov = int(0.8 * len(all_weeks))
    eligible_m = random_classes_with_coverage(mahal_dist, 60, min_cov, rng)
    # Within the random sample, sort rows by mean drift for readability
    class_mean_m = {c: np.nanmean(list(mahal_dist[c].values())) for c in eligible_m}
    eligible_m = sorted(eligible_m, key=class_mean_m.get, reverse=True)

    if eligible_m:
        mat = np.full((len(eligible_m), len(all_weeks)), np.nan)
        for i, c in enumerate(eligible_m):
            for j, wn in enumerate(all_weeks):
                if wn in mahal_dist[c]:
                    mat[i, j] = mahal_dist[c][wn]
        ref_idx = all_weeks.index(ref_wn) if ref_wn in all_weeks else None
        yl = [class_names.get(c, f'cls_{c}') for c in eligible_m]
        heatmap_plot(mat, yl, all_weeks, ref_idx,
                     'Mahalanobis distance from ref distribution',
                     r'Mahalanobis migration distance $D = \sqrt{(\mu_t - \mu_0)^T \Sigma_0^{-1} (\mu_t - \mu_0)}$'
                     '\n(60 random classes, sorted by mean drift within the sample;  '
                     'block colour pattern → discrete jump;  smooth gradient → continuous)',
                     output_dir / 'drift_mahalanobis_heatmap.png')
    next(plots_iter)

    # ── PLOT 4: fragmentation per class ──────────────────────────────────────
    top_frag = random_classes_with_coverage(frag_ratio, args.top_n_classes, min_cov, rng)
    palette  = plt.cm.tab10(np.linspace(0, 1, args.top_n_classes))

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)

    ax = axes[0]
    ax.fill_between(weeks_arr, fr_mean - fr_std, fr_mean + fr_std,
                    alpha=0.2, color='forestgreen')
    ax.plot(weeks_arr, fr_mean, color='forestgreen', lw=2.5, label='Mean across classes')
    ax.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5, label='No fragmentation (ratio = 1)')
    ax.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.6, label=f'Ref week {ref_wn}')
    ax.set_ylabel(r'$\mathrm{Tr}(\Sigma_t)/\mathrm{Tr}(\Sigma_0)$')
    ax.set_title(r'Cluster fragmentation ratio $\mathrm{Tr}(\Sigma_t)/\mathrm{Tr}(\Sigma_0)$'
                 '\n≈1 → cluster migrated as a cohesive unit;  >1 → cluster exploded / split')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, c in enumerate(top_frag):
        wns  = sorted(frag_ratio[c].keys())
        vals = [frag_ratio[c][wn] for wn in wns]
        name = class_names.get(c, f'cls_{c}')
        ax2.plot(wns, vals, lw=1.5, marker='o', markersize=3,
                 color=palette[i], label=f'{name} ({c})')
    ax2.axhline(1.0, color='gray', lw=1, ls='--', alpha=0.5)
    ax2.axvline(ref_wn, color='red', lw=1, ls=':', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('Week number')
    ax2.set_ylabel(r'$\mathrm{Tr}(\Sigma_t)/\mathrm{Tr}(\Sigma_0)$  (log scale)')
    ax2.set_title(f'{args.top_n_classes} random classes — fragmentation ratio over time')
    ax2.legend(fontsize=7, ncol=2); ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    out = output_dir / 'drift_fragmentation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    next(plots_iter)

    # ── PLOT 5: centroid L2 heatmap ───────────────────────────────────────────
    eligible_c = random_classes_with_coverage(centroid_dist, 60, min_cov, rng)
    class_mean_c = {c: np.nanmean(list(centroid_dist[c].values())) for c in eligible_c}
    eligible_c = sorted(eligible_c, key=class_mean_c.get, reverse=True)
    if eligible_c:
        mat = np.full((len(eligible_c), len(all_weeks)), np.nan)
        for i, c in enumerate(eligible_c):
            for j, wn in enumerate(all_weeks):
                if wn in centroid_dist[c]:
                    mat[i, j] = centroid_dist[c][wn]
        ref_idx = all_weeks.index(ref_wn) if ref_wn in all_weeks else None
        yl = [class_names.get(c, f'cls_{c}') for c in eligible_c]
        heatmap_plot(mat, yl, all_weeks, ref_idx,
                     'Centroid L2 distance from reference week',
                     'Per-class centroid L2 drift from reference week\n'
                     '(60 random classes, sorted by mean drift within the sample;  '
                     'block colour pattern → discrete jump;  smooth gradient → continuous)',
                     output_dir / 'drift_heatmap.png')
    next(plots_iter)

    # ── PLOT 6+: class neighbourhood (one per requested ref week) ────────────
    # Focal classes: 5 random classes with high week coverage
    if args.focal_classes:
        focal = [int(x) for x in args.focal_classes.split(',')]
    else:
        focal = random_classes_with_coverage(centroids, 5, min_cov, rng)

    # Anchors are selected once using the main ref_wn (same set across all variants)
    if args.anchor_classes:
        anchors = [int(x) for x in args.anchor_classes.split(',')]
    else:
        focal_set = set(focal)
        votes = {}
        for fc in focal:
            fc_ref = centroids.get(fc, {}).get(ref_wn)
            if fc_ref is None:
                continue
            dists = [(c, float(np.linalg.norm(fc_ref - centroids[c][ref_wn])))
                     for c in centroids
                     if c not in focal_set and ref_wn in centroids.get(c, {})]
            dists.sort(key=lambda x: x[1])
            for rank, (c, _) in enumerate(dists[:args.n_anchors]):
                votes[c] = votes.get(c, 0) + (args.n_anchors - rank)
        anchors = sorted([c for c in votes if c not in focal_set],
                         key=votes.get, reverse=True)[:args.n_anchors]

    for nbhd_ref in nbhd_refs:
        suffix = f'_ref{nbhd_ref}' if len(nbhd_refs) > 1 else ''
        plot_class_neighborhood(
            focal, anchors, centroids, all_weeks, nbhd_ref,
            class_names, output_dir / f'drift_neighbourhood{suffix}.png')
        next(plots_iter)

    # ── PLOT 7: PCA trajectory of N classes (one colour per class) ──────────
    if args.trajectory_classes:
        traj_classes = [int(x) for x in args.trajectory_classes.split(',')]
    else:
        coverage_thresh = int(0.8 * len(all_weeks))
        candidates = [c for c in centroids if len(centroids[c]) >= coverage_thresh]
        rng.shuffle(candidates)
        traj_classes = candidates[:args.trajectory_n]

    plot_pca_trajectory(
        traj_classes, centroids, all_weeks, class_names,
        output_dir / 'drift_pca_trajectory.png')
    next(plots_iter)

    plot_pca_trajectory(
        traj_classes, centroids, all_weeks, class_names,
        output_dir / 'drift_pca_trajectory_ellipses.png',
        npz_files=npz_files,
        ellipse_every=args.trajectory_ellipse_every,
        ellipse_scale=args.trajectory_ellipse_scale,
        ellipse_max_pct=args.trajectory_ellipse_max_pct)
    next(plots_iter)

    plot_pca_trajectory_plotly(
        traj_classes, centroids, all_weeks, class_names,
        output_dir / 'drift_pca_trajectory_ellipses.html',
        npz_files=npz_files,
        ellipse_every=args.trajectory_ellipse_every,
        ellipse_scale=args.trajectory_ellipse_scale,
        ellipse_max_pct=args.trajectory_ellipse_max_pct)
    next(plots_iter)

    plot_pca_trajectory_gif(
        traj_classes, centroids, all_weeks, class_names,
        output_dir / 'drift_pca_trajectory.gif',
        npz_files=npz_files,
        ellipse_every=args.trajectory_ellipse_every,
        ellipse_scale=args.trajectory_ellipse_scale,
        ellipse_max_pct=args.trajectory_ellipse_max_pct,
        frame_ms=args.trajectory_gif_ms,
        dpi=args.trajectory_gif_dpi)
    next(plots_iter)
    plots.close()


if __name__ == '__main__':
    main()
