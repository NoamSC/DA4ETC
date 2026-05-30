#!/usr/bin/env python
"""
Phase 1: t-SNE visualization of one app's latent drift over time.

Layout:
  - Background: all classes from week 1, ~max_bg_per_class samples each,
    colored by class (muted palette) — shows the latent cluster structure.
  - Focal class: same class across all available weeks, colored by week number
    (sequential palette) — shows temporal drift.

Output: interactive Plotly HTML

Usage:
    python plot_tsne_temporal.py
    python plot_tsne_temporal.py --focal_class 151 --output figs/tsne_teams.html
"""

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE


# ── label mapping (index -> name) ────────────────────────────────────────────

def load_class_names(dataset_root):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_per_week_cesnet import load_label_mapping
    mapping, num_classes = load_label_mapping(Path(dataset_root))
    return {v: k for k, v in mapping.items()}, num_classes


# ── data loading ──────────────────────────────────────────────────────────────

def load_week(npz_path, classes=None, max_per_class=None, rng=None):
    """Load embeddings from one npz, optionally subsampling per class."""
    data = np.load(npz_path)
    true = data['true_labels']
    pred = data['pred_labels']
    emb  = data['embeddings']

    if classes is None:
        classes = np.unique(true)

    out_emb, out_true, out_pred = [], [], []
    for cls in classes:
        mask = true == cls
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        out_emb.append(emb[idx])
        out_true.append(true[idx])
        out_pred.append(pred[idx])

    if not out_emb:
        return np.empty((0, emb.shape[1])), np.empty(0, int), np.empty(0, int)
    return (
        np.concatenate(out_emb),
        np.concatenate(out_true),
        np.concatenate(out_pred),
    )


def week_num(path):
    return int(re.search(r'(\d+)$', Path(path).stem).group(1))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir', default='figs/week_1_inference')
    parser.add_argument('--dataset_root',  default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--focal_class',   type=int, default=151,
                        help='Class index to track over time (default: 151 = teams)')
    parser.add_argument('--max_bg_per_class',   type=int, default=30,
                        help='Background samples per class from week 1 (default: 30)')
    parser.add_argument('--max_focal_per_week', type=int, default=120,
                        help='Focal class samples per week (default: 120)')
    parser.add_argument('--reference_week', type=int, default=1,
                        help='Week used for background (default: 1)')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    inference_dir = Path(args.inference_dir)
    output_path = Path(args.output) if args.output else inference_dir / f'tsne_class{args.focal_class}.html'

    print("Loading class names...")
    class_names, num_classes = load_class_names(args.dataset_root)
    focal_name = class_names.get(args.focal_class, f'class_{args.focal_class}')
    print(f"  Focal class: {args.focal_class} = {focal_name}")

    # ── load available weeks ──────────────────────────────────────────────────
    npz_files = sorted(inference_dir.glob('WEEK-2022-*.npz'),
                       key=lambda p: week_num(p))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {inference_dir}")

    ref_file = next((f for f in npz_files if week_num(f) == args.reference_week), npz_files[0])
    ref_week = week_num(ref_file)
    print(f"  Reference (background) week: {ref_week}")
    print(f"  Available weeks: {[week_num(f) for f in npz_files]}")

    # ── background: all classes from reference week ───────────────────────────
    print(f"\nLoading background ({args.max_bg_per_class} samples/class from week {ref_week})...")
    bg_classes = [c for c in range(num_classes) if c != args.focal_class]
    bg_emb, bg_true, bg_pred = load_week(
        ref_file, classes=bg_classes,
        max_per_class=args.max_bg_per_class, rng=rng
    )
    print(f"  Background: {len(bg_emb)} samples across {len(np.unique(bg_true))} classes")

    # ── focal class: across all weeks ─────────────────────────────────────────
    print(f"\nLoading focal class {focal_name} across {len(npz_files)} weeks...")
    focal_embs, focal_weeks = [], []
    for f in npz_files:
        wn = week_num(f)
        emb, _, _ = load_week(f, classes=[args.focal_class],
                               max_per_class=args.max_focal_per_week, rng=rng)
        if len(emb) == 0:
            print(f"  week {wn:02d}: no samples, skipping")
            continue
        focal_embs.append(emb)
        focal_weeks.extend([wn] * len(emb))
        print(f"  week {wn:02d}: {len(emb)} samples")

    focal_emb   = np.concatenate(focal_embs, axis=0)
    focal_weeks = np.array(focal_weeks)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    all_emb = np.concatenate([bg_emb, focal_emb], axis=0)
    print(f"\nRunning t-SNE on {len(all_emb)} points (dim={all_emb.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                random_state=args.seed, n_jobs=-1, verbose=1)
    xy = tsne.fit_transform(all_emb)

    bg_xy    = xy[:len(bg_emb)]
    focal_xy = xy[len(bg_emb):]

    # ── Plotly ────────────────────────────────────────────────────────────────
    print("\nBuilding Plotly figure...")
    fig = go.Figure()

    # Background: one trace per class, muted grey-blue tones
    unique_bg_classes = np.unique(bg_true)
    # Use a fixed grey color for all background — cleaner visually
    fig.add_trace(go.Scattergl(
        x=bg_xy[:, 0], y=bg_xy[:, 1],
        mode='markers',
        marker=dict(size=3, color='#c8c8d0', opacity=0.4),
        text=[class_names.get(int(c), str(c)) for c in bg_true],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Other classes (week 1)',
        legendgroup='background',
        showlegend=True,
    ))

    # Focal class: one trace per week, colored by week (plasma)
    unique_weeks = np.unique(focal_weeks)
    week_min, week_max = unique_weeks.min(), unique_weeks.max()

    import plotly.colors as pc
    colorscale = pc.get_colorscale('Plasma')

    def week_color(wn):
        t = (wn - week_min) / max(week_max - week_min, 1)
        return pc.sample_colorscale(colorscale, t)[0]

    for wn in unique_weeks:
        mask = focal_weeks == wn
        fig.add_trace(go.Scattergl(
            x=focal_xy[mask, 0], y=focal_xy[mask, 1],
            mode='markers',
            marker=dict(
                size=7,
                color=week_color(wn),
                opacity=0.85,
                line=dict(width=0.3, color='white'),
            ),
            name=f'Week {wn:02d}',
            legendgroup=f'week_{wn}',
            hovertemplate=f'<b>{focal_name}</b> — Week {wn}<extra></extra>',
        ))

    fig.update_layout(
        title=dict(
            text=(f'Latent Space: <b>{focal_name}</b> drift over time<br>'
                  f'<sup>Background = all classes, week {ref_week} '
                  f'({args.max_bg_per_class} samples/class) &nbsp;|&nbsp; '
                  f'Colored points = {focal_name} across weeks '
                  f'{week_min}–{week_max}</sup>'),
            font=dict(size=18),
            x=0.5,
        ),
        xaxis=dict(title='t-SNE 1', showgrid=False, zeroline=False),
        yaxis=dict(title='t-SNE 2', showgrid=False, zeroline=False),
        plot_bgcolor='#0f0f14',
        paper_bgcolor='#0f0f14',
        font=dict(color='white'),
        legend=dict(
            itemsizing='constant',
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(255,255,255,0.15)',
            borderwidth=1,
        ),
        width=1100,
        height=750,
    )

    fig.write_html(output_path)
    print(f"\nSaved -> {output_path}")


if __name__ == '__main__':
    main()
