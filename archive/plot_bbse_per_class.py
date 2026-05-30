#!/usr/bin/env python
"""
Grid of per-class plots sorted by training frequency.
Each panel: BBSE drift residual (left, blue) + class % train vs test (right, orange) + F1 (grey).
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
INFERENCE_DIR  = Path('figs/week_1_inference')
DATASET_ROOT   = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
REFERENCE_WEEK = 0
N_COLS         = 6
OUTPUT         = INFERENCE_DIR / 'bbse_per_class_grid.png'

# ── helpers ───────────────────────────────────────────────────────────────────

def load_class_names(dataset_root):
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train_per_week_cesnet import load_label_mapping
        mapping, _ = load_label_mapping(Path(dataset_root))
        return {v: k for k, v in mapping.items()}
    except Exception:
        return {}


def load_weeks(inference_dir):
    files = sorted(
        inference_dir.glob('WEEK-2022-*.npz'),
        key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
    )
    weeks = []
    for f in files:
        wn = int(re.search(r'(\d+)$', f.stem).group(1))
        d  = np.load(f)
        weeks.append((wn, d['true_labels'], d['pred_labels'], d['softmax']))
    return weeks


def build_bbse_reference(ref_true, ref_pred, ref_softmax, num_classes):
    cm = confusion_matrix(ref_true, ref_pred, labels=list(range(num_classes))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C        = cm / row_sums                     # C[i,j] = P(pred=j | true=i)
    C_T_pinv = np.linalg.pinv(C.T)

    h_ref = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = ref_true == c
        if mask.sum() > 0:
            p      = np.clip(ref_softmax[mask], 1e-12, 1.0)
            h_ref[c] = float(-np.sum(p * np.log(p), axis=1).mean())
    return C_T_pinv, h_ref


def bbse_per_class(softmax, pred, num_classes, h_ref):
    """Per-class BBSE residual: actual pseudo-entropy − reference entropy."""
    out = {}
    for c in range(num_classes):
        mask = pred == c
        if mask.sum() == 0 or np.isnan(h_ref[c]):
            out[c] = np.nan
        else:
            p      = np.clip(softmax[mask], 1e-12, 1.0)
            h_act  = float(-np.sum(p * np.log(p), axis=1).mean())
            out[c] = h_act - h_ref[c]
    return out


def class_f1_fast(true, pred, cls):
    tp = int(((pred == cls) & (true == cls)).sum())
    fp = int(((pred == cls) & (true != cls)).sum())
    fn = int(((pred != cls) & (true == cls)).sum())
    d  = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else np.nan


# ── load data ─────────────────────────────────────────────────────────────────

weeks = load_weeks(INFERENCE_DIR)
print(f"Loaded {len(weeks)} weeks")

ref = next(((t, p, s) for wn, t, p, s in weeks if wn == REFERENCE_WEEK), None)
if ref is None:
    print(f"Week {REFERENCE_WEEK} not found — using week {weeks[0][0]}")
    ref = weeks[0][1:]
ref_true, ref_pred, ref_soft = ref
num_classes = ref_soft.shape[1]

class_names  = load_class_names(DATASET_ROOT)
train_counts = np.bincount(ref_true, minlength=num_classes).astype(float)
train_pct    = train_counts / train_counts.sum() * 100

# Sort classes by training frequency, keep only those present
present        = np.where(train_counts > 0)[0]
sorted_classes = present[np.argsort(-train_counts[present])]
print(f"Classes in training: {len(sorted_classes)}")

C_T_pinv, h_ref = build_bbse_reference(ref_true, ref_pred, ref_soft, num_classes)

# ── compute per-week metrics ──────────────────────────────────────────────────

week_nums = []
bbse_w    = {c: [] for c in sorted_classes}
pct_w     = {c: [] for c in sorted_classes}
f1_w      = {c: [] for c in sorted_classes}

for wn, true, pred, softmax in tqdm(weeks, desc='Computing metrics'):
    week_nums.append(wn)
    pc  = bbse_per_class(softmax, pred, num_classes, h_ref)
    cnt = np.bincount(true, minlength=num_classes).astype(float)
    pct = cnt / cnt.sum() * 100
    for c in sorted_classes:
        bbse_w[c].append(pc[c])
        pct_w[c].append(float(pct[c]))
        f1_w[c].append(class_f1_fast(true, pred, c))

week_nums = np.array(week_nums)

# ── plot ──────────────────────────────────────────────────────────────────────

n      = len(sorted_classes)
n_rows = (n + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(n_rows, N_COLS,
                         figsize=(N_COLS * 3.0, n_rows * 2.4),
                         squeeze=False)

xtick_step = max(1, len(week_nums) // 5)

for idx, cls in enumerate(tqdm(sorted_classes, desc='Plotting')):
    row, col = divmod(idx, N_COLS)
    ax_l     = axes[row][col]
    ax_r     = ax_l.twinx()

    name      = class_names.get(int(cls), str(cls))
    bbse_vals = np.array(bbse_w[cls], dtype=float)
    pct_vals  = np.array(pct_w[cls],  dtype=float)
    f1_vals   = np.array(f1_w[cls],   dtype=float)

    # Left axis — BBSE residual
    ax_l.plot(week_nums, bbse_vals, color='steelblue', linewidth=1.3,
              marker='o', markersize=1.8)
    ax_l.axhline(0, color='steelblue', linewidth=0.5, linestyle=':')

    # Right axis — class % (test line + train horizontal) and F1
    ax_r.axhline(train_pct[cls], color='darkorange', linewidth=1.0,
                 linestyle='--', alpha=0.9)
    ax_r.plot(week_nums, pct_vals, color='darkorange', linewidth=1.2,
              marker='s', markersize=1.8, alpha=0.9)
    ax_r.plot(week_nums, f1_vals * max(pct_vals.max(), train_pct[cls], 0.1),
              color='#888888', linewidth=0.8, linestyle='--', alpha=0.65)

    ax_l.set_title(f'#{idx+1}  {name}  ({train_pct[cls]:.2f}%)',
                   fontsize=5.5, pad=2)
    ax_l.set_ylabel('BBSE res.', fontsize=4.5, color='steelblue')
    ax_r.set_ylabel('%  (F1×scale)', fontsize=4.5, color='darkorange')
    ax_l.tick_params(axis='both', labelsize=4, colors='steelblue')
    ax_r.tick_params(axis='both', labelsize=4, colors='darkorange')
    ax_l.tick_params(axis='x', colors='black')
    ax_r.set_ylim(bottom=0)
    ax_l.grid(True, alpha=0.15)
    ax_l.spines['top'].set_visible(False)
    ax_r.spines['top'].set_visible(False)
    ax_l.set_xticks(week_nums[::xtick_step])
    ax_l.tick_params(axis='x', labelsize=3.5, rotation=45)

for idx in range(n, n_rows * N_COLS):
    row, col = divmod(idx, N_COLS)
    axes[row][col].set_visible(False)

legend_handles = [
    Line2D([0], [0], color='steelblue',  lw=1.5,              label='BBSE drift residual (left)'),
    Line2D([0], [0], color='darkorange', lw=1.5, ls='-',      label='% in test week (right)'),
    Line2D([0], [0], color='darkorange', lw=1.5, ls='--',     label='% in train week (right, dashed)'),
    Line2D([0], [0], color='#888888',    lw=1.0, ls='--',     label='F1 × scale (grey)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

fig.suptitle(
    f'BBSE drift residual & class % — model trained on week {REFERENCE_WEEK}  '
    f'({len(weeks)} test weeks: {weeks[0][0]}–{weeks[-1][0]})\n'
    'Panels sorted by training frequency (most common → top-left)',
    fontsize=11, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"\nSaved → {OUTPUT}")
