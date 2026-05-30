#!/usr/bin/env python
"""Plot confusion matrix diffs from pre-computed .npy files."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from train_per_week_cesnet import load_label_mapping

dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
label_indices_mapping, num_classes = load_label_mapping(dataset_root)
class_names = sorted(label_indices_mapping, key=label_indices_mapping.get)

# Load pre-computed confusion matrices
test_weeks = [33, 40, 50]
cms = {}
for tw in test_weeks:
    path = f'cm_33_to_{tw}.npy'
    assert Path(path).exists(), f"Missing {path}"
    cms[tw] = np.load(path)
    print(f"Loaded {path}, shape={cms[tw].shape}")

# Normalize each CM row-wise (same as normalize='true')
cms_norm = {}
for tw, cm in cms.items():
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cms_norm[tw] = cm / row_sums

# Plot diffs: 33->40 minus 33->33, and 33->50 minus 33->33
baseline = cms_norm[33]
for tw in [40, 50]:
    diff = cms_norm[tw] - baseline

    fig, ax = plt.subplots(figsize=(16, 14))
    vmax = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, interpolation='nearest', cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, label='Change in normalized frequency')

    n_classes = len(class_names)
    tick_marks = np.arange(0, n_classes, max(1, n_classes // 20))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([class_names[i] for i in tick_marks], rotation=90, ha='center', fontsize=6)
    ax.set_yticklabels([class_names[i] for i in tick_marks], fontsize=6)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_title(f'Confusion Matrix Diff: (33->{tw}) - (33->33)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    out_path = f'cm_diff_33to{tw}_minus_33to33.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

print("\nDone!")
