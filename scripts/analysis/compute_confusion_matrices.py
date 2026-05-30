#!/usr/bin/env python
"""Compute confusion matrices: all train weeks evaluated on one test week.

Usage:
    python compute_confusion_matrices.py --test_week 10
    python compute_confusion_matrices.py --test_week_idx 0  # same as --test_week 10
"""

# --- repo path bootstrap (added during refactor: keeps flat cross-imports working) ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---


import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

from temporal_generalization import load_model_from_checkpoint, create_week_loader
from training.trainer import validate
from training.visualization import plot_confusion_matrix
from train_per_week_cesnet import load_label_mapping
import torch

ALL_WEEKS = [10, 20, 30, 40, 50]

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--test_week', type=int, choices=ALL_WEEKS)
group.add_argument('--test_week_idx', type=int, choices=range(len(ALL_WEEKS)))
args = parser.parse_args()

test_week = args.test_week if args.test_week is not None else ALL_WEEKS[args.test_week_idx]

experiment_dir = Path('exps/cesnet_multimodal_each_week_train_v01')
dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
output_dir = Path('figs/cms')
output_dir.mkdir(parents=True, exist_ok=True)
device = 'cuda:0'

# Load label mapping
label_indices_mapping, num_classes = load_label_mapping(dataset_root)
class_names = sorted(label_indices_mapping, key=label_indices_mapping.get)

criterion = torch.nn.CrossEntropyLoss()

# Only load models for train_week <= test_week (forward/same-week only)
train_weeks = [tw for tw in ALL_WEEKS if tw <= test_week]
models = {}
for tw in train_weeks:
    checkpoint_path = experiment_dir / f'week_{tw}' / 'weights' / 'best_model.pth'
    config_path = experiment_dir / f'week_{tw}' / 'config.json'
    print(f"Loading model for week {tw}...")
    models[tw] = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, device)

# Load test data once
week_dir = dataset_root / f'WEEK-2022-{test_week}'
print(f"\nLoading test data for week {test_week}...")
loader = create_week_loader(
    week_dir, label_indices_mapping,
    batch_size=64, num_workers=4,
    data_sample_frac=0.1, seed=42,
)

# Evaluate each train model on this test week
for train_week, model in models.items():
    npy_path = output_dir / f'cm_{train_week}_to_{test_week}.npy'
    if npy_path.exists():
        print(f"Skipping {train_week}->{test_week}, already exists")
        continue

    print(f"\nEvaluating train {train_week} -> test {test_week}...")
    val_loss, val_acc, all_labels, all_predictions = validate(
        model, loader, criterion, device, return_features=False
    )
    print(f"  Accuracy: {val_acc:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    np.save(npy_path, cm)
    print(f"  Saved {npy_path}")

    fig = plot_confusion_matrix(
        all_labels, all_predictions, class_names=class_names,
        epoch=None, normalize='true'
    )
    fig.suptitle(f'Train Week {train_week} -> Test Week {test_week}\nAccuracy: {val_acc:.2f}%',
                 fontsize=14, fontweight='bold', y=1.02)
    png_path = output_dir / f'cm_{train_week}_to_{test_week}.png'
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {png_path}")

del loader
print("\nDone!")
