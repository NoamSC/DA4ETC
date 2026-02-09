#!/usr/bin/env python
"""Compute confusion matrices: week_33 model evaluated on weeks 33, 40, 50."""

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

experiment_dir = Path('exps/cesnet_multimodal_each_week_train_v01')
dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
device = 'cuda:0'

# Load label mapping
label_indices_mapping, num_classes = load_label_mapping(dataset_root)
class_names = sorted(label_indices_mapping, key=label_indices_mapping.get)

# Load week_33 model
checkpoint_path = experiment_dir / 'week_33' / 'weights' / 'best_model.pth'
config_path = experiment_dir / 'week_33' / 'config.json'
model = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, device)

criterion = torch.nn.CrossEntropyLoss()

for test_week in [33, 40, 50]:
    week_dir = dataset_root / f'WEEK-2022-{test_week}'
    print(f"\nEvaluating week_33 model on week {test_week}...")

    loader = create_week_loader(
        week_dir, label_indices_mapping,
        batch_size=64, num_workers=4,
        data_sample_frac=0.1, seed=42,
    )

    val_loss, val_acc, all_labels, all_predictions = validate(
        model, loader, criterion, device, return_features=False
    )
    print(f"  Accuracy: {val_acc:.2f}%")

    # Save raw confusion matrix as numpy array
    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_classes))
    np.save(f'cm_33_to_{test_week}.npy', cm)
    print(f"  Saved cm_33_to_{test_week}.npy")

    # Plot and save as PNG
    fig = plot_confusion_matrix(
        all_labels, all_predictions, class_names=class_names,
        epoch=None, normalize='true'
    )
    fig.suptitle(f'Confusion Matrix: Train Week 33 → Test Week {test_week}\nAccuracy: {val_acc:.2f}%',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'cm_33_to_{test_week}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cm_33_to_{test_week}.png")

print("\nDone!")
