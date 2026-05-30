#!/usr/bin/env python
"""
Run a single trained model (default: week_1) across all test weeks and save
per-packet predictions for downstream plotting.

Saves per week: true_labels, pred_labels, softmax (N x C), embeddings (N x D)
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
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from temporal_generalization import (
    load_model_from_checkpoint,
    create_week_loader,
    get_available_weeks,
)
from train_per_week_cesnet import load_label_mapping


def run_inference(model, loader, device):
    model.eval()
    all_true, all_pred, all_softmax, all_embeds = [], [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
            labels = labels.to(device).long()
            inputs = [x.to(device) for x in inputs] if isinstance(inputs, list) else inputs.to(device)

            outputs = model(inputs)
            logits = outputs['class_preds']
            features = outputs['features']

            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_softmax.append(F.softmax(logits, dim=1).cpu().numpy())
            all_embeds.append(features.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_softmax, axis=0),
        np.concatenate(all_embeds, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', default='exps/cesnet_multimodal_each_week_train_v01')
    parser.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--train_week', default='week_1', help='Which trained model to use')
    parser.add_argument('--output_dir', default=None, help='Where to save .npz files (default: figs/<train_week>_inference)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data_sample_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reversed', action='store_true',
                        help='Process weeks in reverse order (for parallel runs)')
    parser.add_argument('--order', default=None,
                        choices=['forward', 'reversed', 'middle_up', 'middle_down'],
                        help='Traversal order across weeks. Run several jobs with '
                             'different orders; the exists() guard partitions the work. '
                             'middle_up: center->last, middle_down: center->first. '
                             'Overrides --reversed if given.')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path('figs') / f'{args.train_week}_inference'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading label mapping from {dataset_root}...")
    label_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"  {num_classes} classes")

    train_week_dir = experiment_dir / args.train_week
    checkpoint_path = train_week_dir / 'weights' / 'best_model.pth'
    config_path = train_week_dir / 'config.json'
    print(f"Loading model: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, args.device)
    print("  Done.")

    test_week_dirs = get_available_weeks(dataset_root)
    order = args.order or ('reversed' if args.reversed else 'forward')
    mid = len(test_week_dirs) // 2
    if order == 'reversed':
        test_week_dirs = list(reversed(test_week_dirs))
    elif order == 'middle_up':      # center -> last
        test_week_dirs = test_week_dirs[mid:]
    elif order == 'middle_down':    # center -> first
        test_week_dirs = list(reversed(test_week_dirs[:mid]))
    print(f"Running over {len(test_week_dirs)} test weeks -> {output_dir} (order={order})\n")

    for week_dir in test_week_dirs:
        week_name = week_dir.name
        out_path = output_dir / f'{week_name}.npz'

        if out_path.exists():
            print(f"  {week_name}: already exists, skipping")
            continue

        loader = create_week_loader(
            week_dir, label_mapping,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_sample_frac=args.data_sample_frac,
            seed=args.seed,
        )
        if loader is None:
            print(f"  {week_name}: no data, skipping")
            continue

        print(f"  {week_name}: running inference...")
        true_labels, pred_labels, softmax, embeddings = run_inference(model, loader, args.device)

        rng = np.random.RandomState(args.seed)
        emb_idx = np.sort(rng.choice(len(true_labels), max(1, len(true_labels) // 10), replace=False))
        np.savez_compressed(
            out_path,
            true_labels=true_labels,
            pred_labels=pred_labels,
            softmax=softmax,
            embeddings=embeddings[emb_idx],
            embedding_indices=emb_idx,
        )
        print(f"    saved {len(true_labels)} samples ({len(emb_idx)} embeddings) -> {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
