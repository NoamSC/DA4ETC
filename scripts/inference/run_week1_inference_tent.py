#!/usr/bin/env python
"""
Run inference with TENT test-time adaptation (entropy minimization).

For each test week the model is reset to its trained weights, then adapts
online batch-by-batch by minimising prediction entropy over the BatchNorm1d
affine parameters (gamma / beta).  All other weights stay frozen.

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
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from temporal_generalization import (
    load_model_from_checkpoint,
    create_week_loader,
    get_available_weeks,
)
from train_per_week_cesnet import load_label_mapping


# ---------------------------------------------------------------------------
# TENT helpers — adapted for BatchNorm1d and dict-returning models
# ---------------------------------------------------------------------------

def configure_model_for_tent(model: nn.Module) -> nn.Module:
    """Switch to train mode, freeze everything except BN affine params."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            # Use current-batch statistics instead of running stats
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_bn_params(model: nn.Module):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for pn, p in m.named_parameters():
                if pn in ('weight', 'bias'):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


@torch.enable_grad()
def _forward_and_adapt(inputs, model, optimizer):
    outputs = model(inputs)
    logits = outputs['class_preds']
    loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


# ---------------------------------------------------------------------------
# Per-week inference
# ---------------------------------------------------------------------------

def run_inference_tent(model, loader, device, original_state, lr=1e-3, steps=1):
    """
    Reset model to original_state, configure for TENT, adapt + infer online.

    Returns (true_labels, pred_labels, softmax, embeddings) as numpy arrays.
    """
    # strict=False: original_state has running_mean/running_var buffers that
    # a previously tent-configured model no longer has (track_running_stats=False).
    # Those buffers are discarded by configure_model_for_tent anyway.
    model.load_state_dict(original_state, strict=False)
    configure_model_for_tent(model)

    params, _ = collect_bn_params(model)
    assert params, "No BatchNorm parameters found — TENT requires BatchNorm layers."
    optimizer = torch.optim.Adam(params, lr=lr)

    all_true, all_pred, all_softmax, all_embeds = [], [], [], []

    for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
        labels = labels.to(device).long()
        inputs = [x.to(device) for x in inputs] if isinstance(inputs, list) else inputs.to(device)

        for _ in range(steps):
            outputs = _forward_and_adapt(inputs, model, optimizer)

        logits = outputs['class_preds']
        features = outputs['features']

        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_softmax.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        all_embeds.append(features.detach().cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_softmax, axis=0),
        np.concatenate(all_embeds, axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', default='exps/cesnet_multimodal_each_week_train_v01')
    parser.add_argument('--dataset_root', default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--train_week', default='week_1', help='Which trained model to use')
    parser.add_argument('--output_dir', default=None,
                        help='Where to save .npz files (default: figs/<train_week>_inference_tent)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data_sample_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_jobs', type=int, default=1,
                        help='Total number of array tasks (default: 1 = no splitting)')
    parser.add_argument('--job_id', type=int, default=0,
                        help='This task index 0..num_jobs-1 (maps to SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--tent_lr', type=float, default=1e-3,
                        help='Learning rate for TENT BN param updates (default: 1e-3)')
    parser.add_argument('--tent_steps', type=int, default=1,
                        help='Gradient steps per batch (default: 1)')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path('figs') / f'{args.train_week}_inference_tent'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading label mapping from {dataset_root}...")
    label_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"  {num_classes} classes")

    checkpoint_path = experiment_dir / args.train_week / 'weights' / 'best_model.pth'
    config_path = experiment_dir / args.train_week / 'config.json'
    print(f"Loading model: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, args.device)
    original_state = deepcopy(model.state_dict())
    print("  Done.")

    test_week_dirs = get_available_weeks(dataset_root)
    if args.num_jobs > 1:
        all_indices = list(range(len(test_week_dirs)))
        chunks = np.array_split(all_indices, args.num_jobs)
        chunk = list(chunks[args.job_id])
        test_week_dirs = [test_week_dirs[i] for i in chunk]
        print(f"Job {args.job_id}/{args.num_jobs}: weeks {test_week_dirs[0].name} – {test_week_dirs[-1].name}")
    print(
        f"Running TENT (lr={args.tent_lr}, steps={args.tent_steps}) over "
        f"{len(test_week_dirs)} test weeks -> {output_dir}\n"
    )

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

        print(f"  {week_name}: running TENT inference...")
        true_labels, pred_labels, softmax, embeddings = run_inference_tent(
            model, loader, args.device, original_state,
            lr=args.tent_lr, steps=args.tent_steps,
        )

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
