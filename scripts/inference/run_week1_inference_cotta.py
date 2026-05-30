#!/usr/bin/env python
"""
Run inference with CoTTA (Continual Test-Time Adaptation).

CoTTA improves on TENT for continual shifts by:
  1. Mean-teacher EMA pseudo-labels (more stable than self-entropy)
  2. Augmentation-averaged pseudo-labels when model confidence is low
  3. Stochastic restoration of weights toward the source model (prevents forgetting)
  4. All parameters are updated (not just BN)

By default the model is reset between weeks (episodic mode) to match
the TENT comparison baseline.  Use --no_reset for true continual adaptation.

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
# CoTTA helpers
# ---------------------------------------------------------------------------

def configure_model_for_cotta(model: nn.Module) -> nn.Module:
    """Train mode, all params grad-enabled; BN layers use batch statistics."""
    model.train()
    model.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def update_ema(ema_model: nn.Module, student: nn.Module, alpha: float) -> nn.Module:
    for ema_p, p in zip(ema_model.parameters(), student.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
    return ema_model


def augment_inputs(inputs, noise_std: float = 0.02):
    """Simple Gaussian noise augmentation for tabular/time-series inputs."""
    ppi, flowstats = inputs
    return [
        ppi + torch.randn_like(ppi) * noise_std,
        flowstats + torch.randn_like(flowstats) * noise_std,
    ]


@torch.enable_grad()
def forward_and_adapt(inputs, student, teacher, anchor, optimizer,
                      model_state, rst_m, ap, n_aug, noise_std):
    """
    One CoTTA step:
      - anchor gives confidence to decide whether to use aug-averaged pseudo-labels
      - teacher (EMA) provides pseudo-labels
      - student is updated by cross-entropy vs. pseudo-labels
      - stochastic restore pulls student back toward source weights
    Returns teacher outputs (dict).
    """
    # anchor confidence (no grad, frozen)
    with torch.no_grad():
        anchor_prob = anchor(inputs)['class_preds'].softmax(1).max(1)[0].mean()

    # standard teacher prediction
    with torch.no_grad():
        teacher_out = teacher(inputs)
    standard_ema_logits = teacher_out['class_preds'].detach()

    # augmentation-averaged teacher prediction
    if anchor_prob < ap:
        aug_logits = []
        for _ in range(n_aug):
            with torch.no_grad():
                aug_logits.append(teacher(augment_inputs(inputs, noise_std))['class_preds'].detach())
        ema_logits = torch.stack(aug_logits).mean(0)
    else:
        ema_logits = standard_ema_logits

    # student forward
    student_out = student(inputs)
    student_logits = student_out['class_preds']

    # cross-entropy student vs. teacher soft labels
    loss = -(ema_logits.softmax(1) * student_logits.log_softmax(1)).sum(1).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # EMA teacher update
    update_ema(teacher, student, alpha=0.99)

    # stochastic restoration toward source weights
    for nm, m in student.named_modules():
        for np_, p in m.named_parameters(recurse=False):
            if p.requires_grad:
                key = f"{nm}.{np_}" if nm else np_
                if key in model_state:
                    mask = (torch.rand(p.shape) < rst_m).to(p.device)
                    with torch.no_grad():
                        p.data.copy_(model_state[key] * mask + p.data * (~mask))

    # return teacher logits as the prediction (more stable)
    return {
        'class_preds': ema_logits,
        'features': student_out['features'],
    }


# ---------------------------------------------------------------------------
# Per-week inference
# ---------------------------------------------------------------------------

def run_inference_cotta(model, loader, device, original_state,
                        lr, rst_m, ap, n_aug, noise_std, steps, reset):
    """
    CoTTA inference for one week.

    reset=True:  model is reset to original_state before this week (episodic).
    reset=False: model continues from wherever it left off (continual).
    """
    if reset:
        model.load_state_dict(original_state, strict=False)

    configure_model_for_cotta(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # teacher and anchor are frozen copies
    teacher = deepcopy(model).to(device)
    for p in teacher.parameters():
        p.detach_()

    anchor = deepcopy(model).to(device)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    # state dict on device for stochastic restore
    state_on_device = {k: v.to(device) for k, v in original_state.items()
                       if 'running_mean' not in k and 'running_var' not in k
                       and 'num_batches_tracked' not in k}

    all_true, all_pred, all_softmax, all_embeds = [], [], [], []

    for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
        labels = labels.to(device).long()
        inputs = [x.to(device) for x in inputs] if isinstance(inputs, list) else inputs.to(device)

        for _ in range(steps):
            outputs = forward_and_adapt(
                inputs, model, teacher, anchor, optimizer,
                state_on_device, rst_m, ap, n_aug, noise_std,
            )

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
    parser.add_argument('--train_week', default='week_1')
    parser.add_argument('--output_dir', default=None,
                        help='Where to save .npz files (default: figs/<train_week>_inference_cotta)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data_sample_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)
    # CoTTA hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cotta_steps', type=int, default=1,
                        help='Gradient steps per batch')
    parser.add_argument('--mt_alpha', type=float, default=0.99,
                        help='EMA smoothing factor for teacher model')
    parser.add_argument('--rst_m', type=float, default=0.01,
                        help='Stochastic restore probability per weight element')
    parser.add_argument('--ap', type=float, default=0.9,
                        help='Anchor confidence threshold for augmentation averaging')
    parser.add_argument('--n_aug', type=int, default=32,
                        help='Number of augmented views for pseudo-label averaging')
    parser.add_argument('--noise_std', type=float, default=0.02,
                        help='Gaussian noise std for input augmentation')
    parser.add_argument('--no_reset', action='store_true',
                        help='Continual mode: do not reset model between weeks')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    dataset_root = Path(args.dataset_root)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path('figs') / f'{args.train_week}_inference_cotta'
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
        chunks = np.array_split(list(range(len(test_week_dirs))), args.num_jobs)
        test_week_dirs = [test_week_dirs[i] for i in list(chunks[args.job_id])]
        print(f"Job {args.job_id}/{args.num_jobs}: weeks {test_week_dirs[0].name} – {test_week_dirs[-1].name}")

    reset = not args.no_reset
    print(
        f"Running CoTTA (lr={args.lr}, rst={args.rst_m}, ap={args.ap}, "
        f"n_aug={args.n_aug}, reset={reset}) over "
        f"{len(test_week_dirs)} weeks -> {output_dir}\n"
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

        print(f"  {week_name}: running CoTTA inference...")
        true_labels, pred_labels, softmax, embeddings = run_inference_cotta(
            model, loader, args.device, original_state,
            lr=args.lr, rst_m=args.rst_m, ap=args.ap,
            n_aug=args.n_aug, noise_std=args.noise_std,
            steps=args.cotta_steps, reset=reset,
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
