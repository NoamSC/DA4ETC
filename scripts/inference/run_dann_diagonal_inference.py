#!/usr/bin/env python
"""
Diagonal inference for the forward-transfer DANN sweep.

The DANN sweep trains ONE model per target week, aligning a fixed source week
(week-16) to that target week, at

    <experiment_dir>/WEEK-2022-16_val_WEEK-2022-{NN}/weights/best_model.pth

The benchmark needs the DIAGONAL of that sweep: for each target week N, load
*that week's* model and evaluate it frozen (plain forward through its class head)
on WEEK-2022-N's TEST split. This is the matched DANN comparator to the frozen
vanilla / TENT / CoTTA runs produced by run_inference.py — the difference is that
here a DIFFERENT model is loaded per week (the one that was DANN-aligned to that
week), instead of one source model evaluated on all weeks.

Saves per week to <output_dir>/WEEK-2022-N.npz:
    true_labels   (N,)
    pred_labels   (N,)
    softmax       (N, C)   (unless --acc_only)

Atomic-save + skip-existing + --num_jobs/--job_id sharding mirror run_inference.py.
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


# ── shared helpers ────────────────────────────────────────────────────────────

def _move(inputs, device):
    return [x.to(device) for x in inputs] if isinstance(inputs, list) else inputs.to(device)


# ── vanilla forward loop (frozen) ─────────────────────────────────────────────

def run_vanilla(model, loader, device):
    """Frozen forward pass — read predictions from the model's class head.

    Identical to run_inference.run_vanilla but does not collect embeddings (the
    diagonal benchmark only needs labels + softmax)."""
    model.eval()
    all_true, all_pred, all_softmax = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
            labels = labels.to(device).long()
            outputs = model(_move(inputs, device))
            logits = outputs['class_preds']
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_softmax.append(F.softmax(logits, 1).cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_softmax))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagonal frozen-forward inference for the DANN forward-transfer sweep")
    parser.add_argument('--experiment_dir', default='exps/cesnet_tls_dann_fwd_w16_v01',
                        help='Sweep root holding per-target-week DANN models '
                             '(WEEK-2022-16_val_WEEK-2022-NN/weights/best_model.pth)')
    parser.add_argument('--dataset_root',   default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--source_week',    default='16',
                        help='Source week the sweep was aligned FROM (default: 16)')
    parser.add_argument('--output_dir',     default='results/inference/dann_fwd_w16_diagonal')
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--data_sample_frac', type=float, default=0.1)
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Controls data loading / shuffling')
    parser.add_argument('--acc_only',       action='store_true',
                        help='Save ONLY true_labels + pred_labels (skip softmax). '
                             'Smaller .npz — for the accuracy-only significance study.')
    parser.add_argument('--device',         default='cuda:0')
    parser.add_argument('--num_jobs',       type=int,   default=1)
    parser.add_argument('--job_id',         type=int,   default=0)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    dataset_root   = Path(args.dataset_root)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("DANN forward-transfer DIAGONAL inference")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  source_week:    WEEK-2022-{args.source_week}")
    print(f"  output_dir:     {output_dir}")

    label_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"  {num_classes} classes")

    all_week_dirs = get_available_weeks(dataset_root)
    if args.num_jobs > 1:
        chunks = np.array_split(range(len(all_week_dirs)), args.num_jobs)
        all_week_dirs = [all_week_dirs[i] for i in list(chunks[args.job_id])]
        print(f"  Job {args.job_id}/{args.num_jobs}: "
              f"{all_week_dirs[0].name} – {all_week_dirs[-1].name}")

    for week_dir in all_week_dirs:
        week_name = week_dir.name                       # e.g. WEEK-2022-18
        week_suffix = week_name.split('-')[-1]          # e.g. 18
        out_path = output_dir / f'{week_name}.npz'

        if out_path.exists():
            print(f"  {week_name}: exists, skipping")
            continue

        # The DIAGONAL: load the model DANN-aligned source→this-week.
        model_run = f'WEEK-2022-{args.source_week}_val_{week_name}'
        ckpt = experiment_dir / model_run / 'weights' / 'best_model.pth'
        cfg  = experiment_dir / model_run / 'config.json'
        if not ckpt.exists() or not cfg.exists():
            print(f"  {week_name}: no DANN checkpoint at {ckpt}, skipping")
            continue

        loader = create_week_loader(
            week_dir, label_mapping,
            batch_size=args.batch_size, num_workers=args.num_workers,
            data_sample_frac=args.data_sample_frac, seed=args.seed,
        )
        if loader is None:
            print(f"  {week_name}: no data, skipping")
            continue

        model = load_model_from_checkpoint(ckpt, cfg, num_classes, args.device)
        print(f"  {week_name}: model {model_run} → eval on {week_name} test")

        true_labels, pred_labels, softmax = run_vanilla(model, loader, args.device)

        # Atomic write: save to a .tmp via a file handle, then Path.replace() to
        # rename atomically. A preempted job leaves only the .tmp (skip-existing
        # checks out_path), so a partial write can never look like a complete .npz.
        tmp_path = out_path.with_name(out_path.name + '.tmp')
        if args.acc_only:
            with open(tmp_path, 'wb') as _fh:
                np.savez_compressed(_fh,
                                    true_labels=true_labels, pred_labels=pred_labels)
            tmp_path.replace(out_path)
            print(f"    saved {len(true_labels)} samples (acc_only) → {out_path}")
        else:
            with open(tmp_path, 'wb') as _fh:
                np.savez_compressed(_fh,
                                    true_labels=true_labels, pred_labels=pred_labels,
                                    softmax=softmax)
            tmp_path.replace(out_path)
            print(f"    saved {len(true_labels)} samples → {out_path}")

        del model, loader

    print("\nDone.")


if __name__ == '__main__':
    main()
