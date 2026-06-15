#!/usr/bin/env python
"""
Unified inference script for vanilla / TENT / CoTTA methods.

Usage:
    python run_inference.py --method vanilla
    python run_inference.py --method tent
    python run_inference.py --method cotta

Saves per week:
    true_labels        (N,)        full predictions
    pred_labels        (N,)        full predictions
    softmax            (N, C)      full predictions
    embeddings         (N//10, D)  10 % random sample, SAME indices across methods
    embedding_indices  (N//10,)    which rows of true_labels the embeddings belong to

The embedding subsample uses --embed_seed (default 0), independent of --seed
(which controls data loading).  Because all methods use the same data loader
seed and frac, true_labels[i] is the same sample in every method run, so
identical embed_seed → identical embedding_indices → directly comparable
embeddings.
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


# ── shared helpers ────────────────────────────────────────────────────────────

def _move(inputs, device):
    return [x.to(device) for x in inputs] if isinstance(inputs, list) else inputs.to(device)


def _pad_singleton(inputs):
    """The adapting methods (TENT/CoTTA/AdaBN) run BatchNorm in train mode, which
    raises `Expected more than 1 value per channel` on a size-1 batch — exactly what
    happens on a week whose sample count leaves a trailing batch of 1 (drop_last is
    False by protocol, so all samples are kept). Duplicate the lone sample to size 2
    so batch-variance is defined (BN eps keeps it finite); callers slice outputs back
    to the original count, so the duplicate never reaches the saved predictions and
    its effect on the BN/adaptation step for that single batch is negligible."""
    if isinstance(inputs, (list, tuple)):
        if inputs[0].shape[0] == 1:
            return [torch.cat([t, t], 0) for t in inputs]
        return inputs
    if inputs.shape[0] == 1:
        return torch.cat([inputs, inputs], 0)
    return inputs


def _subsample_embeddings(embeddings, n_total, embed_seed):
    """Return (subsampled_embeddings, indices) — always 10 % of n_total."""
    n = max(1, n_total // 10)
    idx = np.sort(np.random.RandomState(embed_seed).choice(n_total, n, replace=False))
    return embeddings[idx], idx


# ── vanilla ───────────────────────────────────────────────────────────────────

def run_vanilla(model, loader, device):
    model.eval()
    all_true, all_pred, all_softmax, all_embeds = [], [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
            labels = labels.to(device).long()
            outputs = model(_move(inputs, device))
            logits, feats = outputs['class_preds'], outputs['features']
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_softmax.append(F.softmax(logits, 1).cpu().numpy())
            all_embeds.append(feats.cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_softmax), np.concatenate(all_embeds))


# ── BN-stats diagnostic (no adaptation) ───────────────────────────────────────

def _configure_bnstats(model):
    """Use test-batch BN statistics (like TENT/CoTTA) but perform NO adaptation.

    Isolates the effect of swapping the trained BN running-stats for per-test-batch
    stats from the effect of the adaptation gradient updates. `model.eval()` keeps
    dropout OFF; nulling running_mean/var forces BN to use batch stats even in eval
    (PyTorch BN uses batch stats whenever running_mean/var is None).
    """
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False
            m.running_mean = m.running_var = None
    return model


def run_bnstats(model, loader, device, original_state):
    model.load_state_dict(original_state, strict=False)
    _configure_bnstats(model)
    all_true, all_pred, all_softmax, all_embeds = [], [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
            labels = labels.to(device).long()
            n = labels.shape[0]
            outputs = model(_pad_singleton(_move(inputs, device)))
            logits, feats = outputs['class_preds'][:n], outputs['features'][:n]
            all_true.append(labels.cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_softmax.append(F.softmax(logits, 1).cpu().numpy())
            all_embeds.append(feats.cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_softmax), np.concatenate(all_embeds))


# ── TENT ──────────────────────────────────────────────────────────────────────

def _configure_tent(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None
    # train() above turns dropout ON; TENT reads its predictions from this same
    # model, so leaving dropout active would noise both the entropy gradient and
    # the saved predictions. Disable dropout while keeping BN on batch stats.
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    return model


@torch.enable_grad()
def _tent_step(inputs, model, optimizer):
    out = model(inputs)
    logits = out['class_preds']
    (-(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()).backward()
    optimizer.step(); optimizer.zero_grad()
    return out


def run_tent(model, loader, device, original_state, lr, steps):
    model.load_state_dict(original_state, strict=False)
    _configure_tent(model)
    bn_params = [p for m in model.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                 for p in m.parameters() if p.requires_grad]
    assert bn_params, "TENT: no BatchNorm parameters found"
    optimizer = torch.optim.Adam(bn_params, lr=lr)
    all_true, all_pred, all_softmax, all_embeds = [], [], [], []
    for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
        labels = labels.to(device).long()
        n = labels.shape[0]
        inputs = _pad_singleton(_move(inputs, device))
        for _ in range(steps):
            out = _tent_step(inputs, model, optimizer)
        logits, feats = out['class_preds'][:n], out['features'][:n]
        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.argmax(1).detach().cpu().numpy())
        all_softmax.append(F.softmax(logits, 1).detach().cpu().numpy())
        all_embeds.append(feats.detach().cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_softmax), np.concatenate(all_embeds))


# ── CoTTA ─────────────────────────────────────────────────────────────────────

def _configure_cotta(model):
    model.train()
    model.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.track_running_stats = False
            m.running_mean = m.running_var = None
    return model


def _restore_bn_buffers(model):
    """Re-register BN running buffers that a previous week's _configure_cotta() set
    to None. PyTorch removes a buffer from the module when it's set to None, so a
    later load_state_dict(strict=False) SILENTLY SKIPS the source running_mean/var
    keys (they have no destination) — leaving the buffers absent and BN falling back
    to test-batch stats. Re-registering them first lets load_state_dict repopulate
    the source stats, which the CoTTA anchor (the frozen source model behind the
    confidence gate) requires. Without this the anchor was correct only on the first
    week processed per SLURM shard, making results shard-position-dependent."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.running_mean is None:
            dev = m.weight.device
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=dev))
            m.register_buffer('running_var', torch.ones(m.num_features, device=dev))
            m.register_buffer('num_batches_tracked',
                              torch.tensor(0, dtype=torch.long, device=dev))
            m.track_running_stats = True
    return model


def _ema_update(teacher, student, alpha):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)


def _augment(inputs, std):
    ppi, fs = inputs
    return [ppi + torch.randn_like(ppi) * std, fs + torch.randn_like(fs) * std]


@torch.enable_grad()
def _cotta_step(inputs, student, teacher, anchor, optimizer,
                src_state, rst_m, ap, n_aug, noise_std):
    with torch.no_grad():
        anchor_conf = anchor(inputs)['class_preds'].softmax(1).max(1)[0].mean()
        std_ema = teacher(inputs)['class_preds'].detach()
    if anchor_conf < ap:
        aug_logits = torch.stack([
            teacher(_augment(inputs, noise_std))['class_preds'].detach()
            for _ in range(n_aug)
        ]).mean(0)
        ema_logits = aug_logits
    else:
        ema_logits = std_ema
    out = student(inputs)
    (-(ema_logits.softmax(1) * out['class_preds'].log_softmax(1)).sum(1).mean()).backward()
    optimizer.step(); optimizer.zero_grad()
    _ema_update(teacher, student, alpha=0.99)
    for nm, m in student.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if p.requires_grad:
                key = f"{nm}.{pn}" if nm else pn
                if key in src_state:
                    mask = (torch.rand(p.shape) < rst_m).to(p.device)
                    with torch.no_grad():
                        p.data.copy_(src_state[key] * mask + p.data * (~mask))
    # SAVE the clean (un-augmented) teacher prediction `std_ema`, NOT `ema_logits`.
    # When the confidence gate fires, ema_logits is the MEAN of n_aug noise-augmented
    # teacher passes — that augmented average is a pseudo-LABEL for the self-training
    # loss above, not the model's prediction. Reporting it as the prediction injected
    # augmentation noise into every evaluated sample (the gate fires on ~all 180-class
    # samples), degrading even the source week where adaptation is a no-op.
    return {'class_preds': std_ema, 'features': out['features']}


def run_cotta(model, loader, device, original_state,
              lr, rst_m, ap, n_aug, noise_std, steps, reset):
    if reset:
        model.load_state_dict(original_state, strict=False)
    # anchor = frozen SOURCE model for the confidence gate. It MUST use source BN
    # running stats. A previous week's _configure_cotta() may have nulled the model's
    # BN buffers (and the reset above can't restore deregistered buffers), so rebuild
    # the anchor from a buffer-restored copy loaded with the source state explicitly.
    # eval() then uses source running stats (and dropout off) as the paper intends.
    anchor = deepcopy(model)
    _restore_bn_buffers(anchor)
    anchor.load_state_dict(original_state, strict=False)
    anchor = anchor.to(device).eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    _configure_cotta(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # teacher = EMA of the student; its outputs are the pseudo-labels AND the
    # predictions we save. eval() turns OFF dropout (otherwise every saved
    # prediction is noised by p=0.3 dropout); BN still uses batch stats because
    # _configure_cotta set track_running_stats=False.
    teacher = deepcopy(model).to(device).eval()
    for p in teacher.parameters(): p.detach_()
    src_state = {k: v.to(device) for k, v in original_state.items()
                 if not any(s in k for s in ('running_mean', 'running_var', 'num_batches'))}
    all_true, all_pred, all_softmax, all_embeds = [], [], [], []
    for inputs, labels in tqdm(loader, desc="  batches", leave=False, ncols=100):
        labels = labels.to(device).long()
        n = labels.shape[0]
        inputs = _pad_singleton(_move(inputs, device))
        for _ in range(steps):
            out = _cotta_step(inputs, model, teacher, anchor, optimizer,
                              src_state, rst_m, ap, n_aug, noise_std)
        logits, feats = out['class_preds'][:n], out['features'][:n]
        all_true.append(labels.cpu().numpy())
        all_pred.append(logits.argmax(1).detach().cpu().numpy())
        all_softmax.append(F.softmax(logits, 1).detach().cpu().numpy())
        all_embeds.append(feats.detach().cpu().numpy())
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_softmax), np.concatenate(all_embeds))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # shared
    parser.add_argument('--method', required=True, choices=['vanilla', 'tent', 'cotta', 'bnstats'])
    parser.add_argument('--experiment_dir', default='exps/cesnet_multimodal_each_week_train_v01')
    parser.add_argument('--dataset_root',   default='/home/anatbr/dataset/CESNET-TLS-Year22_v2')
    parser.add_argument('--train_week',     default='week_1')
    parser.add_argument('--output_dir',     default=None)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--num_workers',    type=int,   default=0)
    parser.add_argument('--data_sample_frac', type=float, default=0.1)
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Controls data loading / shuffling')
    parser.add_argument('--acc_only',       action='store_true',
                        help='Save ONLY true_labels + pred_labels (skip softmax + '
                             'embeddings). ~100x smaller .npz — for the multi-seed '
                             'significance study where only accuracy is needed.')
    parser.add_argument('--embed_seed',     type=int,   default=0,
                        help='Controls which 10%% of samples get embeddings saved '
                             '(same value across all methods → same indices)')
    parser.add_argument('--device',         default='cuda:0')
    parser.add_argument('--num_jobs',       type=int,   default=1)
    parser.add_argument('--job_id',         type=int,   default=0)
    # TENT
    parser.add_argument('--tent_lr',        type=float, default=1e-3)
    parser.add_argument('--tent_steps',     type=int,   default=1)
    # CoTTA
    parser.add_argument('--cotta_lr',       type=float, default=1e-3)
    parser.add_argument('--cotta_steps',    type=int,   default=1)
    parser.add_argument('--mt_alpha',       type=float, default=0.99)
    parser.add_argument('--rst_m',          type=float, default=0.01)
    parser.add_argument('--ap',             type=float, default=0.9)
    parser.add_argument('--n_aug',          type=int,   default=32)
    parser.add_argument('--noise_std',      type=float, default=0.02)
    parser.add_argument('--no_reset',       action='store_true',
                        help='CoTTA continual mode: keep adapted state between weeks')
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    dataset_root   = Path(args.dataset_root)
    suffix = {'vanilla': 'inference', 'tent': 'inference_tent', 'cotta': 'inference_cotta',
              'bnstats': 'inference_bnstats'}
    output_dir = (Path(args.output_dir) if args.output_dir
                  else Path('figs') / f'{args.train_week}_{suffix[args.method]}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    label_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"  {num_classes} classes")

    ckpt   = experiment_dir / args.train_week / 'weights' / 'best_model.pth'
    cfg    = experiment_dir / args.train_week / 'config.json'
    model  = load_model_from_checkpoint(ckpt, cfg, num_classes, args.device)
    original_state = deepcopy(model.state_dict())
    print(f"  Loaded model from {ckpt}")

    all_week_dirs = get_available_weeks(dataset_root)
    if args.num_jobs > 1:
        chunks = np.array_split(range(len(all_week_dirs)), args.num_jobs)
        all_week_dirs = [all_week_dirs[i] for i in list(chunks[args.job_id])]
        print(f"  Job {args.job_id}/{args.num_jobs}: "
              f"{all_week_dirs[0].name} – {all_week_dirs[-1].name}")

    cotta_reset = not args.no_reset

    for week_dir in all_week_dirs:
        week_name = week_dir.name
        out_path  = output_dir / f'{week_name}.npz'

        if out_path.exists():
            print(f"  {week_name}: exists, skipping")
            continue

        loader = create_week_loader(
            week_dir, label_mapping,
            batch_size=args.batch_size, num_workers=args.num_workers,
            data_sample_frac=args.data_sample_frac, seed=args.seed,
        )
        if loader is None:
            print(f"  {week_name}: no data, skipping")
            continue

        print(f"  {week_name}: running {args.method}...")

        if args.method == 'vanilla':
            true_labels, pred_labels, softmax, embeddings = run_vanilla(
                model, loader, args.device)

        elif args.method == 'bnstats':
            true_labels, pred_labels, softmax, embeddings = run_bnstats(
                model, loader, args.device, original_state)

        elif args.method == 'tent':
            true_labels, pred_labels, softmax, embeddings = run_tent(
                model, loader, args.device, original_state,
                lr=args.tent_lr, steps=args.tent_steps)

        else:  # cotta
            true_labels, pred_labels, softmax, embeddings = run_cotta(
                model, loader, args.device, original_state,
                lr=args.cotta_lr, rst_m=args.rst_m, ap=args.ap,
                n_aug=args.n_aug, noise_std=args.noise_std,
                steps=args.cotta_steps, reset=cotta_reset)

        # Atomic write: save to a .tmp via a file handle (so numpy doesn't re-append
        # .npz), then Path.replace() to rename atomically. If the job is preempted
        # mid-write, only the .tmp is left (skip-existing checks out_path, so it
        # regenerates) — a partial write can never masquerade as a complete .npz.
        tmp_path = out_path.with_name(out_path.name + '.tmp')
        if args.acc_only:
            with open(tmp_path, 'wb') as _fh:
                np.savez_compressed(
                    _fh,
                    true_labels=true_labels, pred_labels=pred_labels,
                )
            tmp_path.replace(out_path)
            print(f"    saved {len(true_labels)} samples (acc_only) → {out_path}")
        else:
            emb, idx = _subsample_embeddings(embeddings, len(true_labels), args.embed_seed)
            with open(tmp_path, 'wb') as _fh:
                np.savez_compressed(
                    _fh,
                    true_labels=true_labels, pred_labels=pred_labels, softmax=softmax,
                    embeddings=emb, embedding_indices=idx,
                )
            tmp_path.replace(out_path)
            print(f"    saved {len(true_labels)} samples, {len(idx)} embeddings → {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
