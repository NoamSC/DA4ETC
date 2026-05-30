#!/usr/bin/env python
"""
Sanity checks for run_inference.py TENT and CoTTA implementations.

Two sections:
  1. Unit tests (synthetic model, no dataset needed) — run in seconds.
  2. NPZ cross-method consistency checks (requires existing .npz files).

Run:
    python test_inference_sanity.py            # unit tests only
    python test_inference_sanity.py --npz      # + NPZ checks
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── import the functions under test ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from run_inference import (
    _configure_tent, _tent_step,
    _configure_cotta, _ema_update, _augment, _cotta_step,
    run_tent, run_cotta,
    _subsample_embeddings,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


# ── tiny synthetic model ───────────────────────────────────────────────────────

class TinyNet(nn.Module):
    """Two-modality model with BatchNorm1d, matching the real model's interface."""
    def __init__(self, in_dim=32, n_classes=10, feat_dim=16):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, 32)
        self.bn1   = nn.BatchNorm1d(32)
        self.fc2   = nn.Linear(32, feat_dim)
        self.head  = nn.Linear(feat_dim, n_classes)

    def forward(self, inputs):
        x = inputs[0] if isinstance(inputs, list) else inputs
        x = F.relu(self.bn1(self.fc1(x)))
        feats  = self.fc2(x)
        logits = self.head(feats)
        return {'class_preds': logits, 'features': feats}


def make_net(seed=0):
    torch.manual_seed(seed)
    return TinyNet()


def make_batch(bs=32, in_dim=32, n_classes=10, seed=0):
    rng = torch.Generator(); rng.manual_seed(seed)
    ppi  = torch.randn(bs, in_dim,  generator=rng)
    fs   = torch.randn(bs, in_dim // 2, generator=rng)
    labs = torch.randint(0, n_classes, (bs,), generator=rng)
    return [ppi, fs], labs


def fake_loader(n_batches=5, bs=32, in_dim=32, n_classes=10):
    for i in range(n_batches):
        yield make_batch(bs, in_dim, n_classes, seed=i)


# ─────────────────────────────────────────────────────────────────────────────
# TENT unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tent():
    print("\n── TENT ─────────────────────────────────────────────────────────────")
    net = make_net()
    original_state = deepcopy(net.state_dict())

    # 1. configure_tent grad flags
    _configure_tent(net)
    bn_params_with_grad = [
        (n, p) for n, p in net.named_parameters()
        if isinstance(dict(net.named_modules()).get(n.rsplit('.', 1)[0], None),
                      (nn.BatchNorm1d, nn.BatchNorm2d))
        and p.requires_grad
    ]
    non_bn_grads = [
        n for n, p in net.named_parameters()
        if p.requires_grad and not any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
            for nm, m in net.named_modules()
            if nm and n.startswith(nm)
        )
    ]
    # simpler check: BN weight/bias require grad; linear weight/bias do not
    check("TENT: bn1.weight requires_grad",  net.bn1.weight.requires_grad)
    check("TENT: bn1.bias requires_grad",    net.bn1.bias.requires_grad)
    check("TENT: fc1.weight frozen",        not net.fc1.weight.requires_grad)
    check("TENT: head.weight frozen",       not net.head.weight.requires_grad)

    # 2. running stats removed
    check("TENT: bn1.track_running_stats=False", not net.bn1.track_running_stats)
    check("TENT: bn1.running_mean=None",    net.bn1.running_mean is None)
    check("TENT: bn1.running_var=None",     net.bn1.running_var  is None)

    # 3. bn params change, linear params unchanged after one step
    net.load_state_dict(original_state, strict=False)
    _configure_tent(net)
    bn_params = [p for m in net.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                 for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(bn_params, lr=1e-3)
    inputs, _ = make_batch()
    bn_w_before = net.bn1.weight.data.clone()
    fc1_w_before = net.fc1.weight.data.clone()
    _tent_step(inputs, net, optimizer)
    check("TENT: bn1.weight updated after step",
          not torch.allclose(net.bn1.weight.data, bn_w_before))
    check("TENT: fc1.weight unchanged after step",
          torch.allclose(net.fc1.weight.data, fc1_w_before))

    # 4. entropy decreases over multiple steps on the same batch
    net.load_state_dict(original_state, strict=False)
    _configure_tent(net)
    bn_params = [p for m in net.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                 for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(bn_params, lr=1e-2)
    inputs, _ = make_batch(bs=128)

    def entropy(model, x):
        with torch.no_grad():
            logits = model(x)['class_preds']
        p = logits.softmax(1)
        return -(p * p.log()).sum(1).mean().item()

    h0 = entropy(net, inputs)
    for _ in range(20):
        _tent_step(inputs, net, optimizer)
    h1 = entropy(net, inputs)
    check("TENT: entropy decreases over 20 steps", h1 < h0,
          f"before={h0:.4f} after={h1:.4f}")

    # 5. model resets correctly (key check for multi-week loops)
    net.load_state_dict(original_state, strict=False)
    _configure_tent(net)
    bn_params = [p for m in net.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                 for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(bn_params, lr=1e-3)
    for _ in range(10):
        _tent_step(inputs, net, optimizer)
    net.load_state_dict(original_state, strict=False)
    check("TENT: reload restores fc1.weight",
          torch.allclose(net.fc1.weight.data, original_state['fc1.weight']))
    check("TENT: reload restores head.weight",
          torch.allclose(net.head.weight.data, original_state['head.weight']))

    # 6. softmax rows sum to 1
    net.load_state_dict(original_state, strict=False)
    _configure_tent(net)
    bn_params = [p for m in net.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                 for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(bn_params, lr=1e-3)
    true_l, pred_l, softmax, embeds = run_tent(
        net, fake_loader(3), 'cpu', original_state, lr=1e-3, steps=1)
    row_sums = softmax.sum(axis=1)
    check("TENT: softmax rows sum to 1",
          np.allclose(row_sums, 1.0, atol=1e-5),
          f"max deviation={np.abs(row_sums - 1).max():.2e}")
    check("TENT: no NaN in softmax",  not np.isnan(softmax).any())
    check("TENT: no NaN in embeds",   not np.isnan(embeds).any())
    check("TENT: pred_labels in range",
          pred_l.min() >= 0 and pred_l.max() < 10)


# ─────────────────────────────────────────────────────────────────────────────
# CoTTA unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cotta():
    print("\n── CoTTA ────────────────────────────────────────────────────────────")
    net = make_net()
    original_state = deepcopy(net.state_dict())
    device = 'cpu'

    # build full cotta state (mirrors run_cotta setup)
    net.load_state_dict(original_state, strict=False)
    _configure_cotta(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    teacher = deepcopy(net).to(device)
    for p in teacher.parameters(): p.detach_()
    anchor = deepcopy(net).to(device).eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    src_state = {k: v.to(device) for k, v in original_state.items()
                 if not any(s in k for s in ('running_mean', 'running_var', 'num_batches'))}

    # 1. src_state has no BN running stats
    has_stats = any('running_mean' in k or 'running_var' in k for k in src_state)
    check("CoTTA: src_state has no running_mean/var", not has_stats)

    # 2. anchor and teacher start identical to student
    student_params = {n: p.data.clone() for n, p in net.named_parameters()}
    teacher_close = all(
        torch.allclose(tp.data, student_params[n])
        for n, tp in teacher.named_parameters()
    )
    anchor_close = all(
        torch.allclose(ap.data, student_params[n])
        for n, ap in anchor.named_parameters()
    )
    check("CoTTA: teacher starts == student", teacher_close)
    check("CoTTA: anchor starts == student",  anchor_close)

    # 3. anchor never changes across batches
    anchor_snap = {n: p.data.clone() for n, p in anchor.named_parameters()}
    inputs, _ = make_batch()
    for _ in range(5):
        _cotta_step(inputs, net, teacher, anchor, optimizer,
                    src_state, rst_m=0.0, ap=0.9, n_aug=2, noise_std=0.02)
    anchor_unchanged = all(
        torch.allclose(anchor_snap[n], p.data)
        for n, p in anchor.named_parameters()
    )
    check("CoTTA: anchor frozen across 5 batches", anchor_unchanged)

    # 4. teacher diverges from student after EMA update
    teacher_snap = {n: p.data.clone() for n, p in teacher.named_parameters()}
    student_snap = {n: p.data.clone() for n, p in net.named_parameters()}
    _ema_update(teacher, net, alpha=0.99)
    teacher_moved = any(
        not torch.allclose(teacher_snap[n], p.data)
        for n, p in teacher.named_parameters()
    )
    teacher_between = all(
        torch.allclose(
            p.data,
            teacher_snap[n] * 0.99 + student_snap[n] * 0.01,
            atol=1e-6
        )
        for n, p in teacher.named_parameters()
    )
    check("CoTTA: EMA update moves teacher",             teacher_moved)
    check("CoTTA: EMA update follows alpha=0.99 formula", teacher_between)

    # 5. stochastic restore: with rst_m=1.0 student should match src_state
    net2 = make_net(); net2.load_state_dict(original_state); _configure_cotta(net2)
    teacher2 = deepcopy(net2); anchor2 = deepcopy(net2).eval()
    for p in anchor2.parameters(): p.requires_grad_(False)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    torch.manual_seed(0)
    _cotta_step(inputs, net2, teacher2, anchor2, optimizer2,
                src_state, rst_m=1.0, ap=0.9, n_aug=1, noise_std=0.01)
    restored = all(
        torch.allclose(net2.state_dict()[k], src_state[k], atol=1e-6)
        for k in src_state
    )
    check("CoTTA: rst_m=1.0 fully restores student weights", restored)

    # 6. returned class_preds are teacher logits (not student logits).
    #    Force plain-teacher path with ap=-1 (anchor_conf is always >= 0, so
    #    anchor_conf < -1 is never True → else branch: ema_logits = std_ema).
    #    std_ema is computed at the START of the step, before any weight update,
    #    so it should equal teacher(inputs) computed right before the call.
    net3 = make_net(); net3.load_state_dict(original_state); _configure_cotta(net3)
    teacher3 = deepcopy(net3).to(device)
    for p in teacher3.parameters(): p.detach_()
    anchor3 = deepcopy(net3).to(device).eval()
    for p in anchor3.parameters(): p.requires_grad_(False)
    optimizer3 = torch.optim.Adam(net3.parameters(), lr=0.0)  # lr=0: student params unchanged
    teacher3_logits_before = teacher3(inputs)['class_preds'].detach()
    out = _cotta_step(inputs, net3, teacher3, anchor3, optimizer3,
                      src_state, rst_m=0.0, ap=-1.0, n_aug=1, noise_std=0.01)
    check("CoTTA: returned class_preds == teacher logits (not student)",
          torch.allclose(out['class_preds'], teacher3_logits_before, atol=1e-5))

    # 7. student parameters change over batches (learning happening)
    net4 = make_net(); net4.load_state_dict(original_state); _configure_cotta(net4)
    teacher4 = deepcopy(net4)
    anchor4 = deepcopy(net4).eval()
    for p in anchor4.parameters(): p.requires_grad_(False)
    optimizer4 = torch.optim.Adam(net4.parameters(), lr=1e-2)
    w_before = net4.fc1.weight.data.clone()
    for i in range(5):
        inp, _ = make_batch(seed=i)
        _cotta_step(inp, net4, teacher4, anchor4, optimizer4,
                    src_state, rst_m=0.0, ap=0.9, n_aug=2, noise_std=0.02)
    check("CoTTA: student weights change over batches",
          not torch.allclose(net4.fc1.weight.data, w_before))

    # 8. full run_cotta produces valid outputs
    net5 = make_net(); net5.load_state_dict(original_state)
    true_l, pred_l, softmax, embeds = run_cotta(
        net5, fake_loader(3), device, original_state,
        lr=1e-3, rst_m=0.01, ap=0.9, n_aug=2, noise_std=0.02, steps=1, reset=True)
    row_sums = softmax.sum(axis=1)
    check("CoTTA: softmax rows sum to 1",
          np.allclose(row_sums, 1.0, atol=1e-5),
          f"max deviation={np.abs(row_sums - 1).max():.2e}")
    check("CoTTA: no NaN in softmax", not np.isnan(softmax).any())
    check("CoTTA: no NaN in embeds",  not np.isnan(embeds).any())
    check("CoTTA: pred_labels in range",
          pred_l.min() >= 0 and pred_l.max() < 10)


# ─────────────────────────────────────────────────────────────────────────────
# Subsample consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_subsample():
    print("\n── Embedding subsampling ────────────────────────────────────────────")
    N = 1000
    emb = np.random.randn(N, 600).astype(np.float32)
    e1, idx1 = _subsample_embeddings(emb, N, embed_seed=0)
    e2, idx2 = _subsample_embeddings(emb, N, embed_seed=0)
    check("subsample: same seed → same indices",  np.array_equal(idx1, idx2))
    check("subsample: returns 10% of N",          len(idx1) == N // 10)
    check("subsample: embeddings match by index", np.allclose(e1, emb[idx1]))

    # different seeds → different indices (almost surely)
    _, idx3 = _subsample_embeddings(emb, N, embed_seed=1)
    check("subsample: different seed → different indices", not np.array_equal(idx1, idx3))


# ─────────────────────────────────────────────────────────────────────────────
# NPZ cross-method consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_npz_consistency(vanilla_dir, tent_dir, cotta_dir):
    print("\n── NPZ cross-method consistency ─────────────────────────────────────")
    dirs = {
        'vanilla': Path(vanilla_dir),
        'tent':    Path(tent_dir),
        'cotta':   Path(cotta_dir),
    }
    # find weeks present in all three
    week_sets = {m: set(p.stem for p in d.glob('WEEK-*.npz')) for m, d in dirs.items()}
    common = week_sets['vanilla'] & week_sets['tent'] & week_sets['cotta']
    if not common:
        print("  (no weeks present in all three methods — skipping)")
        return

    sample_weeks = sorted(common)[:5]  # check first 5 shared weeks

    all_true_match = True
    all_idx_match  = True
    all_softmax_ok = True
    all_no_nan     = True
    all_shapes_ok  = True

    for week in sample_weeks:
        data = {}
        for m, d in dirs.items():
            try:
                data[m] = np.load(d / f'{week}.npz')
            except Exception as e:
                print(f"  WARN  {week}/{m}: {e}")
                continue

        if len(data) < 3:
            continue

        # true_labels must be identical
        if not (np.array_equal(data['vanilla']['true_labels'], data['tent']['true_labels']) and
                np.array_equal(data['vanilla']['true_labels'], data['cotta']['true_labels'])):
            all_true_match = False
            print(f"  WARN  {week}: true_labels differ across methods")

        # embedding_indices must be identical
        if 'embedding_indices' in data['vanilla'] and 'embedding_indices' in data['tent']:
            if not np.array_equal(data['vanilla']['embedding_indices'],
                                  data['tent']['embedding_indices']):
                all_idx_match = False
                print(f"  WARN  {week}: embedding_indices differ vanilla vs tent")
        if 'embedding_indices' in data['vanilla'] and 'embedding_indices' in data['cotta']:
            if not np.array_equal(data['vanilla']['embedding_indices'],
                                  data['cotta']['embedding_indices']):
                all_idx_match = False
                print(f"  WARN  {week}: embedding_indices differ vanilla vs cotta")

        for m, d in data.items():
            N = len(d['true_labels'])
            sm = d['softmax']
            emb = d['embeddings']

            # softmax rows sum to 1
            if not np.allclose(sm.sum(axis=1), 1.0, atol=1e-4):
                all_softmax_ok = False
                print(f"  WARN  {week}/{m}: softmax rows don't sum to 1")

            # no NaN/Inf
            for key in ('true_labels', 'pred_labels', 'softmax', 'embeddings'):
                arr = d[key]
                if np.isnan(arr).any() or np.isinf(arr).any():
                    all_no_nan = False
                    print(f"  WARN  {week}/{m}: NaN/Inf in {key}")

            # embeddings are ~10% of N
            expected = max(1, N // 10)
            if abs(len(emb) - expected) > expected * 0.05:
                all_shapes_ok = False
                print(f"  WARN  {week}/{m}: embeddings={len(emb)} expected~{expected}")

    check(f"NPZ: true_labels identical across methods ({len(sample_weeks)} weeks)", all_true_match)
    check(f"NPZ: embedding_indices identical across methods",                        all_idx_match)
    check(f"NPZ: softmax rows sum to 1",                                            all_softmax_ok)
    check(f"NPZ: no NaN/Inf in arrays",                                             all_no_nan)
    check(f"NPZ: embeddings ≈ 10% of N",                                            all_shapes_ok)

    # vanilla accuracy on first shared week (sanity that model actually works)
    week = sorted(common)[0]
    d = np.load(dirs['vanilla'] / f'{week}.npz')
    acc = (d['true_labels'] == d['pred_labels']).mean()
    check(f"NPZ: vanilla accuracy > 50% on {week}", acc > 0.5, f"acc={acc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', action='store_true',
                        help='Also run NPZ cross-method consistency checks')
    parser.add_argument('--vanilla_dir', default='figs/week_1_inference')
    parser.add_argument('--tent_dir',    default='figs/week_1_inference_tent')
    parser.add_argument('--cotta_dir',   default='figs/week_1_inference_cotta')
    args = parser.parse_args()

    test_subsample()
    test_tent()
    test_cotta()

    if args.npz:
        test_npz_consistency(args.vanilla_dir, args.tent_dir, args.cotta_dir)

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[32mAll checks passed.\033[0m")
