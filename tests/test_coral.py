"""
Sanity checks for the CORAL loss implementation.

Tests:
1. Loss is zero when source == target (identical distributions)
2. Loss is positive and grows when distributions diverge
3. Gradients flow from CORAL loss back to feature extractor
4. Loss value matches the reference formula manually
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import compute_coral_loss
from models.multimodal_cesnet import Multimodal_CESNET


def test_zero_loss_identical():
    """CORAL loss should be ~0 when source and target are the same tensor."""
    features = torch.randn(64, 128)
    loss = compute_coral_loss(features, features)
    assert loss.item() < 1e-6, f"Expected ~0, got {loss.item()}"
    print(f"[PASS] Zero loss on identical inputs: {loss.item():.2e}")


def test_positive_loss_different():
    """Loss should be positive for different distributions."""
    src = torch.randn(64, 128)
    tgt = torch.randn(64, 128) * 5 + 3  # different scale and mean
    loss = compute_coral_loss(src, tgt)
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    print(f"[PASS] Positive loss on different distributions: {loss.item():.4f}")


def test_loss_increases_with_divergence():
    """Loss should increase as distributions diverge further."""
    src = torch.randn(64, 32)
    losses = []
    for scale in [1.0, 2.0, 5.0, 10.0]:
        tgt = torch.randn(64, 32) * scale
        losses.append(compute_coral_loss(src, tgt).item())
    assert losses == sorted(losses), f"Loss should increase with divergence: {losses}"
    print(f"[PASS] Loss increases with divergence: {[f'{l:.3f}' for l in losses]}")


def test_manual_formula():
    """Verify output matches the formula: ||Cs - Ct||_F^2 / (4*d^2)."""
    torch.manual_seed(0)
    d = 16
    ns, nt = 32, 32
    src = torch.randn(ns, d)
    tgt = torch.randn(nt, d)

    # manual
    src_c = src - src.mean(0)
    tgt_c = tgt - tgt.mean(0)
    cov_s = (src_c.T @ src_c) / (ns - 1)
    cov_t = (tgt_c.T @ tgt_c) / (nt - 1)
    expected = (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)

    got = compute_coral_loss(src, tgt)
    assert abs(got.item() - expected.item()) < 1e-4, \
        f"Formula mismatch: expected {expected.item():.6f}, got {got.item():.6f}"
    print(f"[PASS] Formula matches manual: {got.item():.6f} == {expected.item():.6f}")


def test_gradient_flow_through_model():
    """CORAL loss gradients should reach the feature extractor weights."""
    model = Multimodal_CESNET(
        num_classes=10,
        flowstats_input_size=44,
        ppi_input_channels=3,
        lambda_rgl=0.0,  # no DANN — isolate CORAL
    )
    model.train()

    bs = 8
    src_ppi = torch.randn(bs, 3, 30)
    src_fs  = torch.randn(bs, 44)
    tgt_ppi = torch.randn(bs, 3, 30)
    tgt_fs  = torch.randn(bs, 44)

    model.zero_grad()
    src_out = model([src_ppi, src_fs])
    tgt_out = model([tgt_ppi, tgt_fs])
    loss = compute_coral_loss(src_out['features'], tgt_out['features'])
    loss.backward()

    # Check at least one feature-extractor param has a gradient
    fe_params_with_grad = [
        p for n, p in model.named_parameters()
        if 'ppi' in n or 'flowstats' in n or 'shared' in n
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert fe_params_with_grad, "No gradients reached feature extractor from CORAL loss"
    print(f"[PASS] Gradients flow to feature extractor ({len(fe_params_with_grad)} params with grad)")


def test_gradient_flow_end_to_end_training():
    """CORAL loss should change model params after one optimizer step."""
    torch.manual_seed(42)
    model = Multimodal_CESNET(num_classes=10, flowstats_input_size=44, ppi_input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    bs = 8
    src_ppi, src_fs = torch.randn(bs, 3, 30), torch.randn(bs, 44)
    tgt_ppi, tgt_fs = torch.randn(bs, 3, 30), torch.randn(bs, 44)
    labels = torch.randint(0, 10, (bs,))

    params_before = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    src_out = model([src_ppi, src_fs])
    tgt_out = model([tgt_ppi, tgt_fs])
    loss = criterion(src_out['class_preds'], labels) + compute_coral_loss(src_out['features'], tgt_out['features'])
    loss.backward()
    optimizer.step()

    params_changed = sum(
        not torch.equal(p, pb)
        for p, pb in zip(model.parameters(), params_before)
    )
    assert params_changed > 0, "No parameters changed after optimizer step"
    print(f"[PASS] {params_changed} parameter tensors updated after CORAL + CE backward step")


if __name__ == '__main__':
    print("=" * 60)
    print("CORAL Loss Sanity Checks")
    print("=" * 60)
    test_zero_loss_identical()
    test_positive_loss_different()
    test_loss_increases_with_divergence()
    test_manual_formula()
    test_gradient_flow_through_model()
    test_gradient_flow_end_to_end_training()
    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)
