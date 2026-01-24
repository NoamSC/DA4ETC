"""
Simple test to verify GRL is reversing gradients.
"""
import torch
import torch.nn as nn
from models.configurable_cnn import GradientReversalLayer

print("=" * 80)
print("TESTING GRL GRADIENT REVERSAL")
print("=" * 80)

# Create simple GRL
grl = GradientReversalLayer(lambda_=1.0)

# Create a simple input tensor
x = torch.randn(10, 5, requires_grad=True)
print(f"\nInput tensor shape: {x.shape}")
print(f"Input requires_grad: {x.requires_grad}")

# Forward pass through GRL
y = grl(x)
print(f"\nAfter GRL forward:")
print(f"  Output shape: {y.shape}")
print(f"  Output requires_grad: {y.requires_grad}")
print(f"  Output is input: {y is x}")
print(f"  Output data is input data: {torch.equal(y, x)}")

# Compute a simple loss
loss = y.sum()
print(f"\nLoss: {loss.item()}")

# Backward pass
loss.backward()

# Check gradient
print(f"\nAfter backward:")
print(f"  x.grad is not None: {x.grad is not None}")
if x.grad is not None:
    print(f"  x.grad sum: {x.grad.sum().item()}")
    print(f"  Expected (without GRL): {torch.ones_like(x).sum().item()}")
    print(f"  Expected (with GRL, lambda=1.0): {-torch.ones_like(x).sum().item()}")

    # Check if gradient was reversed
    expected_grad_without_grl = torch.ones_like(x)
    expected_grad_with_grl = -torch.ones_like(x)

    if torch.allclose(x.grad, expected_grad_with_grl):
        print(f"\n✓ GRL IS WORKING! Gradients are reversed (negative)")
    elif torch.allclose(x.grad, expected_grad_without_grl):
        print(f"\n❌ GRL NOT WORKING! Gradients are NOT reversed (positive)")
    else:
        print(f"\n⚠️  Unexpected gradient values")

print("\n" + "=" * 80)
print("TEST 2: GRL with different lambda values")
print("=" * 80)

for lambda_val in [0.1, 1.0, 10.0, 100.0]:
    grl_test = GradientReversalLayer(lambda_=lambda_val)
    x_test = torch.randn(5, 3, requires_grad=True)
    y_test = grl_test(x_test)
    loss_test = y_test.sum()
    loss_test.backward()

    grad_sum = x_test.grad.sum().item()
    expected_sum = -lambda_val * torch.ones_like(x_test).sum().item()

    print(f"\nlambda = {lambda_val:6.1f}:  grad_sum = {grad_sum:8.2f}, expected = {expected_sum:8.2f}")

    if abs(grad_sum - expected_sum) < 1e-3:
        print(f"  ✓ Correct")
    else:
        print(f"  ❌ WRONG!")
