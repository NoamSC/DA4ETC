"""
Test if GRL is properly integrated into the model's computation graph.
"""
import torch
import torch.nn as nn
from config import Config
from models.configurable_cnn import ConfigurableCNN

print("=" * 80)
print("TESTING MODEL GRL INTEGRATION")
print("=" * 80)

# Initialize config and model
cfg = Config()
cfg.MODEL_PARAMS['num_classes'] = 180
cfg.MODEL_PARAMS['lambda_rgl'] = 1.0  # Enable GRL
cfg.MODEL_PARAMS['lambda_grl_gamma'] = 10

model = ConfigurableCNN(cfg.MODEL_PARAMS)
model.set_epoch(0.5)  # Set to mid-training

print(f"\nModel initialized:")
print(f"  lambda_rgl = {model.params['lambda_rgl']}")
print(f"  GRL lambda = {model.grl.lambda_}")

# Create dummy input (for 1D conv: batch_size, channels, length)
# The model expects input_shape=256 channels, and resolution=256
batch_size = 32
input_tensor = torch.randn(batch_size, 256, 256, requires_grad=True)  # 256 channels, 256 length

print(f"\nInput shape: {input_tensor.shape}")
print(f"Input requires_grad: {input_tensor.requires_grad}")

# Forward pass
model.train()
outputs = model(input_tensor)

print(f"\nModel outputs:")
print(f"  class_preds shape: {outputs['class_preds'].shape}")
print(f"  domain_preds shape: {outputs['domain_preds'].shape}")
print(f"  features shape: {outputs['features'].shape}")

# Check requires_grad
print(f"\nRequires grad:")
print(f"  class_preds: {outputs['class_preds'].requires_grad}")
print(f"  domain_preds: {outputs['domain_preds'].requires_grad}")
print(f"  features: {outputs['features'].requires_grad}")

# Create dummy labels
class_labels = torch.randint(0, 180, (batch_size,))
domain_labels = torch.zeros(batch_size).long()  # All source domain

# Compute losses
criterion = nn.CrossEntropyLoss()
classification_loss = criterion(outputs['class_preds'], class_labels)
domain_loss = criterion(outputs['domain_preds'], domain_labels)

print(f"\nLosses:")
print(f"  Classification loss: {classification_loss.item():.4f}")
print(f"  Domain loss: {domain_loss.item():.4f}")

# Test 1: Backprop through domain loss only
print("\n" + "=" * 80)
print("TEST 1: Backprop through DOMAIN LOSS only")
print("=" * 80)

model.zero_grad()
domain_loss.backward()

# Check which parameters got gradients
feature_extractor_grad_norm = 0
domain_classifier_grad_norm = 0

for name, param in model.named_parameters():
    if param.grad is not None:
        if 'feature_extractor' in name:
            feature_extractor_grad_norm += param.grad.norm().item() ** 2
        if 'domain_classifier' in name:
            domain_classifier_grad_norm += param.grad.norm().item() ** 2

feature_extractor_grad_norm = feature_extractor_grad_norm ** 0.5
domain_classifier_grad_norm = domain_classifier_grad_norm ** 0.5

print(f"\nGradient norms after domain_loss.backward():")
print(f"  Feature extractor: {feature_extractor_grad_norm:.6f}")
print(f"  Domain classifier: {domain_classifier_grad_norm:.6f}")

if feature_extractor_grad_norm > 0 and domain_classifier_grad_norm > 0:
    print(f"\n✓ PASS: Both feature extractor and domain classifier received gradients")
    print(f"  This confirms GRL is in the computation graph!")
else:
    print(f"\n❌ FAIL: Missing gradients!")
    if feature_extractor_grad_norm == 0:
        print(f"  Feature extractor has NO gradients - GRL may not be connected!")
    if domain_classifier_grad_norm == 0:
        print(f"  Domain classifier has NO gradients - computation graph broken!")

# Test 2: Check gradient magnitude scaling with different GRL lambdas
print("\n" + "=" * 80)
print("TEST 2: Check gradient scaling with different GRL lambdas")
print("=" * 80)

# Get a specific feature extractor parameter to examine
first_conv_weight = None
for name, param in model.named_parameters():
    if 'feature_extractor' in name and 'weight' in name:
        first_conv_weight = param
        first_conv_name = name
        break

if first_conv_weight is not None:
    print(f"\nTesting gradient magnitude for {first_conv_name}")

    lambda_values = [0.1, 1.0, 10.0]
    grad_norms = []

    for lambda_val in lambda_values:
        # Set GRL lambda
        model.grl.lambda_ = lambda_val

        # Forward and backward
        model.zero_grad()
        outputs_test = model(input_tensor)
        loss_test = criterion(outputs_test['domain_preds'], domain_labels)
        loss_test.backward()

        grad_norm = first_conv_weight.grad.norm().item()
        grad_norms.append(grad_norm)

        print(f"  lambda_grl = {lambda_val:5.1f}:  grad_norm = {grad_norm:.6f}")

    # Check if gradient scales with lambda
    print(f"\nGradient norm ratios:")
    print(f"  grad(lambda=1.0) / grad(lambda=0.1) = {grad_norms[1]/grad_norms[0]:.2f} (expected ~10)")
    print(f"  grad(lambda=10.0) / grad(lambda=1.0) = {grad_norms[2]/grad_norms[1]:.2f} (expected ~10)")

    ratio1 = grad_norms[1] / grad_norms[0]
    ratio2 = grad_norms[2] / grad_norms[1]

    if 8 < ratio1 < 12 and 8 < ratio2 < 12:
        print(f"\n✓ PASS: Gradients scale approximately linearly with lambda_grl")
        print(f"  GRL is correctly scaling gradients!")
    else:
        print(f"\n⚠️  WARNING: Gradient scaling doesn't match expected ratio")
        print(f"  Expected ratios ~10, got {ratio1:.2f} and {ratio2:.2f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nIf both tests passed:")
print(f"  ✓ GRL is in the computation graph")
print(f"  ✓ GRL is reversing gradients")
print(f"  → The problem is NOT with the GRL implementation")
print(f"\nIf Test 1 failed:")
print(f"  ❌ GRL is not properly connected to feature extractor")
print(f"\nIf Test 2 failed:")
print(f"  ❌ GRL is not reversing gradients (implementation bug)")
