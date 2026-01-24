"""
Compare gradient magnitudes from classification loss vs DANN loss.
This will tell us if DANN loss is negligible compared to classification loss.
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from config import Config
from models.configurable_cnn import ConfigurableCNN
from data_utils.csv_dataloader import create_csv_flowpic_loader

print("=" * 80)
print("COMPARING GRADIENT MAGNITUDES: Classification vs DANN")
print("=" * 80)

# Initialize
cfg = Config()
device = cfg.DEVICE
dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v1')

# Load label mapping
with open(dataset_root / 'label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)

# Create loaders with small data fraction for speed
print("\nLoading data...")
train_loader = create_csv_flowpic_loader(
    [dataset_root / 'WEEK-2022-33' / 'train.parquet'],
    batch_size=64,
    num_workers=0,
    shuffle=True,
    resolution=cfg.RESOLUTION,
    data_sample_frac=0.001,
    seed=cfg.SEED,
    label_mapping=label_mapping,
    log_t_axis=False,
    max_dt_ms=4000,
    dataset_format='cesnet_parquet',
    verbose=False
)

test_loader = create_csv_flowpic_loader(
    [dataset_root / 'WEEK-2022-40' / 'test.parquet'],
    batch_size=64,
    num_workers=0,
    shuffle=False,
    resolution=cfg.RESOLUTION,
    data_sample_frac=0.001,
    seed=cfg.SEED,
    label_mapping=label_mapping,
    log_t_axis=False,
    max_dt_ms=4000,
    dataset_format='cesnet_parquet',
    verbose=False
)

# Get batches
source_batch = next(iter(train_loader))
target_batch = next(iter(test_loader))

source_inputs, source_labels = source_batch
target_inputs, _ = target_batch

source_inputs = source_inputs.to(device)
target_inputs = target_inputs.to(device)
source_labels = source_labels.to(device).long()

# Test with different lambda_rgl values
lambda_rgl_values = [1e-8, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

print(f"\nTesting {len(lambda_rgl_values)} different lambda_rgl values...")
print(f"Source batch size: {len(source_inputs)}")
print(f"Target batch size: {len(target_inputs)}")

criterion = nn.CrossEntropyLoss()

results = []

for lambda_rgl in lambda_rgl_values:
    # Create model with this lambda_rgl
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    cfg.MODEL_PARAMS['lambda_rgl'] = lambda_rgl
    cfg.MODEL_PARAMS['lambda_grl_gamma'] = 10

    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(device)
    model.set_epoch(0.02)  # Early in training
    model.train()

    # Forward pass
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    # Compute losses
    classification_loss = criterion(source_outputs['class_preds'], source_labels)

    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    dann_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    dann_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

    lambda_dann = 1.0  # Use the value from the actual experiments

    # Test 1: Gradient from classification loss only
    model.zero_grad()
    classification_loss.backward(retain_graph=True)

    grad_from_class = 0
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grad_from_class += param.grad.norm().item() ** 2
    grad_from_class = grad_from_class ** 0.5

    # Test 2: Gradient from DANN loss only
    model.zero_grad()
    (lambda_dann * dann_loss).backward(retain_graph=True)

    grad_from_dann = 0
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grad_from_dann += param.grad.norm().item() ** 2
    grad_from_dann = grad_from_dann ** 0.5

    # Test 3: Gradient from combined loss (as in actual training)
    model.zero_grad()
    total_loss = classification_loss + lambda_dann * dann_loss
    total_loss.backward()

    grad_from_total = 0
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grad_from_total += param.grad.norm().item() ** 2
    grad_from_total = grad_from_total ** 0.5

    results.append({
        'lambda_rgl': lambda_rgl,
        'grl_lambda': model.grl.lambda_,
        'class_loss': classification_loss.item(),
        'dann_loss': dann_loss.item(),
        'grad_class': grad_from_class,
        'grad_dann': grad_from_dann,
        'grad_total': grad_from_total,
        'ratio': grad_from_dann / grad_from_class if grad_from_class > 0 else 0
    })

# Print results
print("\n" + "=" * 120)
print(f"{'lambda_rgl':>12s}  {'GRL λ':>8s}  {'Class Loss':>10s}  {'DANN Loss':>10s}  "
      f"{'Grad(Class)':>12s}  {'Grad(DANN)':>12s}  {'Grad(Total)':>12s}  {'Ratio':>8s}")
print("=" * 120)

for r in results:
    print(f"{r['lambda_rgl']:12.1e}  {r['grl_lambda']:8.4f}  {r['class_loss']:10.4f}  {r['dann_loss']:10.4f}  "
          f"{r['grad_class']:12.4f}  {r['grad_dann']:12.4f}  {r['grad_total']:12.4f}  {r['ratio']:8.4f}")

print("=" * 120)

# Analysis
print("\nANALYSIS:")
print("-" * 120)

print(f"\n1. Loss values:")
print(f"   Classification loss: {results[0]['class_loss']:.4f}")
print(f"   DANN loss:          {results[0]['dann_loss']:.4f}")
print(f"   lambda_dann:        1.0 (from v13 experiments)")
print(f"   Weighted DANN:      {1.0 * results[0]['dann_loss']:.4f}")

print(f"\n2. Gradient magnitude comparison:")
first_ratio = results[0]['ratio']
last_ratio = results[-1]['ratio']

print(f"   With lambda_rgl = {results[0]['lambda_rgl']:.1e}:")
print(f"     Grad(DANN) / Grad(Class) = {first_ratio:.4f} ({first_ratio*100:.2f}%)")

print(f"   With lambda_rgl = {results[-1]['lambda_rgl']:.1e}:")
print(f"     Grad(DANN) / Grad(Class) = {last_ratio:.4f} ({last_ratio*100:.2f}%)")

print(f"\n3. Gradient scaling with lambda_rgl:")
for i in range(1, len(results)):
    expected_ratio = results[i]['lambda_rgl'] / results[i-1]['lambda_rgl']
    actual_ratio = results[i]['grad_dann'] / results[i-1]['grad_dann']
    print(f"   {results[i-1]['lambda_rgl']:.1e} → {results[i]['lambda_rgl']:.1e}: "
          f"Grad increased by {actual_ratio:.2f}x (expected {expected_ratio:.2f}x)")

print(f"\n4. Impact on total gradient:")
change_in_total = (results[-1]['grad_total'] - results[0]['grad_total']) / results[0]['grad_total'] * 100
print(f"   Total gradient changed by {change_in_total:.2f}% when lambda_rgl went from {results[0]['lambda_rgl']:.1e} to {results[-1]['lambda_rgl']:.1e}")

print("\n" + "=" * 120)
print("CONCLUSION:")
print("=" * 120)

if abs(change_in_total) < 1:
    print("\n❌ CRITICAL FINDING:")
    print(f"   Total gradient changed by only {abs(change_in_total):.2f}% despite 12 orders of magnitude change in lambda_rgl!")
    print(f"   This explains why all experiments produce identical results.")
    print(f"\n   ROOT CAUSE: DANN gradients are NEGLIGIBLE compared to classification gradients")
    print(f"   Even with lambda_rgl=10000, DANN contributes only {last_ratio*100:.2f}% of the gradient magnitude")
elif abs(change_in_total) < 10:
    print("\n⚠️  WARNING:")
    print(f"   Total gradient changed by only {abs(change_in_total):.2f}%")
    print(f"   DANN gradients may be too weak to significantly affect training")
else:
    print("\n✓ DANN gradients are significant")
    print(f"   Total gradient changed by {abs(change_in_total):.2f}%")
    print(f"   Different lambda_rgl values SHOULD produce different results")
