"""
Test if different GRL lambdas produce different parameter updates after one optimizer step.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from config import Config
from models.configurable_cnn import ConfigurableCNN
from data_utils.csv_dataloader import create_csv_flowpic_loader

print("=" * 120)
print("TESTING IF DIFFERENT LAMBDA_RGL VALUES PRODUCE DIFFERENT PARAMETER UPDATES")
print("=" * 120)

# Initialize
cfg = Config()
device = cfg.DEVICE
dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v1')

# Load label mapping
with open(dataset_root / 'label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

num_classes = len(label_mapping)

# Create loaders
print("\nLoading data...")
train_loader = create_csv_flowpic_loader(
    [dataset_root / 'WEEK-2022-33' / 'train.parquet'],
    batch_size=64,
    num_workers=0,
    shuffle=True,
    resolution=cfg.RESOLUTION,
    data_sample_frac=0.001,
    seed=42,  # SAME SEED
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
    seed=42,  # SAME SEED
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

criterion = nn.CrossEntropyLoss()
lambda_dann = 1.0

# Test three different lambda_rgl values
lambda_rgl_values = [1e-8, 1.0, 10000.0]

print(f"\nRunning {len(lambda_rgl_values)} experiments with different lambda_rgl values...")
print(f"All using SAME seed (42), SAME data, SAME architecture\n")

results = {}

for lambda_rgl in lambda_rgl_values:
    print(f"\n{'='*120}")
    print(f"EXPERIMENT: lambda_rgl = {lambda_rgl}")
    print(f"{'='*120}")

    # Create model with this lambda_rgl
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    cfg.MODEL_PARAMS['lambda_rgl'] = lambda_rgl
    cfg.MODEL_PARAMS['lambda_grl_gamma'] = 10

    # IMPORTANT: Use SAME seed for model initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(device)
    model.set_epoch(0.02)  # Early in training
    model.train()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # Get initial parameters
    first_param = None
    first_param_name = None
    for name, param in model.named_parameters():
        if 'feature_extractor.0.weight' in name:
            first_param = param.data.clone()
            first_param_name = name
            break

    print(f"\nInitial {first_param_name}[0,0,:5] = {first_param[0,0,:5]}")

    # Forward pass
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    # Compute losses
    classification_loss = criterion(source_outputs['class_preds'], source_labels)

    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    dann_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    dann_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

    total_loss = classification_loss + lambda_dann * dann_loss

    print(f"\nLosses:")
    print(f"  Classification: {classification_loss.item():.4f}")
    print(f"  DANN:          {dann_loss.item():.4f}")
    print(f"  Total:         {total_loss.item():.4f}")
    print(f"  GRL Lambda:    {model.grl.lambda_:.6f}")

    # Backward and optimizer step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Get updated parameters
    updated_param = None
    for name, param in model.named_parameters():
        if name == first_param_name:
            updated_param = param.data.clone()
            break

    # Compute parameter change
    param_change = (updated_param - first_param).abs().sum().item()

    print(f"\nAfter optimization step:")
    print(f"  Updated {first_param_name}[0,0,:5] = {updated_param[0,0,:5]}")
    print(f"  Parameter change (L1 norm): {param_change:.6f}")

    results[lambda_rgl] = {
        'initial_param': first_param.cpu(),
        'updated_param': updated_param.cpu(),
        'param_change': param_change,
        'total_loss': total_loss.item()
    }

# Compare results
print(f"\n\n{'='*120}")
print("COMPARISON")
print(f"{'='*120}")

print(f"\nParameter changes:")
for lambda_rgl in lambda_rgl_values:
    print(f"  lambda_rgl = {lambda_rgl:10.1e}:  L1 change = {results[lambda_rgl]['param_change']:.6f}")

# Check if parameters are identical
param1 = results[lambda_rgl_values[0]]['updated_param']
param2 = results[lambda_rgl_values[1]]['updated_param']
param3 = results[lambda_rgl_values[2]]['updated_param']

diff_1_2 = (param1 - param2).abs().sum().item()
diff_1_3 = (param1 - param3).abs().sum().item()
diff_2_3 = (param2 - param3).abs().sum().item()

print(f"\nParameter differences after one step:")
print(f"  |params({lambda_rgl_values[0]:.1e}) - params({lambda_rgl_values[1]:.1e})| = {diff_1_2:.8f}")
print(f"  |params({lambda_rgl_values[0]:.1e}) - params({lambda_rgl_values[2]:.1e})| = {diff_1_3:.8f}")
print(f"  |params({lambda_rgl_values[1]:.1e}) - params({lambda_rgl_values[2]:.1e})| = {diff_2_3:.8f}")

print(f"\n{'='*120}")
print("CONCLUSION")
print(f"{'='*120}\n")

if diff_1_3 < 1e-6:
    print("❌ CRITICAL BUG FOUND!")
    print(f"   Parameters are IDENTICAL (diff = {diff_1_3:.2e}) despite vastly different lambda_rgl!")
    print(f"   This explains why training results are bit-exact identical!\n")
    print(f"   Possible causes:")
    print(f"   1. Bug in model initialization (params being shared somehow)")
    print(f"   2. Bug in optimizer (not using gradients)")
    print(f"   3. Bug in backward pass (gradients not flowing)")
elif diff_1_3 < 0.01:
    print("⚠️  WARNING:")
    print(f"   Parameters are VERY similar (diff = {diff_1_3:.6f})")
    print(f"   This might explain why training results converge to same values")
    print(f"   Adam optimizer may be normalizing gradient differences")
else:
    print("✓ Parameters are DIFFERENT as expected")
    print(f"   Difference: {diff_1_3:.6f}")
    print(f"   The problem is NOT with parameter updates after one step")
    print(f"\n   If training still produces identical results, the issue must be:")
    print(f"   1. Checkpoints being loaded incorrectly")
    print(f"   2. Experiments using different configs than saved")
    print(f"   3. Data loading issues")
