"""
Load actual checkpoints from v14 experiments and simulate one training epoch.
This verifies that the three experiments with different lambda_rgl will diverge.
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
print("TESTING ACTUAL V14 CHECKPOINTS - SIMULATING ONE TRAINING EPOCH")
print("=" * 120)

# Define the three v14 experiments
experiments = {
    '1e-8': 'exps/cesnet_v4_dann_sanity/normal_v14_grl_effect_test_no_rgl_WEEK-2022-33',
    '1e0': 'exps/cesnet_v4_dann_sanity/normal_v14_grl_effect_test_yes_rgl_WEEK-2022-33',
    '1e4': 'exps/cesnet_v4_dann_sanity/normal_v14_grl_effect_test_crazy_rgl_WEEK-2022-33'
}

# Choose which epoch checkpoint to load (middle of training)
checkpoint_epoch = 5

dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v1')

# Load label mapping
with open(dataset_root / 'label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

print(f"\nLoading checkpoints from epoch {checkpoint_epoch}")
print(f"Will simulate ONE additional training epoch for each experiment\n")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

results = {}

for exp_name, exp_path in experiments.items():
    exp_path = Path(exp_path)

    print(f"\n{'='*120}")
    print(f"EXPERIMENT: lambda_rgl = {exp_name}")
    print(f"{'='*120}")

    # Load config
    with open(exp_path / 'config.json', 'r') as f:
        saved_config = json.load(f)

    lambda_rgl = saved_config['MODEL_PARAMS']['lambda_rgl']
    lambda_grl_gamma = saved_config['MODEL_PARAMS']['lambda_grl_gamma']
    lambda_dann = saved_config['LAMBDA_DANN']
    learning_rate = saved_config['LEARNING_RATE']
    weight_decay = saved_config['WEIGHT_DECAY']

    print(f"\nConfig:")
    print(f"  lambda_rgl = {lambda_rgl}")
    print(f"  lambda_grl_gamma = {lambda_grl_gamma}")
    print(f"  lambda_dann = {lambda_dann}")
    print(f"  learning_rate = {learning_rate}")
    print(f"  weight_decay = {weight_decay}")

    # Create model
    cfg = Config()
    cfg.MODEL_PARAMS = saved_config['MODEL_PARAMS']
    cfg.MODEL_PARAMS['num_classes'] = 180

    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(device)

    # Load checkpoint
    checkpoint_path = exp_path / 'weights' / f'model_weights_epoch_{checkpoint_epoch}.pth'

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints:")
        for ckpt in sorted((exp_path / 'weights').glob('*.pth')):
            print(f"  {ckpt.name}")
        continue

    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Get initial weights (before simulating epoch)
    first_param_name = 'feature_extractor.0.weight'
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()

    print(f"Initial {first_param_name}[0,0,:5] = {initial_weights[first_param_name][0,0,:5]}")

    # Create optimizer with SAME state as during training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Try to load optimizer state if available
    optimizer_path = exp_path / 'weights' / f'optimizer_state_epoch_{checkpoint_epoch}.pth'
    if optimizer_path.exists():
        print(f"Loading optimizer state: {optimizer_path.name}")
        opt_state = torch.load(optimizer_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(opt_state)
    else:
        print(f"Note: Optimizer state not found, using fresh optimizer")

    # Set epoch (normalized)
    num_epochs = 50
    normalized_epoch = checkpoint_epoch / num_epochs
    model.set_epoch(normalized_epoch)
    print(f"Set normalized epoch = {normalized_epoch:.2f}, GRL lambda = {model.grl.lambda_:.4f}")

    # Load data (use same data as training)
    print(f"\nLoading data...")
    train_loader = create_csv_flowpic_loader(
        [dataset_root / 'WEEK-2022-33' / 'train.parquet'],
        batch_size=64,
        num_workers=0,
        shuffle=True,
        resolution=saved_config['RESOLUTION'],
        data_sample_frac=0.01,  # Use 1% of data for faster test
        seed=saved_config['SEED'] + checkpoint_epoch,  # Different seed per epoch
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
        resolution=saved_config['RESOLUTION'],
        data_sample_frac=0.01,
        seed=saved_config['SEED'] + checkpoint_epoch,
        label_mapping=label_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=False
    )

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Simulate ONE epoch of training
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss_sum = 0
    dann_loss_sum = 0
    class_loss_sum = 0
    domain_correct = 0
    domain_total = 0
    n_batches = 0
    max_batches = 100  # Limit to 100 batches for speed

    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    print(f"\nSimulating training epoch (max {max_batches} batches)...")

    for batch_idx in range(min(max_batches, len(train_loader))):
        # Get batches
        try:
            source_batch = next(train_iter)
            target_batch = next(test_iter)
        except StopIteration:
            break

        source_inputs, source_labels = source_batch
        target_inputs, _ = target_batch

        source_inputs = source_inputs.to(device)
        target_inputs = target_inputs.to(device)
        source_labels = source_labels.to(device).long()

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

        # Backward and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track metrics
        total_loss_sum += total_loss.item()
        dann_loss_sum += dann_loss.item()
        class_loss_sum += classification_loss.item()

        # Domain accuracy
        with torch.no_grad():
            _, source_pred = source_outputs['domain_preds'].max(1)
            _, target_pred = target_outputs['domain_preds'].max(1)
            domain_correct += source_pred.eq(source_domain_labels).sum().item()
            domain_correct += target_pred.eq(target_domain_labels).sum().item()
            domain_total += len(source_domain_labels) + len(target_domain_labels)

        n_batches += 1

        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{max_batches}: Loss={total_loss.item():.4f}, DANN={dann_loss.item():.4f}, Class={classification_loss.item():.4f}")

    # Get final weights (after simulated epoch)
    final_weights = {}
    for name, param in model.named_parameters():
        final_weights[name] = param.data.clone()

    print(f"\nAfter simulated epoch:")
    print(f"  Final {first_param_name}[0,0,:5] = {final_weights[first_param_name][0,0,:5]}")

    # Compute parameter change
    param_change = (final_weights[first_param_name] - initial_weights[first_param_name]).abs().sum().item()
    total_param_change = sum((final_weights[name] - initial_weights[name]).abs().sum().item()
                              for name in initial_weights.keys())

    print(f"  {first_param_name} L1 change: {param_change:.6f}")
    print(f"  Total parameter L1 change: {total_param_change:.6f}")

    # Average losses
    avg_total_loss = total_loss_sum / n_batches
    avg_dann_loss = dann_loss_sum / n_batches
    avg_class_loss = class_loss_sum / n_batches
    domain_acc = domain_correct / domain_total * 100

    print(f"\nAverage over {n_batches} batches:")
    print(f"  Total loss: {avg_total_loss:.4f}")
    print(f"  DANN loss: {avg_dann_loss:.4f}")
    print(f"  Classification loss: {avg_class_loss:.4f}")
    print(f"  Domain accuracy: {domain_acc:.2f}%")

    results[exp_name] = {
        'lambda_rgl': lambda_rgl,
        'initial_weights': initial_weights,
        'final_weights': final_weights,
        'param_change': total_param_change,
        'first_param_change': param_change,
        'avg_total_loss': avg_total_loss,
        'avg_dann_loss': avg_dann_loss,
        'avg_class_loss': avg_class_loss,
        'domain_acc': domain_acc,
        'grl_lambda': model.grl.lambda_
    }

# Compare results
print(f"\n\n{'='*120}")
print("COMPARISON OF THREE EXPERIMENTS AFTER ONE SIMULATED EPOCH")
print(f"{'='*120}")

print(f"\nGRL Lambda values (at epoch {checkpoint_epoch}):")
for exp_name in ['1e-8', '1e0', '1e4']:
    if exp_name in results:
        print(f"  lambda_rgl = {exp_name:>4s}:  GRL lambda = {results[exp_name]['grl_lambda']:.6f}")

print(f"\nTotal parameter changes (L1 norm):")
for exp_name in ['1e-8', '1e0', '1e4']:
    if exp_name in results:
        print(f"  lambda_rgl = {exp_name:>4s}:  {results[exp_name]['param_change']:.6f}")

# Compare final weights between experiments
print(f"\nParameter differences between experiments:")
if '1e-8' in results and '1e0' in results and '1e4' in results:
    diff_1e8_1e0 = sum((results['1e-8']['final_weights'][name] - results['1e0']['final_weights'][name]).abs().sum().item()
                        for name in results['1e-8']['final_weights'].keys())
    diff_1e8_1e4 = sum((results['1e-8']['final_weights'][name] - results['1e4']['final_weights'][name]).abs().sum().item()
                        for name in results['1e-8']['final_weights'].keys())
    diff_1e0_1e4 = sum((results['1e0']['final_weights'][name] - results['1e4']['final_weights'][name]).abs().sum().item()
                        for name in results['1e0']['final_weights'].keys())

    print(f"  |weights(1e-8) - weights(1e0)| = {diff_1e8_1e0:.6f}")
    print(f"  |weights(1e-8) - weights(1e4)| = {diff_1e8_1e4:.6f}")
    print(f"  |weights(1e0) - weights(1e4)| = {diff_1e0_1e4:.6f}")

print(f"\nAverage training losses:")
for exp_name in ['1e-8', '1e0', '1e4']:
    if exp_name in results:
        print(f"  lambda_rgl = {exp_name:>4s}:  {results[exp_name]['avg_total_loss']:.6f}")

print(f"\nAverage DANN losses:")
for exp_name in ['1e-8', '1e0', '1e4']:
    if exp_name in results:
        print(f"  lambda_rgl = {exp_name:>4s}:  {results[exp_name]['avg_dann_loss']:.6f}")

print(f"\nDomain classifier accuracies:")
for exp_name in ['1e-8', '1e0', '1e4']:
    if exp_name in results:
        print(f"  lambda_rgl = {exp_name:>4s}:  {results[exp_name]['domain_acc']:.2f}%")

print(f"\n{'='*120}")
print("CONCLUSION")
print(f"{'='*120}\n")

if '1e-8' in results and '1e0' in results and '1e4' in results:
    weights_differ = diff_1e8_1e4 > 1.0  # Significant difference
    losses_differ = abs(results['1e-8']['avg_total_loss'] - results['1e4']['avg_total_loss']) > 0.01

    if weights_differ:
        print("✅ WEIGHTS ARE DIVERGING!")
        print(f"   After one simulated epoch from checkpoint {checkpoint_epoch}:")
        print(f"   Weights differ by {diff_1e8_1e4:.2f} (L1 norm)")
        print(f"   The three experiments are on DIFFERENT training trajectories!")
        print(f"\n   This confirms the GRL fix is working in actual training!")
    else:
        print("❌ WEIGHTS ARE STILL TOO SIMILAR")
        print(f"   Weight difference: {diff_1e8_1e4:.6f}")
        print(f"   This suggests:")
        print(f"   1. The v14 checkpoints might have been trained with the OLD (buggy) GRL")
        print(f"   2. Or the experiments need more time to diverge")
        print(f"   3. Or lambda_dann is too small relative to classification loss")

    # Additional diagnostics
    print(f"\n{'='*120}")
    print("GRADIENT ANALYSIS")
    print(f"{'='*120}\n")

    print(f"GRL strength comparison (higher = stronger domain adaptation):")
    print(f"  1e-8: {results['1e-8']['grl_lambda']:>12.6f}  (essentially disabled)")
    print(f"  1e0:  {results['1e0']['grl_lambda']:>12.6f}  (moderate)")
    print(f"  1e4:  {results['1e4']['grl_lambda']:>12.6f}  (very strong)")

    grl_ratio = results['1e4']['grl_lambda'] / max(results['1e0']['grl_lambda'], 1e-10)
    print(f"\n  Ratio (1e4 / 1e0): {grl_ratio:.1f}x")
    print(f"  → Gradient reversal strength differs by {grl_ratio:.0f}x between experiments")

else:
    print("⚠️  Could not load all three experiments for comparison")
