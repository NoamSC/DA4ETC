"""
Test suite to verify DANN gradient flow and detect common silent bugs.

This script performs the following tests:
1. Verifies gradient flow from domain loss to feature extractor
2. Tests domain classifier capability with frozen features (λ=0)
3. Monitors domain logits statistics (mean, std)
4. Tracks gradient norms on feature extractor from domain loss
5. Validates GRL connection and sign
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

from models.configurable_cnn import ConfigurableCNN
from config import Config


def test_gradient_flow(model, source_batch, target_batch, device, criterion):
    """
    Test 1: Verify gradients flow from domain loss to feature extractor.

    Returns:
        dict: Contains gradient norms and statistics
    """
    print("\n" + "="*80)
    print("TEST 1: Gradient Flow Verification")
    print("="*80)

    model.train()
    source_inputs, source_labels = source_batch
    target_inputs, _ = target_batch

    source_inputs = source_inputs.to(device)
    target_inputs = target_inputs.to(device)
    source_labels = source_labels.to(device).long()

    # Zero gradients
    model.zero_grad()

    # Forward pass
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    # Create domain labels
    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    # Compute ONLY domain loss (no classification loss)
    domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

    # Backward pass
    domain_loss.backward()

    # Collect gradient statistics from feature extractor
    gradient_stats = {}
    total_norm = 0

    print("\nGradient norms in feature extractor:")
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            gradient_stats[name] = grad_norm
            print(f"  {name:50s}: {grad_norm:12.6f}")

    total_norm = np.sqrt(total_norm)
    gradient_stats['total_feature_extractor_norm'] = total_norm

    print(f"\n  {'TOTAL FEATURE EXTRACTOR GRADIENT NORM':50s}: {total_norm:12.6f}")

    # Check if gradients are flowing
    if total_norm < 1e-6:
        print("\n  ⚠️  WARNING: Feature extractor gradients are nearly ZERO!")
        print("  This indicates the domain loss is NOT connected to the feature extractor.")
        print("  Possible causes:")
        print("    - GRL not applied correctly")
        print("    - Domain classifier not using feature extractor outputs")
        print("    - Computation graph broken")
    elif total_norm > 0.01:
        print("\n  ✓ PASS: Gradients are flowing to feature extractor")
    else:
        print("\n  ⚠️  WARNING: Gradients are very small (might be an issue)")

    return gradient_stats


def test_domain_logits_statistics(model, source_batch, target_batch, device):
    """
    Test 2: Monitor domain logits statistics.

    Returns:
        dict: Logits statistics (mean, std, etc.)
    """
    print("\n" + "="*80)
    print("TEST 2: Domain Logits Statistics")
    print("="*80)

    model.eval()
    source_inputs, _ = source_batch
    target_inputs, _ = target_batch

    source_inputs = source_inputs.to(device)
    target_inputs = target_inputs.to(device)

    with torch.no_grad():
        source_outputs = model(source_inputs)
        target_outputs = model(target_inputs)

        source_logits = source_outputs['domain_preds']
        target_logits = target_outputs['domain_preds']

        all_logits = torch.cat([source_logits, target_logits], dim=0)

        stats = {
            'source_mean': source_logits.mean().item(),
            'source_std': source_logits.std().item(),
            'target_mean': target_logits.mean().item(),
            'target_std': target_logits.std().item(),
            'all_mean': all_logits.mean().item(),
            'all_std': all_logits.std().item(),
        }

        print("\nSource domain logits:")
        print(f"  Mean: {stats['source_mean']:10.4f}")
        print(f"  Std:  {stats['source_std']:10.4f}")

        print("\nTarget domain logits:")
        print(f"  Mean: {stats['target_mean']:10.4f}")
        print(f"  Std:  {stats['target_std']:10.4f}")

        print("\nAll domain logits:")
        print(f"  Mean: {stats['all_mean']:10.4f}")
        print(f"  Std:  {stats['all_std']:10.4f}")

        # Check for degenerate behavior
        if stats['all_std'] < 0.1:
            print("\n  ⚠️  WARNING: Logits have very low standard deviation!")
            print("  This suggests the domain classifier is not learning.")
            print("  Possible causes:")
            print("    - Disconnected computation graph")
            print("    - Learning rate too low")
            print("    - Initialization issue")
        else:
            print("\n  ✓ PASS: Logits show reasonable variance")

    return stats


def test_frozen_features_domain_classifier(model, source_loader, target_loader, device,
                                           num_batches=100, lr=1e-3):
    """
    Test 3: Train domain classifier with FROZEN features (λ=0 equivalent).

    If domain classifier can't exceed 50-60% accuracy with frozen features,
    there's likely a data/labels/architecture issue.

    If it jumps to high accuracy, the domain signal exists and the issue is
    with GRL scheduling / loss weights / optimizer dynamics.

    Returns:
        dict: Training history and final accuracy
    """
    print("\n" + "="*80)
    print("TEST 3: Domain Classifier with Frozen Features")
    print("="*80)
    print("\nTraining domain classifier with frozen feature extractor...")
    print(f"Using {num_batches} batches, learning rate = {lr}")
    print("NOTE: For untrained models, features are random - domain classifier may not learn well.")

    # Freeze feature extractor
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # Temporarily disable GRL by setting lambda to 0
    original_lambda = model.grl.lambda_
    model.grl.lambda_ = 0.0

    # Setup optimizer for domain classifier only
    domain_params = []
    for name, param in model.named_parameters():
        if 'domain_classifier' in name or 'domain_output' in name:
            domain_params.append(param)

    optimizer = torch.optim.Adam(domain_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    accuracies = []
    losses = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for batch_idx in tqdm(range(num_batches), desc="Training domain classifier"):
        try:
            source_inputs, _ = next(source_iter)
            target_inputs, _ = next(target_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            source_inputs, _ = next(source_iter)
            target_inputs, _ = next(target_iter)

        source_inputs = source_inputs.to(device)
        target_inputs = target_inputs.to(device)

        # Forward pass
        optimizer.zero_grad()
        source_outputs = model(source_inputs)
        target_outputs = model(target_inputs)

        # Domain labels
        source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
        target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

        # Loss
        loss = criterion(source_outputs['domain_preds'], source_domain_labels)
        loss += criterion(target_outputs['domain_preds'], target_domain_labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Track accuracy
        with torch.no_grad():
            _, source_pred = source_outputs['domain_preds'].max(1)
            _, target_pred = target_outputs['domain_preds'].max(1)
            correct = source_pred.eq(source_domain_labels).sum().item()
            correct += target_pred.eq(target_domain_labels).sum().item()
            total = len(source_domain_labels) + len(target_domain_labels)
            acc = 100.0 * correct / total

            accuracies.append(acc)
            losses.append(loss.item())

    # Restore original state
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
    model.grl.lambda_ = original_lambda

    final_acc = np.mean(accuracies[-10:])  # Average of last 10 batches

    print(f"\nFinal domain classifier accuracy (frozen features): {final_acc:.2f}%")

    if final_acc < 55:
        print("\n  ⚠️  FAIL: Domain classifier cannot distinguish domains!")
        print("  Accuracy is near random (50%). Possible causes:")
        print("    - Data loading issue (source/target labels swapped?)")
        print("    - Insufficient domain shift in the data")
        print("    - Architecture too weak for domain classification")
        print("    - Domain labels incorrect")
    elif final_acc < 70:
        print("\n  ⚠️  WARNING: Domain classifier has weak performance")
        print("  This suggests limited domain shift or architecture issues.")
    else:
        print("\n  ✓ PASS: Domain classifier can distinguish domains!")
        print("  If DANN is not working, the issue is likely:")
        print("    - GRL scheduling (lambda too small/large)")
        print("    - Loss weight balance (lambda_dann)")
        print("    - Optimizer dynamics (learning rate, momentum)")

    return {
        'accuracies': accuracies,
        'losses': losses,
        'final_accuracy': final_acc
    }


def test_grl_sign(model, source_batch, target_batch, device, criterion):
    """
    Test 4: Verify GRL has correct sign (reverses gradients).

    Returns:
        dict: Gradient comparison results
    """
    print("\n" + "="*80)
    print("TEST 4: GRL Sign Verification")
    print("="*80)

    source_inputs, source_labels = source_batch
    target_inputs, _ = target_batch

    source_inputs = source_inputs.to(device)
    target_inputs = target_inputs.to(device)
    source_labels = source_labels.to(device).long()

    # IMPORTANT: Temporarily set GRL lambda to 1.0 for this test
    # to ensure gradients are large enough to measure
    original_lambda = model.grl.lambda_
    model.grl.lambda_ = 1.0
    print(f"\nTemporarily setting GRL lambda to 1.0 (was {original_lambda:.6f})")

    # Test 1: Gradients WITH GRL (lambda = 1.0)
    model.train()
    model.zero_grad()

    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)
    domain_loss.backward()

    # Store gradients WITH GRL
    grads_with_grl = {}
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grads_with_grl[name] = param.grad.clone()

    # Test 2: Gradients WITHOUT GRL (set lambda = 0)
    model.grl.lambda_ = 0.0

    model.zero_grad()
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)
    domain_loss.backward()

    # Store gradients WITHOUT GRL
    grads_without_grl = {}
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grads_without_grl[name] = param.grad.clone()

    # Restore original lambda
    model.grl.lambda_ = original_lambda

    # Compare signs
    print("\nChecking gradient sign reversal (lambda=1.0 vs lambda=0.0):")
    all_correct_sign = True
    num_checked = 0

    for name in grads_with_grl.keys():
        if name in grads_without_grl:
            # Compute dot product (should be negative if signs are reversed)
            dot_product = (grads_with_grl[name] * grads_without_grl[name]).sum().item()
            norm_with = grads_with_grl[name].norm().item()
            norm_without = grads_without_grl[name].norm().item()

            # Skip if gradients are too small to measure
            if norm_with < 1e-8 or norm_without < 1e-8:
                print(f"  - {name:50s}: SKIP (gradients too small)")
                continue

            # Cosine similarity
            cos_sim = dot_product / (norm_with * norm_without + 1e-8)

            sign_correct = cos_sim < -0.5  # Should be close to -1
            all_correct_sign = all_correct_sign and sign_correct
            num_checked += 1

            status = "✓" if sign_correct else "✗"
            print(f"  {status} {name:50s}: cosine_sim = {cos_sim:8.4f} (norm_with={norm_with:.6f}, norm_without={norm_without:.6f})")

    if num_checked == 0:
        print("\n  ⚠️  WARNING: No gradients large enough to check!")
        print("  This suggests domain classifier is not connected to feature extractor.")
        all_correct_sign = False
    elif all_correct_sign:
        print("\n  ✓ PASS: GRL correctly reverses gradients")
    else:
        print("\n  ⚠️  FAIL: GRL is NOT reversing gradients correctly!")
        print("  Check GRL implementation in models/configurable_cnn.py")

    return {
        'all_correct_sign': all_correct_sign,
        'num_checked': num_checked
    }


def visualize_results(frozen_results, output_dir):
    """
    Visualize test results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    axes[0].plot(frozen_results['accuracies'], label='Domain Classifier Accuracy')
    axes[0].axhline(y=50, color='r', linestyle='--', label='Random Baseline (50%)')
    axes[0].set_xlabel('Batch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Domain Classifier Training (Frozen Features)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot loss
    axes[1].plot(frozen_results['losses'], label='Domain Loss', color='orange')
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Domain Loss (Frozen Features)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = output_dir / 'dann_gradient_flow_tests.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def run_all_tests(model, source_loader, target_loader, device, output_dir='test_results', num_frozen_batches=100):
    """
    Run all DANN gradient flow tests.
    """
    print("\n" + "="*80)
    print("DANN GRADIENT FLOW TEST SUITE")
    print("="*80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"GRL Lambda: {model.grl.lambda_}")

    criterion = nn.CrossEntropyLoss()

    # Get sample batches
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    source_batch = next(source_iter)
    target_batch = next(target_iter)

    results = {}

    # Test 1: Gradient flow
    results['gradient_flow'] = test_gradient_flow(
        model, source_batch, target_batch, device, criterion
    )

    # Test 2: Domain logits statistics
    results['logits_stats'] = test_domain_logits_statistics(
        model, source_batch, target_batch, device
    )

    # Test 3: Frozen features domain classifier
    results['frozen_features'] = test_frozen_features_domain_classifier(
        model, source_loader, target_loader, device, num_batches=num_frozen_batches, lr=1e-3
    )

    # Test 4: GRL sign verification
    results['grl_sign'] = test_grl_sign(
        model, source_batch, target_batch, device, criterion
    )

    # Visualize results
    visualize_results(results['frozen_features'], output_dir)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total_norm = results['gradient_flow']['total_feature_extractor_norm']
    frozen_acc = results['frozen_features']['final_accuracy']
    grl_sign_ok = results['grl_sign']['all_correct_sign']
    grl_num_checked = results['grl_sign'].get('num_checked', 0)
    logits_std = results['logits_stats']['all_std']

    print(f"\n1. Gradient Flow:")
    print(f"   - Feature extractor gradient norm: {total_norm:.6f}")
    if total_norm > 0.01:
        print(f"   - Status: ✓ PASS")
    elif total_norm > 0.0001:
        print(f"   - Status: ⚠  WARNING (very small - check GRL lambda: {model.grl.lambda_:.6f})")
    else:
        print(f"   - Status: ✗ FAIL (disconnected or broken)")

    print(f"\n2. Domain Logits Statistics:")
    print(f"   - Logits std: {logits_std:.4f}")
    if logits_std > 0.1:
        print(f"   - Status: ✓ PASS")
    else:
        print(f"   - Status: ✗ FAIL (classifier not learning)")

    print(f"\n3. Frozen Features Domain Classifier:")
    print(f"   - Final accuracy: {frozen_acc:.2f}%")
    if frozen_acc > 70:
        print(f"   - Status: ✓ PASS (strong domain signal)")
    elif frozen_acc > 55:
        print(f"   - Status: ⚠  WARNING (weak domain signal or untrained features)")
    else:
        print(f"   - Status: ⚠  EXPECTED for untrained model (random features)")
        print(f"   - Test with a trained checkpoint to verify domain signal exists")

    print(f"\n4. GRL Sign Verification:")
    print(f"   - Checked {grl_num_checked} layers")
    if grl_num_checked == 0:
        print(f"   - Status: ✗ FAIL (no gradients to check - disconnected)")
    elif grl_sign_ok:
        print(f"   - Status: ✓ PASS")
    else:
        print(f"   - Status: ✗ FAIL (wrong sign)")

    print("\n" + "="*80)

    return results


if __name__ == "__main__":
    # Import data loading utilities
    from data_utils.csv_dataloader import create_csv_flowpic_loader
    from train_per_week_cesnet import load_label_mapping
    import json
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Test DANN gradient flow")
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/anatbr/dataset/CESNET-TLS-Year22_v1',
        help='Path to CESNET-TLS-Year22 dataset root'
    )
    parser.add_argument(
        '--source_week',
        type=int,
        default=33,
        help='Source week number (e.g., 33 for WEEK-2022-33)'
    )
    parser.add_argument(
        '--target_week',
        type=int,
        default=40,
        help='Target week number (e.g., 40 for WEEK-2022-40)'
    )
    parser.add_argument(
        '--data_frac',
        type=float,
        default=0.01,
        help='Fraction of data to use for testing (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint to load (optional, for testing trained models)'
    )
    parser.add_argument(
        '--num_frozen_batches',
        type=int,
        default=100,
        help='Number of batches for frozen features test (default: 100)'
    )
    args = parser.parse_args()

    # Initialize config
    config = Config()

    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")

    # Get dataset root
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    # Load label mapping
    print("Loading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"Found {num_classes} classes")

    # Load model
    config.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(config.MODEL_PARAMS).to(device)
    model.set_epoch(0.5)  # Set to middle of training to get reasonable GRL lambda

    # Load checkpoint if provided
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("Checkpoint loaded successfully!")
        else:
            print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
            print("Continuing with random initialization...")

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"GRL Lambda: {model.grl.lambda_}")

    # Load data
    print("\nLoading data...")
    source_week = f"WEEK-2022-{args.source_week:02d}"
    target_week = f"WEEK-2022-{args.target_week:02d}"

    source_dir = dataset_root / source_week
    target_dir = dataset_root / target_week

    train_path = source_dir / 'train.parquet'
    test_path = target_dir / 'test.parquet'

    if not train_path.exists() or not test_path.exists():
        raise ValueError(
            f"Data files not found!\n"
            f"  Train: {train_path} (exists: {train_path.exists()})\n"
            f"  Test: {test_path} (exists: {test_path.exists()})"
        )

    # Create data loaders (using same parameters as train_per_week_cesnet.py)
    train_loader = create_csv_flowpic_loader(
        [train_path],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        resolution=config.RESOLUTION,
        data_sample_frac=args.data_frac,
        seed=config.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=True
    )

    test_loader = create_csv_flowpic_loader(
        [test_path],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        resolution=config.RESOLUTION,
        data_sample_frac=args.data_frac,
        seed=config.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train ({source_week}): {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"  Test ({target_week}): {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")

    # Run tests
    output_dir = Path("test_results") / f"{source_week}_to_{target_week}"
    results = run_all_tests(
        model=model,
        source_loader=train_loader,
        target_loader=test_loader,
        device=device,
        output_dir=output_dir,
        num_frozen_batches=args.num_frozen_batches
    )

    print("\nTests completed!")
