"""
Debug script to verify GRL lambda actually affects training.

This script checks:
1. If domain loss is being computed
2. If domain loss contributes to total loss
3. If GRL lambda affects the gradients
4. If changing lambda_rgl produces different training behavior
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.configurable_cnn import ConfigurableCNN
from config import Config
from data_utils.csv_dataloader import create_csv_flowpic_loader
from train_per_week_cesnet import load_label_mapping


def check_loss_computation(model, source_batch, target_batch, device, lambda_dann):
    """
    Check if domain loss is being computed and contributes to total loss.
    """
    print("="*80)
    print("TEST: Loss Computation")
    print("="*80)

    model.train()
    source_inputs, source_labels = source_batch
    target_inputs, _ = target_batch

    source_inputs = source_inputs.to(device)
    target_inputs = target_inputs.to(device)
    source_labels = source_labels.to(device).long()

    # Forward pass
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    # Check if domain predictions exist
    if 'domain_preds' not in source_outputs or 'domain_preds' not in target_outputs:
        print("\n❌ CRITICAL ERROR: Model not producing domain predictions!")
        print("   Check if lambda_rgl > 0 in config")
        return False

    print(f"\n✓ Model produces domain predictions")
    print(f"  Source domain preds shape: {source_outputs['domain_preds'].shape}")
    print(f"  Target domain preds shape: {target_outputs['domain_preds'].shape}")

    # Compute classification loss
    criterion = nn.CrossEntropyLoss()
    classification_loss = criterion(source_outputs['class_preds'], source_labels)

    # Compute domain loss
    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

    # Compute total loss (as in training loop)
    total_loss = classification_loss + lambda_dann * domain_loss

    print(f"\n✓ Loss values:")
    print(f"  Classification loss: {classification_loss.item():.4f}")
    print(f"  Domain loss:        {domain_loss.item():.4f}")
    print(f"  lambda_dann:        {lambda_dann}")
    print(f"  Weighted domain:    {(lambda_dann * domain_loss).item():.4f}")
    print(f"  Total loss:         {total_loss.item():.4f}")

    # Check if domain loss contributes
    if lambda_dann == 0:
        print(f"\n⚠️  WARNING: lambda_dann = 0, domain loss does NOT contribute!")
        print(f"   This means DANN is effectively disabled.")
        return False

    contribution = (lambda_dann * domain_loss.item()) / total_loss.item() * 100
    print(f"\n✓ Domain loss contributes {contribution:.1f}% of total loss")

    if contribution < 1:
        print(f"⚠️  WARNING: Domain loss contribution is very small ({contribution:.2f}%)")
        print(f"   Consider increasing lambda_dann (currently {lambda_dann})")

    return True


def compare_gradients_with_different_lambda(model, source_batch, target_batch, device, lambda_dann):
    """
    Compare gradients with different GRL lambda values.
    This directly tests if lambda_rgl affects the gradients.
    """
    print("\n" + "="*80)
    print("TEST: GRL Lambda Impact on Gradients")
    print("="*80)

    criterion = nn.CrossEntropyLoss()

    # Test with different lambda values
    lambda_values = [1e-8, 0.01, 0.1, 1.0]
    gradient_norms = {}

    for test_lambda in lambda_values:
        print(f"\nTesting with lambda_rgl = {test_lambda}")

        # Set GRL lambda
        model.grl.lambda_ = test_lambda

        # Forward pass
        model.train()
        model.zero_grad()

        source_inputs, source_labels = source_batch
        target_inputs, _ = target_batch

        source_inputs = source_inputs.to(device)
        target_inputs = target_inputs.to(device)
        source_labels = source_labels.to(device).long()

        source_outputs = model(source_inputs)
        target_outputs = model(target_inputs)

        # Compute losses
        classification_loss = criterion(source_outputs['class_preds'], source_labels)

        source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
        target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

        domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
        domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

        # Total loss
        total_loss = classification_loss + lambda_dann * domain_loss

        # Backward
        total_loss.backward()

        # Measure gradients on feature extractor
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if 'feature_extractor' in name and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2

        total_grad_norm = np.sqrt(total_grad_norm)
        gradient_norms[test_lambda] = total_grad_norm

        print(f"  Feature extractor gradient norm: {total_grad_norm:.6f}")

    # Check if gradients are different
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)

    norms_list = list(gradient_norms.values())
    max_norm = max(norms_list)
    min_norm = min(norms_list)

    print(f"\nGradient norm range: {min_norm:.6f} to {max_norm:.6f}")

    # Check if norms are significantly different
    if max_norm > 0:
        variation = (max_norm - min_norm) / max_norm * 100
        print(f"Variation: {variation:.2f}%")

        if variation < 1:
            print(f"\n❌ FAIL: Gradient norms are nearly identical!")
            print(f"   Changing lambda_rgl has NO EFFECT on gradients.")
            print(f"\n   Possible causes:")
            print(f"   1. lambda_dann = 0 (domain loss not used)")
            print(f"   2. GRL not in computation graph")
            print(f"   3. Training loop not using domain loss")
            print(f"   4. Domain loss weight too small compared to classification loss")
            return False
        else:
            print(f"\n✓ PASS: Gradients change with lambda_rgl")
            print(f"   GRL is working correctly!")
            return True
    else:
        print(f"\n❌ FAIL: No gradients detected!")
        return False


def trace_backward_path(model, source_batch, target_batch, device):
    """
    Trace backward path to verify GRL is in the computation graph.
    """
    print("\n" + "="*80)
    print("TEST: Backward Path Tracing")
    print("="*80)

    model.train()
    model.grl.lambda_ = 1.0

    source_inputs, _ = source_batch
    target_inputs, _ = target_batch

    source_inputs = source_inputs.to(device)
    target_inputs = target_inputs.to(device)

    # Forward with gradient tracking
    model.zero_grad()
    source_outputs = model(source_inputs)
    target_outputs = model(target_inputs)

    # Check if tensors require grad
    print(f"\nChecking gradient requirements:")
    print(f"  Source features require_grad: {source_outputs['features'].requires_grad}")
    print(f"  Source domain_preds require_grad: {source_outputs['domain_preds'].requires_grad}")

    # Compute domain loss
    criterion = nn.CrossEntropyLoss()
    source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
    target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

    domain_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
    domain_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

    print(f"\nDomain loss: {domain_loss.item():.4f}")
    print(f"Domain loss requires_grad: {domain_loss.requires_grad}")

    # Backward
    domain_loss.backward()

    # Check which parameters received gradients
    print(f"\nParameters with gradients:")
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1
            if 'feature_extractor' in name:
                print(f"  ✓ {name}: grad_norm={param.grad.norm().item():.6f}")

    print(f"\nTotal parameters with gradients: {params_with_grad}")

    if params_with_grad == 0:
        print(f"\n❌ FAIL: No parameters received gradients!")
        return False
    else:
        print(f"\n✓ PASS: Gradients flowing through the network")
        return True


def check_config_values(config):
    """
    Check if config has correct DANN settings.
    """
    print("\n" + "="*80)
    print("TEST: Configuration Check")
    print("="*80)

    print(f"\nDANN Configuration:")
    print(f"  lambda_rgl:       {config.MODEL_PARAMS.get('lambda_rgl', 'NOT SET')}")
    print(f"  lambda_grl_gamma: {config.MODEL_PARAMS.get('lambda_grl_gamma', 'NOT SET')}")
    print(f"  lambda_dann:      {config.LAMBDA_DANN}")

    issues = []

    if config.MODEL_PARAMS.get('lambda_rgl', 0) <= 0:
        issues.append("lambda_rgl is 0 or negative - DANN is disabled!")

    if config.LAMBDA_DANN == 0:
        issues.append("lambda_dann is 0 - domain loss not used in training!")

    if config.LAMBDA_DANN < 0.001:
        issues.append(f"lambda_dann is very small ({config.LAMBDA_DANN}) - domain loss has minimal impact")

    if issues:
        print(f"\n❌ Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✓ Configuration looks correct")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug GRL impact on training")
    parser.add_argument('--dataset_root', type=str,
                       default='/home/anatbr/dataset/CESNET-TLS-Year22_v1')
    parser.add_argument('--source_week', type=int, default=33)
    parser.add_argument('--target_week', type=int, default=40)
    args = parser.parse_args()

    # Initialize config
    config = Config()
    device = config.DEVICE

    print(f"Using device: {device}\n")

    # Check configuration first
    config_ok = check_config_values(config)

    # Load data
    dataset_root = Path(args.dataset_root)
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)

    source_week = f"WEEK-2022-{args.source_week:02d}"
    target_week = f"WEEK-2022-{args.target_week:02d}"

    source_dir = dataset_root / source_week
    target_dir = dataset_root / target_week

    print("Loading data...")
    train_loader = create_csv_flowpic_loader(
        [source_dir / 'train.parquet'],
        batch_size=256,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        resolution=config.RESOLUTION,
        data_sample_frac=0.001,
        seed=config.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=False
    )

    test_loader = create_csv_flowpic_loader(
        [target_dir / 'test.parquet'],
        batch_size=256,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        resolution=config.RESOLUTION,
        data_sample_frac=0.001,
        seed=config.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=False
    )

    # Get batches
    source_batch = next(iter(train_loader))
    target_batch = next(iter(test_loader))

    # Load model
    config.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(config.MODEL_PARAMS).to(device)
    model.set_epoch(0.5)

    print(f"\nModel initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  GRL lambda: {model.grl.lambda_:.6f}\n")

    # Run diagnostic tests
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC TESTS")
    print("="*80)

    test_results = {}

    # Test 1: Configuration
    test_results['config'] = config_ok

    # Test 2: Loss computation
    test_results['loss_computation'] = check_loss_computation(
        model, source_batch, target_batch, device, config.LAMBDA_DANN
    )

    # Test 3: Backward path
    test_results['backward_path'] = trace_backward_path(
        model, source_batch, target_batch, device
    )

    # Test 4: Gradient variation with lambda
    test_results['gradient_variation'] = compare_gradients_with_different_lambda(
        model, source_batch, target_batch, device, config.LAMBDA_DANN
    )

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    print(f"\nTest Results:")
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(test_results.values())

    if all_passed:
        print(f"\n✓ ALL TESTS PASSED!")
        print(f"\nGRL is working correctly. If training results are identical,")
        print(f"the issue is likely in the training script, not the model.")
        print(f"\nCheck:")
        print(f"  1. Training script actually uses lambda_dann in loss computation")
        print(f"  2. Different experiments use different config files")
        print(f"  3. Model checkpoints are not being overwritten")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"\nThis explains why changing lambda_rgl has no effect.")
        print(f"\nFix the failed tests above to enable DANN.")
