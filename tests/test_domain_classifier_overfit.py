"""
Test if domain classifier can overfit with frozen feature extractor.

This script trains the domain classifier to overfit on a small subset of data
with the feature extractor frozen. If it can achieve near-100% accuracy,
it confirms:
1. Domain classifier architecture has sufficient capacity
2. Domain signal exists in the features
3. Gradients flow properly
4. Optimizer works correctly

If it cannot overfit, there's a fundamental issue with the setup.
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from models.configurable_cnn import ConfigurableCNN
from config import Config
from data_utils.csv_dataloader import create_csv_flowpic_loader
from train_per_week_cesnet import load_label_mapping


def train_domain_classifier_to_overfit(
    model,
    source_loader,
    target_loader,
    device,
    num_batches=10,
    num_epochs=100,
    lr=1e-3,
    output_dir='test_results/overfit_test'
):
    """
    Train domain classifier to overfit on a small batch of data.

    Args:
        model: The model with feature extractor to freeze
        source_loader: Source domain data loader
        target_loader: Target domain data loader
        device: Device to run on
        num_batches: Number of batches to use for overfitting (small!)
        num_epochs: Number of epochs to train
        lr: Learning rate
        output_dir: Directory to save results

    Returns:
        dict: Training history and results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DOMAIN CLASSIFIER OVERFITTING TEST")
    print("="*80)
    print(f"\nGoal: Train domain classifier to 100% accuracy on {num_batches} batches")
    print(f"This tests if the domain classifier has sufficient capacity and gradients flow properly.\n")

    # Freeze feature extractor
    print("Freezing feature extractor...")
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # Disable GRL (we want normal gradient flow for this test)
    original_lambda = model.grl.lambda_
    model.grl.lambda_ = 0.0
    print(f"Setting GRL lambda to 0 (was {original_lambda:.6f})")

    # Setup optimizer for domain classifier only
    domain_params = []
    for name, param in model.named_parameters():
        if ('domain_classifier' in name or 'domain_output' in name) and param.requires_grad:
            domain_params.append(param)

    num_trainable = sum(p.numel() for p in domain_params)
    print(f"\nTrainable domain classifier parameters: {num_trainable:,}")

    optimizer = torch.optim.Adam(domain_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Collect a fixed set of batches to overfit on
    print(f"\nCollecting {num_batches} batches of data...")
    source_batches = []
    target_batches = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for i in range(num_batches):
        try:
            source_batch = next(source_iter)
            target_batch = next(target_iter)
        except StopIteration:
            print(f"Warning: Only collected {i} batches (not enough data)")
            break

        source_batches.append(source_batch)
        target_batches.append(target_batch)

    total_samples = sum(len(b[0]) for b in source_batches) + sum(len(b[0]) for b in target_batches)
    print(f"Collected {len(source_batches)} batches with {total_samples} total samples")

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'epoch': []
    }

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        # Train on the same batches repeatedly
        for source_batch, target_batch in zip(source_batches, target_batches):
            source_inputs, _ = source_batch
            target_inputs, _ = target_batch

            source_inputs = source_inputs.to(device)
            target_inputs = target_inputs.to(device)

            # Forward pass
            optimizer.zero_grad()
            source_outputs = model(source_inputs)
            target_outputs = model(target_inputs)

            # Domain labels (0 = source, 1 = target)
            source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
            target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

            # Compute loss
            loss = criterion(source_outputs['domain_preds'], source_domain_labels)
            loss += criterion(target_outputs['domain_preds'], target_domain_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()

            with torch.no_grad():
                _, source_pred = source_outputs['domain_preds'].max(1)
                _, target_pred = target_outputs['domain_preds'].max(1)
                epoch_correct += source_pred.eq(source_domain_labels).sum().item()
                epoch_correct += target_pred.eq(target_domain_labels).sum().item()
                epoch_total += len(source_domain_labels) + len(target_domain_labels)

        # Calculate metrics
        avg_loss = epoch_loss / len(source_batches)
        acc = 100.0 * epoch_correct / epoch_total

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc)
        history['epoch'].append(epoch + 1)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch < 5 or acc > 99:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")

        # Early stopping if we achieve near-perfect accuracy
        if acc >= 99.9:
            print(f"\n✓ Achieved 99.9%+ accuracy at epoch {epoch+1}!")
            break

    # Final evaluation
    final_acc = history['train_acc'][-1]
    final_loss = history['train_loss'][-1]

    print("\n" + "="*80)
    print("OVERFITTING TEST RESULTS")
    print("="*80)
    print(f"\nFinal metrics after {len(history['epoch'])} epochs:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.2f}%")

    if final_acc >= 99.0:
        print("\n✓ PASS: Domain classifier can overfit!")
        print("  This confirms:")
        print("    - Architecture has sufficient capacity")
        print("    - Domain signal exists in the features")
        print("    - Gradients flow properly through domain classifier")
        print("    - Optimizer works correctly")
    elif final_acc >= 90.0:
        print("\n⚠  WARNING: Partial overfitting (90-99%)")
        print("  The classifier is learning but not perfectly.")
        print("  Possible issues:")
        print("    - Learning rate might be too low")
        print("    - Need more epochs")
        print("    - Architecture might be slightly underpowered")
    elif final_acc >= 70.0:
        print("\n⚠  WARNING: Poor overfitting (70-90%)")
        print("  The classifier is not overfitting as expected.")
        print("  Possible issues:")
        print("    - Architecture too weak")
        print("    - Gradient flow issues")
        print("    - Optimizer settings")
    else:
        print("\n✗ FAIL: Cannot overfit!")
        print("  This suggests a fundamental problem:")
        print("    - Gradients may not be flowing properly")
        print("    - Architecture may be broken")
        print("    - Optimizer may be misconfigured")

    # Restore original state
    for param in model.feature_extractor.parameters():
        param.requires_grad = True
    model.grl.lambda_ = original_lambda

    # Visualize results
    visualize_overfitting_results(history, output_dir)

    return history


def visualize_overfitting_results(history, output_dir):
    """Create visualization of overfitting training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    axes[0].plot(history['epoch'], history['train_acc'], linewidth=2, color='blue')
    axes[0].axhline(y=100, color='green', linestyle='--', label='Perfect (100%)')
    axes[0].axhline(y=99, color='orange', linestyle='--', label='Target (99%)')
    axes[0].axhline(y=50, color='red', linestyle='--', label='Random (50%)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Domain Classifier Overfitting - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])

    # Plot loss
    axes[1].plot(history['epoch'], history['train_loss'], linewidth=2, color='red')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Domain Classifier Overfitting - Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    save_path = output_dir / 'domain_classifier_overfit.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test domain classifier overfitting capability")
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
        help='Source week number'
    )
    parser.add_argument(
        '--target_week',
        type=int,
        default=40,
        help='Target week number'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='Number of batches to overfit on (default: 10)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--data_frac',
        type=float,
        default=0.01,
        help='Fraction of data to load (default: 0.01)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Optional checkpoint to load (to test with trained features)'
    )
    args = parser.parse_args()

    # Initialize config
    config = Config()
    device = config.DEVICE
    print(f"Using device: {device}\n")

    # Get dataset root
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    # Load label mapping
    print("Loading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"Found {num_classes} classes\n")

    # Load model
    config.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(config.MODEL_PARAMS).to(device)
    model.set_epoch(0.5)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("Checkpoint loaded successfully!\n")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Continuing with random initialization...\n")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} total parameters\n")

    # Load data
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

    print("Loading data...")
    # Use larger batch size for overfitting test
    batch_size = 256

    train_loader = create_csv_flowpic_loader(
        [train_path],
        batch_size=batch_size,
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
        batch_size=batch_size,
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

    print(f"\nDataset loaded:")
    print(f"  Source ({source_week}): {len(train_loader.dataset):,} samples")
    print(f"  Target ({target_week}): {len(test_loader.dataset):,} samples")
    print()

    # Run overfitting test
    output_dir = Path("test_results") / f"overfit_{source_week}_to_{target_week}"
    history = train_domain_classifier_to_overfit(
        model=model,
        source_loader=train_loader,
        target_loader=test_loader,
        device=device,
        num_batches=args.num_batches,
        num_epochs=args.num_epochs,
        lr=args.lr,
        output_dir=output_dir
    )

    print("\nOverfitting test completed!")
