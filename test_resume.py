#!/usr/bin/env python3
"""
Test script to verify resume functionality.
This creates a dummy training run and tests resuming from checkpoint.
"""

import torch
import torch.nn as nn
from pathlib import Path
import shutil
from torch.utils.data import TensorDataset, DataLoader

from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_model
from training.must_trainer import train_must
from config import Config

def create_dummy_data(num_samples=100, num_features=256, num_classes=5):
    """Create dummy data for testing."""
    X = torch.randn(num_samples, 1, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader

def test_standard_training_resume():
    """Test resume functionality for standard training."""
    print("\n" + "="*60)
    print("TESTING STANDARD TRAINING RESUME")
    print("="*60 + "\n")

    cfg = Config()
    device = torch.device('cpu')  # Use CPU for testing

    # Create test directories
    test_exp_path = Path('test_exps/standard_training_test')
    if test_exp_path.exists():
        shutil.rmtree(test_exp_path)
    test_exp_path.mkdir(parents=True, exist_ok=True)

    weights_dir = test_exp_path / 'weights'
    plots_dir = test_exp_path / 'plots'
    weights_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Create dummy data
    train_loader = create_dummy_data()
    test_loader = create_dummy_data()

    # Create model
    model_params = {
        'conv_type': '1d',
        'input_shape': 256,
        'num_classes': 5,
        'conv_layers': [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 16,
        'dropout_prob': 0.1,
        'use_batch_norm': True,
        'lambda_rgl': 0,
        'dann_fc_out_features': 16,
        'lambda_grl_gamma': 10,
    }

    model = ConfigurableCNN(model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    label_mapping = {f'class_{i}': i for i in range(5)}

    # Train for 2 epochs
    print("PHASE 1: Initial training (2 epochs)")
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
        weights_save_dir=weights_dir,
        plots_save_dir=plots_dir,
        label_mapping=label_mapping
    )

    # Check that checkpoint exists
    epoch_2_checkpoint = weights_dir / 'model_weights_epoch_2.pth'
    assert epoch_2_checkpoint.exists(), "Epoch 2 checkpoint not found!"
    print("\n✓ Phase 1 completed - checkpoint saved")

    # Resume from checkpoint
    print("\nPHASE 2: Resume training (epoch 3-4)")
    model2 = ConfigurableCNN(model_params).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

    train_model(
        model=model2,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer2,
        num_epochs=4,
        device=device,
        weights_save_dir=weights_dir,
        plots_save_dir=plots_dir,
        label_mapping=label_mapping,
        resume_checkpoint_path=epoch_2_checkpoint,
        resume_from_epoch=2
    )

    # Check that new checkpoints exist
    epoch_4_checkpoint = weights_dir / 'model_weights_epoch_4.pth'
    assert epoch_4_checkpoint.exists(), "Epoch 4 checkpoint not found!"
    print("\n✓ Phase 2 completed - resumed successfully")

    # Cleanup
    shutil.rmtree(test_exp_path)
    print("\n✓ Standard training resume test PASSED\n")


def test_must_training_resume():
    """Test resume functionality for MUST training."""
    print("\n" + "="*60)
    print("TESTING MUST TRAINING RESUME")
    print("="*60 + "\n")

    cfg = Config()
    device = torch.device('cpu')  # Use CPU for testing

    # Create test directories
    test_exp_path = Path('test_exps/must_training_test')
    if test_exp_path.exists():
        shutil.rmtree(test_exp_path)
    test_exp_path.mkdir(parents=True, exist_ok=True)

    weights_dir = test_exp_path / 'weights'
    plots_dir = test_exp_path / 'plots'
    weights_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Create dummy data
    source_loader = create_dummy_data()
    target_loader = create_dummy_data()

    # Create models
    model_params = {
        'conv_type': '1d',
        'input_shape': 256,
        'num_classes': 5,
        'conv_layers': [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 16,
        'dropout_prob': 0.1,
        'use_batch_norm': True,
        'lambda_rgl': 0,
        'dann_fc_out_features': 16,
        'lambda_grl_gamma': 10,
    }

    teacher_model = ConfigurableCNN(model_params).to(device)
    student_model = ConfigurableCNN(model_params).to(device)
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    label_mapping = {f'class_{i}': i for i in range(5)}

    # MUST params - very short for testing
    must_params = {
        'iterations': 100,
        'alpha': 0.5,
        'pseudo_threshold': 0.75,
        'warm_start_epochs': 1,  # Just 1 epoch for testing
        'eval_every': 50,
        'target_batches_per_iter': 2,
        'optimizer': 'adam',
        'momentum': 0.9,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }

    # Train for 100 iterations
    print("PHASE 1: Initial training (100 iterations)")
    train_must(
        teacher_model=teacher_model,
        student_model=student_model,
        source_loader=source_loader,
        target_loader=target_loader,
        criterion=criterion,
        teacher_optimizer=teacher_optimizer,
        student_optimizer=student_optimizer,
        must_params=must_params,
        device=device,
        weights_save_dir=weights_dir,
        plots_save_dir=plots_dir,
        label_mapping=label_mapping
    )

    # Check that checkpoint exists
    iter_50_checkpoint = weights_dir / 'must_checkpoint_iter_50.pth'
    assert iter_50_checkpoint.exists(), "Iteration 50 checkpoint not found!"
    print("\n✓ Phase 1 completed - checkpoint saved")

    # Resume from checkpoint
    print("\nPHASE 2: Resume training (iteration 51-200)")
    teacher_model2 = ConfigurableCNN(model_params).to(device)
    student_model2 = ConfigurableCNN(model_params).to(device)
    teacher_optimizer2 = torch.optim.Adam(teacher_model2.parameters(), lr=0.001)
    student_optimizer2 = torch.optim.Adam(student_model2.parameters(), lr=0.001)

    # Load checkpoint
    checkpoint = torch.load(iter_50_checkpoint, weights_only=False)

    must_params['iterations'] = 200

    train_must(
        teacher_model=teacher_model2,
        student_model=student_model2,
        source_loader=source_loader,
        target_loader=target_loader,
        criterion=criterion,
        teacher_optimizer=teacher_optimizer2,
        student_optimizer=student_optimizer2,
        must_params=must_params,
        device=device,
        weights_save_dir=weights_dir,
        plots_save_dir=plots_dir,
        label_mapping=label_mapping,
        resume_checkpoint=checkpoint,
        resume_from_iteration=51
    )

    # Check that new checkpoints exist
    iter_150_checkpoint = weights_dir / 'must_checkpoint_iter_150.pth'
    assert iter_150_checkpoint.exists(), "Iteration 150 checkpoint not found!"
    print("\n✓ Phase 2 completed - resumed successfully")

    # Cleanup
    shutil.rmtree(test_exp_path)
    print("\n✓ MUST training resume test PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RESUME FUNCTIONALITY TEST SUITE")
    print("="*60)

    try:
        test_standard_training_resume()
        test_must_training_resume()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")

        # Clean up test directory
        if Path('test_exps').exists():
            shutil.rmtree('test_exps')

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
