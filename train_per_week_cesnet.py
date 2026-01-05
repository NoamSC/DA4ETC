"""
Train a model on each week of CESNET-TLS-Year22 dataset with 256x256 flowpics.

This script trains a separate model for each week using the pre-split train.parquet
and test.parquet files in each WEEK-2022-XX directory.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from data_utils.csv_dataloader import create_csv_flowpic_loader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_model
from training.utils import set_seed, save_config_to_json
from config import Config


def load_label_mapping(dataset_root):
    """Load label mapping from CESNET dataset root."""
    label_mapping_path = dataset_root / 'label_mapping.json'
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    # The label_mapping.json maps app_name (string) -> integer_id
    # The parquet files have APP column containing string app names
    # We need to create label_indices_mapping: app_name (string) -> index
    # This maps directly from what's in the APP column to final label indices
    app_names = sorted(label_mapping.keys())
    label_indices_mapping = {app_name: i for i, app_name in enumerate(app_names)}

    return label_indices_mapping, len(app_names)


def get_week_directories(dataset_root):
    """Get all week directories sorted by week number."""
    week_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('WEEK-2022-')])
    return week_dirs


def train_week(cfg, week_dir, label_indices_mapping, num_classes, override=False, val_week_dir=None):
    """
    Train a model on a single week's data.

    Args:
        cfg: Configuration object
        week_dir: Path to the week directory for training (e.g., WEEK-2022-18)
        label_indices_mapping: Mapping from original label indices to new indices
        num_classes: Number of classes
        override: If True, delete existing experiment and start fresh
        val_week_dir: Path to the week directory for validation (default: same as week_dir)

    Returns:
        dict: Training results/metrics
    """
    # Default validation week to training week if not specified
    if val_week_dir is None:
        val_week_dir = week_dir

    week_name = week_dir.name
    val_week_name = val_week_dir.name

    if val_week_dir == week_dir:
        print(f"\n{'='*80}")
        print(f"Training on {week_name}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Training on {week_name}, Validating on {val_week_name}")
        print(f"{'='*80}\n")

    # Set up paths
    train_path = week_dir / 'train.parquet'
    test_path = val_week_dir / 'test.parquet'

    if not train_path.exists() or not test_path.exists():
        if val_week_dir == week_dir:
            print(f"Warning: Skipping {week_name} - missing train.parquet or test.parquet")
        else:
            print(f"Warning: Skipping - missing train ({train_path.exists()}) or test ({test_path.exists()})")
        return None

    # Create experiment directory using config's experiment name format
    if val_week_dir == week_dir:
        experiment_path = cfg.BASE_EXPERIMENTS_PATH / cfg.EXPERIMENT_NAME.format(week_name)
    else:
        experiment_path = cfg.BASE_EXPERIMENTS_PATH / cfg.EXPERIMENT_NAME.format(f"{week_name}_val_{val_week_name}")

    # Check for existing experiment
    existing_checkpoint = None
    resume_from_epoch = 0

    if experiment_path.exists() and not override:
        print(f"\n{'='*60}")
        print(f"EXISTING EXPERIMENT FOUND: {experiment_path}")
        print(f"{'='*60}")

        weights_save_dir = experiment_path / 'weights'

        # Look for the latest epoch checkpoint
        epoch_files = sorted(weights_save_dir.glob('model_weights_epoch_*.pth'))
        if epoch_files:
            latest_checkpoint = epoch_files[-1]
            print(f"Found checkpoint: {latest_checkpoint.name}")
            # Extract epoch number from filename
            epoch_num = int(latest_checkpoint.stem.split('_')[-1])
            existing_checkpoint = latest_checkpoint
            resume_from_epoch = epoch_num
            print(f"Will resume from epoch {resume_from_epoch + 1}")
        else:
            print("No checkpoint found, starting from scratch")

        print(f"{'='*60}\n")
    elif experiment_path.exists() and override:
        print(f"\n{'='*60}")
        print(f"OVERRIDE MODE: Deleting existing experiment: {experiment_path}")
        print(f"{'='*60}\n")
        import shutil
        shutil.rmtree(experiment_path)

    # Create directories
    experiment_path.mkdir(parents=True, exist_ok=True)
    weights_save_dir = experiment_path / 'weights'
    plots_save_dir = experiment_path / 'plots'
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")

    # Create data loaders
    print(f"Loading data from {week_name}...")
    train_loader = create_csv_flowpic_loader(
        [train_path],
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        resolution=cfg.RESOLUTION,
        data_sample_frac=cfg.TRAIN_DATA_FRAC,
        seed=cfg.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=True
    )

    test_loader = create_csv_flowpic_loader(
        [test_path],
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        resolution=cfg.RESOLUTION,
        data_sample_frac=cfg.VAL_DATA_FRAC,
        seed=cfg.SEED,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train ({week_name}): {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    if val_week_dir == week_dir:
        print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")
    else:
        print(f"  Test ({val_week_name}): {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")

    # Create model
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Create a simple label mapping for visualization (just use indices)
    label_mapping = {i: str(i) for i in range(num_classes)}

    # Train the model
    print(f"\nStarting training for {week_name}...")
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        weights_save_dir=weights_save_dir,
        plots_save_dir=plots_save_dir,
        label_mapping=label_mapping,
        lambda_mmd=cfg.LAMBDA_MMD,
        mmd_bandwidths=cfg.MMD_BANDWIDTHS,
        lambda_dann=cfg.LAMBDA_DANN,
        resume_checkpoint_path=existing_checkpoint,
        resume_from_epoch=resume_from_epoch,
        train_per_epoch_data_frac=cfg.TRAIN_PER_EPOCH_DATA_FRAC,
        seed=cfg.SEED
    )

    print(f"\n✓ Completed training for {week_name}")
    print(f"  Results saved to: {experiment_path}")

    return {'week': week_name, 'experiment_path': str(experiment_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Train a model on each week of CESNET-TLS-Year22 dataset with 256x256 flowpics"
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/anatbr/dataset/CESNET-TLS-Year22',
        help='Path to CESNET-TLS-Year22 dataset root'
    )
    parser.add_argument(
        '--start_week',
        type=int,
        default=None,
        help='Start from this week number (e.g., 18 for WEEK-2022-18). Default: start from first week'
    )
    parser.add_argument(
        '--end_week',
        type=int,
        default=None,
        help='End at this week number (inclusive). Default: process all weeks'
    )
    parser.add_argument(
        '--week',
        type=int,
        default=None,
        help='Train only on this specific week number (e.g., 18 for WEEK-2022-18)'
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing experiments and start from scratch'
    )
    parser.add_argument(
        '--train_data_frac',
        type=float,
        default=1e-2,
        help='Fraction to load from train.parquet (default: 0.01)'
    )
    parser.add_argument(
        '--val_data_frac',
        type=float,
        default=None,
        help='Fraction to load from test.parquet (default: train_data_frac * train_per_epoch_data_frac)'
    )
    parser.add_argument(
        '--train_per_epoch_data_frac',
        type=float,
        default=1.0,
        help='Fraction of loaded training data to use per epoch (default: 1.0)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=15,
        help='Number of training epochs (default: 15)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-3,
        help='Learning rate (default: 3e-3)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name (will be formatted with week name). Default: use value from config.py'
    )
    parser.add_argument(
        '--lambda_rgl',
        type=float,
        default=None,
        help='Lambda RGL for domain adaptation (default: use value from config.py)'
    )
    parser.add_argument(
        '--lambda_dann',
        type=float,
        default=None,
        help='Lambda DANN for domain adaptation (default: use value from config.py)'
    )

    args = parser.parse_args()

    # Create configuration
    cfg = Config()
    cfg.RESOLUTION = 256  # Force 256x256 flowpics
    cfg.TRAIN_DATA_FRAC = args.train_data_frac
    cfg.VAL_DATA_FRAC = args.val_data_frac if args.val_data_frac is not None else (args.train_data_frac * args.train_per_epoch_data_frac)
    cfg.TRAIN_PER_EPOCH_DATA_FRAC = args.train_per_epoch_data_frac
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_EPOCHS = args.num_epochs
    cfg.LEARNING_RATE = args.learning_rate

    # Override experiment name if provided
    if args.exp_name is not None:
        cfg.EXPERIMENT_NAME = args.exp_name

    # Override lambda parameters if provided
    if args.lambda_rgl is not None:
        cfg.MODEL_PARAMS['lambda_rgl'] = args.lambda_rgl

    if args.lambda_dann is not None:
        cfg.LAMBDA_DANN = args.lambda_dann

    # Set seed for reproducibility
    set_seed(cfg.SEED)

    # Get dataset root
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    # Load label mapping
    print("Loading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)
    print(f"Found {num_classes} classes")

    # Get week directories
    week_dirs = get_week_directories(dataset_root)
    print(f"\nFound {len(week_dirs)} week directories")

    # Filter weeks based on arguments
    if args.week is not None:
        # Train on specific week
        week_name = f"WEEK-2022-{args.week:02d}"
        week_dirs = [d for d in week_dirs if d.name == week_name]
        if not week_dirs:
            raise ValueError(f"Week {week_name} not found in dataset")
        print(f"Training on specific week: {week_name}")
    else:
        # Filter by start/end week
        if args.start_week is not None:
            start_name = f"WEEK-2022-{args.start_week:02d}"
            week_dirs = [d for d in week_dirs if d.name >= start_name]

        if args.end_week is not None:
            end_name = f"WEEK-2022-{args.end_week:02d}"
            week_dirs = [d for d in week_dirs if d.name <= end_name]

        print(f"Processing {len(week_dirs)} weeks")

    # Train on each week
    results = []
    for i, week_dir in enumerate(week_dirs, 1):
        print(f"\n{'#'*80}")
        print(f"# Week {i}/{len(week_dirs)}: {week_dir.name}")
        print(f"{'#'*80}")

        result = train_week(
            cfg=cfg,
            week_dir=week_dir,
            label_indices_mapping=label_indices_mapping,
            num_classes=num_classes,
            override=args.override
        )

        if result is not None:
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully trained on {len(results)} weeks")
    print(f"\nResults:")
    for result in results:
        print(f"  {result['week']}: {result['experiment_path']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
