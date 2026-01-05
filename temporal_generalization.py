"""
Temporal Generalization Evaluation
===================================

Evaluates how models trained on week T generalize to future weeks (T+1, T+2, ...).

Usage:
------
Command line:
    python temporal_generalization.py \
        --experiment_dir exps/cesnet_v3 \
        --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22 \
        --output exps/cesnet_v3/cesnet_v3_results.json \
        --batch_size 64 \
        --data_sample_frac 0.1 \
        --device cuda:0

As a module:
    from temporal_generalization import evaluate_temporal_generalization, save_results
    from train_per_week_cesnet import load_label_mapping

    label_mapping, num_classes = load_label_mapping('/path/to/dataset')
    results = evaluate_temporal_generalization(
        experiment_dir='exps/cesnet_v3',
        dataset_root='/path/to/dataset',
        label_indices_mapping=label_mapping,
        num_classes=num_classes
    )
    save_results(results, 'results.json')

In Jupyter notebook:
    See temporal_generalization_analysis.ipynb for full analysis with visualizations
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from models.configurable_cnn import ConfigurableCNN
from data_utils.csv_dataloader import create_csv_flowpic_loader
from training.trainer import validate


def load_model_from_checkpoint(checkpoint_path, config_path, num_classes, device='cuda:0'):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model weights (.pth file)
        config_path: Path to config.json file
        num_classes: Number of classes (not stored in config)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add num_classes to model params
    model_params = config['MODEL_PARAMS'].copy()
    model_params['num_classes'] = num_classes

    # Create model
    model = ConfigurableCNN(model_params).to(device)

    # Load weights
    if checkpoint_path.suffix == '.pth' and 'best_model' in checkpoint_path.name:
        # Best model checkpoint includes extra metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Regular epoch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def create_week_loader(week_dir, label_indices_mapping, batch_size=64, num_workers=4,
                       resolution=256, data_sample_frac=0.1, seed=42):
    """
    Create a data loader for a specific week.

    Args:
        week_dir: Path to week directory (e.g., WEEK-2022-33)
        label_indices_mapping: Label mapping dictionary
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
        resolution: FlowPic resolution
        data_sample_frac: Fraction of data to load
        seed: Random seed

    Returns:
        DataLoader for the week's test data
    """
    test_path = week_dir / 'test.parquet'

    if not test_path.exists():
        return None

    loader = create_csv_flowpic_loader(
        [test_path],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        resolution=resolution,
        data_sample_frac=data_sample_frac,
        seed=seed,
        label_mapping=label_indices_mapping,
        log_t_axis=False,
        max_dt_ms=4000,
        dataset_format='cesnet_parquet',
        verbose=False
    )

    return loader


def evaluate_model_on_loader(model, loader, device='cuda:0'):
    """
    Evaluate a model on a data loader using the existing validation function.

    Args:
        model: Model to evaluate (should already be in eval mode)
        loader: Data loader to evaluate on
        device: Device to run evaluation on

    Returns:
        dict with metrics: accuracy, loss, total_samples
    """
    if loader is None:
        return None

    criterion = torch.nn.CrossEntropyLoss()

    # Use the existing validate function from training.trainer
    val_loss, val_acc, all_labels, all_predictions = validate(
        model, loader, criterion, device, return_features=False
    )

    return {
        'accuracy': val_acc,
        'loss': val_loss,
        'total_samples': len(all_labels)
    }


def get_available_weeks(dataset_root):
    """
    Get list of available week directories.

    Args:
        dataset_root: Root directory of CESNET dataset

    Returns:
        List of week directory paths
    """
    dataset_root = Path(dataset_root)
    week_dirs = sorted([d for d in dataset_root.iterdir()
                       if d.is_dir() and d.name.startswith('WEEK-2022-')])
    return week_dirs


def evaluate_temporal_generalization(experiment_dir, dataset_root, label_indices_mapping,
                                     num_classes, batch_size=64, num_workers=4, resolution=256,
                                     data_sample_frac=0.1, seed=42, device='cuda:0',
                                     checkpoint_name='best_model.pth'):
    """
    Evaluate how models trained on each week generalize to future weeks.

    Memory-efficient version: loads all models once, then iterates through test weeks
    loading one week's data at a time.

    Args:
        experiment_dir: Directory containing trained models (e.g., exps/cesnet_v3)
        dataset_root: Root directory of CESNET dataset
        label_indices_mapping: Label mapping dictionary
        num_classes: Number of classes in the dataset
        batch_size: Batch size for evaluation
        num_workers: Number of worker processes
        resolution: FlowPic resolution
        data_sample_frac: Fraction of data to use per week
        seed: Random seed
        device: Device to run evaluation on
        checkpoint_name: Name of checkpoint file to load (default: best_model.pth)

    Returns:
        dict mapping (train_week, test_week) -> metrics
    """
    experiment_dir = Path(experiment_dir)
    dataset_root = Path(dataset_root)

    # Get available test weeks
    test_week_dirs = get_available_weeks(dataset_root)
    test_week_names = [d.name for d in test_week_dirs]

    print(f"\nFound {len(test_week_names)} weeks with data: {test_week_names[0]} to {test_week_names[-1]}")

    # Find all trained models
    train_week_dirs = sorted([d for d in experiment_dir.iterdir()
                             if d.is_dir() and d.name.startswith('WEEK-2022-')])

    print(f"Found {len(train_week_dirs)} trained models")

    # Step 1: Load all models once
    print("\nLoading all trained models...")
    models = {}
    for train_week_dir in tqdm(train_week_dirs, desc="Loading models"):
        train_week_name = train_week_dir.name

        checkpoint_path = train_week_dir / 'weights' / checkpoint_name
        config_path = train_week_dir / 'config.json'

        if not checkpoint_path.exists() or not config_path.exists():
            print(f"  Skipping {train_week_name} - missing checkpoint or config")
            continue

        model = load_model_from_checkpoint(checkpoint_path, config_path, num_classes, device)
        models[train_week_name] = model

    print(f"Loaded {len(models)} models\n")

    results = {}

    # Step 2: Iterate through test weeks, evaluate all models on each week
    # Outer loop: test weeks (load one week's data at a time)
    # Inner loop: models (already loaded, just evaluate)
    print("Evaluating models on all test weeks...")
    for test_week_dir in tqdm(test_week_dirs, desc="Test weeks", position=0):
        test_week_name = test_week_dir.name

        # Load this week's data once
        loader = create_week_loader(
            test_week_dir, label_indices_mapping, batch_size, num_workers,
            resolution, data_sample_frac, seed
        )

        if loader is None:
            continue

        # Evaluate all models on this week's data
        for train_week_name, model in tqdm(models.items(), desc=f"  Models on {test_week_name}", position=1, leave=False):
            # Evaluate
            metrics = evaluate_model_on_loader(model, loader, device)

            if metrics is not None:
                results[(train_week_name, test_week_name)] = metrics

        # Delete loader to free memory
        del loader

    return results


def extract_week_number(week_name):
    """Extract week number from week name (e.g., 'WEEK-2022-33' -> 33)"""
    return int(week_name.split('-')[-1])


def results_to_dataframe(results):
    """
    Convert results dictionary to a pandas DataFrame for easy analysis.

    Args:
        results: Output from evaluate_temporal_generalization()

    Returns:
        DataFrame with columns: train_week, test_week, train_week_num, test_week_num,
                                accuracy, loss, total_samples
    """
    import pandas as pd

    data = []
    for (train_week, test_week), metrics in results.items():
        data.append({
            'train_week': train_week,
            'test_week': test_week,
            'train_week_num': extract_week_number(train_week),
            'test_week_num': extract_week_number(test_week),
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'total_samples': metrics['total_samples']
        })

    df = pd.DataFrame(data)
    df = df.sort_values(['train_week_num', 'test_week_num'])

    return df


def save_results(results, output_path):
    """
    Save results to a JSON file.

    Args:
        results: Output from evaluate_temporal_generalization()
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_results = {
        f"{train_week}_{test_week}": metrics
        for (train_week, test_week), metrics in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")


def load_results(results_path):
    """
    Load results from a JSON file.

    Args:
        results_path: Path to results file

    Returns:
        dict mapping (train_week, test_week) -> metrics
    """
    with open(results_path, 'r') as f:
        serializable_results = json.load(f)

    # Convert back to tuple keys
    results = {}
    for key, metrics in serializable_results.items():
        train_week, test_week = key.rsplit('_WEEK-2022-', 1)
        train_week = train_week
        test_week = 'WEEK-2022-' + test_week
        results[(train_week, test_week)] = metrics

    return results


def main():
    """
    Main function to run temporal generalization evaluation from command line.
    """
    import argparse
    from train_per_week_cesnet import load_label_mapping

    parser = argparse.ArgumentParser(
        description="Evaluate temporal generalization of trained models"
    )
    parser.add_argument(
        '--experiment_dir',
        type=str,
        default='exps/cesnet_v3',
        help='Directory containing trained models (default: exps/cesnet_v3)'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/home/anatbr/dataset/CESNET-TLS-Year22',
        help='Root directory of CESNET dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON (default: {experiment_dir}_temporal_generalization_results.json)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loader workers (default: 8)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=256,
        help='FlowPic resolution (default: 256)'
    )
    parser.add_argument(
        '--data_sample_frac',
        type=float,
        default=0.1,
        help='Fraction of data to use per week (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run on (default: cuda:0)'
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='best_model.pth',
        help='Checkpoint filename to load (default: best_model.pth)'
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        exp_name = Path(args.experiment_dir).name
        args.output = f'exps/{exp_name}_temporal_generalization_results.json'

    print("="*80)
    print("Temporal Generalization Evaluation")
    print("="*80)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Dataset root:         {args.dataset_root}")
    print(f"Output:               {args.output}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Data sample frac:     {args.data_sample_frac}")
    print(f"Device:               {args.device}")
    print("="*80)

    # Load label mapping
    print("\nLoading label mapping...")
    label_indices_mapping, num_classes = load_label_mapping(Path(args.dataset_root))
    print(f"Number of classes: {num_classes}")

    # Run evaluation
    results = evaluate_temporal_generalization(
        experiment_dir=args.experiment_dir,
        dataset_root=args.dataset_root,
        label_indices_mapping=label_indices_mapping,
        num_classes=num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        data_sample_frac=args.data_sample_frac,
        seed=args.seed,
        device=args.device,
        checkpoint_name=args.checkpoint_name
    )

    # Save results
    save_results(results, args.output)

    # Print summary
    print("\n" + "="*80)
    print("Evaluation Complete")
    print("="*80)
    print(f"Total evaluations: {len(results)}")
    print(f"Results saved to:  {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
