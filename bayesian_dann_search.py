#!/usr/bin/env python3
"""
Simple Bayesian hyperparameter search for DANN parameters.
Each SLURM array task runs this script, which uses Optuna to intelligently
pick the next hyperparameters to try (avoiding duplicates with parallel runs).

optuna-dashboard sqlite:///exps/dann_search/optuna_study.db
"""

import argparse
from pathlib import Path
import torch
import optuna

from config import Config
from train_per_week_cesnet import train_week, load_label_mapping


def objective(trial, train_week_num=33, val_week_num=None, exp_name="cesnet_dann_search_v1", baseline=False):
    """
    Optuna objective function - trains model with sampled hyperparameters.

    Args:
        trial: Optuna trial object that provides hyperparameter suggestions
        train_week_num: Week number to train on (default: 33)
        val_week_num: Week number to validate on (default: same as train_week_num)
        exp_name: Experiment name prefix (default: cesnet_dann_search_v1)
        baseline: If True, use lambda_dann=0 and lambda_grl_gamma=0 (no DANN)

    Returns:
        Best validation accuracy achieved
    """
    # Default validation week to training week if not specified
    if val_week_num is None:
        val_week_num = train_week_num

    # Sample hyperparameters using Bayesian optimization
    if baseline:
        lambda_rgl = 0.0
        lambda_dann = 0.0
        trial.set_user_attr('baseline', True)
    else:
        lambda_rgl = trial.suggest_float('lambda_rgl', 0.001, 10.0, log=True)
        lambda_dann = trial.suggest_float('lambda_dann', 0.001, 10.0, log=True)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"  lambda_rgl:  {lambda_rgl:.6f}")
    print(f"  lambda_dann: {lambda_dann:.6f}")
    print(f"  Train week:  {train_week_num}")
    print(f"  Val week:    {val_week_num}")
    print(f"{'='*70}\n")

    # Configure training
    cfg = Config()
    cfg.EXPERIMENT_NAME = f"{exp_name}/trial_{trial.number:03d}/{{}}"
    cfg.MODEL_PARAMS['lambda_rgl'] = lambda_rgl
    cfg.LAMBDA_DANN = lambda_dann
    cfg.NUM_EPOCHS = 30
    cfg.TRAIN_DATA_FRAC = 1.0
    cfg.TRAIN_PER_EPOCH_DATA_FRAC = 0.1
    cfg.VAL_DATA_FRAC = 0.1  # Use 10% of validation data (same as cesnet_v3)

    # Set up paths
    dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v1')
    train_week_dir = dataset_root / f'WEEK-2022-{train_week_num:02d}'
    val_week_dir = dataset_root / f'WEEK-2022-{val_week_num:02d}'

    # Load label mapping
    label_indices_mapping, num_classes = load_label_mapping(dataset_root)

    # Train the model with cross-week validation
    result = train_week(
        cfg=cfg,
        week_dir=train_week_dir,
        label_indices_mapping=label_indices_mapping,
        num_classes=num_classes,
        override=False,
        val_week_dir=val_week_dir
    )

    # Extract best validation accuracy from training history
    experiment_path = Path(result['experiment_path'])
    history_path = experiment_path / 'plots' / 'training_history.pth'

    history = torch.load(history_path)
    val_accuracies = history['val_accuracies']
    best_val_acc = max(val_accuracies)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} completed")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Single trial for Bayesian DANN hyperparameter search"
    )
    parser.add_argument(
        '--trial_index',
        type=int,
        required=True,
        help='SLURM array task index'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='sqlite:///exps/dann_search/optuna_study.db',
        help='Optuna storage URL (default: SQLite database)'
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default='dann_search',
        help='Optuna study name (default: dann_search)'
    )
    parser.add_argument(
        '--train_week',
        type=int,
        default=None,
        help='Week number to train on (default: None)'
    )
    parser.add_argument(
        '--val_week',
        type=int,
        default=None,
        help='Week number to validate on (default: same as train_week)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='cesnet_dann_search_v1',
        help='Experiment name prefix (default: cesnet_dann_search_v1)'
    )
    parser.add_argument(
        '--n_random_trials',
        type=int,
        default=30,
        help='Number of random sampling trials before switching to Bayesian optimization (default: 30)'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Run a baseline trial with lambda_dann=0 and lambda_rgl=0 (no DANN)'
    )
    args = parser.parse_args()

    # Create storage directory if needed
    # Extract directory from storage URL
    if args.storage.startswith('sqlite:///'):
        db_path = args.storage.replace('sqlite:///', '')
        storage_path = Path(db_path).parent
        storage_path.mkdir(parents=True, exist_ok=True)

    val_week_display = args.val_week if args.val_week is not None else args.train_week

    print(f"\n{'='*70}")
    print(f"DANN Bayesian Hyperparameter Search - Trial Runner")
    print(f"{'='*70}")
    print(f"SLURM Array Task ID: {args.trial_index}")
    print(f"Study Name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Train Week: {args.train_week}")
    print(f"Val Week: {val_week_display}")
    print(f"Random Trials: {args.n_random_trials}")
    print(f"{'='*70}\n")

    # Determine sampler based on number of completed trials
    # Use RandomSampler for first n_random_trials, then switch to TPE
    # Check existing study to see how many trials have been completed

    # Retry logic to handle concurrent database creation
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                direction='maximize',
                sampler=optuna.samplers.RandomSampler(),  # Start with random sampler
                load_if_exists=True  # Load existing study if it exists
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6, 8 seconds
                print(f"Database connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to connect to database after {max_retries} attempts")
                raise

    # If baseline requested, check if it already ran; if so, run a regular trial instead
    if args.baseline:
        baseline_done = any(
            t.user_attrs.get('baseline') and t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
        if baseline_done:
            print("Baseline already completed, running a regular trial instead.")
            args.baseline = False

    # Determine which sampler to use
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed >= args.n_random_trials:
        # Switch to Bayesian optimization (TPE)
        print(f"Switching to Bayesian optimization (TPE) - {n_completed} random trials completed")
        study.sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_random_trials)
    else:
        print(f"Using random sampling - {n_completed}/{args.n_random_trials} random trials completed")

    # Run one trial
    # Optuna will automatically:
    # 1. Check what trials have been completed or are running
    # 2. Use the appropriate sampler (random or Bayesian) to pick hyperparameters
    # 3. Avoid sampling hyperparameters that are currently being evaluated
    study.optimize(
        lambda trial: objective(trial, train_week_num=args.train_week, val_week_num=args.val_week, exp_name=args.exp_name, baseline=args.baseline),
        n_trials=1
    )

    # Print current best
    print(f"\n{'='*70}")
    print(f"Current Best (across all {len(study.trials)} trials)")
    print(f"{'='*70}")
    print(f"  lambda_rgl:      {study.best_params['lambda_rgl']:.6f}")
    print(f"  lambda_dann:     {study.best_params['lambda_dann']:.6f}")
    print(f"  Val Accuracy:    {study.best_value:.2f}%")
    print(f"  Best Trial:      {study.best_trial.number}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
