import os
import tempfile
import torch
from pathlib import Path
from csv import writer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import trange, tqdm
from datetime import datetime, timedelta
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from data_utils.csv_dataloader import create_csv_flowpic_loader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_model
from training.utils import set_seed, save_config_to_json, group_chunks_by_interval
from config import Config


def create_resampled_dataloader(source_loader, target_loader):
    """
    Create a new DataLoader from source_loader that matches the label distribution of target_loader.
    
    Args:
        source_loader: DataLoader to resample from
        target_loader: DataLoader whose distribution to match
        
    Returns:
        DataLoader: New dataloader with resampled data
    """
    # Get class distributions
    source_class_counts = source_loader.dataset.get_class_counts()
    target_class_counts = target_loader.dataset.get_class_counts()
    
    # Calculate total samples for normalization
    source_total = sum(source_class_counts.values())
    target_total = sum(target_class_counts.values())
    
    # Calculate sampling weights based on target distribution
    sampling_weights = {}
    for class_id in source_class_counts:
        if class_id in target_class_counts and source_class_counts[class_id] > 0:
            # Weight = (target_proportion / source_proportion)
            target_prop = target_class_counts[class_id] / target_total
            source_prop = source_class_counts[class_id] / source_total
            sampling_weights[class_id] = target_prop / source_prop
        else:
            sampling_weights[class_id] = 0.0
    
    # Create sample weights for each item in source dataset
    sample_weights = []
    for i in range(len(source_loader.dataset)):
        _, label = source_loader.dataset[i]
        sample_weights.append(sampling_weights[float(label)])
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create the resampled DataLoader
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(target_loader.dataset),  # Same number as target
        replacement=True
    )
    
    resampled_loader = DataLoader(
        source_loader.dataset,
        sampler=sampler,
        batch_size=target_loader.batch_size,
        num_workers=source_loader.num_workers
    )
    
    return resampled_loader


def get_dataloader_distribution(dataloader, max_batches=None):
    """
    Get the actual label distribution from a dataloader by iterating through it.
    
    Args:
        dataloader: DataLoader to analyze
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        dict: Class counts
    """
    class_counts = defaultdict(int)
    
    for batch_idx, (_, labels) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
            
        for label in labels:
            class_counts[float(label)] += 1
    
    return dict(class_counts)


def print_distribution_comparison(source_loader, target_loader, resampled_loader, 
                                apps_id_df=None, label_indices_mapping=None, 
                                max_batches_for_validation=None):
    """
    Print before/after comparison of label distributions.
    
    Args:
        source_loader: Original DataLoader
        target_loader: Target distribution DataLoader  
        resampled_loader: Resampled DataLoader
        apps_id_df: Optional DataFrame with app names for better readability
        label_indices_mapping: Mapping from original to new label indices
        max_batches_for_validation: Limit batches for resampled validation (None for all)
    """
    print("=" * 90)
    print("LABEL DISTRIBUTION COMPARISON")
    print("=" * 90)
    
    # Get distributions
    source_dist = source_loader.dataset.get_class_counts()
    target_dist = target_loader.dataset.get_class_counts()
    
    # For resampled loader, we need to actually iterate (since it's sampled)
    if max_batches_for_validation:
        print(f"Calculating resampled distribution (sampling {max_batches_for_validation} batches)...")
    else:
        print("Calculating resampled distribution (this may take a moment)...")
    
    resampled_dist = get_dataloader_distribution(resampled_loader, max_batches_for_validation)
    
    # Get all unique classes
    all_classes = set(source_dist.keys()) | set(target_dist.keys()) | set(resampled_dist.keys())
    
    # Calculate totals
    source_total = sum(source_dist.values())
    target_total = sum(target_dist.values())
    resampled_total = sum(resampled_dist.values())
    
    print(f"\nDataset sizes:")
    print(f"  Source (original): {source_total:,} samples")
    print(f"  Target: {target_total:,} samples") 
    print(f"  Resampled ({'partial' if max_batches_for_validation else 'full'} validation): {resampled_total:,} samples")
    
    print(f"\n{'Class':<8} {'Source':<15} {'Target':<15} {'Resampled':<15} {'App Name':<30}")
    print("-" * 90)
    
    for class_id in sorted(all_classes):
        # Get counts
        source_count = source_dist.get(class_id, 0)
        target_count = target_dist.get(class_id, 0)
        resampled_count = resampled_dist.get(class_id, 0)
        
        # Calculate percentages
        source_pct = (source_count / source_total * 100) if source_total > 0 else 0
        target_pct = (target_count / target_total * 100) if target_total > 0 else 0
        resampled_pct = (resampled_count / resampled_total * 100) if resampled_total > 0 else 0
        
        # Get app name if available
        app_name = "Unknown"
        if apps_id_df is not None:
            try:
                # Try to find the original label index for this class
                original_label_idx = class_id
                if label_indices_mapping is not None:
                    # Find the original index by reversing the mapping
                    for orig_idx, new_idx in label_indices_mapping.items():
                        if new_idx == class_id:
                            original_label_idx = orig_idx
                            break
                
                if original_label_idx in apps_id_df.index:
                    app_name = apps_id_df.loc[original_label_idx, 'names']
            except Exception as e:
                app_name = f"Error: {str(e)[:20]}"
        
        print(f"{int(class_id):<8} "
              f"{source_count:>6}({source_pct:>5.1f}%) "
              f"{target_count:>6}({target_pct:>5.1f}%) "
              f"{resampled_count:>6}({resampled_pct:>5.1f}%) "
              f"{app_name:<30}")
    
    print("=" * 90)
    
    # Add summary statistics
    if not max_batches_for_validation:
        print("\nDistribution Match Quality:")
        total_diff = 0
        for class_id in all_classes:
            target_pct = (target_dist.get(class_id, 0) / target_total * 100) if target_total > 0 else 0
            resampled_pct = (resampled_dist.get(class_id, 0) / resampled_total * 100) if resampled_total > 0 else 0
            total_diff += abs(target_pct - resampled_pct)
        
        print(f"  Total absolute difference: {total_diff:.2f}% (lower is better)")
        print("=" * 90)


def measure_data_drift_exp(cfg, train_dfs_path, test_dfs_paths, test_names=None, use_resampling=False):
    
    if test_names is None:
        test_names = map(str, range(len(test_dfs_paths)))

    set_seed(cfg.SEED)

    label_whitelist = cfg.LABEL_WHITELIST
    apps_id_df = pd.read_csv('data/app_id_mapping.csv', index_col=0)
    # apps_id_df = pd.read_csv('/home/anatbr/students/noamshakedc/da4etc/data/app_id_mapping.csv', index_col=0)
    apps_id_df = apps_id_df[apps_id_df.index.isin(label_whitelist)]
    label_mapping = {row['names']: i for i, row in apps_id_df.reset_index().iterrows()}
    label_indices_mapping = {label_index: new_label_index for
                            new_label_index, (label_index, row)
                            in enumerate(apps_id_df.iterrows())
                            if label_index in label_whitelist}

    num_classes = len(label_mapping)

    # Create the original train loader    
    original_train_loader = create_csv_flowpic_loader(train_dfs_path, batch_size=cfg.BATCH_SIZE, num_workers=0,
                                            shuffle=True, resolution=cfg.RESOLUTION, sample_frac=cfg.SAMPLE_FRAC_FROM_CSVS,
                                            label_mapping=label_indices_mapping, log_t_axis=False) 

    
    for test_name, test_dfs_path in zip(test_names, test_dfs_paths):
        print(f"Running experiment for test set: {test_name}")
        print(f"Using resampling: {'YES' if use_resampling else 'NO'}")
        
        cfg.MODEL_PARAMS['num_classes'] = num_classes
        model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        experiment_path = Path(str(cfg.EXPERIMENT_PATH).format(test_name))
        # if use_resampling:
        #     experiment_path = experiment_path.parent / f"{experiment_path.name}_resampled"
        
        experiment_path.mkdir(parents=True, exist_ok=True)

        weights_save_dir = experiment_path / 'weights'
        plots_save_dir = experiment_path / 'plots'
        weights_save_dir.mkdir(parents=True, exist_ok=True)
        plots_save_dir.mkdir(parents=True, exist_ok=True)

        save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")

        test_loader = create_csv_flowpic_loader(test_dfs_path, batch_size=cfg.BATCH_SIZE, num_workers=0,
                                                shuffle=False, resolution=cfg.RESOLUTION, sample_frac=cfg.SAMPLE_FRAC_FROM_CSVS,
                                                label_mapping=label_indices_mapping, log_t_axis=False)

        # Choose which train loader to use
        if use_resampling:
            print("\n" + "="*50)
            print("APPLYING RESAMPLING TO MATCH TARGET DISTRIBUTION")
            print("="*50)
            
            train_loader = create_resampled_dataloader(original_train_loader, test_loader)
            
            # Print distribution comparison (quick validation for performance)
            print_distribution_comparison(
                source_loader=original_train_loader,
                target_loader=test_loader, 
                resampled_loader=train_loader,
                apps_id_df=apps_id_df,
                label_indices_mapping=label_indices_mapping,
                max_batches_for_validation=None
            )
            
            print(f"\nUsing resampled training loader:")
            print(f"  Original train dataset: {len(original_train_loader.dataset):,} samples")
            print(f"  Target test dataset: {len(test_loader.dataset):,} samples")
            print(f"  Resampled train: Will generate {len(test_loader.dataset):,} samples per epoch")
            
        else:
            print("\nUsing original training loader (no resampling)")
            train_loader = original_train_loader
            
            print(f"  Train dataset: {len(train_loader.dataset):,} samples")
            print(f"  Test dataset: {len(test_loader.dataset):,} samples")

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
            # adapt_batch_norm=cfg.ADAPT_BATCH_NORM,
        )
    
        # training_history_dict = torch.load(experiment_path / 'plots' / 'training_history.pth', weights_only=False)
        # max_val_accuracy = np.max(training_history_dict['val_accuracies'])
        # return max_val_accuracy

# if __name__ == "__main__":
#     cfg = Config()
#     parent = Path("/home/anatbr/dataset/Allot/allot_hourly_chunks")
#     interval = timedelta(days=1)
#     start_date = datetime(2024, 9, 5, 9)

#     grouped = group_chunks_by_interval(parent, interval, start_date=start_date)
#     print(f"Number of groups: {len(grouped)}")
#     print(f"groups: {[g.strftime('%Y_%m_%d_%H_%M') for g in grouped.keys()]}")
#     train_dfs_path = grouped[start_date]
    
#     test_dfs_paths = [grouped[g] for g in grouped if g != start_date]
#     test_names = [g.strftime('%Y_%m_%d_%H_%M') for g in grouped if g != start_date]

#     measure_data_drift_exp(cfg, train_dfs_path, test_dfs_paths, test_names=test_names)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one test split from the data drift experiment.")
    parser.add_argument('--test_index', type=int, required=True, help='Index of the test group to evaluate.')
    parser.add_argument('--use_cheat_to_match_target_label_distribution', action='store_true', 
                        help='Resample training data to match target test distribution (for analysis purposes).')
    args = parser.parse_args()

    cfg = Config()
    parent = Path("/home/anatbr/dataset/Allot/allot_hourly_chunks")
    interval = timedelta(days=1)
    start_date = datetime(2024, 9, 5, 9)

    grouped = group_chunks_by_interval(parent, interval, start_date=start_date)

    print(f"Number of groups: {len(grouped)}")
    all_groups = list(grouped.items())

    train_dfs_path = grouped[start_date]
    test_groups = [(g, grouped[g]) for g in grouped] # if g != start_date
    print(f"number of test groups: {len(test_groups)}")
    
    # Validate test_index
    if args.test_index < 0 or args.test_index >= len(test_groups):
        raise ValueError(f"test_index {args.test_index} out of range (0 to {len(test_groups)-1})")

    test_name, test_dfs_path = test_groups[args.test_index]

    measure_data_drift_exp(
        cfg=cfg,
        train_dfs_path=train_dfs_path,
        test_dfs_paths=[test_dfs_path],
        test_names=[test_name.strftime('%Y_%m_%d_%H_%M')],
        use_resampling=args.use_cheat_to_match_target_label_distribution
    )