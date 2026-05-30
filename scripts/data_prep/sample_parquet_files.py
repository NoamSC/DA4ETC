"""
Sample data from CESNET parquet files for analysis.

This script efficiently samples from large parquet files without loading
the entire file into memory by using PyArrow's row group filtering.
"""

# --- repo path bootstrap (added during refactor: keeps flat cross-imports working) ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---


import argparse
from pathlib import Path
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm


def sample_from_parquet(parquet_path: str, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Efficiently sample n_samples rows from a parquet file.

    Uses PyArrow to read metadata first, then samples specific row groups
    to avoid loading the entire file.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    if total_rows == 0:
        return pd.DataFrame()

    # If file has fewer rows than requested, take all
    if total_rows <= n_samples:
        return parquet_file.read().to_pandas()

    # Generate random row indices to sample
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(total_rows, size=n_samples, replace=False)
    sample_indices = np.sort(sample_indices)

    # Read the file in batches and collect sampled rows
    # This is more memory efficient than reading all then sampling
    sampled_dfs = []
    current_row = 0
    sample_idx_pos = 0

    for i in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(i)
        row_group_rows = row_group.num_rows
        row_group_end = current_row + row_group_rows

        # Find indices that fall within this row group
        indices_in_group = []
        while sample_idx_pos < len(sample_indices) and sample_indices[sample_idx_pos] < row_group_end:
            local_idx = sample_indices[sample_idx_pos] - current_row
            indices_in_group.append(local_idx)
            sample_idx_pos += 1

        if indices_in_group:
            # Read only this row group
            row_group_df = parquet_file.read_row_group(i).to_pandas()
            sampled_dfs.append(row_group_df.iloc[indices_in_group])

        current_row = row_group_end

        # Early exit if we've collected all samples
        if sample_idx_pos >= len(sample_indices):
            break

    if not sampled_dfs:
        return pd.DataFrame()

    return pd.concat(sampled_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Sample data from CESNET parquet files")
    parser.add_argument("--dataset-root", type=str,
                        default="/home/anatbr/dataset/CESNET-TLS-Year22_v2",
                        help="Root directory of CESNET dataset")
    parser.add_argument("--samples-per-file", type=int, default=1000,
                        help="Number of samples to take from each parquet file (default: 1000)")
    parser.add_argument("--output", type=str, default="sampled_data.parquet",
                        help="Output file path (default: sampled_data.parquet)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    # Find all parquet files
    parquet_files = sorted(dataset_root.glob("**/*.parquet"))[::10]
    print(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        print("No parquet files found!")
        return

    # Sample from each file
    all_samples = []
    total_sampled = 0

    for parquet_path in tqdm(parquet_files, desc="Sampling parquet files"):
        try:
            df = sample_from_parquet(
                str(parquet_path),
                n_samples=args.samples_per_file,
                seed=args.seed
            )
            if len(df) > 0:
                # Add source file info for reference
                df['_source_file'] = parquet_path.name
                df['_source_week'] = parquet_path.parent.name
                all_samples.append(df)
                total_sampled += len(df)
        except Exception as e:
            print(f"Warning: Failed to sample from {parquet_path}: {e}")

    if not all_samples:
        print("No samples collected!")
        return

    # Combine all samples
    print(f"\nCombining {total_sampled} samples from {len(all_samples)} files...")
    combined_df = pd.concat(all_samples, ignore_index=True)

    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Save to parquet
    combined_df.to_parquet(args.output, index=False)
    print(f"\nSaved to: {args.output}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total samples: {len(combined_df)}")
    print(f"Weeks represented: {combined_df['_source_week'].nunique()}")
    print(f"Files sampled: {combined_df['_source_file'].nunique()}")

    if '_source_week' in combined_df.columns:
        print("\nSamples per week:")
        print(combined_df['_source_week'].value_counts().sort_index())


if __name__ == "__main__":
    main()
