"""
Split CESNET-TLS-Year22 dataset into train/test sets per week.

For each week, randomly samples 30% of rows as test set and remaining 70% as train set.
Creates two consolidated parquet files per week: train.parquet and test.parquet
"""

import os
import json
import random
from pathlib import Path
from typing import List, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def get_week_directories(dataset_root: Path) -> List[Path]:
    """Get all WEEK-* directories sorted chronologically."""
    week_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('WEEK-')]
    return sorted(week_dirs)


def get_daily_parquet_files(week_dir: Path) -> List[Path]:
    """Get all daily parquet files in a week directory."""
    parquet_files = []
    for day_dir in week_dir.iterdir():
        if day_dir.is_dir():
            for file in day_dir.iterdir():
                if file.suffix == '.parquet' and file.name.startswith('flows-'):
                    parquet_files.append(file)
    return sorted(parquet_files)


def load_and_merge_parquet_files(parquet_files: List[Path]) -> pa.Table:
    """Load and merge multiple parquet files into a single table with unified schema."""
    if not parquet_files:
        return None

    # Read all tables
    tables = []
    for file_path in tqdm(parquet_files, desc="  Loading files", leave=False):
        table = pq.read_table(file_path)
        tables.append(table)

    # Unify schemas by casting all tables to match the first table's schema
    # This handles inconsistencies like TLS_JA3 being double vs string
    target_schema = tables[0].schema
    unified_tables = []

    for i, table in enumerate(tables):
        if table.schema != target_schema:
            # Cast table to target schema
            try:
                table = table.cast(target_schema)
            except pa.ArrowInvalid:
                # If casting fails, try a more lenient approach
                print(f"    Warning: Schema mismatch in file {i}, attempting field-by-field conversion...")
                # Convert to pandas and back to use target schema
                import pandas as pd
                df = table.to_pandas()
                table = pa.Table.from_pandas(df, schema=target_schema)
        unified_tables.append(table)

    # Concatenate all tables
    combined_table = pa.concat_tables(unified_tables)

    return combined_table


def split_table_train_test(table: pa.Table, test_ratio: float = 0.3, seed: int = 42) -> Tuple[pa.Table, pa.Table]:
    """Split a PyArrow table into train and test sets by randomly sampling rows."""
    n_rows = len(table)
    n_test = int(n_rows * test_ratio)

    # Create random indices
    np.random.seed(seed)
    indices = np.random.permutation(n_rows)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Sort indices for better locality (faster access)
    test_indices = np.sort(test_indices)
    train_indices = np.sort(train_indices)

    # Use PyArrow take to select rows
    train_table = table.take(train_indices)
    test_table = table.take(test_indices)

    return train_table, test_table


def process_week(week_dir: Path, test_ratio: float = 0.3, seed: int = 42):
    """Process a single week: split into train/test and create merged parquet files."""
    # Get all daily parquet files
    parquet_files = get_daily_parquet_files(week_dir)

    if not parquet_files:
        return week_dir.name, "skipped", 0, 0

    # Load and merge all files into single table
    combined_table = load_and_merge_parquet_files(parquet_files)

    if combined_table is None or len(combined_table) == 0:
        return week_dir.name, "skipped", 0, 0

    # Split rows into train/test
    train_table, test_table = split_table_train_test(combined_table, test_ratio, seed)

    # Create output paths
    train_output = week_dir / "train.parquet"
    test_output = week_dir / "test.parquet"

    # Save train/test tables
    pq.write_table(train_table, train_output)
    pq.write_table(test_table, test_output)

    # Save split metadata
    metadata = {
        "test_ratio": test_ratio,
        "seed": seed,
        "n_train_rows": len(train_table),
        "n_test_rows": len(test_table),
        "n_total_rows": len(combined_table),
        "source_files": [f.name for f in parquet_files]
    }

    metadata_path = week_dir / "train_test_split.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return week_dir.name, "completed", len(train_table), len(test_table)


def main():
    # Configuration
    dataset_root = Path("../../../dataset/CESNET-TLS-Year22")
    test_ratio = 0.3
    seed = 42
    n_workers = 1

    print("CESNET-TLS-Year22 Train/Test Split")
    print("=" * 60)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Test ratio: {test_ratio:.0%}")
    print(f"Random seed: {seed}")
    print(f"Workers: {n_workers}")

    # Get all week directories
    week_dirs = get_week_directories(dataset_root)
    print(f"\nFound {len(week_dirs)} weeks to process")

    # Process weeks in parallel using joblib with tqdm progress bar
    # Use different seed for each week to avoid correlated train/test splits
    print(f"\nProcessing weeks in parallel with {n_workers} workers...")
    results = Parallel(n_jobs=n_workers)(
        delayed(process_week)(week_dir, test_ratio, seed + i)
        for i, week_dir in enumerate(tqdm(week_dirs, desc="Processing weeks"))
    )

    # Summary
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("\nSummary:")
    completed = [r for r in results if r and r[1] == "completed"]
    skipped = [r for r in results if r and r[1] == "skipped"]
    print(f"  Completed: {len(completed)} weeks")
    print(f"  Skipped: {len(skipped)} weeks")
    if completed:
        total_train_rows = sum(r[2] for r in completed)
        total_test_rows = sum(r[3] for r in completed)
        print(f"  Total train rows: {total_train_rows:,}")
        print(f"  Total test rows: {total_test_rows:,}")
        print(f"  Total rows: {total_train_rows + total_test_rows:,}")


if __name__ == "__main__":
    main()
