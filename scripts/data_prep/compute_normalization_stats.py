"""
Compute per-cell normalization statistics for CESNET dataset.

This script analyzes the distribution of each feature in the dataset to enable
per-cell normalization. It computes statistics for:

PPI (Per-Packet Information): Shape (3, 30)
  - Channel 0: PPI_IPT (inter-packet time)
  - Channel 1: PPI_DIRECTIONS (packet directions)
  - Channel 2: PPI_SIZES (packet sizes)

Flowstats: Shape (44,)
  - 0: BYTES
  - 1: BYTES_REV
  - 2: PACKETS
  - 3: PACKETS_REV
  - 4: DURATION
  - 5: PPI_LEN
  - 6: PPI_DURATION
  - 7: PPI_ROUNDTRIPS
  - 8-15: PHIST_SRC_SIZES (8 bins)
  - 16-23: PHIST_DST_SIZES (8 bins)
  - 24-31: PHIST_SRC_IPT (8 bins)
  - 32-39: PHIST_DST_IPT (8 bins)
  - 40: FLOW_ENDREASON_IDLE
  - 41: FLOW_ENDREASON_ACTIVE
  - 42: FLOW_ENDREASON_END
  - 43: FLOW_ENDREASON_OTHER

Output:
  - Saves statistics to a .npz file for use in normalization layers
  - Prints summary statistics to console
"""

# --- repo path bootstrap (added during refactor: keeps flat cross-imports working) ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---


import json
import sys
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('/home/anatbr/students/noamshakedc/da4etc')

from data_utils.cesnet_dataloader import create_parquet_loader


# Feature names for interpretability
PPI_CHANNEL_NAMES = ['PPI_IPT', 'PPI_DIRECTIONS', 'PPI_SIZES']

FLOWSTATS_NAMES = [
    'BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV',
    'DURATION', 'PPI_LEN', 'PPI_DURATION', 'PPI_ROUNDTRIPS',
    'PHIST_SRC_SIZES_0', 'PHIST_SRC_SIZES_1', 'PHIST_SRC_SIZES_2', 'PHIST_SRC_SIZES_3',
    'PHIST_SRC_SIZES_4', 'PHIST_SRC_SIZES_5', 'PHIST_SRC_SIZES_6', 'PHIST_SRC_SIZES_7',
    'PHIST_DST_SIZES_0', 'PHIST_DST_SIZES_1', 'PHIST_DST_SIZES_2', 'PHIST_DST_SIZES_3',
    'PHIST_DST_SIZES_4', 'PHIST_DST_SIZES_5', 'PHIST_DST_SIZES_6', 'PHIST_DST_SIZES_7',
    'PHIST_SRC_IPT_0', 'PHIST_SRC_IPT_1', 'PHIST_SRC_IPT_2', 'PHIST_SRC_IPT_3',
    'PHIST_SRC_IPT_4', 'PHIST_SRC_IPT_5', 'PHIST_SRC_IPT_6', 'PHIST_SRC_IPT_7',
    'PHIST_DST_IPT_0', 'PHIST_DST_IPT_1', 'PHIST_DST_IPT_2', 'PHIST_DST_IPT_3',
    'PHIST_DST_IPT_4', 'PHIST_DST_IPT_5', 'PHIST_DST_IPT_6', 'PHIST_DST_IPT_7',
    'FLOW_ENDREASON_IDLE', 'FLOW_ENDREASON_ACTIVE',
    'FLOW_ENDREASON_END', 'FLOW_ENDREASON_OTHER',
]


class OnlineStatistics:
    """Welford's online algorithm for computing mean and variance in one pass."""

    def __init__(self, shape):
        self.shape = shape
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)  # sum of squared differences
        self.min_val = np.full(shape, np.inf, dtype=np.float64)
        self.max_val = np.full(shape, -np.inf, dtype=np.float64)

        # For percentile computation, store reservoir samples
        self.reservoir_size = 100000
        self.reservoir = None
        self.seen = 0

    def update(self, batch):
        """Update statistics with a batch of data. batch shape: (B, *shape)"""
        batch = batch.numpy().astype(np.float64)

        for sample in batch:
            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            delta2 = sample - self.mean
            self.M2 += delta * delta2

            self.min_val = np.minimum(self.min_val, sample)
            self.max_val = np.maximum(self.max_val, sample)

        # Reservoir sampling for percentiles
        flat_batch = batch.reshape(batch.shape[0], -1)
        for sample in flat_batch:
            self.seen += 1
            if self.reservoir is None:
                self.reservoir = np.zeros((self.reservoir_size, sample.shape[0]), dtype=np.float64)

            if self.seen <= self.reservoir_size:
                self.reservoir[self.seen - 1] = sample
            else:
                j = np.random.randint(0, self.seen)
                if j < self.reservoir_size:
                    self.reservoir[j] = sample

    @property
    def variance(self):
        if self.count < 2:
            return np.zeros(self.shape, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)

    def get_percentiles(self, percentiles=[1, 5, 25, 50, 75, 95, 99]):
        """Compute percentiles from reservoir samples."""
        if self.reservoir is None:
            return {p: np.zeros(self.shape) for p in percentiles}

        n_samples = min(self.seen, self.reservoir_size)
        data = self.reservoir[:n_samples].reshape(n_samples, *self.shape)

        result = {}
        for p in percentiles:
            result[p] = np.percentile(data, p, axis=0)
        return result


def compute_statistics(loader, max_batches=None, desc="Computing statistics"):
    """Compute per-cell statistics for PPI and flowstats."""

    # Initialize online statistics trackers
    ppi_stats = OnlineStatistics((3, 30))
    flowstats_stats = OnlineStatistics((44,))

    num_batches = len(loader) if max_batches is None else min(max_batches, len(loader))

    for i, ((ppi_batch, flowstats_batch), _) in enumerate(tqdm(loader, total=num_batches, desc=desc)):
        if max_batches is not None and i >= max_batches:
            break

        ppi_stats.update(ppi_batch)
        flowstats_stats.update(flowstats_batch)

    return ppi_stats, flowstats_stats


def print_ppi_statistics(stats):
    """Print formatted PPI statistics."""
    print("\n" + "=" * 80)
    print("PPI Statistics (shape: 3 channels x 30 time steps)")
    print("=" * 80)

    percentiles = stats.get_percentiles()

    for c, name in enumerate(PPI_CHANNEL_NAMES):
        print(f"\n--- Channel {c}: {name} ---")
        print(f"  Mean:   min={stats.mean[c].min():.4e}, max={stats.mean[c].max():.4e}, avg={stats.mean[c].mean():.4e}")
        print(f"  Std:    min={stats.std[c].min():.4e}, max={stats.std[c].max():.4e}, avg={stats.std[c].mean():.4e}")
        print(f"  Min:    {stats.min_val[c].min():.4e}")
        print(f"  Max:    {stats.max_val[c].max():.4e}")
        print(f"  P1:     {percentiles[1][c].min():.4e} - {percentiles[1][c].max():.4e}")
        print(f"  P50:    {percentiles[50][c].min():.4e} - {percentiles[50][c].max():.4e}")
        print(f"  P99:    {percentiles[99][c].min():.4e} - {percentiles[99][c].max():.4e}")


def print_flowstats_statistics(stats):
    """Print formatted flowstats statistics."""
    print("\n" + "=" * 80)
    print("Flowstats Statistics (44 features)")
    print("=" * 80)

    percentiles = stats.get_percentiles()

    # Group by feature type for cleaner output
    groups = [
        ("Basic flow stats", range(0, 8)),
        ("PHIST_SRC_SIZES (8 bins)", range(8, 16)),
        ("PHIST_DST_SIZES (8 bins)", range(16, 24)),
        ("PHIST_SRC_IPT (8 bins)", range(24, 32)),
        ("PHIST_DST_IPT (8 bins)", range(32, 40)),
        ("Flow end reasons", range(40, 44)),
    ]

    for group_name, indices in groups:
        print(f"\n--- {group_name} ---")
        for i in indices:
            name = FLOWSTATS_NAMES[i]
            print(f"  {name:25s}: mean={stats.mean[i]:12.4e}, std={stats.std[i]:12.4e}, "
                  f"min={stats.min_val[i]:12.4e}, max={stats.max_val[i]:12.4e}")


def save_statistics(ppi_stats, flowstats_stats, output_path):
    """Save statistics to npz file for use in normalization."""
    ppi_percentiles = ppi_stats.get_percentiles()
    flowstats_percentiles = flowstats_stats.get_percentiles()

    np.savez(
        output_path,
        # PPI statistics
        ppi_mean=ppi_stats.mean.astype(np.float32),
        ppi_std=ppi_stats.std.astype(np.float32),
        ppi_min=ppi_stats.min_val.astype(np.float32),
        ppi_max=ppi_stats.max_val.astype(np.float32),
        ppi_p1=ppi_percentiles[1].astype(np.float32),
        ppi_p5=ppi_percentiles[5].astype(np.float32),
        ppi_p25=ppi_percentiles[25].astype(np.float32),
        ppi_p50=ppi_percentiles[50].astype(np.float32),
        ppi_p75=ppi_percentiles[75].astype(np.float32),
        ppi_p95=ppi_percentiles[95].astype(np.float32),
        ppi_p99=ppi_percentiles[99].astype(np.float32),
        # Flowstats statistics
        flowstats_mean=flowstats_stats.mean.astype(np.float32),
        flowstats_std=flowstats_stats.std.astype(np.float32),
        flowstats_min=flowstats_stats.min_val.astype(np.float32),
        flowstats_max=flowstats_stats.max_val.astype(np.float32),
        flowstats_p1=flowstats_percentiles[1].astype(np.float32),
        flowstats_p5=flowstats_percentiles[5].astype(np.float32),
        flowstats_p25=flowstats_percentiles[25].astype(np.float32),
        flowstats_p50=flowstats_percentiles[50].astype(np.float32),
        flowstats_p75=flowstats_percentiles[75].astype(np.float32),
        flowstats_p95=flowstats_percentiles[95].astype(np.float32),
        flowstats_p99=flowstats_percentiles[99].astype(np.float32),
        # Metadata
        ppi_channel_names=np.array(PPI_CHANNEL_NAMES),
        flowstats_names=np.array(FLOWSTATS_NAMES),
        n_samples=ppi_stats.count,
    )
    print(f"\nStatistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics for CESNET dataset")
    parser.add_argument("--dataset-root", type=str,
                        default="/home/anatbr/dataset/CESNET-TLS-Year22_v2",
                        help="Root directory of CESNET dataset")
    parser.add_argument("--weeks", type=str, nargs="+", default=None,
                        help="Specific weeks to analyze (e.g., WEEK-2022-30). If not specified, uses all weeks.")
    parser.add_argument("--sample-frac", type=float, default=0.01,
                        help="Fraction of data to sample (default: 0.01 = 1%%)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Maximum number of batches to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for loading (default: 256)")
    parser.add_argument("--output", type=str, default="normalization_stats.npz",
                        help="Output file path (default: normalization_stats.npz)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers (default: 4)")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    # Load label mapping
    label_mapping_path = dataset_root / "label_mapping.json"
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    print(f"Loaded {len(label_mapping)} classes from label_mapping.json")

    # Find parquet files
    if args.weeks:
        week_dirs = [dataset_root / week for week in args.weeks]
    else:
        week_dirs = sorted(dataset_root.glob("WEEK-*"))

    parquet_files = []
    for week_dir in week_dirs:
        files = sorted(week_dir.glob("**/*.parquet"))
        parquet_files.extend(files)

    print(f"Found {len(parquet_files)} parquet files across {len(week_dirs)} weeks")

    if not parquet_files:
        print("No parquet files found!")
        return

    # Create dataloader
    loader = create_parquet_loader(
        parquet_files=[str(f) for f in parquet_files],
        label_mapping=label_mapping,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for statistics
        num_workers=args.num_workers,
        data_sample_frac=args.sample_frac,
    )

    print(f"Dataset size: {len(loader.dataset)} samples")
    print(f"Number of batches: {len(loader)}")

    # Compute statistics
    ppi_stats, flowstats_stats = compute_statistics(loader, max_batches=args.max_batches)

    print(f"\nProcessed {ppi_stats.count} samples")

    # Print statistics
    print_ppi_statistics(ppi_stats)
    print_flowstats_statistics(flowstats_stats)

    # Save statistics
    save_statistics(ppi_stats, flowstats_stats, args.output)

    # Print usage example
    print("\n" + "=" * 80)
    print("Usage Example for Normalization")
    print("=" * 80)
    print("""
# Load statistics
stats = np.load('normalization_stats.npz')

# For z-score normalization:
ppi_normalized = (ppi - stats['ppi_mean']) / (stats['ppi_std'] + 1e-8)
flowstats_normalized = (flowstats - stats['flowstats_mean']) / (stats['flowstats_std'] + 1e-8)

# For min-max normalization to [0, 1]:
ppi_normalized = (ppi - stats['ppi_min']) / (stats['ppi_max'] - stats['ppi_min'] + 1e-8)

# For robust normalization (using percentiles):
ppi_normalized = (ppi - stats['ppi_p50']) / (stats['ppi_p95'] - stats['ppi_p5'] + 1e-8)
""")


if __name__ == "__main__":
    main()
