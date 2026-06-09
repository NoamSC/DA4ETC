"""
Prepare a CESNET-QUIC22 week as a CESNET-TLS-Year22-compatible dataset root.

The multimodal pipeline (data_utils/cesnet_dataloader.create_parquet_loader) and the
training entrypoint (scripts/train/train_per_week_cesnet.py) were written for the
CESNET-TLS-Year22 layout. CESNET-QUIC22 differs in three ways:

  1. PPI is a single combined string column ``[[IPT]; [DIR]; [SIZE]]`` instead of the
     pre-split ``PPI_IPT`` / ``PPI_DIRECTIONS`` / ``PPI_SIZES`` columns. (Same channel
     order, so we just split it.)
  2. QUIC has only 3 flow-end reasons (IDLE / ACTIVE / OTHER); TLS-Year22 has a 4th,
     ``FLOW_ENDREASON_END``. The 44-dim flowstats vector expects all four, so we add
     ``FLOW_ENDREASON_END = False`` (constant 0 -> harmless after normalization).
  3. No per-week train/test split, no ``label_mapping.json``. We build both here.

Output layout (drop-in for ``--dataset_root``):

    <out_root>/
    ├── label_mapping.json
    └── WEEK-2022-<NN>/
        ├── train.parquet      (70%)
        └── test.parquet       (30%)

After running this, compute normalization stats and train:

    python scripts/data_prep/compute_normalization_stats.py --dataset_root <out_root> ...
    python scripts/train/train_per_week_cesnet.py --multimodal --week <NN> --dataset_root <out_root> ...
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
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Columns we read from the raw QUIC parquet (only what the multimodal loader needs).
QUIC_READ_COLUMNS = [
    "APP", "PPI",
    "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
    "DURATION", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS",
    "PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT",
    "FLOW_ENDREASON_IDLE", "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_OTHER",
]


def get_daily_parquet_files(week_dir: Path):
    files = []
    for day_dir in sorted(week_dir.iterdir()):
        if day_dir.is_dir():
            files.extend(sorted(day_dir.glob("flows-*.parquet")))
    return files


def build_label_mapping(stats_path: Path, include_background: bool):
    """Build the app -> index mapping from the dataset stats file.

    By default uses the 102 service apps only (standard CESNET-QUIC22 classification);
    optionally also includes the 3 background classes.
    """
    stats = json.loads(stats_path.read_text())
    apps = sorted(stats["apps"].keys())
    if include_background:
        apps = sorted(apps + list(stats["backgrounds"].keys()))
    return {app: i for i, app in enumerate(apps)}


def _parse_ppi(s):
    """Parse the combined PPI string ``[[IPT];[DIR];[SIZE]]`` into three lists.

    Returns (ipt, directions, sizes) or None if the PPI is missing/malformed.
    """
    if not isinstance(s, str) or not s:
        return None
    try:
        # QUIC stores rows separated by ';' inside the brackets in the docs, but the
        # parquet conversion uses standard JSON-style nested lists with commas.
        ppi = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(ppi, list) or len(ppi) != 3:
        return None
    return ppi[0], ppi[1], ppi[2]


PHIST_COLUMNS = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]


def _parse_phist(v):
    """Parse a PHIST histogram (string ``"[...]"`` or list) into a list of 8 ints."""
    if isinstance(v, str):
        v = json.loads(v)
    v = list(v)
    assert len(v) == 8, f"PHIST must have 8 bins, got {len(v)}"
    return v


def transform_day(df: pd.DataFrame, label_set: set) -> pd.DataFrame:
    """Filter to known apps and convert a daily QUIC frame to the TLS-compatible schema."""
    df = df[df["APP"].isin(label_set)].copy()
    if len(df) == 0:
        return df

    parsed = df["PPI"].map(_parse_ppi)
    keep = parsed.notna()
    df = df[keep]
    parsed = parsed[keep]

    df["PPI_IPT"] = parsed.map(lambda p: p[0])
    df["PPI_DIRECTIONS"] = parsed.map(lambda p: p[1])
    df["PPI_SIZES"] = parsed.map(lambda p: p[2])
    df = df.drop(columns=["PPI"])

    # PHIST_* are stored as string-encoded lists in QUIC parquet; parse to native lists
    # of 8 so the prepared schema matches TLS-Year22 (and avoids re-parsing at load).
    for col in PHIST_COLUMNS:
        df[col] = df[col].map(_parse_phist)

    # QUIC lacks FLOW_ENDREASON_END; add as constant False to fill the 44-dim vector.
    df["FLOW_ENDREASON_END"] = False

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quic_root", type=Path,
                    default=Path("/home/anatbr/dataset/cesnet-quic22_parquet"),
                    help="Root of converted QUIC parquet (contains W-2022-XX/<day>/flows-*.parquet)")
    ap.add_argument("--stats_file", type=Path,
                    default=Path("/home/anatbr/dataset/cesnet-quic22/stats-dataset.json"),
                    help="Dataset stats JSON used to build the label mapping")
    ap.add_argument("--out_root", type=Path,
                    default=Path("/home/anatbr/dataset/cesnet-quic22-prepared"),
                    help="Output dataset root (TLS-Year22-compatible)")
    ap.add_argument("--week", type=int, default=44,
                    help="QUIC week number (44-47). Output dir is WEEK-2022-<week>.")
    ap.add_argument("--test_frac", type=float, default=0.30, help="Test split fraction")
    ap.add_argument("--include_background", action="store_true",
                    help="Include the 3 background classes (default: 102 apps only)")
    ap.add_argument("--row_group_size", type=int, default=100_000)
    args = ap.parse_args()

    src_week_dir = args.quic_root / f"W-2022-{args.week:02d}"
    assert src_week_dir.is_dir(), f"QUIC week dir not found: {src_week_dir}"

    label_mapping = build_label_mapping(args.stats_file, args.include_background)
    label_set = set(label_mapping)
    print(f"Label set: {len(label_mapping)} classes "
          f"({'with' if args.include_background else 'without'} background)")

    out_week_dir = args.out_root / f"WEEK-2022-{args.week:02d}"
    out_week_dir.mkdir(parents=True, exist_ok=True)

    # Persist label mapping at the dataset root (load_label_mapping only uses the keys).
    label_map_path = args.out_root / "label_mapping.json"
    label_map_path.write_text(json.dumps(label_mapping, indent=2))
    print(f"Wrote {label_map_path}")

    daily_files = get_daily_parquet_files(src_week_dir)
    assert daily_files, f"No daily parquet files under {src_week_dir}"
    print(f"Found {len(daily_files)} daily files in {src_week_dir.name}")

    # Deterministic split, week-dependent (mirrors TLS split_cesnet_train_test convention).
    rng = np.random.RandomState(42 + args.week)

    train_path = out_week_dir / "train.parquet"
    test_path = out_week_dir / "test.parquet"
    train_writer = test_writer = None
    schema = None
    n_train = n_test = 0
    seen_labels = set()

    try:
        for f in tqdm(daily_files, desc="Days"):
            df = pq.read_table(f, columns=QUIC_READ_COLUMNS).to_pandas()
            df = transform_day(df, label_set)
            if len(df) == 0:
                continue
            seen_labels.update(df["APP"].unique())

            mask = rng.rand(len(df)) >= args.test_frac  # True -> train
            df_train = df[mask].reset_index(drop=True)
            df_test = df[~mask].reset_index(drop=True)

            for sub, path in ((df_train, train_path), (df_test, test_path)):
                if len(sub) == 0:
                    continue
                table = pa.Table.from_pandas(sub, preserve_index=False)
                if schema is None:
                    schema = table.schema
                else:
                    table = table.cast(schema)
                if path == train_path:
                    if train_writer is None:
                        train_writer = pq.ParquetWriter(train_path, schema)
                    train_writer.write_table(table, row_group_size=args.row_group_size)
                    n_train += len(sub)
                else:
                    if test_writer is None:
                        test_writer = pq.ParquetWriter(test_path, schema)
                    test_writer.write_table(table, row_group_size=args.row_group_size)
                    n_test += len(sub)
    finally:
        if train_writer is not None:
            train_writer.close()
        if test_writer is not None:
            test_writer.close()

    # Save split metadata (parallels TLS train_test_split.json).
    meta = {
        "source_week": src_week_dir.name,
        "test_frac": args.test_frac,
        "seed": 42 + args.week,
        "n_train": n_train,
        "n_test": n_test,
        "num_classes": len(label_mapping),
        "classes_present": len(seen_labels),
        "include_background": args.include_background,
    }
    (out_week_dir / "train_test_split.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDone. train={n_train:,}  test={n_test:,}")
    print(f"Classes present in data: {len(seen_labels)}/{len(label_mapping)}")
    print(f"  {train_path}")
    print(f"  {test_path}")
    if len(seen_labels) < len(label_mapping):
        missing = sorted(label_set - seen_labels)
        print(f"NOTE: {len(missing)} mapped apps have no rows this week: {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
