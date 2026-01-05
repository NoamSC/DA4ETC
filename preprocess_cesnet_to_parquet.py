"""
Convert CESNET CSV.xz files to Parquet format with pre-parsed PPI data.

This script reads .csv.xz files, parses the PPI column, and saves the result
as .parquet files in the same directory. This significantly speeds up data loading.

Usage:
    python preprocess_cesnet_to_parquet.py
"""

from pathlib import Path
import pandas as pd
import ast
from tqdm import tqdm
import argparse


def safe_parse_ppi(x):
    """Parse PPI string to extract IPT and sizes."""
    if not isinstance(x, str):
        return None, None

    x = x.strip()
    if not (x.startswith('[') and x.endswith(']')):
        return None, None

    try:
        ppi = ast.literal_eval(x)
        if isinstance(ppi, list) and len(ppi) >= 3:
            # PPI format: [ipt_list, direction_list, size_list, aux_list]
            ipt = ppi[0] if isinstance(ppi[0], list) else []
            sizes = ppi[2] if isinstance(ppi[2], list) else []
            return ipt, sizes
    except (ValueError, SyntaxError):
        pass

    return None, None


def process_csv_file(csv_path, overwrite=False):
    """Process a single CSV file and save as Parquet."""
    # Replace .csv.xz with .parquet (not just .xz)
    if str(csv_path).endswith('.csv.xz'):
        parquet_path = Path(str(csv_path).replace('.csv.xz', '.parquet'))
    else:
        parquet_path = csv_path.with_suffix('.parquet')

    # Skip if parquet already exists and not overwriting
    if parquet_path.exists() and not overwrite:
        print(f"Skipping {csv_path.name} (parquet already exists)")
        return False

    try:
        # Read CSV with decompression
        df = pd.read_csv(csv_path, compression='xz')

        # Parse PPI column
        print(f"  Parsing PPI for {len(df):,} rows...")
        parsed = df['PPI'].apply(safe_parse_ppi)
        df['PPI_IPT'] = parsed.apply(lambda x: x[0])
        df['PPI_SIZES'] = parsed.apply(lambda x: x[1])

        # Drop original PPI column to save space
        df = df.drop(columns=['PPI'])

        # Filter out rows with invalid PPI
        original_len = len(df)
        df = df[df['PPI_IPT'].notna() & df['PPI_SIZES'].notna()]
        if len(df) < original_len:
            print(f"  Filtered out {original_len - len(df):,} invalid rows")

        # Save as Parquet without compression for faster reading
        df.to_parquet(parquet_path, index=False)

        # Print size comparison
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {parquet_path.name}")
        print(f"  Size: {csv_size_mb:.2f} MB (csv.xz) -> {parquet_size_mb:.2f} MB (parquet)")

        return True

    except Exception as e:
        print(f"Error processing {csv_path.name}: {e}")
        return False


def process_csv_file_wrapper(args):
    """Wrapper for multiprocessing."""
    csv_file, overwrite, root_dir = args
    try:
        result = process_csv_file(csv_file, overwrite)
        return (result, csv_file.relative_to(root_dir))
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        return (None, csv_file.relative_to(root_dir))


def main():
    parser = argparse.ArgumentParser(description='Convert CESNET CSV.xz files to Parquet')
    parser.add_argument('--root-dir', type=str,
                        default='../../../dataset/CESNET-TLS-Year22',
                        help='Root directory containing CSV.xz files')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing parquet files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process (for testing)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    ROOT_DIR = Path(args.root_dir)

    # Find all CSV.xz files
    csv_files = list(ROOT_DIR.rglob('*.csv.xz'))

    if args.limit:
        csv_files = csv_files[:args.limit]

    print(f"Found {len(csv_files)} CSV.xz files in {ROOT_DIR}")
    print(f"Overwrite mode: {args.overwrite}")
    print(f"Workers: {args.workers}")
    print()

    # Process files in parallel
    processed = 0
    skipped = 0
    failed = 0

    if args.workers > 1:
        # Multiprocessing mode
        from multiprocessing import Pool

        # Prepare arguments for each worker
        worker_args = [(csv_file, args.overwrite, ROOT_DIR) for csv_file in csv_files]

        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_csv_file_wrapper, worker_args),
                total=len(csv_files),
                desc="Converting files"
            ))

        # Count results
        for result, rel_path in results:
            if result is True:
                processed += 1
            elif result is False:
                skipped += 1
            else:
                failed += 1
    else:
        # Single-process mode (useful for debugging)
        for csv_file in tqdm(csv_files, desc="Converting files"):
            print(f"\nProcessing: {csv_file.relative_to(ROOT_DIR)}")
            result = process_csv_file(csv_file, overwrite=args.overwrite)

            if result is True:
                processed += 1
            elif result is False:
                skipped += 1
            else:
                failed += 1

    print("\n" + "="*60)
    print("Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {len(csv_files)}")


if __name__ == '__main__':
    main()
