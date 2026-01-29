"""
verify_parquet_schema.py

Validate (and optionally FIX) that all CESNET *.parquet files produced by
preprocess_cesnet_to_parquet.py share the same schema.

Usage:
  # just validate
  python verify_parquet_schema.py --root-dir ../../../dataset/CESNET-TLS-Year22_v2

  # validate a few files
  python verify_parquet_schema.py --root-dir ... --limit 20

  # validate + rewrite nonconforming files in-place
  python verify_parquet_schema.py --root-dir ... --fix --overwrite

Notes:
- This checks Arrow/Parquet dtypes and required columns.
- Optionally checks list lengths for histogram columns by sampling rows.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


# -----------------------------
# Canonical schema definition
# -----------------------------
def expected_schema() -> pa.Schema:
    """
    Canonical schema for CESNET parquet after preprocessing.

    Important corrections vs the draft:
    - PHIST_* should be list<int64> (not string)
    - PPI_DIRECTIONS should exist and be list<int64>
    - PPI_IPT / PPI_SIZES are list<int64>
    - TIME_FIRST / TIME_LAST kept as string (since CSV source); you can change to timestamp if you prefer.
    """
    i64 = pa.int64()
    f64 = pa.float64()
    s = pa.string()
    b = pa.bool_()
    li64 = pa.list_(i64)

    return pa.schema([
        pa.field("ID", i64),
        pa.field("SRC_IP", s),
        pa.field("DST_IP", s),
        pa.field("DST_ASN", i64),
        pa.field("DST_PORT", i64),
        pa.field("PROTOCOL", i64),
        pa.field("TLS_SNI", s),
        pa.field("TLS_JA3", s),
        pa.field("TIME_FIRST", s),
        pa.field("TIME_LAST", s),
        pa.field("DURATION", f64),
        pa.field("BYTES", i64),
        pa.field("BYTES_REV", i64),
        pa.field("PACKETS", i64),
        pa.field("PACKETS_REV", i64),
        pa.field("PPI_LEN", i64),
        pa.field("PPI_DURATION", f64),
        pa.field("PPI_ROUNDTRIPS", i64),
        pa.field("APP", s),
        pa.field("CATEGORY", s),

        pa.field("FLAG_CWR", i64),
        pa.field("FLAG_CWR_REV", i64),
        pa.field("FLAG_ECE", i64),
        pa.field("FLAG_ECE_REV", i64),
        pa.field("FLAG_URG", i64),
        pa.field("FLAG_URG_REV", i64),
        pa.field("FLAG_ACK", i64),
        pa.field("FLAG_ACK_REV", i64),
        pa.field("FLAG_PSH", i64),
        pa.field("FLAG_PSH_REV", i64),
        pa.field("FLAG_RST", i64),
        pa.field("FLAG_RST_REV", i64),
        pa.field("FLAG_SYN", i64),
        pa.field("FLAG_SYN_REV", i64),
        pa.field("FLAG_FIN", i64),
        pa.field("FLAG_FIN_REV", i64),

        pa.field("FLOW_ENDREASON_IDLE", b),
        pa.field("FLOW_ENDREASON_ACTIVE", b),
        pa.field("FLOW_ENDREASON_END", b),
        pa.field("FLOW_ENDREASON_OTHER", b),

        pa.field("PHIST_SRC_SIZES", li64),
        pa.field("PHIST_DST_SIZES", li64),
        pa.field("PHIST_SRC_IPT", li64),
        pa.field("PHIST_DST_IPT", li64),

        pa.field("PPI_IPT", li64),
        pa.field("PPI_DIRECTIONS", li64),
        pa.field("PPI_SIZES", li64),
    ])


# -----------------------------
# Helpers
# -----------------------------
def schema_as_map(schema: pa.Schema) -> Dict[str, pa.DataType]:
    return {f.name: f.type for f in schema}


def compare_schema(
    actual: pa.Schema,
    expected: pa.Schema,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, problems)
    - strict=True: require exact column set (no missing, no extras)
    - strict=False: allow extras, but still require all expected columns and matching dtypes for them
    """
    probs: List[str] = []
    a = schema_as_map(actual)
    e = schema_as_map(expected)

    missing = [k for k in e.keys() if k not in a]
    if missing:
        probs.append(f"Missing columns: {missing}")

    if strict:
        extras = [k for k in a.keys() if k not in e]
        if extras:
            probs.append(f"Extra columns: {extras}")

    # compare dtypes for shared cols
    for name, etype in e.items():
        if name not in a:
            continue
        atype = a[name]
        if not atype.equals(etype):
            probs.append(f"Type mismatch: {name}: actual={atype} expected={etype}")

    return (len(probs) == 0), probs


def _cast_column_to_list_int64(arr: pa.Array) -> pa.Array:
    """
    Ensure a column is list<int64>. Handles:
    - list<int32>/list<float>/etc by casting values
    - string columns like "[0,1,2,...]" by TRYING to parse (best-effort)
    """
    # Already list?
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        values = arr.flatten()
        # Cast values to int64 (best-effort)
        values_i64 = pc.cast(values, pa.int64(), safe=False)
        # Rebuild list with same offsets
        offsets = arr.offsets if hasattr(arr, "offsets") else arr.offsets
        return pa.ListArray.from_arrays(offsets, values_i64)

    # If it's string, try JSON-ish parse: NOT perfect for all formats.
    # You can tighten this if needed.
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        # Very conservative: if parsing fails, keep nulls.
        # We transform strings to list by: strip brackets, split by comma.
        s = arr
        # remove [ ] and spaces
        s = pc.replace_substring_regex(s, r"^\s*\[\s*", "")
        s = pc.replace_substring_regex(s, r"\s*\]\s*$", "")
        # split
        lst = pc.split_pattern(s, pattern=",")
        # trim and cast
        # lst is list<string>; flatten, trim, cast to int64, then rebuild
        flat = pc.list_flatten(lst)
        flat = pc.utf8_trim_whitespace(flat)
        flat_i64 = pc.cast(flat, pa.int64(), safe=False)
        offsets = pc.list_value_length(lst)  # lengths
        # build offsets array from lengths
        # offsets is lengths; convert to cumulative offsets
        lens = offsets.to_numpy(zero_copy_only=False)
        cum = [0]
        total = 0
        for L in lens:
            total += int(L)
            cum.append(total)
        return pa.ListArray.from_arrays(pa.array(cum, type=pa.int32()), flat_i64)

    # fallback: try direct cast (likely to fail)
    return pc.cast(arr, pa.list_(pa.int64()), safe=False)


def fix_table_to_expected(table: pa.Table, expected: pa.Schema) -> pa.Table:
    """
    Cast/align an Arrow Table to expected schema:
    - Ensures all expected columns exist (creates null columns if missing)
    - Casts primitive cols to expected dtypes
    - Ensures PHIST_* and PPI_* are list<int64>
    - Drops extra cols (keeps only expected columns)
    """
    cols = {}
    for field in expected:
        name = field.name
        et = field.type
        if name not in table.column_names:
            cols[name] = pa.nulls(table.num_rows, type=et)
            continue

        col = table[name]
        arr = col.combine_chunks()

        # list<int64> columns
        if pa.types.is_list(et):
            cols[name] = _cast_column_to_list_int64(arr)
        else:
            cols[name] = pc.cast(arr, et, safe=False)

    fixed = pa.table(cols, schema=expected)
    return fixed


def check_list_lengths(
    parquet_path: Path,
    columns: List[str],
    expected_len: int,
    max_rows: int = 2000
) -> Optional[str]:
    """
    Sample up to max_rows and verify list lengths equals expected_len.
    Returns an error string if failed, else None.
    """
    pf = pq.ParquetFile(parquet_path)
    # read small sample
    table = pf.read(columns=columns)
    if table.num_rows == 0:
        return "File has 0 rows"

    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)

    for c in columns:
        if c not in table.column_names:
            return f"Missing {c} for list-length check"
        arr = table[c].combine_chunks()
        if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
            return f"{c} is not a list type (got {arr.type})"
        lens = pc.list_value_length(arr)
        # allow nulls; check only non-null
        non_null = pc.invert(pc.is_null(lens))
        bad = pc.any(pc.and_(non_null, pc.not_equal(lens, expected_len)))
        if bad.as_py():
            # find one example
            idx = pc.index(lens, pc.scalar(expected_len, lens.type), start=0)
            # idx is first occurrence; we want first mismatch:
            # simplest: scan python-side small sample
            ln = lens.to_pylist()
            for i, L in enumerate(ln):
                if L is None:
                    continue
                if int(L) != expected_len:
                    return f"{c} has non-{expected_len} length at row {i}: {L}"
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Verify (and optionally fix) CESNET Parquet schema consistency.")
    ap.add_argument("--root-dir", type=str, required=True, help="Root directory containing parquet files")
    ap.add_argument("--pattern", type=str, default="*.parquet", help="Glob pattern (default: *.parquet)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of files checked")
    ap.add_argument("--strict", action="store_true", default=False,
                    help="Require exact column set (no extra cols). Default: allow extras.")
    ap.add_argument("--check-hist-lens", action="store_true", default=False,
                    help="Also sample rows and verify PHIST_* have length 8.")
    ap.add_argument("--check-ppi-lens", action="store_true", default=False,
                    help="Also sample rows and verify PPI_* have same length as PPI_LEN for sampled rows (basic check).")
    ap.add_argument("--fix", action="store_true", default=False,
                    help="Rewrite files that don't match schema by casting to the canonical schema.")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Allow overwriting existing parquet files when --fix is used.")
    args = ap.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        print(f"ERROR: root-dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob(args.pattern))
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)

    exp = expected_schema()

    ok_count = 0
    bad_count = 0
    fixed_count = 0

    hist_cols = ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]

    for p in files:
        try:
            pf = pq.ParquetFile(p)
            actual = pf.schema_arrow

            ok, probs = compare_schema(actual, exp, strict=args.strict)

            # optional checks
            if ok and args.check_hist_lens:
                err = check_list_lengths(p, hist_cols, expected_len=8, max_rows=2000)
                if err is not None:
                    ok = False
                    probs.append(f"Histogram length check failed: {err}")

            # (optional) PPI length sanity: compare list lengths to PPI_LEN for first N rows
            if ok and args.check_ppi_lens:
                table = pf.read(columns=["PPI_LEN", "PPI_IPT", "PPI_DIRECTIONS", "PPI_SIZES"]).slice(0, 2000)
                pl = table["PPI_LEN"].combine_chunks()
                for c in ["PPI_IPT", "PPI_DIRECTIONS", "PPI_SIZES"]:
                    arr = table[c].combine_chunks()
                    lens = pc.list_value_length(arr)
                    # compare where both non-null
                    mask = pc.and_(pc.invert(pc.is_null(pl)), pc.invert(pc.is_null(lens)))
                    mismatch = pc.any(pc.and_(mask, pc.not_equal(pc.cast(lens, pa.int64()), pc.cast(pl, pa.int64()))))
                    if mismatch.as_py():
                        ok = False
                        probs.append(f"PPI length check failed: {c} length != PPI_LEN in sample")
                        break

            if ok:
                ok_count += 1
                continue

            bad_count += 1
            print(f"\n❌ SCHEMA MISMATCH: {p}")
            for pr in probs:
                print(f"  - {pr}")

            if args.fix:
                if not args.overwrite:
                    print("  (Not fixing because --overwrite not set)")
                    continue

                # read full table and cast
                table = pf.read()
                fixed = fix_table_to_expected(table, exp)

                # overwrite in-place
                pq.write_table(fixed, p, compression="snappy")
                fixed_count += 1
                print("  ✅ Fixed and overwrote file.")

        except Exception as e:
            bad_count += 1
            print(f"\n❌ ERROR reading {p}: {e}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"Checked:  {len(files)}")
    print(f"OK:       {ok_count}")
    print(f"Bad:      {bad_count}")
    if args.fix:
        print(f"Fixed:    {fixed_count}")
    print("=" * 70)

    # non-zero exit if anything bad (useful for CI)
    if bad_count > 0 and not args.fix:
        sys.exit(2)
    if bad_count > 0 and args.fix and fixed_count < bad_count:
        # some were bad but not fixed (e.g., overwrite not set)
        sys.exit(3)


if __name__ == "__main__":
    main()
