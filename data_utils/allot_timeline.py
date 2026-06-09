"""
Global time axis for the Allot dataset.

The Allot dataset is stored as 4 capture periods (``domain_0``..``domain_3``),
each a folder of hourly ``chunk_<YYYY-MM-DD_HH-MM>`` files. The periods do not
overlap in time (Sept 2024 -> Mar 2025), so concatenating all chunks sorted by
their filename timestamp yields one chronological timeline.

We slice that timeline into fixed-size *windows* of ~``window_frac`` of the
chunks each (default 2%, giving ~52 windows -- intentionally close to CESNET's
~53 weeks so the "train on a couple of weeks, infer on the rest" workflow maps
over directly). Training uses two windows (an early one and one ~1/4 in);
inference runs the trained model over every window.

Both training and inference import the SAME functions here so they agree on what
each window contains.
"""

import re
from pathlib import Path

ALLOT_ROOT = Path("/home/anatbr/dataset/Allot/allot_hourly_chunks_parquets")
WINDOW_FRAC = 0.02  # fraction of all chunks per window
_TS_RE = re.compile(r"chunk_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})")


def _timestamp_key(path):
    m = _TS_RE.search(Path(path).name)
    # Fall back to the raw name so sorting is still deterministic if the regex misses.
    return m.group(1) if m else Path(path).name


def list_chunks_sorted(root=ALLOT_ROOT, suffix=".parquet"):
    """Return all chunk file paths across all domains, sorted chronologically."""
    root = Path(root)
    files = list(root.glob(f"domain_*/*{suffix}"))
    if not files:  # fall back to CSV layout
        files = list(root.glob("domain_*/*.csv"))
    assert files, f"No chunk files found under {root}"
    return sorted(files, key=_timestamp_key)


def window_size(n_chunks, window_frac=WINDOW_FRAC):
    return max(1, round(window_frac * n_chunks))


def make_windows(files, window_frac=WINDOW_FRAC):
    """Split an ordered file list into consecutive windows. Returns list[list[Path]]."""
    w = window_size(len(files), window_frac)
    return [files[i:i + w] for i in range(0, len(files), w)]


def num_windows(root=ALLOT_ROOT, window_frac=WINDOW_FRAC):
    return len(make_windows(list_chunks_sorted(root), window_frac))


def quarter_window_index(root=ALLOT_ROOT, window_frac=WINDOW_FRAC):
    """Index of the window ~1/4 of the way through the timeline ('week 16' equivalent)."""
    return round(0.25 * num_windows(root, window_frac))


def get_training_slice(slice_name, root=ALLOT_ROOT, window_frac=WINDOW_FRAC):
    """
    Return (window_index, [files]) for a named training slice.

      'early'   -> window 0, with the first (possibly partial) chunk dropped
                   ('week 1' equivalent, start but not chunk 0)
      'quarter' -> the window ~1/4 of the way in ('week 16' equivalent)
    """
    files = list_chunks_sorted(root, )
    windows = make_windows(files, window_frac)
    if slice_name == "early":
        idx = 0
        chunks = windows[idx][1:]  # drop the first, potentially-partial chunk
    elif slice_name == "quarter":
        idx = quarter_window_index(root, window_frac)
        chunks = windows[idx]
    else:
        raise ValueError(f"Unknown slice_name {slice_name!r}; expected 'early' or 'quarter'")
    assert chunks, f"Training slice {slice_name!r} is empty"
    return idx, chunks
