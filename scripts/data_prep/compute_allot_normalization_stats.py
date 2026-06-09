"""
Compute z-score normalization statistics for the Allot dataset (multimodal).

Mirrors scripts/data_prep/compute_normalization_stats.py but uses the Allot
multimodal loader, which derives flowstats from packet sequences. Output is a
.npz with the SAME keys the loader/model expect:
  - ppi_mean, ppi_std             shape (3, 30)
  - flowstats_mean, flowstats_std shape (FLOWSTATS_DIM,)

Stats are computed PER TRAINING SLICE (early / quarter) from that slice's own
chunks -- the same data the model is trained on. The training script
(train_allot_multimodal.py) calls compute_and_save_norm_stats() automatically
when the .npz is missing, so you normally do not need to run this by hand.

Manual use:
  python scripts/data_prep/compute_allot_normalization_stats.py \
      --slice early --output exps/allot_multimodal/early_eq/normalization_stats.npz
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils.allot_multimodal_dataloader import (
    create_allot_multimodal_loader, FLOWSTATS_DIM, FLOWSTATS_NAMES, PPI_LEN, PPI_CHANNELS,
)
from data_utils.allot_timeline import get_training_slice
# Reuse the online statistics machinery from the CESNET script.
from compute_normalization_stats import OnlineStatistics


def compute_and_save_norm_stats(files, label_mapping, out_path,
                                sample_frac=0.1, batch_size=256, num_workers=4):
    """Compute z-score stats over `files` and save a CESNET-compatible .npz."""
    loader = create_allot_multimodal_loader(
        files=[str(f) for f in files],
        label_mapping=label_mapping,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        data_sample_frac=sample_frac,
        drop_last=False,
    )
    ppi_stats = OnlineStatistics((PPI_CHANNELS, PPI_LEN))
    flowstats_stats = OnlineStatistics((FLOWSTATS_DIM,))
    for (ppi_batch, flowstats_batch), _ in tqdm(loader, desc="norm-stats"):
        ppi_stats.update(ppi_batch)
        flowstats_stats.update(flowstats_batch)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        ppi_mean=ppi_stats.mean.astype(np.float32),
        ppi_std=ppi_stats.std.astype(np.float32),
        flowstats_mean=flowstats_stats.mean.astype(np.float32),
        flowstats_std=flowstats_stats.std.astype(np.float32),
        flowstats_names=np.array(FLOWSTATS_NAMES),
        n_samples=ppi_stats.count,
    )
    print(f"Saved stats ({ppi_stats.count} samples) to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compute Allot multimodal normalization stats")
    parser.add_argument("--slice", dest="slice_name", choices=["early", "quarter"], required=True)
    parser.add_argument("--sample-frac", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    from data_utils.allot_multimodal_dataloader import build_allot_label_mapping
    _, files = get_training_slice(args.slice_name)
    print(f"Slice {args.slice_name!r}: {len(files)} chunks")
    label_mapping = build_allot_label_mapping(files)
    print(f"Found {len(label_mapping)} appId classes")
    compute_and_save_norm_stats(
        files, label_mapping, args.output,
        sample_frac=args.sample_frac, batch_size=args.batch_size, num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
