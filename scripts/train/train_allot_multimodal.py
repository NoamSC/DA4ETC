"""
Train the multi-modal CNN (models.multimodal_cesnet.Multimodal_CESNET) on a time
slice of the Allot dataset.

Allot chunks only carry packet-sequence (PPI) information, so the flowstats
modality is DERIVED from the sequences by the Allot loader (see
data_utils/allot_multimodal_dataloader.py). The model is built with
flowstats_input_size = FLOWSTATS_DIM (not the CESNET 44).

The global Allot timeline (all 4 domains concatenated chronologically, see
data_utils/allot_timeline.py) is sliced into ~2% windows. This script trains on
ONE such slice:

  --slice early    -> window 0 minus its first (partial) chunk   ("week 1" eq.)
  --slice quarter  -> the window ~1/4 of the way in              ("week 16" eq.)

producing a model under exps/allot_multimodal/<slice>_eq/ that the Allot
inference script then runs across all windows.

Normalization stats are computed from the slice's own chunks and cached at
<experiment>/normalization_stats.npz (auto-computed if missing).

Example:
  python scripts/train/train_allot_multimodal.py --slice early --override
  python scripts/train/train_allot_multimodal.py --slice quarter --override
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import json
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils.allot_multimodal_dataloader import (
    create_allot_multimodal_loader, build_allot_label_mapping, FLOWSTATS_DIM,
)
from data_utils.allot_timeline import get_training_slice, WINDOW_FRAC
from models.multimodal_cesnet import Multimodal_CESNET
from training.trainer import train_model
from training.utils import set_seed, save_config_to_json
from compute_allot_normalization_stats import compute_and_save_norm_stats
from config import Config


def main():
    parser = argparse.ArgumentParser(description="Train multimodal CNN on an Allot time slice")
    parser.add_argument("--slice", dest="slice_name", choices=["early", "quarter"], required=True,
                        help="early = 'week 1' eq.; quarter = 'week 16' eq.")
    parser.add_argument("--window-frac", type=float, default=WINDOW_FRAC,
                        help="Fraction of all chunks per window (must match inference)")
    parser.add_argument("--experiment-name", type=str, default="allot_multimodal/{slice}_eq")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction of the slice's chunks (chronologically last) held out for validation")
    parser.add_argument("--norm-sample-frac", type=float, default=0.2,
                        help="Sampling fraction when computing normalization stats")
    # Training-budget overrides. config.py may sit in "debug" mode (tiny fracs), so we
    # set explicit, sensible defaults here rather than inheriting them.
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-data-frac", type=float, default=1.0,
                        help="Fraction of each training chunk to load")
    parser.add_argument("--train-per-epoch-frac", type=float, default=1.0,
                        help="Fraction of loaded training data used per epoch (slices are small -> 1.0)")
    parser.add_argument("--val-data-frac", type=float, default=0.2,
                        help="Fraction of each validation chunk to load")
    parser.add_argument("--override", action="store_true",
                        help="Delete any existing experiment dir and start fresh")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.SEED)
    # Apply CLI budget overrides so the run does not depend on config.py's debug values.
    cfg.NUM_EPOCHS = args.num_epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.TRAIN_DATA_FRAC = args.train_data_frac
    cfg.TRAIN_PER_EPOCH_DATA_FRAC = args.train_per_epoch_frac
    cfg.VAL_DATA_FRAC = args.val_data_frac

    window_idx, slice_files = get_training_slice(args.slice_name, window_frac=args.window_frac)
    print(f"Slice {args.slice_name!r} -> global window {window_idx}, {len(slice_files)} chunks")

    # Chronological train/val split within the slice (filenames are timestamped).
    n_val = max(1, round(len(slice_files) * args.val_frac))
    if n_val >= len(slice_files):
        n_val = len(slice_files) - 1  # always keep >=1 train chunk
    train_files = [str(f) for f in slice_files[:len(slice_files) - n_val]]
    val_files = [str(f) for f in slice_files[len(slice_files) - n_val:]]
    print(f"  {len(train_files)} train / {len(val_files)} val chunks")

    label_mapping = build_allot_label_mapping(train_files)
    num_classes = len(label_mapping)
    print(f"  {num_classes} appId classes in training data")

    experiment_path = cfg.BASE_EXPERIMENTS_PATH / args.experiment_name.format(slice=args.slice_name)
    if experiment_path.exists() and args.override:
        print(f"OVERRIDE: removing {experiment_path}")
        shutil.rmtree(experiment_path)
    weights_save_dir = experiment_path / "weights"
    plots_save_dir = experiment_path / "plots"
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")
    with open(experiment_path / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    # Record exactly which chunks were used (so inference can exclude them as "seen").
    with open(experiment_path / "slice_manifest.json", "w") as f:
        json.dump({
            "slice_name": args.slice_name,
            "window_index": window_idx,
            "window_frac": args.window_frac,
            "train_files": train_files,
            "val_files": val_files,
        }, f, indent=2)

    # Normalization stats from this slice's training chunks (cached).
    norm_stats_path = experiment_path / "normalization_stats.npz"
    if not norm_stats_path.exists():
        print("Computing normalization stats from training slice...")
        compute_and_save_norm_stats(
            train_files, label_mapping, norm_stats_path,
            sample_frac=args.norm_sample_frac, num_workers=cfg.NUM_WORKERS,
        )

    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=FLOWSTATS_DIM,
        ppi_input_channels=3,
        lambda_rgl=cfg.MODEL_PARAMS.get("lambda_rgl", 0.0),
        lambda_grl_gamma=cfg.MODEL_PARAMS.get("lambda_grl_gamma", 10.0),
    ).to(cfg.DEVICE)

    train_loader = create_allot_multimodal_loader(
        files=train_files, label_mapping=label_mapping,
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
        data_sample_frac=cfg.TRAIN_DATA_FRAC, seed=cfg.SEED,
        normalization_stats=norm_stats_path,
    )
    val_loader = create_allot_multimodal_loader(
        files=val_files, label_mapping=label_mapping,
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
        data_sample_frac=cfg.VAL_DATA_FRAC, seed=cfg.SEED,
        normalization_stats=norm_stats_path, drop_last=False,
    )
    print(f"Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    viz_label_mapping = {i: str(i) for i in range(num_classes)}

    print(f"\nTraining multimodal CNN on Allot slice '{args.slice_name}'...")
    train_model(
        model=model, train_loader=train_loader, test_loader=val_loader,
        criterion=criterion, optimizer=optimizer, num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE, weights_save_dir=weights_save_dir, plots_save_dir=plots_save_dir,
        label_mapping=viz_label_mapping, lambda_mmd=cfg.LAMBDA_MMD, mmd_bandwidths=cfg.MMD_BANDWIDTHS,
        lambda_dann=cfg.LAMBDA_DANN, lambda_coral=cfg.LAMBDA_CORAL,
        train_per_epoch_data_frac=cfg.TRAIN_PER_EPOCH_DATA_FRAC, seed=cfg.SEED,
        enable_profiler=cfg.ENABLE_PROFILER,
    )
    print(f"\n✓ Completed Allot multimodal training (slice '{args.slice_name}', window {window_idx})")
    print(f"  Checkpoint dir: {weights_save_dir}")


if __name__ == "__main__":
    main()
