"""DANN (domain-adversarial) training of the multimodal CNN on a closed-world
(proprietary, anonymized) time slice — the DANN variant on the proprietary
closed-world set.

Mirrors train_allot_multimodal.py but adds a gradient-reversal domain head aligning
the labeled SOURCE slice (clean-regime quarter / Week-16 equivalent) to UNLABELED
FUTURE windows (transductive UDA, the most favorable setting for DANN — same protocol
as the CESNET diagonal DANN). The model already carries a domain_classifier and GRL
(models/multimodal_cesnet.py via models/dann_utils.py); train_model uses test_loader
as the DANN target when lambda_dann>0 (one combined source+target forward so BatchNorm
sees the mixed distribution — the BN fix documented in UDA_BENCHMARK_STATUS.md).

The target loader is later windows of the timeline (unlabeled w.r.t. DANN — labels are
only used to MONITOR val accuracy for model selection, never in the adversarial loss).

Example:
  python scripts/train/train_allot_multimodal_dann.py --slice quarter \
      --lambda_dann 0.1 --lambda_grl_gamma 10 --target-windows 20 27 34 41 --override
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
from data_utils.allot_timeline import (
    get_training_slice, make_windows, list_chunks_sorted, WINDOW_FRAC,
)
from models.multimodal_cesnet import Multimodal_CESNET
from training.trainer import train_model
from training.utils import set_seed, save_config_to_json
from compute_allot_normalization_stats import compute_and_save_norm_stats
from config import Config


def main():
    ap = argparse.ArgumentParser(description="DANN training of multimodal CNN on a closed-world slice")
    ap.add_argument("--slice", dest="slice_name", choices=["early", "quarter"], default="quarter")
    ap.add_argument("--window-frac", type=float, default=WINDOW_FRAC)
    ap.add_argument("--experiment-name", type=str, default="allot_multimodal/{slice}_eq_dann")
    ap.add_argument("--target-windows", type=int, nargs="+", default=[20, 27, 34, 41],
                    help="Forward windows used as the UNLABELED DANN target domain")
    ap.add_argument("--lambda_dann", type=float, default=0.1)
    ap.add_argument("--lambda_grl_gamma", type=float, default=10.0)
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-data-frac", type=float, default=1.0)
    ap.add_argument("--target-data-frac", type=float, default=0.3)
    ap.add_argument("--norm-sample-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--override", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    set_seed(args.seed)
    cfg.SEED = args.seed
    cfg.NUM_EPOCHS = args.num_epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.TRAIN_DATA_FRAC = args.train_data_frac
    cfg.TRAIN_PER_EPOCH_DATA_FRAC = 1.0
    cfg.LAMBDA_DANN = args.lambda_dann

    window_idx, slice_files = get_training_slice(args.slice_name, window_frac=args.window_frac)
    train_files = [str(f) for f in slice_files]
    print(f"Slice {args.slice_name!r} -> source window {window_idx}, {len(train_files)} chunks")

    # Build label mapping from the SOURCE slice (defines the closed-world output space).
    label_mapping = build_allot_label_mapping(train_files)
    num_classes = len(label_mapping)
    print(f"  {num_classes} classes in source slice")

    # UNLABELED target = concatenation of chunks from chosen future windows.
    all_windows = make_windows(list_chunks_sorted(), args.window_frac)
    target_files = []
    for w in args.target_windows:
        assert w > window_idx, f"target window {w} must be forward of source {window_idx}"
        target_files += [str(f) for f in all_windows[w]]
    print(f"  DANN target = future windows {args.target_windows} -> {len(target_files)} chunks")

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
    with open(experiment_path / "slice_manifest.json", "w") as f:
        json.dump({"slice_name": args.slice_name, "window_index": window_idx,
                   "window_frac": args.window_frac, "train_files": train_files,
                   "target_windows": args.target_windows,
                   "lambda_dann": args.lambda_dann,
                   "lambda_grl_gamma": args.lambda_grl_gamma, "seed": args.seed}, f, indent=2)

    norm_stats_path = experiment_path / "normalization_stats.npz"
    if not norm_stats_path.exists():
        print("Computing normalization stats from source slice...")
        compute_and_save_norm_stats(train_files, label_mapping, norm_stats_path,
                                    sample_frac=args.norm_sample_frac, num_workers=cfg.NUM_WORKERS)

    model = Multimodal_CESNET(
        num_classes=num_classes, flowstats_input_size=FLOWSTATS_DIM, ppi_input_channels=3,
        lambda_rgl=args.lambda_dann, lambda_grl_gamma=args.lambda_grl_gamma,
    ).to(cfg.DEVICE)

    train_loader = create_allot_multimodal_loader(
        files=train_files, label_mapping=label_mapping, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, data_sample_frac=cfg.TRAIN_DATA_FRAC,
        seed=cfg.SEED, normalization_stats=norm_stats_path)
    # Target loader: unlabeled future windows. Labels exist but are NOT used by the DANN
    # loss (train_one_epoch only uses target domain labels = 1); they enter only the
    # val-accuracy monitor for model selection (transductive, favorable to DANN).
    target_loader = create_allot_multimodal_loader(
        files=target_files, label_mapping=label_mapping, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, data_sample_frac=args.target_data_frac,
        seed=cfg.SEED, normalization_stats=norm_stats_path, drop_last=False)
    print(f"Source: {len(train_loader.dataset):,} samples; Target: {len(target_loader.dataset):,} samples")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    viz_label_mapping = {i: str(i) for i in range(num_classes)}

    print(f"\nDANN training (lambda_dann={args.lambda_dann}, gamma={args.lambda_grl_gamma})...")
    train_model(
        model=model, train_loader=train_loader, test_loader=target_loader,
        criterion=criterion, optimizer=optimizer, num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE, weights_save_dir=weights_save_dir, plots_save_dir=plots_save_dir,
        label_mapping=viz_label_mapping, lambda_mmd=0.0, mmd_bandwidths=cfg.MMD_BANDWIDTHS,
        lambda_dann=args.lambda_dann, lambda_coral=0.0,
        train_per_epoch_data_frac=1.0, seed=cfg.SEED, enable_profiler=False)
    print(f"\n✓ DANN training done; checkpoint dir: {weights_save_dir}")


if __name__ == "__main__":
    main()
