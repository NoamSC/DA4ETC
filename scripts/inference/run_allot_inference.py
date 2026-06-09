#!/usr/bin/env python
"""
Run a trained Allot multimodal model across the ENTIRE Allot timeline and save
per-window predictions (the "infer on the rest of the data" step).

Mirrors scripts/inference/run_week1_inference.py, but for the Allot dataset:
  - windows come from data_utils/allot_timeline (must use the same --window-frac
    as training);
  - the model + flowstats dim (FLOWSTATS_DIM, not the CESNET 44) and the label
    mapping come from the training experiment dir;
  - flowstats are DERIVED from packet sequences by the Allot loader.

For each window it saves <output_dir>/window_<idx>.npz with:
  true_labels, pred_labels, softmax (N x C), embeddings (subset), embedding_indices,
  plus window_index / start_ts / end_ts / n_total metadata.

Windows whose chunks were used for training are still scored (in-distribution
reference, like CESNET inferring the week-1 model on week 1). The training
window index is recorded in <experiment_dir>/slice_manifest.json so it can be
excluded downstream.

Example (one model over all windows):
  python scripts/inference/run_allot_inference.py \
      --experiment_dir exps/allot_multimodal/early_eq

Parallelism: pass --num-shards N --shard-id K to process only windows where
(idx % N == K); combined with the exists() guard, several array tasks cover all
windows without overlap.
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[2]
for _p in [_root, *sorted((_root / 'scripts').glob('*'))]:
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# --- end bootstrap ---

import re
import json
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.allot_multimodal_dataloader import create_allot_multimodal_loader, FLOWSTATS_DIM
from data_utils.allot_timeline import list_chunks_sorted, make_windows, WINDOW_FRAC
from models.multimodal_cesnet import Multimodal_CESNET
# Reuse the (model/loader-agnostic) TENT & CoTTA implementations from the CESNET
# unified inference script. They operate on a model + loader, so they work on the
# Allot loader unchanged (CoTTA's _augment already unpacks [ppi, flowstats]).
from run_inference import run_vanilla, run_tent, run_cotta, run_bnstats

_TS_RE = re.compile(r"chunk_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})")


def _ts(path):
    m = _TS_RE.search(Path(path).name)
    return m.group(1) if m else Path(path).name


def load_allot_model(experiment_dir, num_classes, device):
    """Build Multimodal_CESNET with the Allot flowstats dim and load best_model.pth."""
    ckpt_path = Path(experiment_dir) / "weights" / "best_model.pth"
    config_path = Path(experiment_dir) / "config.json"
    cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_params = cfg.get("MODEL_PARAMS", {})
    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=FLOWSTATS_DIM,
        ppi_input_channels=3,
        lambda_rgl=model_params.get("lambda_rgl", 0.0),
        lambda_grl_gamma=model_params.get("lambda_grl_gamma", 10.0),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_slim_npz(out_path, true_labels, pred_labels, softmax, embeddings,
                  meta, emb_frac=0.01, top_k=5, seed=42):
    """Save a compact result: int16 labels + top-k class idx/prob + small embedding
    subsample. ~12 MB/window vs ~225 MB for full softmax+embeddings."""
    k = min(top_k, softmax.shape[1])
    top_idx = np.argpartition(softmax, -k, axis=1)[:, -k:]                 # (N, k) unsorted
    top_prob = np.take_along_axis(softmax, top_idx, axis=1)
    order = np.argsort(-top_prob, axis=1)                                  # sort desc per row
    top_idx = np.take_along_axis(top_idx, order, axis=1).astype(np.int16)
    top_prob = np.take_along_axis(top_prob, order, axis=1).astype(np.float16)

    n = len(true_labels)
    n_emb = max(1, int(n * emb_frac))
    emb_idx = np.sort(np.random.RandomState(seed).choice(n, n_emb, replace=False))
    np.savez_compressed(
        out_path,
        true_labels=true_labels.astype(np.int16),
        pred_labels=pred_labels.astype(np.int16),
        topk_idx=top_idx, topk_prob=top_prob,
        embeddings=embeddings[emb_idx].astype(np.float16), embedding_indices=emb_idx,
        **meta,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True,
                        help="e.g. exps/allot_multimodal/early_eq")
    parser.add_argument("--method", choices=["vanilla", "tent", "cotta", "bnstats"], default="vanilla")
    parser.add_argument("--output_dir", default=None,
                        help="default: <experiment_dir>/{inference,inference_tent,inference_cotta,inference_bnstats}")
    parser.add_argument("--window-frac", type=float, default=WINDOW_FRAC,
                        help="MUST match the value used for training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_sample_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle batches. For TTA this decorrelates time-ordered "
                             "Allot batches so BN/entropy adaptation sees class-diverse "
                             "batches (vanilla is order-independent).")
    # TENT
    parser.add_argument("--tent_lr", type=float, default=1e-3)
    parser.add_argument("--tent_steps", type=int, default=1)
    # CoTTA
    parser.add_argument("--cotta_lr", type=float, default=1e-3)
    parser.add_argument("--cotta_steps", type=int, default=1)
    parser.add_argument("--rst_m", type=float, default=0.01)
    parser.add_argument("--ap", type=float, default=0.9)
    parser.add_argument("--n_aug", type=int, default=32)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--no_reset", action="store_true",
                        help="CoTTA continual mode: keep adapted state across windows")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    suffix = {"vanilla": "inference", "tent": "inference_tent", "cotta": "inference_cotta",
              "bnstats": "inference_bnstats"}
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / suffix[args.method]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Method: {args.method} -> {output_dir}")

    label_mapping = json.loads((experiment_dir / "label_mapping.json").read_text())
    label_mapping = {int(k): int(v) for k, v in label_mapping.items()}
    num_classes = len(label_mapping)
    norm_stats_path = experiment_dir / "normalization_stats.npz"
    assert norm_stats_path.exists(), f"Missing {norm_stats_path}"
    print(f"{num_classes} classes; norm stats: {norm_stats_path}")

    if args.shuffle:
        torch.manual_seed(args.seed)  # reproducible batch order for TTA adaptation

    model = load_allot_model(experiment_dir, num_classes, args.device)
    original_state = deepcopy(model.state_dict())  # TTA methods reset/anchor to this
    print(f"Loaded model from {experiment_dir/'weights'/'best_model.pth'}")

    windows = make_windows(list_chunks_sorted(), args.window_frac)
    print(f"{len(windows)} windows; this shard handles idx %% {args.num_shards} == {args.shard_id}\n")

    for idx, win_files in enumerate(windows):
        if idx % args.num_shards != args.shard_id:
            continue
        out_path = output_dir / f"window_{idx:02d}.npz"
        if out_path.exists():
            print(f"  window {idx:02d}: exists, skipping")
            continue

        # Rows whose appId is not in the training label mapping are dropped by the
        # loader (closed-world). A window with no known-class rows is skipped.
        try:
            loader = create_allot_multimodal_loader(
                files=[str(f) for f in win_files], label_mapping=label_mapping,
                batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers,
                data_sample_frac=args.data_sample_frac, seed=args.seed,
                normalization_stats=norm_stats_path, drop_last=False,
            )
        except AssertionError as e:
            print(f"  window {idx:02d}: no usable data ({e}); skipping")
            continue

        print(f"  window {idx:02d} ({_ts(win_files[0])}..{_ts(win_files[-1])}): {args.method}...")
        if args.method == "vanilla":
            true_labels, pred_labels, softmax, embeddings = run_vanilla(model, loader, args.device)
        elif args.method == "bnstats":
            true_labels, pred_labels, softmax, embeddings = run_bnstats(
                model, loader, args.device, original_state)
        elif args.method == "tent":
            true_labels, pred_labels, softmax, embeddings = run_tent(
                model, loader, args.device, original_state, args.tent_lr, args.tent_steps)
        else:  # cotta
            true_labels, pred_labels, softmax, embeddings = run_cotta(
                model, loader, args.device, original_state,
                lr=args.cotta_lr, rst_m=args.rst_m, ap=args.ap, n_aug=args.n_aug,
                noise_std=args.noise_std, steps=args.cotta_steps, reset=not args.no_reset)

        save_slim_npz(
            out_path, true_labels, pred_labels, softmax, embeddings,
            meta=dict(window_index=idx, start_ts=_ts(win_files[0]),
                      end_ts=_ts(win_files[-1]), n_total=len(true_labels)),
            seed=args.seed,
        )
        acc = (true_labels == pred_labels).mean() if len(true_labels) else float("nan")
        print(f"    saved {len(true_labels)} samples (slim), acc={acc:.3f} -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
