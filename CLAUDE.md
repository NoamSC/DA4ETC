# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo produces all results for the paper *"The Illusion of Continuous Drift: Disentangling Discrete Covariate Shift from Label Shift in Encrypted Traffic for Unsupervised Monitoring"*. Core thesis: drift in encrypted-traffic classification is **discrete and per-class** (asynchronous "teleportations"), not continuous/global, so global UDA/TTA (DANN, TENT, CoTTA) suffers negative transfer. The paper's primary contribution is a **BBSE-corrected entropy residual** that tracks hidden model degradation while ignoring benign label-shift (volume) volatility.

## Environment & execution

- Python interpreter (conda env `ml2`): `/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python`. The login shell is `tcsh`.
- **Anything longer than a few minutes must run via SLURM** (`sbatch slurm_files/<job>.slurm`), never as a login-node background process. Monitor with `squeue -u $USER`; logs land in `logs/`. SLURM scripts hardcode the `ml2` python path and `cd` to the repo root.
- Run all entrypoints **from the repo root**: `python scripts/<stage>/<script>.py …`. Each script has a sys.path bootstrap that adds the repo root + `scripts/*` so library imports (`config`, `data_utils`, `models`, `training`) and cross-script imports resolve.
- Disk quota on this netapp tree has been hit before. The bloat is per-epoch checkpoints under `exps/` — keep `best_model.pth` + a couple of recent epochs per week, prune the rest when space runs low.

## Common commands

```bash
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Train a per-week model (FlowPic CNN; supports --lambda_dann/--lambda_rgl/--lambda_grl_gamma for DANN)
$PY scripts/train/train_per_week_cesnet.py --week 18 --train_data_frac 0.01
# Resume happens automatically from checkpoints; --override deletes and restarts.

# Frozen-source inference across all weeks, with optional test-time adaptation
$PY scripts/inference/run_inference.py --method vanilla|bnstats|tent|cotta ...

# Paper analyses
$PY scripts/analysis/auroc_anomaly_detection.py        # Sec. V-C primary result (AUROC/FAR table)
$PY scripts/analysis/precision_isolation_ablation.py   # Sec. VI per-class isolation ablation
$PY scripts/analysis/run_temporal_generalization.py    # source-week model vs all weeks

# Tests: most are standalone scripts, some are pytest-style
$PY tests/test_coral.py                  # standalone (has __main__)
$PY -m pytest tests/test_configurable_mmcnn.py   # pytest-style
```

SLURM array jobs in `slurm_files/` parallelize the per-week work (e.g. `run_weekly_training.slurm` trains weeks 0–52; inference scripts shard with `--num-shards/--shard-id`).

## Architecture

Pipeline stages map to `scripts/` subdirectories: `data_prep/` → `train/` → `inference/` → `analysis/` → `viz/`. Library code lives in `config.py` (the only root module — a `Config` dataclass holding all hyperparameters, including DA lambdas), `data_utils/` (parquet/FlowPic dataloaders, `cesnet_labels.load_label_mapping` is canonical), `models/`, and `training/` (trainer + `domain_adaptation_methods/` with MMD/DANN/CORAL/BBSE).

Two model backbones:
- `models/configurable_cnn.py` — FlowPic (256×256) CNN, used by `train_per_week_cesnet.py`.
- `models/multimodal_cesnet.py` — `Multimodal_CESNET` (PPI Conv1d + flowstats MLP), the backbone for the UDA/TTA benchmark and Allot work.

The benchmark protocol: train one source model on a fixed week, evaluate it **frozen** on every later week with `data_sample_frac=0.1, seed=42` and shared input normalization (`normalization_stats.npz`), so per-week samples are identical across methods. Inference outputs go to `results/inference/<run_name>/` as per-week/window `.npz` files (true_labels, pred_labels, softmax, embeddings); analysis scripts consume those. Figures go to `figs/`; experiment checkpoints/tensorboard to `exps/<EXPERIMENT_NAME>/<week>/`.

## Datasets

- **CESNET-TLS-Year22** (primary): `/home/anatbr/dataset/CESNET-TLS-Year22/`, 53 weekly snapshots `WEEK-2022-00..52`, 180 classes, each week has `train.parquet`/`test.parquet` (70/30, seed = 42 + week_index).
- **CESNET-QUIC22** (secondary): week-44 source, 102 classes, 4 eval weeks.
- **Allot** (private, closed-world): windowed timeline via `data_utils/allot_timeline.py`; flowstats are derived from packet sequences by the loader (dim differs from CESNET's 44).

## Critical facts (easy to get wrong)

- **The TLS source week is WEEK-2022-01, not 00.** Every model scores highest on test-week-00 because week 00 is an easy week — do not infer the source week from peak accuracy.
- **Week 10 is a documented sensor artifact** (new ipfixprobe exporter skipping retransmitted packets), i.e. ground-truth covariate shift — not application drift. Because of it the paper uses **two reference weeks**: Week-1 (pre-artifact) for the decomposition / estimator-breakdown / 52-week monitoring analyses, and Week-16 (healthy regime, Macro F1 peaks 0.806) for clean-regime analyses (teleportation visualizations, label-shift robustness). The benchmark is likewise run with both week-1 and week-16 sources. Keep new experiments consistent with this split.
- Results-state ground truth lives in the root status markdown files (`PROJECT_STATUS.md`, `UDA_BENCHMARK_STATUS.md`, `*_HANDOFF.md`, `*_RESULTS.md`) and `paper/sota_benchmark_section.md`. Several are untracked working documents — check them before re-deriving numbers or re-running expensive jobs.
- **Related-work summaries live in `paper/litrature/SUMMARIES.md`** (34 thesis-focused per-paper notes, grouped by role — datasets, ETC models, drift evidence, label-shift/BBSE family, online label shift, global UDA/TTA baselines, open-set/continual learning, interpretability — each tagged Core/Supporting/Background). Each entry's Citation line ends with the `\bibitem` key in backticks. Consult these before writing related work or citing a paper, instead of re-reading the PDFs. PDFs sit alongside in `paper/litrature/` (named by citation key).
- `archive/` holds obsolete one-offs plus vendored cotta/tent code that is not run.
