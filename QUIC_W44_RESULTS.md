# CESNET-QUIC22 — week-44 multimodal model: temporal generalization & TTA

Train the multimodal model (`Multimodal_CESNET`: PPI conv1d + flowstats MLP) on the
**first** CESNET-QUIC22 week and evaluate how it generalizes to later weeks, comparing
no-adaptation vs test-time adaptation (TENT / CoTTA) vs test-time BatchNorm
recalibration (AdaBN). CESNET-QUIC22 has 4 weeks: **W-2022-44** (train) and 45/46/47
(drift targets).

---

## TL;DR findings

1. **The model drifts with time.** Vanilla accuracy falls from **0.899** (in-distribution
   wk44) to **~0.66** by wk47.
2. **Naive TTA does not help — and on average hurts.** TENT collapses on wk45
   (0.800 → 0.537); CoTTA over-adapts (worst in-distribution: 0.741).
3. **The cheap winner on drifted weeks is AdaBN** — just recomputing BatchNorm
   statistics on each test batch, *zero* gradient updates: **+6 pts** over vanilla on
   wk46/47, with no collapse.
4. **"TTA hurts in-distribution" was mostly an artifact:** for TENT it was the small
   inference batch (64); at batch 256 it lands on the no-adaptation BN floor. For CoTTA
   it's genuine over-adaptation (augmentation-averaged pseudo-labels under a high
   confidence gate).

---

## Setup

| | |
|---|---|
| Dataset | CESNET-QUIC22, prepared to TLS-Year22 schema (see `QUIC_W44_MULTIMODAL_HANDOFF.md`) |
| Prepared root | `/home/anatbr/students/noamshakedc/cesnet-quic22-prepared/` |
| Classes | 102 service apps (background classes excluded) |
| Train week | W-2022-44 — 21.0 M train / 9.0 M test flows (70/30 split) |
| Drift weeks | 45 (27.3 M), 46 (21.6 M), 47 (28.3 M) train flows |
| Model | `Multimodal_CESNET`, flowstats=44, ppi=(3,30); 30 epochs, batch 256, lr 3e-3, `--train_data_frac 0.2` |
| Best val acc (wk44) | **89.90 %** (epoch 25) |
| Inference | frozen wk44 model over all 4 weeks, `--data_sample_frac 0.1 --seed 42` (identical samples across methods) |
| Normalization | shared `normalization_stats.npz` (PPI (3,30) + flowstats (44,)) used by train & all inference |

**Methods compared** (all from the same wk44 checkpoint):
- `vanilla` — frozen model, `eval()` (trained BN running-stats).
- `bnstats` — **AdaBN**: BN uses test-batch stats, *no* gradient updates, dropout off.
- `tent` / `cotta` — entropy / consistency test-time adaptation (batch 64).
- `tent_bs256` / `cotta_bs256` — same, at batch 256 (matches training batch).

---

## Results

### Accuracy by week
| wk | vanilla | bnstats (AdaBN) | tent | cotta | tent bs256 | cotta bs256 |
|----|---------|---------|------|-------|-----------|-------------|
| 44 (in-dist) | **0.899** | 0.881 | 0.858 | 0.741 | 0.877 | 0.801 |
| 45 | **0.800** | 0.781 | 0.537 | 0.669 | 0.708 | 0.732 |
| 46 | 0.663 | **0.723** | 0.660 | 0.692 | 0.694 | 0.714 |
| 47 | 0.659 | **0.716** | 0.662 | 0.694 | 0.676 | 0.703 |
| **avg** | 0.755 | **0.775** | 0.679 | 0.699 | 0.739 | 0.738 |

Δ vs vanilla (avg acc): bnstats **+0.020**, tent −0.076, cotta −0.056, tent_bs256 −0.017, cotta_bs256 −0.018.

### Macro-F1 by week (102 classes, unweighted)
| wk | vanilla | bnstats | tent | cotta | tent bs256 | cotta bs256 |
|----|---------|---------|------|-------|-----------|-------------|
| 44 | **0.736** | 0.712 | 0.662 | 0.652 | 0.701 | 0.682 |
| 45 | **0.691** | 0.662 | 0.394 | 0.589 | 0.594 | 0.626 |
| 46 | **0.633** | 0.624 | 0.504 | 0.583 | 0.581 | 0.602 |
| 47 | **0.637** | 0.626 | 0.514 | 0.580 | 0.558 | 0.593 |
| **avg** | **0.674** | 0.656 | 0.519 | 0.601 | 0.609 | 0.626 |

> **Caveat:** AdaBN wins on *accuracy* for drifted weeks but is flat-to-slightly-worse on
> *macro-F1* (0.656 vs 0.674 avg) — it helps common classes, marginally hurts rare ones.

### Interpretation (why TTA hurts in-distribution)
- **BN-stat swap alone** (`bnstats` wk44 0.881 vs vanilla 0.899) costs only ~2 pts.
- **TENT @ bs256** (0.877) ≈ `bnstats` (0.881) → TENT's in-distribution harm was almost
  entirely **small-batch BN noise** (64 samples / 102 classes), not the entropy updates.
- **CoTTA** loses ~6 pts to batch size (0.741 → 0.801 @ bs256) but stays ~8 pts below
  `bnstats` even at bs256 → its **adaptation machinery** (augmentation-averaged teacher
  pseudo-labels under the `ap=0.9` gate + consistency drift) actively degrades an
  already-correct model.
- **On drift**, plain BN recalibration recovers most of the loss for free; full TTA
  mostly underperforms this trivial baseline (TENT even collapses on wk45).

Artifacts: `results/inference/quic_w44_summary/quic_w44_summary.json` +
`quic_w44_accuracy_by_week.png`.

---

## Reproduce

### One command (full pipeline, SLURM DAG)
```bash
bash slurm_files/submit_quic_pipeline.sh        # submits prep→stats→train→inference→summary
squeue -u $USER | grep -i quic                  # watch
tail -60 "$(ls -t logs/*quic_summary*.out | head -1)"   # final 6-method table
```
Everything runs via SLURM with `afterok`/`afterany` dependencies — no login-node jobs.
Wall-clock: prep ~30 min/week (parallel array), stats ~20 min, training ~4 h (30 ep,
requeues/resumes if preempted), inference 20 min–3 h/method (CoTTA slowest), summary <1 min.

### Stage-by-stage (if you want to run/resubmit one piece)
| Stage | Script | Partition | Notes |
|-------|--------|-----------|-------|
| Prep weeks 44-47 | `slurm_files/run_quic_prepare.slurm` | cpu-killable | array 0-3; idempotent |
| Normalization stats | `slurm_files/run_quic_norm_stats.slurm` | cpu-killable | writes `<root>/normalization_stats.npz` |
| Train (wk44) | `slurm_files/run_quic_multimodal_train.slurm` | killable (gpu) | → `exps/cesnet_quic22_multimodal_v01/WEEK-2022-44/weights/best_model.pth` |
| Inference: vanilla | `slurm_files/run_quic_inference.slurm` | killable (gpu) | |
| Inference: TENT/CoTTA | `slurm_files/run_quic_tta_inference.slurm` | killable (gpu) | array 0=tent, 1=cotta |
| Inference: AdaBN | `slurm_files/run_quic_bnstats.slurm` | killable (gpu) | `--method bnstats` |
| Inference: TTA @ bs256 | `slurm_files/run_quic_tta_bs256.slurm` | killable (gpu) | array; batch 256 |
| Summary (6-method) | `slurm_files/run_quic_tta_summary.slurm` | cpu-killable | table + JSON + plot |

Each inference job prints its own per-week accuracy at the end of its `.out` log and
writes `results/inference/quic_w44_inference{,_tent,_cotta,_bnstats,_tent_bs256,_cotta_bs256}/WEEK-2022-{44..47}.npz`
(each: `true_labels, pred_labels, softmax, embeddings`).

### Key source
- `scripts/data_prep/prepare_quic_week.py` — QUIC → TLS-schema converter (splits `PPI`,
  parses `PHIST_*`, adds `FLOW_ENDREASON_END`, 70/30 split, builds `label_mapping.json`).
- `scripts/inference/run_inference.py` — `--method {vanilla,bnstats,tent,cotta}`
  (dropout-mode fixes per `TTA_RERUN_HANDOFF.md`; `bnstats` = AdaBN baseline).
- `scripts/analysis/quic_tta_summary.py` — per-week acc + macro-F1, deltas, JSON + plot.

### Re-run just the analysis on existing `.npz`
```bash
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
$PY scripts/analysis/quic_tta_summary.py \
    --results_dir results/inference --out_dir results/inference/quic_w44_summary
```

---

## Notes / next levers
- AdaBN with a running momentum buffer (instead of per-batch) may stabilize macro-F1.
- CoTTA: lower `--cotta_lr`, lower `--ap` gate, smaller `--noise_std` — its aug machinery
  is the main in-distribution culprit.
- Fairness: all methods share `--seed 42 --data_sample_frac 0.1` → identical per-week
  evaluated samples (see `TTA_RERUN_HANDOFF.md` "no easy bugs" checklist).
- These docs (`QUIC_W44_*.md`) are untracked working-tree notes — commit if you want them kept.
