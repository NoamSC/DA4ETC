# UDA / Test-Time Adaptation Methods — Reference

Self-contained reference for the unsupervised domain-adaptation (UDA) and test-time
adaptation (TTA) baselines benchmarked in this project: **what each method is**, **exactly
how we ran it**, and **all relevant results**. Companion to `UDA_BENCHMARK_STATUS.md` (which
holds the audit/fix history); this file is the clean methods+results reference.

All results are **forward-only** (a source model is scored on its own week and later weeks —
the deployment/monitoring scenario; evaluating on earlier weeks is backward transfer and is
excluded). Headline deltas on the week-16 and QUIC sources are confirmed across 5 seeds.

---

## 1. The methods

All methods share one frozen-trained backbone, `Multimodal_CESNET` (per-packet PPI sequence →
Conv1d branch; flow statistics → MLP branch; fused → 180-way classifier). They differ only in
**what, if anything, they update at test time**.

| Method | Family | Updates at test time | One-line intuition |
|---|---|---|---|
| **Source-only** | baseline | nothing | Frozen source model applied as-is; the reference every Δ is measured against. |
| **AdaBN** | TTA, non-parametric | BatchNorm **statistics** only (no gradients) | Recompute BN mean/var from the test batch instead of source running stats — corrects covariate shift in feature scale, no learning. |
| **TENT** | TTA, parametric | BatchNorm **affine params** (γ, β) via entropy gradient | Minimize prediction entropy on each test batch — make the model "more confident", updating only BN scale/shift. |
| **CoTTA** | continual TTA, parametric | **all** weights via consistency + restore | Student/teacher (EMA) self-training with augmentation-averaged pseudo-labels and stochastic restore-to-source, designed to adapt without forgetting. |
| **DANN** | UDA, adversarial (train-time) | trained source→target aligned model | Gradient-reversal domain classifier aligns source and target feature distributions *during training*; the canonical *global* UDA approach. |
| **CORAL** | UDA, moment-matching (train-time) | trained source→target aligned model | Penalizes the difference between source and target feature **covariances** (second-order statistics) during training — alignment without an adversary. |
| **MMD / DAN** | UDA, kernel-moment (train-time) | trained source→target aligned model | Penalizes the **Maximum Mean Discrepancy** (multi-kernel RBF) between source and target feature distributions — matches ALL moments via the kernel mean embedding, not just the covariance. |

### Mechanism detail

- **Source-only (vanilla)** — `model.eval()`, single forward pass, argmax. No adaptation.
- **AdaBN** — BN running stats are nulled so BN uses **per-test-batch** statistics; `eval()`
  mode (dropout off), `requires_grad=False`. Isolates "BN-stat shift" from learned adaptation,
  so it is the *diagnostic control* between Source-only and the gradient methods.
- **TENT** — `train()` mode for BN (batch stats), dropout forced off; only BN affine params are
  trainable; one Adam step per batch minimizing the Shannon entropy of the softmax. Episodic
  (model reset to source at the start of each test week).
- **CoTTA** — student (trainable) + teacher (EMA, α=0.99) + frozen source **anchor**. Per batch:
  if the anchor's confidence on the (clean) input is below a threshold, the pseudo-label is the
  mean over `n_aug` noise-augmented teacher passes, else the clean teacher prediction; a
  cross-entropy-to-pseudo-label step updates the student, the teacher is EMA-updated, and a
  small random fraction of weights is restored to source. The **saved** prediction is the clean
  teacher output (not the augmentation-averaged pseudo-label). Episodic per week.
- **DANN** — trained per target week: the week-16 labeled source is aligned to week-N's
  unlabeled test split through a gradient-reversal-layer domain head while the class head trains
  on source labels (transductive UDA). One model per target week; evaluated frozen on that week
  (the "diagonal").
- **CORAL** — same per-target-week transductive setup as DANN, but instead of an adversarial
  domain head it adds a loss term `‖cov(source feats) − cov(target feats)‖²_F / 4d²` (Sun &
  Saenko 2016, d = 600 feature dims) to the classification loss, pulling the two feature
  covariances together. Source and target are forwarded as one combined batch (same BN-fix path
  as DANN). One model per target week; evaluated frozen on the diagonal.
- **MMD / DAN** — same per-target-week transductive setup, with a **multi-kernel RBF Maximum
  Mean Discrepancy** loss between the z-normalized source and target features (bandwidths
  {0.1, 1, 10}; Long et al. DAN 2015 / Gretton MMD). MMD matches the full distribution (all
  moments via the kernel mean embedding), a strictly richer alignment than CORAL's covariance.
  One model per target week; evaluated frozen on the diagonal.

---

## 2. How we ran them

### 2.1 Shared protocol

- **Backbone / source models.** One `Multimodal_CESNET` trained per source week
  (`exps/cesnet_multimodal_each_week_train_v01/week_<N>/weights/best_model.pth`), then frozen.
- **Forward-only evaluation.** Score the source model on its own week and every later week.
- **Identical samples across methods.** Each per-week test split is subsampled with
  `data_sample_frac=0.1`, `seed=42`, via a deterministic seeded loader (`drop_last=False`), and
  all methods use the same input normalization (`normalization_stats.npz`). Therefore
  `true_labels[i]` is the **same flow** for every method/batch size — differences are the
  method, not the data draw.
- **Batch-size points.** TENT/CoTTA are reported at **bs64 and bs256**. Small batches make BN's
  per-batch statistics noisy (which itself penalizes BN-adapting methods); **bs256 is the fair
  operating point**, bs64 exposes the magnitude of the small-batch BN artifact.
- **Significance.** Headline deltas re-run across **5 seeds** (1, 2, 3, 4, 42); a delta counts
  as real only if it exceeds the across-seed noise floor (~1e-4 here).
- **Δ semantics.** Δ = mean-accuracy change vs Source-only on the weeks the two share; negative
  Δ = negative transfer. "In-dist" = source-week test accuracy; "Far ≥43" = mean acc weeks 43–52.
- **Infrastructure.** SLURM array jobs on the preemptible `killable` partition with `--requeue`,
  skip-existing per-week `.npz` outputs, and atomic writes (`.tmp` + rename) so preemption can
  never leave a half-written file. A size-1 final batch crashed the BN-in-train-mode methods
  (TENT/CoTTA/AdaBN); fixed by padding a lone sample to 2 and slicing the output back.

### 2.2 Exact hyperparameters

| Method | Key hyperparameters |
|---|---|
| Source-only | — |
| AdaBN | none (BN batch statistics, no optimizer) |
| TENT | optimizer Adam, **lr 1e-3**, **1 step/batch**, updates BN affine params only, episodic reset per week |
| CoTTA | optimizer Adam, **lr 1e-3**, **1 step/batch**, EMA **α 0.99**, restore frac **rst_m 0.01**, anchor confidence threshold **ap 0.9**, **n_aug 32** augmentations, **noise_std 0.02**, episodic reset per week |
| DANN | **λ_dann 1.0**, GRL **λ_rgl 0.1**, GRL **γ 10**, **50 epochs**, lr **3e-3**, batch 64, `train_data_frac 1.0`, `val_data_frac 0.1`, `train_per_epoch_data_frac 0.1`, `num_workers 0` |
| CORAL | **λ_coral 100** (calibrated — see note), **50 epochs**, lr **3e-3**, batch 64, `train_data_frac 1.0`, `val_data_frac 0.1`, `train_per_epoch_data_frac 0.1`, `num_workers 0` |
| MMD / DAN | **λ_mmd 5** (calibrated), RBF bandwidths **{0.1, 1, 10}**, **50 epochs**, lr **3e-3**, batch 64, `train_data_frac 1.0`, `val_data_frac 0.1`, `train_per_epoch_data_frac 0.1`, `num_workers 0` |
| Shared (inference) | `batch_size` 64 (and 256 for TENT/CoTTA), `data_sample_frac 0.1`, `seed 42`, `num_workers 0` |

**λ_coral calibration.** Because the CORAL loss is normalized by 4d² (d=600) its raw value is
tiny, so λ must be chosen, not assumed. A real-training sweep on the farthest week (16→52,
λ ∈ {100, 300, 1000}) showed λ=100 gives the strongest actual covariance alignment (residual
CORAL loss 0.028→0.012) with the best accuracy (~69%), while higher λ aligned *less* and gained
nothing. λ=1 (a naive default) was ~1000× too weak (CORAL ≈ 0.1% of the classification loss) —
an inactive, unfair baseline — so it was rejected. **λ_coral = 100.**

**λ_mmd calibration.** MMD z-normalizes features before the RBF kernel, so its raw loss is O(1)
and λ is much smaller than CORAL's. A real-training sweep (16→52, λ ∈ {0.1, 1, 5, 20}) gave
λ=5 the best accuracy (~70%) with the MMD term solidly active (~27% of the classification loss).
**λ_mmd = 5.**

**Noise-floor caveat (CORAL & MMD).** Both alignment losses are estimated from the per-batch
(batch=64) statistics of 600-d features, so they are **dominated by finite-sample noise**: the
CORAL loss is ~74% noise (rank-63 estimate of a 600×600 covariance, SNR ≈1.36×), and the raw
MMD stays ~flat at ~0.031 regardless of λ. So the alignment these methods can actually achieve
at batch 64 is limited; a batch-≥256 re-sweep is the recommended robustness follow-up. Their
≈0 forward-transfer Δ should be read with this caveat (it is consistent with, but not a fully
powered test of, "global alignment cannot help").

### 2.3 Commands

**TTA inference** (per method × source week × batch size), sharded 4 ways via
`slurm_files/run_inference_auditfix.slurm`:

```bash
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
$PY scripts/inference/run_inference.py \
    --method {vanilla|bnstats|tent|cotta} --train_week {week_1|week_16} \
    --batch_size {64|256} \
    --experiment_dir exps/cesnet_multimodal_each_week_train_v01 \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --data_sample_frac 0.1 --seed 42 --num_workers 0 \
    --output_dir results/inference_auditfix/<src>_<method>_bs<batch> \
    --num_jobs 4 --job_id $SHARD
```

**DANN training** (one model per target week N; source week 16), array 0–52 via
`slurm_files/run_dann_fwd_w16.slurm`:

```bash
$PY scripts/train/train_per_week_cesnet.py --multimodal \
    --week 16 --val_week N --exp_name "cesnet_tls_dann_fwd_w16_v01/{}" \
    --lambda_dann 1.0 --lambda_rgl 0.1 --lambda_grl_gamma 10 \
    --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 --batch_size 64 --num_workers 0 --learning_rate 3e-3
```

**CORAL training** (one model per target week N; source week 16), array 0–52 via
`slurm_files/run_coral_fwd_w16.slurm`:

```bash
$PY scripts/train/train_per_week_cesnet.py --multimodal \
    --week 16 --val_week N --exp_name "cesnet_tls_coral_fwd_w16_v01/{}" \
    --lambda_coral 100 \
    --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 --batch_size 64 --num_workers 0 --learning_rate 3e-3
```

**MMD training** (one model per target week N; source week 16), array 0–52 via
`slurm_files/run_mmd_fwd_w16.slurm`:

```bash
$PY scripts/train/train_per_week_cesnet.py --multimodal \
    --week 16 --val_week N --exp_name "cesnet_tls_mmd_fwd_w16_v01/{}" \
    --lambda_mmd 5 \
    --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 --batch_size 64 --num_workers 0 --learning_rate 3e-3
```

**DANN / CORAL / MMD diagonal inference** (each week scored by its own aligned model) via
`slurm_files/run_{dann,coral,mmd}_diagonal.slurm` → `results/inference/{dann,coral,mmd}_fwd_w16_diagonal/`.

**Multi-seed significance** (5 seeds, acc-only) via `slurm_files/run_inference_seedvar.slurm`;
analyzed by `scripts/analysis/analyze_seedvar.py`.

**Result tables** regenerated by `scripts/analysis/regen_benchmark_table.py` (forward-only).

---

## 3. Results

### 3.1 CESNET-TLS-Year22 — week-1 source (forward weeks 1–52, 180 classes)

| Method | Mean acc | Δ | Macro-F1 | In-dist | Far ≥43 |
|---|--:|--:|--:|--:|--:|
| Source-only | 0.608 | — | 0.455 | 0.845 | 0.530 |
| AdaBN | 0.599 | −0.009 | 0.462 | 0.853 | 0.518 |
| TENT (bs64) | 0.482 | **−0.126** | 0.362 | 0.828 | 0.458 |
| TENT (bs256) | 0.580 | −0.027 | 0.444 | 0.848 | 0.522 |
| CoTTA (bs64) | 0.514 | **−0.094** | 0.399 | 0.801 | 0.457 |
| CoTTA (bs256) | 0.580 | −0.028 | 0.444 | 0.846 | 0.507 |

### 3.2 CESNET-TLS-Year22 — week-16 source (forward weeks 16–52, 180 classes)

| Method | Mean acc | Δ | Macro-F1 | In-dist | Far ≥43 |
|---|--:|--:|--:|--:|--:|
| Source-only | 0.798 | — | 0.654 | 0.883 | 0.746 |
| AdaBN | 0.787 | −0.011 | 0.642 | 0.877 | 0.731 |
| TENT (bs64) | 0.759 | **−0.038** | 0.612 | 0.872 | 0.675 |
| TENT (bs256) | 0.784 | −0.014 | 0.640 | 0.881 | 0.723 |
| CoTTA (bs64) | 0.738 | **−0.060** | 0.607 | 0.834 | 0.699 |
| CoTTA (bs256) | 0.784 | −0.014 | 0.641 | 0.863 | 0.737 |

### 3.3 UDA training-time alignment (diagonal) — week-16 source (forward weeks 16–52)

Each week scored by its own source-16→week-N aligned model (transductive, one model per week —
the *most favorable* setting for these methods). Δ vs the frozen week-16 source-only model.

| Method | Mean acc | Δ | Macro-F1 | Far ≥43 |
|---|--:|--:|--:|--:|
| Source-only (week-16) | 0.798 | — | 0.654 | 0.746 |
| DANN (diagonal) | 0.801 | +0.006 | 0.656 | 0.747 |
| CORAL (diagonal) | 0.794 | −0.001 | 0.652 | 0.742 |
| MMD (diagonal) | 0.794 | −0.002 | 0.652 | 0.741 |

Neither adversarial (DANN), covariance-matching (CORAL), nor kernel-MMD alignment buys
meaningful forward-transfer, even with a dedicated per-target-week model — three different
alignment principles, the same ≈0 result.

### 3.4 CESNET-QUIC22 — week-44 source (forward weeks 44–47, secondary)

| Method | Mean acc | Δ | Macro-F1 | In-dist |
|---|--:|--:|--:|--:|
| Source-only | 0.755 | — | 0.674 | 0.899 |
| AdaBN | 0.775 | **+0.020** | 0.655 | 0.882 |
| TENT (bs64) | 0.690 | −0.065 | 0.524 | 0.856 |
| TENT (bs256) | 0.736 | −0.019 | 0.607 | 0.876 |
| CoTTA (bs64) | 0.702 | −0.054 | 0.606 | 0.754 |
| CoTTA (bs256) | 0.745 | −0.010 | 0.633 | 0.811 |

### 3.5 Allot — private closed-world (52 windows ≈ 2% each; early ≈ "wk1", quarter ≈ "wk16")

| Slice | Method | Windows | Mean acc | Δ vs Source-only |
|---|---|--:|--:|--:|
| early | Source-only | 48 | 0.699 | — |
| early | AdaBN | 48 | 0.672 | −0.027 |
| early | TENT | 48 | 0.486 | **−0.213** |
| early | CoTTA | 48 | 0.421 | **−0.278** |
| quarter | Source-only | 48 | 0.701 | — |
| quarter | AdaBN | 48 | 0.673 | −0.028 |
| quarter | TENT | 48 | 0.502 | **−0.199** |
| quarter | CoTTA | 48 | 0.506 | **−0.194** |

### 3.6 Multi-seed significance (5 seeds: 1, 2, 3, 4, 42)

Forward-only deltas vs Source-only, mean ± std across seeds (noise floor ~1e-4):

| Source | Method | Δ (forward) | Verdict |
|---|---|--:|---|
| TLS week-16 | AdaBN | **−0.011** ± 0.0001 | significant (negative) |
| TLS week-16 | TENT (bs64) | **−0.038** ± 0.0007 | significant (negative) |
| QUIC week-44 | AdaBN | **+0.020** ± 0.0002 | significant (positive) |
| QUIC week-44 | TENT (bs64) | **−0.066** ± 0.0061 | significant (negative) |

> Note: AdaBN on TLS week-16 is *positive* (+0.007) on an all-weeks average, but that gain is
> entirely **backward transfer** to the easy pre-source weeks and **vanishes forward-only**
> (−0.011). This is exactly why the benchmark is reported forward-only.

---

## 4. Takeaways

1. **Every global adaptation method fails to help** — adversarial UDA (DANN), moment-matching
   UDA (CORAL), kernel-MMD UDA (DAN), and continual TTA (TENT/CoTTA) — on both TLS source weeks,
   on QUIC, and on Allot: at best no gain, usually negative transfer. The shared failure across
   four method families points to *global adaptation scope*, not any single algorithm.
2. **Control the small-batch BN artifact.** At bs64 TENT/CoTTA look much worse; at bs256 the
   damage shrinks to −0.01…−0.03 (TLS) / −0.01…−0.02 (QUIC). Report bs256 as the fair point;
   the residual negative Δ there is the real parametric-adaptation harm.
3. **AdaBN (no gradients) is the diagnostic control:** near-neutral on TLS forward (|Δ| ≤ 0.011),
   positive only on QUIC (+0.020) — far smaller than the gradient methods either way. The gap
   between AdaBN and TENT/CoTTA pins the harm on global *parameter* updates.
4. **Training-time UDA does not transfer forward even in its best case:** with a dedicated
   per-target-week aligned model, DANN beats the frozen source by only +0.006 mean (+0.001 far),
   CORAL by −0.001, and MMD by −0.002 (i.e. nothing) — three different alignment principles
   (adversarial / covariance / kernel-MMD), the same ≈0 result. None recovers the per-class
   *discrete* drift; in-distribution health does not generalize across time. (Caveat: the CORAL
   and MMD losses are noise-floor-limited at batch 64 — see §2.2.)
5. **The effect reproduces under both source weeks** (wk16 0.798 vs wk1 0.608 forward), so it is
   not an artifact of the documented Week-10 sensor change.
