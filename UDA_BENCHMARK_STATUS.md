# UDA / CTTA Benchmark Status — "how SOTA adaptation (doesn't) work on this task"

_Generated from on-disk inference outputs (`frac=0.1`, `seed=42`, identical per-week/window
samples across methods). All numbers are real (validated `np.load`, no 0-byte stubs).
Last updated 2026-06-14._

> ## ⚠️ VALIDITY AUDIT (2026-06-14) — the tables below are UNDER REVIEW, do not cite yet
>
> A per-method audit found a real defect in **every** row of the benchmark. Three code
> fixes have been applied (working tree, not yet committed); the affected runs must be
> regenerated before the numbers can be trusted. Severity ranges from "magnitude inflated"
> to "result is meaningless."
>
> ### 🔧 UPDATE 2026-06-15 — DANN root cause found (BatchNorm); earlier DANN framings RETRACTED
>
> The real DANN bug was **BatchNorm + separate source/target forward passes**:
> `train_one_epoch` forwarded source then target as two separate `train()`-mode passes, so
> BN normalized each batch by its OWN mean/var and **erased the covariate shift** before the
> domain classifier — pinning domain accuracy at exactly 50% / loss 2·ln2 **even though the
> covariate shift is real**. Fix (`training/trainer.py`): forward source+target as ONE
> combined batch so BN sees the mixed distribution. **Verified**: on a fresh test (wk16→wk44,
> wk16→wk52) domain-classifier accuracy jumped from 50% to **67–71%** (dann_loss 1.1 < 1.386).
>
> **Retractions:**
> - The **"no global covariate gap"** conclusion is **WITHDRAWN** — it was this BN bug.
>   Covariate shift exists; DANN trains normally once source+target share a BN batch.
> - The **normalization-artifact** hypothesis and the norm/no-norm **contrast experiment**
>   are **WITHDRAWN / moot** (no-norm also showed 50% because BN, not input normalization,
>   caused it).
> - The "Resolved: transductive `--val_week`" note below still holds for the *target-week*
>   choice, but `--val_week` alone was **not sufficient** — the BN fix was the missing piece.
>
> The buggy sweep was preserved as `exps/cesnet_tls_dann_fwd_w16_v01_BUGGY_preBNfix`; a clean
> 52-target re-run (BN fix + `num_workers=0`) is in flight (job 551646, convergence drain).
> DANN accuracy numbers in the tables below are stale pending that re-run + diagonal inference.
>
> ### Findings (all independently verified against on-disk artifacts/code)
>
> 1. **🔴 DANN — adversarial training never happened (result invalid).** The old
>    `run_dann_train.slurm` passed no `--val_week`, so `train_per_week_cesnet.py` defaulted
>    the DANN **target domain to the source week's own test split** — a random 70/30 split
>    of the *same* week (seed 42+week), i.e. no domain gap. The week-16 tensorboard
>    (`exps/cesnet_tls_dann_grl01_v01/WEEK-2022-01/`) confirms it empirically:
>    `Accuracy/domain_classifier` = **49.9–50.1%** and `Loss/dann` = **1.3877 ≈ 2·ln2**
>    (random binary), flat for all 50 epochs. The reversed gradient into the feature
>    extractor is ≈0, so DANN ≈ a source-only model with dead heads. The DANN row and the
>    "global alignment doesn't generalize forward" narrative built on it are unsupported.
>
> 2. **🔴 CoTTA — saved predictions are augmentation-averaged, not clean.** In
>    `run_inference.py` `_cotta_step`, when the confidence gate fires (`anchor_conf < ap`,
>    `ap=0.9` — fires on ~every 180-class sample), the returned `class_preds` was
>    `ema_logits` = mean of `n_aug` **noise-augmented** teacher passes. That augmented
>    average is a self-training pseudo-LABEL, not the prediction; reporting it injected
>    augmentation noise into every evaluated sample and dropped accuracy ~6.5 pts even on
>    CoTTA's own source week, where adaptation is a no-op.
>
> 3. **🔴 CoTTA — episodic reset silently fails to restore BN buffers.** `_configure_cotta`
>    sets `running_mean = running_var = None`, which *deregisters* those buffers; the next
>    week's `load_state_dict(original_state, strict=False)` then silently skips the source
>    running-stats keys. The **anchor** (`deepcopy(model)` for the confidence gate) was
>    therefore built on test-batch stats, not source stats, on every week except the first
>    per SLURM shard → results depended on shard position.
>
> 4. **🟠 Shared TTA harness — non-deterministic eval batching (TENT + CoTTA + AdaBN).**
>    The eval loader (`temporal_generalization.py:create_week_loader`, multimodal branch)
>    used `shuffle=True` with no generator + `drop_last=True`. `seed=42` fixed only *which
>    rows* are sampled, not batch composition/order. Since these methods normalize on
>    per-batch BN stats, each prediction depended on randomized batch-mates →
>    (a) the protocol's "`true_labels[i]` is the same sample across methods" invariant was
>    false for batch composition; (b) `drop_last` discarded a different tail per run/batch
>    size, so **bs64 and bs256 rows weren't scored on the same population** — the
>    bs64-vs-bs256 story was confounded with BN-stat noise.
>
> 5. **🟡 AdaBN / TENT method code is sound** (AdaBN: eval + no-grad; TENT: BN-affine-only
>    optimizer). Their only problem is finding #4. But: AdaBN's headline deltas (wk16
>    +0.007, QUIC +0.020) are **smaller than run-to-run noise** → not reproducible, sign
>    could flip — a problem since AdaBN is the paper's "≈neutral control." TENT's bs64
>    negative-transfer **magnitude** is inflated (noisy 64-sample BN stats over 180 classes
>    replacing trained running stats); direction is defensible, magnitude is not.
>
> ### Fixes applied (working tree)
>
> - **Deterministic, full-population eval loader.** Added a `generator` param to
>   `create_parquet_loader` (`data_utils/cesnet_dataloader.py`); `create_week_loader`
>   (`scripts/analysis/temporal_generalization.py`) now passes a seeded
>   `torch.Generator().manual_seed(seed)` and `drop_last=False`. Batch composition is now
>   randomized (representative BN stats) **and** identical across methods and batch sizes;
>   the full population is scored everywhere.
> - **CoTTA clean prediction.** `_cotta_step` now saves `std_ema` (the clean un-augmented
>   teacher forward, already computed) as `class_preds`; augmentation stays strictly in the
>   pseudo-label/loss path.
> - **CoTTA anchor reset.** New `_restore_bn_buffers()` re-registers the nulled BN buffers
>   before the anchor's `load_state_dict`, so the confidence gate uses source stats on
>   every week regardless of shard position.
> - **Valid forward-transfer DANN — one-flag fix, no code change.** The cross-week DANN
>   capability already existed and was already used (e.g. `cesnet_v11_dann_search_13_to_33`,
>   dirs `WEEK-2022-13_val_WEEK-2022-33`, domain acc ~96% = real alignment). The benchmark
>   row was broken *only* because `run_dann_train.slurm` omitted `--val_week`. New
>   `slurm_files/run_dann_fwd_w16.slurm` simply passes `--week 16 --val_week N` (source
>   week 16 aligned to week N via the existing, proven path), one model per target week
>   (the DANN-row diagonal). No trainer changes. _(An earlier draft added a
>   `--dann_target_week` arg for unsupervised selection + a disjoint train-split target;
>   reverted as redundant — `--val_week` matches the prior methodology.)_
>
> ### Re-runs required before the tables are valid (NOT yet launched — pending review)
>
> - **All TTA rows** (AdaBN/TENT/CoTTA, both source weeks + QUIC + Allot): re-run
>   `run_inference.py` with the fixed loader; CoTTA additionally needs the prediction +
>   reset fixes. Then re-attribute the bs64-vs-bs256 framing on a clean, common population.
> - **DANN:** `sbatch slurm_files/run_dann_fwd_w16.slurm`, then a per-week-model inference
>   pass that loads `WEEK-2022-16_val_WEEK-2022-NN` and scores week NN (the diagonal).
>   `run_inference.py` needs a small "per-target-week model" mode for this — **not yet
>   built** (deferred until the retrain is underway).
>
> ### Resolved: DANN target = transductive `--val_week` (matches prior methodology)
>
> Decided to use the existing `--val_week` path: source week 16 aligned to week N's **test
> split** (the same split scored downstream — transductive UDA, as in the prior
> `13→33`/`33→40` runs), with model selection on that split. This is optimistic *in DANN's
> favor* (it sees the target's unlabeled data and is selected on it), which only
> strengthens a "DANN shows negative transfer anyway" claim. The alternative (disjoint
> train-split target + source-only selection) was implemented then reverted as redundant.

Mirrors the §V SOTA matrix (DANN / TENT / CoTTA + AdaBN control). **CORAL: cut** (out of
scope, per decision). **Targeted adaptation (ours): not produced** (separate framing).

## Protocol
Single source model trained on one week, evaluated frozen on every other week.
AdaBN/TENT/CoTTA adapt online per test week (episodic, reset per week); DANN is a trained
source model (gradient-reversal domain head, `lambda_rgl=0.1, gamma=10`) forward-evaluated
with its class head. Shared multimodal backbone (`Multimodal_CESNET`: PPI Conv1d + flowstats
MLP). Δ = mean accuracy change vs the no-adaptation source-only model on shared weeks;
negative Δ = negative transfer. "In-dist" = the source week; "far" = mean of weeks 43–52.

CESNET-TLS-Year22 is run with **two source weeks** because of the documented Week-10
monitoring-infrastructure change: **week 1** (pre-artifact) and **week 16** (post-artifact,
healthy regime).

---

## CESNET-TLS-Year22 — week-1 source (53 weeks, 180 classes)

| Method | Family | Scope | Mean acc | Δ | Macro-F1 | In-dist | Far ≥43 |
|---|---|---|--:|--:|--:|--:|--:|
| Source-only | — | none | **0.613** | — | 0.461 | 0.845 | 0.530 |
| AdaBN | CTTA, non-param | global stats | 0.605 | −0.009 | 0.469 | 0.853 | 0.519 |
| TENT (bs64) | CTTA | global, entropy | 0.483 | **−0.130** | 0.367 | 0.834 | 0.434 |
| TENT (bs256) | CTTA | global, entropy | 0.585 | −0.028 | 0.450 | 0.847 | 0.522 |
| CoTTA (bs64) | CTTA | global, consistency | 0.509 | **−0.104** | 0.393 | 0.780 | 0.446 |
| CoTTA (bs256) | CTTA | global, consistency | 0.572 | −0.041 | 0.437 | 0.833 | 0.498 |
| DANN | UDA | global, adversarial | 0.545 | **−0.069** | 0.420 | 0.863 | 0.429 |

## CESNET-TLS-Year22 — week-16 source (53 weeks, 180 classes)

| Method | Family | Scope | Mean acc | Δ | Macro-F1 | In-dist | Far ≥43 |
|---|---|---|--:|--:|--:|--:|--:|
| Source-only | — | none | **0.759** | — | 0.622 | 0.883 | 0.746 |
| AdaBN | CTTA, non-param | global stats | 0.766 | **+0.007** | 0.626 | 0.876 | 0.731 |
| TENT (bs64) | CTTA | global, entropy | 0.713 | −0.046 | 0.580 | 0.872 | 0.677 |
| TENT (bs256) | CTTA | global, entropy | 0.752 | −0.007 | 0.617 | 0.880 | 0.721 |
| CoTTA (bs64) | CTTA | global, consistency | 0.689 | −0.070 | 0.568 | 0.817 | 0.688 |
| CoTTA (bs256) | CTTA | global, consistency | 0.746 | −0.012 | 0.610 | 0.853 | 0.731 |
| DANN | UDA | global, adversarial | 0.747 | −0.012 | 0.617 | 0.882 | 0.728 |

## Allot — private (52 ~2% windows; early≈"wk1", quarter≈"wk16"; closed-world)

| Slice | Method | Windows | Mean acc | Δ vs vanilla |
|---|---|--:|--:|--:|
| early | Source-only | 48 | 0.699 | — |
| early | AdaBN | 48 | 0.672 | -0.027 |
| early | TENT | 48 | 0.486 | **-0.213** |
| early | CoTTA | 48 | 0.421 | **-0.278** |
| quarter | Source-only | 48 | 0.701 | — |
| quarter | AdaBN | 48 | 0.673 | -0.028 |
| quarter | TENT | 48 | 0.502 | **-0.199** |
| quarter | CoTTA | 48 | 0.506 | **-0.194** |

(All methods now at 48/48 windows.)

## CESNET-QUIC22 — week-44 source (4 weeks, secondary) — from QUIC_W44_RESULTS.md

| Method | Mean acc | Δ | Macro-F1 |
|---|--:|--:|--:|
| Source-only | 0.755 | — | 0.674 |
| AdaBN | 0.775 | **+0.020** | 0.656 |
| TENT (bs64) | 0.679 | −0.076 | 0.519 |
| TENT (bs256) | 0.739 | −0.017 | 0.609 |
| CoTTA (bs64) | 0.699 | −0.056 | 0.601 |
| CoTTA (bs256) | 0.738 | −0.018 | 0.626 |

---

## Findings

1. **Every gradient-based global adaptation method shows negative transfer**, across UDA
   (DANN) and CTTA (TENT/CoTTA), on both TLS source weeks and on Allot. The shared failure
   across method families points to *global adaptation scope*, not any single algorithm.

2. **The small-batch BN artifact is large and must be controlled.** At batch 64 TENT/CoTTA
   look catastrophic (−0.10 to −0.13 on wk1), but most of that is small-batch BatchNorm
   noise: at batch 256 the damage shrinks to −0.01…−0.04 (TLS) and ≈−0.02 (QUIC). Report
   bs256 as the fair operating point; the residual negative Δ at bs256 is the real
   parametric-adaptation harm.

3. **AdaBN (non-parametric) is the diagnostic control**: ≈neutral or net-positive on CESNET
   (wk1 −0.009, wk16 **+0.007**, QUIC **+0.020**), mildly negative on Allot (−0.03). The
   gap between AdaBN (no grad) and TENT/CoTTA (grad) pins the harm on global *parameter*
   updates, motivating targeted per-application adaptation.

4. **The effect reproduces under both source weeks.** The healthier week-16 source has
   higher absolute accuracy (0.759 vs 0.613) and milder—but still present—negative transfer,
   confirming the phenomenon is not an artifact of the week-10 sensor break.

5. **DANN** is fine in-distribution (in-dist ≈ source-only or higher: 0.863/0.882) but worst
   on the far weeks (0.429 wk1), i.e. global source-alignment does not generalize forward —
   the negative transfer is temporal.

## Label-shift estimator variants of the isolation detector (2026-06-11)

All are classical label-shift correctors: each estimates only the target label marginal
P(Y) under the assumption that P(X|Y) is invariant (zero covariate shift) — which is the
shared failure mode the residual weaponizes. Implementations: repo BBSE + the `abstention`
library (Kundaje lab) with `cvxpy` for RLLS, both installed in `ml2`.

- **BBSE (trunc. pinv)** — Lipton'18 black-box shift estimation: inverts the *hard-prediction*
  confusion matrix C^T w = q via truncated pseudo-inverse (rcond=1e-2); the paper's "Ours".
- **BBSE-soft** — same inversion but on the *soft* (expected-softmax) confusion matrix; uses
  the same trunc. pinv because abstention's unregularized solve is singular at 180 classes.
- **RLLS** — Azizzadenesheli'19: BBSE recast as ℓ2-regularized constrained least squares
  (cvxpy QP), trading a little bias for a well-conditioned weight estimate.
- **SLD-EM / MLLS** — Saerens'02: EM fixed point that reweights the model's softmax by the
  current prior estimate until the implied target prior converges; uses probabilities, not a
  confusion matrix.
- **MLLS + BCTS** — Alexandari'20: same EM, but on bias-corrected temperature-scaled softmax
  (calibration fitted once on the reference week); the "hard-to-beat" calibrated variant.
- **Naive (uncorrected)** — no estimator: entropy residual vs frozen reference composition;
  upper bound on clean data, collapses under benign label shift (panel B/C).

Clean-data per-class detection AUROC (panel-A protocol):

| Detector            | wk-1 ref | wk-16 ref |
|---------------------|---------:|----------:|
| Naive (uncorrected) |    0.942 |     0.967 |
| RLLS                |    0.912 |     0.935 |
| SLD-EM / MLLS       |    0.904 |     0.934 |
| MLLS + BCTS         |    0.904 |     0.934 |
| BBSE-soft           |    0.899 |     0.903 |
| BBSE (ours)         |    0.889 |     0.903 |

Under INJECTED extreme label shift (panel-B protocol: clean weeks resampled to LogUniform
prior tilts; negatives = healthy classes under shift; figure
`panels/fig_isolation_A_roc_estimators_labelshift.png`):

| Detector            | wk1 1000× | wk16 1000× |
|---------------------|----------:|-----------:|
| Naive (uncorrected) |     0.806 |      0.739 |
| BBSE (ours)         | **0.891** |      0.825 |
| BBSE-soft           | **0.910** |  **0.829** |
| RLLS                |     0.897 |      0.816 |
| SLD-EM / MLLS (=+BCTS) |  0.846 |      0.784 |

Week-level Table I (auroc_anomaly_detection.py, ref wk0, viral regime — published rows
reproduce exactly): BBSE 0.987 (FAR@95 4.6%), BBSE-soft 0.988 (6.5%), RLLS 0.982 (7.4%),
MLLS+BCTS 0.953 (17.6%), SLD-EM 0.946 (25.0%), uncorrected 0.959 (25.0%), MFWDD 0.677.

**Takeaway:** estimator choice barely matters on clean data (all within ~0.03), and under
contamination the ordering FLIPS to favor the confusion-matrix family: naive collapses
(0.94→0.81, 0.97→0.74), BBSE/BBSE-soft are nearly invariant, RLLS close behind, EM-family
clearly weaker (SLD-EM ties uncorrected FAR@95 under viral spikes). Estimation accuracy and
detection robustness dissociate: in Fig-7 terms the EM family is the *most accurate*
estimator under pure label shift (L1 3-4e-4 vs BBSE 9e-4 at s=4) yet the *least robust*
detector. BBSE stays the right "ours"; BBSE-soft is a hair better and is the only defensible
swap. BCTS calibration is a no-op here (temperature ≈ 1; source model well-calibrated).
Caveat: wk-16 ground truth counts pre-artifact weeks 0–9 as degraded (reverse side of the
week-10 sensor change). Paper figs regenerated with variants: Fig 8 (+SLD-EM, +MLLS+BCTS),
Fig 10 (+RLLS ρ=-0.994 — ties BBSE; +SLD-EM ρ=-0.991), Fig 7 (+SLD-EM, +MLLS+BCTS),
Fig 11/Table I (+4 detector rows). Week-16 mirror (figs/ref16/fig_mirror_effect.png):
RLLS -0.932, BBSE -0.927 ≈ oracle -0.926, uncorrected -0.925 — but **SLD-EM FLIPS to
ρ=+0.62**: EM absorbs the covariate-shift entropy inflation into a hallucinated label
shift and goes quiet exactly where the model is broken (invisible on the wk-1 ref where
EM scored -0.991). Definitive: keep BBSE as "ours"; EM-family is disqualified as the
residual's estimator. Forward-only wk16 mirror (figs/ref16/fig_mirror_effect_fwd.png,
weeks 17-52): BBSE -0.988, RLLS -0.992 ≈ oracle -0.991 — and SLD-EM flat at +0.03, so
EM fails even in the clean teleportation regime. Mirror under EXTREME injected shift
(figs/ref16/fig_mirror_extreme_fwd_{10x,100x}.png, 1 draw/week, seed 42): correction
value grows with severity — uncorrected -0.983→-0.938(10×)→-0.845(100×) vs BBSE
-0.988→-0.951→-0.892 (≥ oracle at both severities); RLLS ties BBSE; EM far behind.
NOTE: Fig 8 & Fig 11/Table I use reference week 0 but their captions claim Week-1 —
fix the captions, not the runs.

Script: `scripts/analysis/isolation_estimator_variants.py`; outputs in `figs/isolation/`
(wk-1, reuses base ablation cache) and `figs/isolation_w16/` (wk-16, self-computed gt);
figure `panels/fig_isolation_A_roc_estimators.png` + `estimator_variants_metrics.json`.

## Provenance (output dirs)
- TLS wk1: `results/inference/week_1_inference{,_bnstats,_tent_fixed,_tent_bs256,_cotta_fixed,_cotta_bs256}/`, `dann_w01src_inference/`
- TLS wk16: `results/inference/week_16_inference{,_bnstats,_tent_fixed,_tent_bs256,_cotta_fixed,_cotta_bs256}/`, `dann_w16src_inference/`
- Allot: `exps/allot_multimodal/{early,quarter}_eq/{inference,inference_bnstats,inference_tent,inference_cotta}/`
- QUIC: `results/inference/quic_w44_*`
- Recompute: load each `.npz` (`true_labels`,`pred_labels`), mean per-week acc + sklearn macro-F1, Δ on shared weeks.

## Dataset label-frequency bias & the undo recipe (2026-06-11)

CESNET-TLS-Year22 was recorded with **dynamic per-service sampling** that deliberately
flattens the label distribution (Hynek et al. 2024, "Flow sampling"): services sorted by
traffic volume — top 5% (9/180) kept 1:15, middle 35% (63) at 1:9→1:2 by prevalence,
bottom 60% (108) unsampled; the final uniform 1:10 is proportion-neutral. Consequence:
dataset label proportions UNDERSTATE real network label shift, so any axis derived from
label priors (weekly shift magnitudes, Fig 7's "natural weekly range" band, synthetic
shift magnitudes) is compressed.

**Undo:** `reconstruct_sampling_weights()` in `archive/plot_paper_figures.py` rebuilds the
per-service factors w from `stats-dataset.json` (rank → band factor; iterate rank↔factor
to a fixed point since downsampling can reorder ranks at band boundaries; exact dynamic
ratios are unpublished → approximation, footnote it in the paper). Then
`p_network ∝ p_dataset ⊙ w`.

**Recipe for any proportion-based graph:** map EVERY proportion vector (train prior,
weekly priors, estimator outputs, realized synthetic priors) through the same
`f(p)=p⊙w/Σ` BEFORE computing distances — never remap a dataset-space error. The static
prior stays on the y=x diagonal in any space. It is a pure re-projection: no model
reruns, only replots (Fig 7's `_vec` cache stores per-draw prior vectors for this).

**Scope:** label-proportion quantities only. P(X|Y), per-class F1, confusion matrices and
the teleportation analyses are unaffected by class-level subsampling. Network-mix
aggregate F1 would instead need importance-weighted evaluation (`reweighted_macro_f1`).

Same rework switched Fig 7's axes to **TV distance in % of traffic** (0.5·L1 — "share of
traffic that changed application"), replacing the unintuitive per-class MAE.

**Result (2026-06-12, both Fig 7 versions regenerated, per-draw vectors cached in
`..._em_vec.npz` so further tweaks are free replots):** the conclusion is invariant to the
correction. As recorded: natural range 5.3–30.4% TV, max induced 99.4% (3.3×); at s=4
prior/naive/BBSE/RLLS/EM/BCTS = 87.2/14.2/8.2/5.2/3.7/2.3%. Debiased: natural 4.4–28.6%,
max 99.3% (3.5×); s=4 = 85.2/17.1/7.5/6.3/2.5/1.7%. Identical method ordering in both
spaces (~11× estimator-vs-prior gap) → the flattened-sampling objection is neutralized
without changing any claim. Figures: `figs/fig_estimation_robustness_severity_ref16.png`
and `..._debiased.png`.
