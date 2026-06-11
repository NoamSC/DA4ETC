---
name: research-review
description: Review a change, script, or experiment for EXPERIMENTAL-VALIDITY problems, not just code bugs — train/test leakage, preprocessing/normalization mistakes, evaluation-protocol violations, seed/sample inconsistency, metric misuse under label shift, and the da4etc-specific pitfalls (source-week, week-10 artifact, frozen-source benchmark). Use after finishing something important, before trusting a number, or before putting a result in the paper. Complements /code-review (which finds code bugs); this finds research bugs.
---

# research-review

A research result can be *code-correct* and still *scientifically wrong*. This skill hunts the second kind: silent validity bugs that inflate or distort results. Run it on a diff, a single script, an experiment dir, or a number that's about to go in the paper.

## How to run

1. Determine scope: the current `git diff` (default), a named script under `scripts/`, an experiment under `exps/`/`results/`, or a specific claim/number the user names.
2. Read the relevant code paths end-to-end — data loading → preprocessing → split → model → eval → metric. Don't skim; leakage hides in the seams between stages.
3. Work the checklist below. For each finding give: **severity** (critical / high / medium / low), the exact `file:line`, *why it biases the result*, and the fix.
4. End with a short verdict: is the result trustworthy as stated, or must something be re-run?

## Checklist

### Data leakage (most important)
- **Normalization/standardization fit on the wrong data.** Stats must come from **train only** (or the shared frozen `normalization_stats.npz`), never refit on test, never on train+test together. Confirm the same frozen stats are used across every compared method.
- **Train/test contamination.** Same flows/sessions/time-windows appearing in both splits; per-week split must use the documented `seed = 42 + week_index`. Watch for dedup that runs after the split.
- **Target/feature leakage.** A feature that encodes the label or future information (e.g. a flowstat derived using post-hoc knowledge). For TTA, confirm test labels are NOT used to adapt (only unsupervised signals are allowed).
- **Selection on the test set.** Hyperparameters, early-stopping, thresholds (e.g. AUROC/FAR operating points), or week choices tuned on the same data they're evaluated on.

### Preprocessing / sampling consistency
- The benchmark requires **identical per-week samples across methods**: `data_sample_frac=0.1`, `seed=42`, shared normalization. Any method compared with a different frac/seed/normalization is not comparable — flag it.
- Label mapping must be the canonical `cesnet_labels.load_label_mapping`; a re-derived mapping can silently permute classes.
- Class set / dimensionality mismatches (e.g. Allot flowstats dim ≠ CESNET's 44) handled explicitly, not broadcast away.

### Evaluation protocol
- **Frozen source.** Benchmark models are evaluated frozen on later weeks — confirm no accidental grad/BN update except where the TTA method legitimately requires it (bnstats/TENT/CoTTA update specific params only).
- **Metric choice under label shift.** Accuracy is misleading when class priors move; macro-F1 is the protocol metric. Check the reported metric matches the claim.
- **Estimand correctness.** For BBSE / label-shift / entropy-residual results, confirm the confusion matrix / weights are estimated on the right split and the residual is BBSE-corrected (the paper's core contribution) — not raw entropy.

### Reproducibility & stats
- Seeds set for numpy/torch/cuda where results depend on them; reported as such.
- Multi-week / multi-class comparisons: is a headline number cherry-picked from one week, or is the trend robust? Note variance, not just the peak.

### da4etc-specific traps (from CLAUDE.md "Critical facts")
- **TLS source week is WEEK-2022-01, not 00.** Every model peaks on test-week-00 because it's an easy week — a result that "discovers" week-00 as best is just re-confirming this, not a finding.
- **Week-10 is a sensor artifact** (ipfixprobe exporter change), i.e. ground-truth covariate shift, not application drift. Treating week-10 as drift, or including it in clean-regime analyses, is a bug. The paper uses two reference weeks: **Week-1** (decomposition/monitoring) and **Week-16** (clean-regime). Check the analysis uses the right one.
- Results ground truth lives in the root status markdowns and `paper/sota_benchmark_section.md` — cross-check a number against those before trusting a freshly computed one.

## Output

Group findings by severity, lead with anything **critical** (a result that must be re-run or retracted). If nothing is wrong, say so plainly and state what you checked — a clean bill from this skill is a claim, so make it falsifiable by listing the checks that passed.
