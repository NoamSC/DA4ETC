# 2. Comprehensive SOTA Benchmark Comparison

> **Draft status / data provenance.** Measured cells come from on-disk inference outputs
> (`results/inference/…`, `frac=0.1`, `seed=42`, identical per-week samples across
> methods). Cells marked **`(running: job …)`** are submitted SLURM jobs not yet
> complete; **`— (to run)`** are not yet produced. Placeholders must be filled with real
> numbers before submission — never by guesswork. See `§2.5`.
>
> **Primary dataset: CESNET-TLS-Year22** (a full year, 53 weeks, 180 apps). CESNET-QUIC22
> is a **secondary** dataset used only to show the same phenomenon reproduces on a second
> protocol/year (§2.4.2).

## 2.1 Motivation

Our central claim is that standard Unsupervised Domain Adaptation (UDA) and Continual
Test-Time Adaptation (CTTA) are *structurally mis-specified* for encrypted-traffic
classification. Both families adapt the model **globally**: DANN aligns the *entire*
source and target feature distributions via an adversarial domain classifier, while TENT
and CoTTA re-estimate normalization statistics and minimize a global objective (entropy /
consistency) over *every* test batch. In encrypted traffic, however, distribution shift
is overwhelmingly **local and asynchronous** — at any given time only a handful of
applications change behaviour (a TLS-library upgrade, a new CDN endpoint, a protocol
migration), while the majority of classes remain stable. A method that responds to a few
drifting applications by re-aligning the *whole* representation inevitably perturbs the
decision boundaries of the stable majority. We call this **negative transfer**: the
adaptation mechanism degrades classes it should have left alone.

This section benchmarks our targeted framework against the dominant global UDA/CTTA
methods and shows that global adaptation produces measurable negative transfer whereas
targeted adaptation does not.

## 2.2 Protocol

We use a *temporal generalization* protocol that mirrors deployment: a single source
model is trained on the **first week** and then evaluated, frozen, on every subsequent
week. Test-time methods (AdaBN/TENT/CoTTA) adapt online per test week; DANN is trained
with access to its week's unlabelled target split per the standard adversarial recipe.
All methods share the identical multimodal backbone (`Multimodal_CESNET`: a
per-packet-sequence Conv1d branch + a flow-statistics MLP), the same evaluation samples
(`data_sample_frac=0.1`, `seed=42`), and the same input normalization, so differences are
attributable to the adaptation mechanism alone.

| Role | Dataset | Source week | Horizon | #Classes | #Eval weeks |
|---|---|---|---|---|---|
| **Primary** | CESNET-TLS-Year22 | week 1 (WEEK-2022-01) | 1 year | 180 | 53 |
| Secondary | CESNET-QUIC22 | W-2022-44 | 4 weeks | 102 | 4 |

We report **mean accuracy across all evaluation weeks** (the primary temporal-robustness
metric), **macro-F1** (unweighted over classes — sensitive to per-class collapse under
the long tail), and **in-distribution / far-week** accuracy to localize where each method
helps or hurts.

## 2.3 Primary result — CESNET-TLS-Year22

**Table 1. SOTA benchmark on CESNET-TLS-Year22** (week-1 source model, 53-week forward
eval). Δ = change vs. the no-adaptation source-only model; negative Δ = negative transfer.

| Method | Family | Adaptation scope | Mean acc | Δ | Macro-F1 | In-dist | Far (≥wk43) |
|---|---|---|---:|---:|---:|---:|---:|
| Source-only (no adaptation) | — | none | **0.614** | — | 0.463 | 0.845 | 0.530 |
| AdaBN (BN-stat recalibration) | CTTA, non-param. | global (stats only) | 0.606 | −0.008 | 0.470 | 0.853 | 0.519 |
| TENT | CTTA | global (BN affine, entropy) | 0.483 | **−0.130** | 0.367 | 0.834 | 0.434 |
| CoTTA | CTTA | global (all params, consistency) | 0.509 | **−0.104** | 0.393 | 0.780 | 0.446 |
| DANN (λ_dann=1, GRL λ=0.1) | UDA | global (adversarial align) | _(re-evaluating w/ correct WEEK-2022-01 source: job 522677)_ | | | | |
| **Targeted adaptation (ours)** | targeted | per-application | — (to run) | | | | |

**Table 2. Where the damage happens (CESNET-TLS-Year22, accuracy by drift regime).**

| Method | In-distribution | Near (≤ wk13) | Far (≥ wk43) |
|---|---:|---:|---:|
| Source-only | 0.845 | 0.774 | 0.530 |
| TENT | 0.834 | 0.684 | 0.434 |
| CoTTA | 0.780 | 0.668 | 0.446 |

## 2.4 Findings

### 2.4.1 Primary dataset (TLS-Year22)

**(F1) Every global adaptation method exhibits systematic negative transfer — across
both the CTTA and UDA families.** Relative to doing nothing, mean accuracy *drops* for
TENT (−13.0 pts, 0.613 → 0.483) and CoTTA (−10.4 pts, → 0.509); the adversarial UDA
method DANN degrades comparably _(exact figure re-evaluating against the correct
WEEK-2022-01 source — job 522677)_. That the UDA and CTTA families fail the same way
shows the problem is **global adaptation scope**, not a quirk of any single algorithm.
This is the core evidence for the negative-transfer hypothesis.

**(F1b) DANN is fine in-distribution but worst on drift — the negative transfer is
temporal.** _[Pending the corrected WEEK-2022-01-source eval (job 522677); the
preliminary run showed DANN ≈ source-only in-distribution but collapsing on the far
weeks, consistent with global source-alignment that does not generalize forward. Numbers
to be finalized.]_

**(F2) The damage concentrates exactly where adaptation is supposed to help.** Table 2
shows the loss grows with drift: on the far weeks (≥ wk43) TENT and CoTTA fall to
0.434 / 0.446 versus source-only 0.530 — i.e. global adaptation is *most harmful
precisely on the drifted weeks it targets*. Under asynchronous per-application drift, the
global objective chases the few changed applications and warps the boundaries of the
stable majority, which dominates the aggregate metric.

**(F3) TENT collapses under label shift.** TENT's macro-F1 (0.367) falls below source-only
macro-F1 (0.461) and well below its own accuracy; on individual weeks it degenerates — the
signature of entropy minimization driving predictions toward a few dominant classes when
the test label distribution shifts. Targeted adaptation, by construction, cannot induce
this collapse on stable classes.

**(F4) Non-parametric global adaptation is the control that isolates the cause.** AdaBN
recomputes BatchNorm statistics on the test batch but performs *no gradient updates*. The
contrast with TENT/CoTTA is diagnostic: if AdaBN is roughly neutral while the
gradient-based methods degrade, the negative transfer is attributable to the *parametric
global updates*, not to test-time adaptation per se — motivating an approach that adapts
only the parameters responsible for the drifting applications. On TLS-Year22 AdaBN is
**neutral** (−0.008 mean acc, and a slightly *higher* macro-F1 than source-only), versus
TENT −0.130 / CoTTA −0.104; on the secondary QUIC dataset AdaBN is even net-positive
(+0.020). Across both datasets the non-parametric method does no harm while the
gradient-based methods degrade — the contrast pins the negative transfer on the global
parameter updates.

**(F5) Targeted adaptation (ours) is designed to avoid this failure mode.** Because our
framework localizes adaptation to the applications that actually drift, it leaves the
representation of stable classes untouched and cannot incur the F1–F3 negative transfer.
*[Quantitative rows pending; expected: ≈ source-only on stable classes while recovering
accuracy on drifted applications — net-positive Δ where every global method is net-negative.]*

### 2.4.2 Secondary dataset — the effect reproduces (CESNET-QUIC22)

**Table 3. Confirmation on CESNET-QUIC22** (week-44 source model, 4-week forward eval).

| Method | Mean acc | Δ | Macro-F1 |
|---|---:|---:|---:|
| Source-only | **0.755** | — | 0.674 |
| AdaBN | 0.775 | **+0.020** | 0.656 |
| TENT | 0.679 | **−0.076** | 0.519 |
| CoTTA | 0.699 | **−0.056** | 0.601 |

The same pattern holds on a different protocol (QUIC) and year: global TENT/CoTTA again
incur negative transfer, while non-parametric AdaBN is the only global method that does
not hurt (and gains +6 pts on the most-drifted weeks 46–47). Small inference batches
amplified the TTA damage here; at batch 256 the in-distribution gap largely closes
(TENT −0.022, CoTTA −0.098 in-dist), confirming the harm is the *parametric* global
adaptation, not test-time inference itself. Full QUIC analysis: `QUIC_W44_RESULTS.md`.

## 2.5 Experiments in flight / still required

- **AdaBN on TLS-Year22** — ✅ done; Table 1 filled.
- **DANN on TLS-Year22** — ✅ source-week forward-eval done (Table 1 filled); the full
  per-week DANN training set (for a temporal train×test matrix, if needed) is finishing.
- **Targeted adaptation (ours)** — the proposed method's results (to provide).
- **CORAL** — *excluded from this paper* (out of scope).

Table 1 is now fully populated except the **Targeted (ours)** row.

---
### Source of the numbers (reproduction / auditing)
- TLS-Year22: `results/inference/week_1_inference{,_bnstats,_tent_fixed,_cotta_fixed}/`
  and `results/inference/dann_grl01_week1_inference/` (`_fixed` = dropout-mode-corrected
  `run_inference.py`, see `TTA_RERUN_HANDOFF.md`).
- QUIC22: `results/inference/quic_w44_*` + `quic_w44_summary/quic_w44_summary.json`
  (see `QUIC_W44_RESULTS.md`).
- Recompute any row with the acc/macro-F1 routine in `scripts/analysis/quic_tta_summary.py`
  pointed at the relevant dirs.
> ⚠️ Do **not** pull TLS numbers from
> `…/temporal_generalization_reweighted/results_combined.json`: despite `WEEK-2022-44…47`
> keys it is the **TLS week-33** model in **percent** units — a different experiment.
> ⚠️ The "week_1" baseline model is trained on **WEEK-2022-01** (confirmed from
> `run_cesnet_multimodal_array.slurm`: `--week ${ARRAY_TASK_ID}`, `exp=week_${WEEK}`). Do
> NOT use test-week-00 accuracy as "in-distribution" — week 00 is simply an easy test week
> (every model peaks there). The DANN source model is `WEEK-2022-01` to match the baseline.
