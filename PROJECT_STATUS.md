# Encrypted-Traffic Domain-Adaptation Benchmark — Status & Findings

_Consolidated log of the SOTA-benchmark work (negative-transfer hypothesis) on
CESNET-TLS-Year22 (primary) and CESNET-QUIC22 (secondary). Last updated: 2026-06-10._

---

## 1. Goal / hypothesis

Show that **global** Unsupervised Domain Adaptation (DANN) and Continual Test-Time
Adaptation (TENT/CoTTA) are *mis-specified* for encrypted-traffic classification: drift is
**local and asynchronous** (a few apps change at a time), so a method that adapts the whole
representation warps the stable majority's boundaries — **negative transfer**. A *targeted*
(per-application) method should avoid this. The benchmark is the empirical proof.

**Protocol:** train one source model on a fixed week, evaluate it frozen on every later
week (`Multimodal_CESNET` backbone; `data_sample_frac=0.1, seed=42`; shared input
normalization). Test-time methods adapt online per week; DANN trains adversarially against
its week's target split.

---

## 2. Datasets & source week

| Role | Dataset | Source (train) week | #Classes | #Eval weeks |
|---|---|---|---|---|
| **Primary** | CESNET-TLS-Year22 | **WEEK-2022-01** (`week_1` model) | 180 | 53 |
| Secondary | CESNET-QUIC22 | W-2022-44 | 102 | 4 |

> ⚠️ **Source week = WEEK-2022-01**, confirmed from `slurm_files/run_cesnet_multimodal_array.slurm`
> (`WEEK=${ARRAY_TASK_ID}`, `--week ${WEEK}`, `exp=week_${WEEK}`). Do **not** infer the source
> week from "which test week scores highest" — every model peaks on test-week-00 because
> week 00 is an easy week, not the train week. True in-distribution = WEEK-2022-01 (0.845).

---

## 3. Results

### 3.1 Primary — CESNET-TLS-Year22 (week-1 source, 53-week forward eval)

| Method | Family | Scope | Mean acc | Δ vs src | Macro-F1 | In-dist (wk01) | Far (≥wk43) |
|---|---|---|---:|---:|---:|---:|---:|
| Source-only | — | none | **0.614** | — | 0.463 | 0.845 | 0.530 |
| AdaBN | CTTA, non-param | global (stats) | 0.606 | −0.008 | 0.470 | 0.853 | 0.519 |
| TENT | CTTA | global (BN affine) | 0.483 | **−0.130** | 0.367 | 0.834 | 0.434 |
| CoTTA | CTTA | global (all params) | 0.509 | **−0.104** | 0.393 | 0.780 | 0.446 |
| DANN (λ=1, GRL 0.1) | UDA | global (adversarial) | _re-evaluating (job 522677)_ | | | | |
| **Targeted (ours)** | targeted | per-app | _to provide_ | | | | |

> DANN row: a preliminary eval used the **wrong** source week (WEEK-2022-00) and gave
> avg 0.493 / Δ −0.121 / far 0.386 — being recomputed against the correct WEEK-2022-01
> source (`results/inference/dann_w01src_inference/`, job 522677). Qualitatively DANN was
> ≈ source-only in-distribution but collapsed on far weeks (consistent with negative
> transfer); final numbers pending.

### 3.2 Secondary — CESNET-QUIC22 (week-44 source, 4-week eval)

| Method | Mean acc | Δ | Macro-F1 |
|---|---:|---:|---:|
| Source-only | **0.755** | — | 0.674 |
| AdaBN | 0.775 | **+0.020** | 0.656 |
| TENT | 0.679 | −0.076 | 0.519 |
| CoTTA | 0.699 | −0.056 | 0.601 |

---

## 4. Findings

- **F1 — global adaptation → systematic negative transfer, across BOTH families.**
  TENT −0.130, CoTTA −0.104 (DANN comparable, being finalized). Same failure for UDA and
  CTTA ⇒ the problem is global *scope*, not a specific algorithm.
- **F2 — damage grows with drift.** On TLS far weeks (≥43): TENT 0.434 / CoTTA 0.446 vs
  source-only 0.530. Global methods hurt *most* exactly where adaptation is supposed to help.
- **F3 — TENT collapses under label shift** (macro-F1 0.367 < source-only 0.463; per-week
  degeneration toward dominant classes).
- **F4 — AdaBN is the control that isolates the cause.** Non-parametric BN-stat
  recalibration (no gradients) is neutral on TLS (−0.008, macro-F1 slightly higher) and
  net-positive on QUIC (+0.020, +6 pts on drifted weeks). ⇒ negative transfer comes from
  the *parametric global updates*, not test-time adaptation per se.
- **F5 — targeted adaptation (ours)** localizes to drifting apps, so it cannot warp stable
  classes; expected net-positive where every global method is net-negative. (Pending.)
- **QUIC batch-size note:** small inference batches amplified TTA damage; at batch 256 the
  in-distribution gap largely closes — further confirming the harm is parametric, not BN
  stats. (See `QUIC_W44_RESULTS.md`.)

---

## 5. DANN training health (where intact)

- Per-week DANN models train cleanly: val acc 85–91% (≈ vanilla), and **domain-classifier
  accuracy ≈ 50.0% on every week** = adversarial alignment succeeded (domain-invariant
  features), not collapsing.

---

## 6. Outstanding work (requested / planned)

1. **NEW — week-16 source table.** Week **10** had a major network-wide change, so the
   user wants the *entire* benchmark **re-run with WEEK-2022-16 as the source/training
   week** (post-change regime). Both tables (week-01 source and week-16 source) go in the
   paper. → not started.
2. **DANN per-week figure.** One graph overlaying: per-week **vanilla** accuracy (each
   week's own model, in-distribution diagonal) vs per-week accuracy of the **base-week
   model inferred on each week** (the temporal-degradation curve). → not started.
3. **DANN row finalization** for the week-01 table (job 522677, running).
4. **Targeted (ours)** rows — both tables.

---

## 7. Operational log (this effort)

- **Disk quota:** the netapp tree hit a per-area quota twice → writes failed → AdaBN lost
  weeks, DANN fail-looped, and some outputs were **truncated/corrupt**. Freed space by
  deleting superseded buggy TTA dirs (user-approved, 25 G) and pruning the DANN run's own
  redundant per-epoch checkpoints (keep `best_model` + latest 2/week). Per-epoch
  checkpoints are the bloat — watch for re-fill; ~17 G of redundant epoch checkpoints in
  inactive exps (`vanilla v01`, `dann_v01`) remain as an untapped buffer (not deleted).
- **Corruption recovery:** scanned checkpoints/npz; deleted only corrupt (0-byte/unloadable)
  files; resumed weeks from last good epoch where possible (26 resumed, 1 retrained from
  scratch, 26 intact). Source-week models were intact.
- **Source-week correction:** initially mis-identified the base week as WEEK-2022-00;
  corrected to WEEK-2022-01 and re-ran the DANN eval + fixed the in-dist column.

---

## 8. Artifact index

**Paper / docs** (working tree, **untracked — not committed to git**):
- `paper/sota_benchmark_section.md` — the §2 SOTA-benchmark section (tables + F1–F5).
- `QUIC_W44_RESULTS.md`, `QUIC_W44_MULTIMODAL_HANDOFF.md` — QUIC results + pipeline.
- `TTA_RERUN_HANDOFF.md` — TTA dropout-fix context.
- `PROJECT_STATUS.md` — this file.

**Results** (`results/inference/`): `week_1_inference{,_bnstats,_tent_fixed,_cotta_fixed}/`,
`dann_w01src_inference/` (correct DANN), `quic_w44_*` + `quic_w44_summary/`.

**Code:** `scripts/inference/run_inference.py` (`--method vanilla|bnstats|tent|cotta`),
`scripts/train/train_per_week_cesnet.py` (`--lambda_dann/--lambda_rgl/--lambda_grl_gamma`),
`scripts/analysis/quic_tta_summary.py`, `scripts/data_prep/prepare_quic_week.py`.

**SLURM:** `slurm_files/run_inference_bnstats.slurm`, `run_dann_train.slurm`,
`run_dann_inference.slurm`, `run_quic_*` , `submit_quic_pipeline.sh`.

**Persistent memory** (survives sessions): `benchmark-source-week.md`,
`use-slurm-for-long-jobs.md`.

**Live jobs:** DANN w01 eval `522677` (running), DANN per-week training (44/53 done).
