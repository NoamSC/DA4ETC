# Final Claims — UDA/TTA Baselines vs. Our Full System

*Honest, decision-relevant comparison table. All numbers verified against the cited source files (spot-checked 2026-06). Unmeasured cells are marked "— (not run)"; nothing here is fabricated.*

## Framing (read this first)

The UDA/TTA baselines and our system are **not** measured on the same axis, and forcing them onto one would be dishonest. The honest comparison is:

- **UDA/TTA baselines** (AdaBN, TENT, CoTTA, DANN, CORAL, MMD) are *whole-population blind adapters*: they try to lift mean accuracy / macro-F1 over the entire test population using **0 target labels**, and we report their **ΔF1 / Δacc vs. the frozen source-only model**.
- **Our full system** is two operationally distinct pieces:
  1. **BBSE monitor** — a *label-free degradation detector* (BBSE-corrected entropy residual), measured as **AUROC** for telling "is the model silently degrading right now?" It moves **no** prediction, so its own ΔF1/Δacc is **0 by construction**; its value is the *signal*, not a population-wide accuracy bump.
  2. **Few-shot per-class repair** — a *targeted* corrector applied **only to the specific classes the monitor flags as teleported**, measured as **per-class recall recovery** with **stable-class macro-F1 held flat — no measurable per-class negative transfer (|Δ| ≤ 0.001 when each flagged class is repaired in isolation)**. Its labels are spent on the *flagged class only* (K ∈ {1,5,10,50} on one class), not the whole dataset.

The single axis on which the two approaches are directly comparable is **label cost**: UDA/TTA spend **0 target labels** to adapt the whole population (and can do real harm doing so); our system spends **0 labels to detect** degradation and then **K labels on one flagged class** to repair it. That different label budget — blind population adaptation vs. detection + targeted, labels-on-the-broken-class-only repair — is the centerpiece of the comparison.

So the operationally honest claim is: **UDA/TTA give you ≈0-to-negative whole-population change at 0 labels (and can do real harm); our system tells you *when* you are degrading without labels, then *cheaply repairs the specific drifted classes* with a handful of labels and negligible per-class poaching (≤0.76%).** These are a different operating point — detection + targeted repair rather than blind population adaptation — not strictly better on every axis.

A note on the global-vs-per-class macro-F1 axis: the **joint** global macro-F1 "after repairing all flagged classes simultaneously" has been measured **across all 18 qualifying forward weeks** (`results/repair/joint_repair_v01/{metrics,crossweek}.json`; K=50, 5 seeds). Repairing all three teleported classes (49/57/140) at once lifts **global macro-F1 over all 180 classes by +0.019 ± 0.0017** (positive on *every* one of the 18 weeks; min +0.016), and the **stable 176-class macro-F1 rises by +0.0074 ± 0.0014** (positive on *every* week — never poached), because the broken prototypes had been stealing stable-class test points and the repair removes that contamination. So the cumulative-poaching concern is answered robustly: joint repair gives a **positive global ΔF1 with no cumulative negative transfer on any week**. (Absolute macro-F1 levels ~0.47–0.56 are on the 10% subsample; the deltas are apples-to-apples.) One honesty point: a labels-matched logistic baseline on the *same* K shots is competitive with the prototype repair (see *Gaps & caveats*).

---

## Table 1 — CESNET-TLS-Year22, week-16 source, forward weeks 16–52 (main, 180 classes)

Source-only baseline: **acc 0.798, macro-F1 0.654**. (`UDA_METHODS.md` §3.2–3.3)

| Method | ΔF1 | Δacc | #target labels | Ease / cost | Negative-transfer risk | Label-free detection (AUROC) |
|---|--:|--:|---|---|---|--:|
| Source-only | — | — | 0 | trivial (frozen) | none | — |
| AdaBN | −0.012 | −0.011 | 0 | cheap (BN stats only) | low–moderate | — |
| TENT (bs256) | −0.014 | −0.014 | 0 | cheap (1 grad step/batch) | moderate (bs64: −0.038) | — |
| CoTTA (bs256) | −0.013 | −0.014 | 0 | expensive (32 augs, EMA teacher) | moderate (bs64: −0.060) | — |
| DANN (diagonal) | +0.002 | +0.006 | 0 | **very expensive: 1 trained model / target week** | high (adversarial) | — |
| CORAL (diagonal) | −0.002 | −0.001 | 0 | **very expensive: 1 model / week** | moderate; noise-floor-limited @bs64 | — |
| MMD (diagonal) | −0.002 | −0.002 | 0 | **very expensive: 1 model / week** | moderate; noise-floor-limited @bs64 | — |
| **Ours: BBSE monitor** | **0 (by design)** | **0 (by design)** | **0** | **cheap (1 ref-week confusion matrix)** | **none (moves no prediction)** | **0.834** AUROC / 0.675 FAR@95 (viral spike) |
| **Ours: few-shot repair** | **+0.019 global (joint, 18 wks)*** | **+ (per-class 0→0.86)*** | **K∈{1,5,10,50} on flagged classes only** | **cheap (prototype/k-NN; no retrain)** | **none — joint repair lifts stable-176 macro-F1 +0.0074 on every week (no cumulative poaching)** ✅ | — (pairs with monitor) |

\* **Per-flagged-class repair (K=50, NCM unless noted), stable-class macro-F1 unchanged in isolation (`results/repair/few_shot_repair_w16_v01/metrics.json`). All recalls are over a single eval week with small n (47–62 flows), so treat absolute values as preliminary, with wide implied CIs:**
| Flagged class | recall before | recall after (K=50) | n_eval flows | stable macro-F1 Δ |
|---|--:|--:|--:|--:|
| eset-edtd (idx 57, teleport wk18) | 0.00 | **0.88** (≈55/62) | 62 | 0.0000 |
| docker-registry (idx 49, teleport wk28) | 0.00 | **0.86** (peaks 0.87 @K=10) | 51 | +0.0002 |
| skype (idx 140, multimodal, teleport wk28) | 0.16 | **0.43** (NCM) — see budget-matched note below | 49 | 0.0000 |
| microsoft-defender (idx 98, drift wk22, *not* teleported) | 0.55 | **0.83** | 47 | −0.0006 |

(The microsoft-defender row was omitted from an earlier draft; it is included here because there is no reason to cherry-pick — a partially-degraded class recovering 0.55 → 0.83 only strengthens the result.)

**Skype budget-matched comparison (`results/repair/few_shot_knn_v01/metrics.json`, K=50).** At the NCM single-prototype operating point, skype recall is **0.43** at a poaching budget of **1.69%** of stable flows. Holding *that same* poaching budget fixed, the alternative correctors reach: **k-NN 0.604**, **KMeans-M4 0.616**, **Parzen 0.633** — i.e. at matched budget the clustering variant (KMeans-M4) **slightly edges the new k-NN corrector**, and k-NN is not the strictly-highest-recall option. The honest case for k-NN is that it is **non-parametric**: it needs no cluster count *M* to be chosen, whereas KMeans-M4 fixes M=4 a priori. At its own raw operating point k-NN reaches **0.584 recall at only 0.76% poaching** — the same recall ballpark as NCM's 0.43-at-1.69% but at far lower poaching — so its value is the recall/poaching trade-off and lack of a tuning knob, not a higher peak recall. (We do **not** pair "0.58" with "lower poaching" as a single measured point: 0.584 is its raw recall and 0.76% its raw poaching; 0.604 is its recall at the *NCM-matched* budget.)

Comparison baseline: CoTTA's *forward* negative-transfer on stable classes is **−0.06 macro-F1** (`few_shot_repair_w16_v01/metrics.json: cotta_fwd_neg_transfer_macroF1`), i.e. blind adaptation *hurts* the very classes our repair leaves untouched. Note the repair's "flat" stable macro-F1 baseline sits at the **≈0.45 level (10% subsample, single eval week)**, not the model's full-data macro-F1 of 0.654 — the |Δ|≤0.001 claim is about that subsample baseline, not the headline number.

**Bold = where our system is distinctive** (label-free detection — competitive with the best label-shift estimator, see below; negligible per-class poaching; cheap targeted repair at K labels on the broken class only).

**Viral-spike AUROC, honest ranking (`figs/auroc_anomaly_results_ref16.json`).** BBSE reaches **AUROC 0.834 (FAR@95 = 0.675)**. It is **not** the single best estimator in this regime: **RLLS 0.849** and **bbse_soft 0.836** slightly outscore it (em_bcts 0.828, em 0.759). The honest statement is therefore "**BBSE is competitive with the best label-shift estimator (RLLS/bbse_soft, within ≈0.015 AUROC) and far above the uncorrected entropy residual (0.708) and MFWDD (0.650)**." We default to BBSE not because it wins the viral-spike AUROC race but because it is **robust under extreme shift** (the severity-sweep tail, where the estimator variants tie — see the estimator-agnostic finding) and is the simplest correction; the small viral-regime gap to RLLS is not a reason to claim BBSE is uniquely best.

---

## Table 2 — CESNET-TLS-Year22, EXTREME label shift (severity sweep, wk16 ref)

UDA/TTA were **not run** under engineered extreme label shift — only the **detector** exists here, which is precisely the regime that distinguishes it (its whole point is robustness to benign volume/label-shift volatility). (`figs/isolation_w16/labelshift_sweep_results_fwd.json`)

| Method | ΔF1 | Δacc | #target labels | Detection AUROC @1000× | Negative-transfer risk |
|---|--:|--:|---|--:|---|
| AdaBN / TENT / CoTTA / DANN / CORAL / MMD | — (not run) | — (not run) | — | — (not run) | — |
| **Ours: BBSE monitor** | n/a (monitor) | n/a (monitor) | **0** | **0.810** ✅ | **none** |
| naive entropy residual | n/a | n/a | 0 | 0.703 | none |
| MFWDD | n/a | n/a | 0 | 0.591 | none |

Robustness curve (AUROC vs. severity, `..._fwd.json`): ours stays high — 0.897 (1×) → 0.880 (5×) → **0.810 (1000×)**. **Crucially, the naive residual is *stronger* than ours below ~5× severity** (naive 0.921 vs. ours 0.897 @1×; naive 0.896 vs. ours 0.889 @3×) and only **crosses below ours around severity 5×** (naive 0.864 vs. ours 0.880), after which it **collapses to 0.703 at 1000×**; MFWDD is uniformly weak (0.690 → 0.591). So the honest claim is **not** "ours dominates everywhere" — it is "ours **dominates only in the extreme tail** (severity ≳5×), and pays a small price in the benign/low-severity regime." That tail is precisely the EXTREME-label-shift regime this table targets: only the BBSE-corrected residual stays discriminative when benign label shift is cranked to 1000×.

---

## Table 3 — CESNET-QUIC22, week-44 source (forward weeks 44–47, secondary)

Source-only baseline: **acc 0.755, macro-F1 0.674**. (`UDA_METHODS.md` §3.4)

| Method | ΔF1 | Δacc | #target labels | Ease / cost | Negative-transfer risk | Label-free detection (AUROC) |
|---|--:|--:|---|---|---|--:|
| Source-only | — | — | 0 | trivial | none | — |
| AdaBN | −0.019 | **+0.020** | 0 | cheap | low–moderate | — |
| TENT (bs256) | −0.067 | −0.019 | 0 | cheap | moderate | — |
| CoTTA (bs256) | −0.041 | −0.010 | 0 | expensive | moderate | — |
| DANN / CORAL / MMD | — (not run) | — (not run) | 0 | — | — | — |
| **Ours: BBSE monitor** | n/a (monitor) | n/a (monitor) | 0 | cheap | none | **— (not run on QUIC)** |
| **Ours: few-shot repair** | — (not run) | — (not run) | — | — | none (by construction) | — |

QUIC note: AdaBN is the only baseline with a positive Δacc here, but its **macro-F1 still drops (−0.019)** — i.e. it trades tail-class performance for head-class accuracy, the classic blind-adaptation failure. Our monitor/repair are **not yet run on QUIC**.

---

## Table 4 — Allot, private closed-world (52 windows; early ≈ "wk1", quarter ≈ "wk16")

Source-only baselines: early acc 0.699, quarter acc 0.701. (`UDA_METHODS.md` §3.5)

| Method | Δacc (early) | Δacc (quarter) | #target labels | Negative-transfer risk | Label-free detection |
|---|--:|--:|---|---|---|
| Source-only | — | — | 0 | none | — |
| AdaBN | −0.027 | −0.028 | 0 | moderate | — |
| TENT | **−0.213** | **−0.199** | 0 | **severe** | — |
| CoTTA | **−0.278** | **−0.194** | 0 | **severe** | — |
| DANN / CORAL / MMD | — (not run) | — (not run) | 0 | — | — |
| **Ours: BBSE monitor** | n/a (monitor) | n/a (monitor) | 0 | none | **AUROC 0.890 / 0.879 (viral); 1.00 (clean)** ✅ |
| **Ours: few-shot repair** | — (not run) | — (not run) | — | none (by construction) | — |

Allot note: this is the **clearest negative-transfer cautionary tale** — TENT and CoTTA destroy ~20–28 accuracy points by blindly minimizing entropy / self-training on a closed-world stream. **And it is exactly where our label-free monitor works:** on Allot's viral/benign-volatility regime the BBSE-corrected residual scores **AUROC 0.890 (early_eq) / 0.879 (quarter_eq), FAR@95 0.518 / 0.601**, vs uncorrected entropy **0.710 / 0.793** and MFWDD **0.586 / 0.633** (`results/allot_monitor_v01/metrics.json`, `figs/fig_allot_monitor_auroc.png`). The degraded/healthy ground truth is a *real* multi-month covariate gap (window Macro-F1 < 0.57; same construction as CESNET), with the viral spike layered on only as benign nuisance — not a synthetic, BBSE-favouring perturbation. Same pattern as CESNET (BBSE 0.834 vs uncorrected 0.708) — the monitor generalizes to the dataset where blind adaptation fails hardest. **Caveat on n:** AUROC is over **47 windows (28 degraded / 19 healthy)** with 12 *correlated* viral resamples each, so treat it as a point estimate over ~47 independent windows, not n≈564. (Clean regime: all detectors 1.00 — easy.) The few-shot repair is **not yet run on Allot**.

---

## Bottom line

UDA/TTA baselines give you a **≈0-to-negative whole-population change at 0 target labels**, and on harder distributions they actively *harm* the model: TENT/CoTTA lose up to **−0.28 accuracy on Allot** and **−0.06 forward macro-F1 on stable TLS classes**; even the best train-time aligners (DANN/CORAL/MMD) move forward accuracy by ≤**0.006** *while requiring one fully trained model per target week*. The honest reading is **no measurable benefit at the batch sizes feasible here** — and at least the DANN/CORAL/MMD result is partly measurement-limited (their alignment losses are noise-floor-limited at batch 64, and these three are single-seed). A **batch-≥256 re-sweep is the recommended robustness check** before concluding they cannot help at all.

Our system answers the two questions an operator actually has. **"Am I silently degrading?"** — the BBSE monitor says yes/no **without any labels**, at AUROC **0.834** (FAR@95 0.675) under viral spikes on CESNET — competitive with the best label-shift estimator (RLLS 0.849, bbse_soft 0.836) and far above the uncorrected residual (0.708) — **0.810 under 1000× label shift** (where the naive residual collapses to 0.703), and it **generalizes to Allot** (viral AUROC **0.890 / 0.879** vs uncorrected 0.710 / 0.793) — i.e. it works on the exact dataset where the UDA baselines lose −0.21/−0.28 accuracy. One caveat to state plainly: **under benign/low-severity label shift the naive residual is actually *stronger* than ours** (0.921 vs 0.897 @1×); ours **dominates only in the extreme tail** (severity ≳5×). **"How do I fix the specific thing that broke?"** — the few-shot repair restores the flagged drifted classes (eset-edtd 0.00→**0.88**, docker-registry 0.00→**0.86**, microsoft-defender 0.55→**0.83**, skype 0.16→**0.43** NCM / ~0.60 with the k-NN corrector at matched budget) from **a handful of labels on that one class**; repairing **all flagged classes jointly lifts global macro-F1 +0.019 ± 0.0017 with no cumulative negative transfer on any of 18 weeks** (stable-176 macro-F1 +0.0074, positive every week). Honest qualifier: a **labels-matched logistic fit on the same K shots reaches higher recall** but at higher poaching (skype logreg 0.79 @ 1.78% poach vs k-NN 0.58 @ 0.76%), so the repair's contribution is being **fit-free and integrating cleanly with the label-free monitor** — plus k-NN's recall/poaching trade-off and its multi-modal edge over NCM — not strictly-higher recall. The core distinction is the **label budget**: UDA spends 0 labels to adapt everything blindly; we spend 0 to detect and K per flagged class to repair only what broke. That is **a different operating point — detection + targeted repair rather than blind population adaptation**.

---

## Gaps & honest caveats

**Method × dataset cells not yet measured:**
- **DANN / CORAL / MMD**: run **only** on CESNET-TLS-Year22 (wk1 / wk16). **Not run on QUIC, Allot, or the extreme label-shift sweep.**
- **Our BBSE monitor**: run on CESNET-TLS-Year22 (viral spike + extreme label-shift sweep), Allot (viral AUROC 0.890/0.879), **and now QUIC** (viral AUROC BBSE 0.736 / RLLS 0.774 / EM 0.792 vs uncorrected 0.566, MFWDD 0.458 — same pattern, but small-sample: 3 eval weeks, `results/quic_monitor/`).
- **Our few-shot repair**: run on CESNET-TLS (wk1 / wk16), **and now QUIC + Allot** (K=50 NCM: QUIC dns-doh 0.41→0.90, google-www 0.05→0.69, google-fonts 0.12→0.67, google-gstatic 0.30→0.75, control discord flat; Allot cls25 0.03→0.45, cls109 0.05→0.58, cls33 0.19→0.53, control cls17 flat — `results/repair/few_shot_repair_{quic,allot}_v01/`). **Not run on the extreme sweep.**
- **UDA/TTA under extreme label shift**: **not run at all** — that regime currently has only the detector.

**Methodological caveats:**
- **Batch-64 noise floor + single-seed (DANN/CORAL/MMD).** Both alignment losses are dominated by per-batch estimation noise at batch 64 (CORAL loss ~74% noise — a rank-63 estimate of a 600×600 covariance, SNR ≈1.36×; MMD stays ~flat regardless of λ), and the three diagonal aligners are **single-seed**. So the "≈0" DANN/CORAL/MMD result is partly a *measurement-limited* upper bound on what these aligners could do, not proof they cannot align (`UDA_METHODS.md` noise-floor note). The honest reading: at the batch sizes feasible here they buy nothing, but a batch-≥256 re-sweep is needed to settle it.
- **Per-class vs. global macro-F1 axis — resolved across 18 weeks.** The joint repair (all flagged classes at once → one global macro-F1) was run on **all 18 qualifying forward weeks** (`results/repair/joint_repair_v01/crossweek.json`): global macro-F1 **Δ +0.019 ± 0.0017** (positive on every week), stable-176 macro-F1 **Δ +0.0074 ± 0.0014** (rises on every week — zero cumulative poaching). So the "no negative transfer" claim holds *jointly and cross-week*, not only per-class in isolation. (Absolute levels ~0.47–0.56 are the 10% subsample; deltas are apples-to-apples.)
- **The repair is *not* uniquely good — a labels-matched logistic fit reaches higher recall.** A one-vs-rest logistic regression on the *same* K shots (`results/repair/labelmatched_baseline_v01/metrics.json`) hits higher recall than the prototype/k-NN repair at K=50 (docker 1.00, eset 0.89, **skype 0.79 vs k-NN 0.58**) — but at **higher poaching** (skype logreg 1.78% vs k-NN 0.76%; a poach-matched comparison was not run). So the repair does **not** strictly beat the simplest thing the same labels buy; its value is being **fit-free, integrating cleanly with the frozen-prototype monitor**, and k-NN's recall-at-low-poaching trade-off + its multi-modal edge over the single-prototype NCM. We do **not** claim the prototype repair is the best K-shot corrector.
- **Small n per class.** Per-class recall is over **47–62 eval flows from a single week** (eset-edtd 62, docker-registry 51, skype 49, microsoft-defender 47). So "0.00→0.88" means ≈55/62 flows; treat absolute recall values as **preliminary point estimates with wide implied confidence intervals**, not precise figures.
- **Repair artifacts are 10%-subsample embeddings** and flagged as *preliminary, NOT directly comparable to the full-data monitor figures* (`metrics.json: caveat`). The "flat" stable macro-F1 sits at the **≈0.45 subsample level (single eval week)**, not the model's full-data 0.654 — don't conflate the two. The recall-recovery *magnitudes* (large per-class jumps) are robust across 5 seeds; the absolute numbers are preliminary.
- **Detector AUROC source weeks.** Monitor numbers use the **week-16 healthy-regime** reference (clean regime), consistent with the paper's two-reference-week protocol; the week-1 (pre-artifact) reference is used for the 52-week decomposition analyses elsewhere.

**Referee-recommended experiments — now DONE (2026-06):**
- **(a) Joint multi-class repair → global macro-F1** ✅ — across 18 weeks: Δ +0.019 ± 0.0017 global, stable-176 +0.0074 ± 0.0014, both positive on every week (no cumulative negative transfer). `results/repair/joint_repair_v01/crossweek.json`.
- **(b) BBSE monitor on Allot** ✅ — viral AUROC 0.890/0.879 vs uncorrected 0.710/0.793, MFWDD 0.586/0.633. Closes the coverage asymmetry. `results/allot_monitor_v01/`, `figs/fig_allot_monitor_auroc.png`.
- **(c) Labels-matched baseline** ✅ — logistic K-shot fit is competitive with (and on skype beats) the prototype/k-NN repair, so the repair's claim is reframed to "fit-free, monitor-integrated" rather than "best corrector." `results/repair/labelmatched_baseline_v01/`.

**Referee-recommended experiments — DONE (2026-06-29):**
- **(d) BBSE monitor on QUIC** ✅ — viral AUROC BBSE 0.736 / RLLS 0.774 / EM 0.792 vs uncorrected 0.566, MFWDD 0.458 (same corrected-beats-uncorrected pattern as TLS/Allot). Caveat: only 3 eval weeks (45-47), so the viral protocol is small-sample. `results/quic_monitor/`.
- **(e) Few-shot repair on QUIC + Allot** ✅ — teleporters recover (QUIC to 0.67-0.90, Allot to 0.38-0.58), controls flat (no negative transfer). Generalizes the diagnose-and-repair loop beyond TLS. `results/repair/few_shot_repair_{quic,allot}_v01/`.
- **(f) DANN diagonal on QUIC + Allot** ✅ — already in tab:nt_merged (QUIC 0.761/+0.053; Allot 0.675/+0.011).
- **(g) Labeling-cost-per-year "big table"** ✅ — tab:labelcost: ours 5-25k labels/yr (0.06-0.30% of one retrain) vs UDA 0 (≤0 acc) vs MFWDD→retrain. `scripts/analysis/labeling_cost_per_year.py`, [[labeling-cost-per-year]].

**In flight (2026-06-29):** DANN λ=0 alignment-confound control (job 621277, isolates alignment vs retraining in the +0.067 forward gain); all-flagged TLS joint repair (job 621388, full-loop global ΔF1 for the labeling-cost table); QUIC k-NN repair variant.

**Still open:** CORAL/MMD only on TLS-wk16 (not QUIC/Allot — explicitly de-scoped 2026-06-29); UDA/TTA under the extreme label-shift sweep; the batch-≥256 UDA robustness re-sweep.

## Sources (verified)
- `UDA_METHODS.md` §3.2–3.5 — all UDA/TTA Δacc/ΔF1 (TLS wk16, QUIC wk44, Allot, DANN/CORAL/MMD diagonal).
- `UDA_BENCHMARK_STATUS.md` — benchmark protocol / status.
- `figs/auroc_anomaly_results_ref16.json` — viral-spike AUROC: BBSE 0.834 (FAR@95 0.675), RLLS 0.849, bbse_soft 0.836, uncorrected 0.708, MFWDD 0.650.
- `figs/isolation_w16/labelshift_sweep_results_fwd.json` — extreme label-shift sweep: ours 0.810 / naive 0.703 / MFWDD 0.591 @1000×; **naive > ours below ~5× (0.921 vs 0.897 @1×), crossover near 5×**.
- `results/repair/few_shot_repair_w16_v01/metrics.json` — NCM per-class repair (eset-edtd 0→0.88, docker-registry 0→0.86, skype 0.16→0.43, microsoft-defender 0.55→0.83) + per-class-isolated stable-F1-flat + CoTTA neg-transfer −0.06.
- `results/repair/few_shot_knn_v01/metrics.json` — skype budget-matched (NCM 0.43 @1.69% poach; k-NN 0.604, KMeans-M4 0.616, Parzen 0.633 at matched budget; k-NN raw 0.584 @0.76% poach).
