# Critic-loop log

Each iteration: launch independent critic subagents on different facets → summarize →
apply fixes → recompile → commit tex+pdf. This file logs findings + dispositions.

## Iteration 1 (2026-06-21) — 6 critics: thesis, rigor, writing, figures, related-work, anonymity

### MUST-FIX applied
- **Abstract FAR caveat** (thesis): the 4.6% / 5.4×–14× is the Week-1 stream (includes the
  Week-10 sensor event). Abstract now notes "same ordering reproduced in a clean-regime
  replication that excludes the dataset's one sensor event"; repair scoped to "proof of concept".
- **Causal softening** (thesis): "discrete drift → global adaptation fails" changed from an
  assertion to "predicts trouble… consistent with the negative transfer we measure", with the
  AdaBN control localizing harm.
- **Novelty reframe vs HOLMES** (related-work): re-anchored the contribution on "separating
  covariate degradation from benign label shift + calibration against hidden F1", not "first
  per-class label-free monitor".
- **DANN scoping** (rigor/writing/figures/anon): prose no longer claims DANN holds on Week-1/QUIC;
  scoped to Week-16 + closed-world; red "?" table cells noted as pending (SLURM 598444–598447).
- **"volume" disambiguated** in case study (thesis): per-flow signature (covariate) vs class
  prevalence (label shift).
- **MFWDD-style baseline** (rigor): footnote clarifying it is a reimplementation in MFWDD's spirit
  on the output (softmax) distribution, not raw input features; benchmarks the idea, not the system.

### Applied (should/nice)
- "SLD-EM inverts (ρ=+0.030)" → "fails to track (ρ≈0)" in all 3 places (ρ≈0 is no-correlation).
- Repair honesty: skype only 0.16→0.43; non-interference is by-construction; "proof of concept".
- 6 missing citations added + cited: DDM (gama2004ddm), ADWIN (bifet2007adwin), quantification
  survey (gonzalez2017quant), TTA survey (liang2024ttasurvey), OSR survey (geng2021osr), MSP
  (hendrycks2017msp), energy (liu2020energy).
- Tics removed: "Crucially", "We state plainly that", "in short", "deserves separate treatment".
- Conclusion hedge collapsed; metaphor unified to "two-sided instrument".
- OOD-sweep figure path flattened (figs/ root) for Overleaf.

### Deferred / flagged (open items)
- **[BACKBONE PROVENANCE]** (rigor MUST-FIX): the Week-16 viral figure (Fig 12) was generated on
  the FlowPic CNN, not mm-CESNET. Agent ad4fd3d2 is investigating + re-running on the multimodal
  backbone if cheap. Sync Table II + Fig 12 numbers when it reports. **NEEDS RESOLUTION before submit.**
- **[FIG 14 ref-week-0 + "ours" legends]** (figures): fig_auroc_anomaly.png still titled "ref Week 0"
  and 4 figures still show "ours". Earlier figure agent relabeled the isolation/sweep figs but did
  not finish the auroc ref-1 regen. To redo.
- **[DANN red cells]**: filled automatically when SLURM 598446/598447 land (watcher bjlqw4ue2).
- Vendor dates (TeamViewer/Skype/Opera) — already hedged as coincidence; verify before camera-ready.
- 179-vs-180 class count; regime1 caption s∈[0,4] vs TV-% axis; fig:density weak (cut candidate).
- bs256 effect is mild on TLS — consider foregrounding that the dramatic deltas are bs64/closed-world.

## Iteration 2 (2026-06-21) — applied remaining writing/rigor findings
- bs256-honesty foregrounding; Week-1 split sizes (43/9) + rest claim on ordering;
  conclusion 'actively harmful' -> 'does not help' (batch-scoped); tics removed.

## Iteration 3 (2026-06-21) — 2 critics: method/math correctness, adversarial-reject
### MUST-FIX applied (math critic — HIGHEST VALUE)
- **Residual equation now matches the implementation.** Old eq wrote term2 as
  H(w_t) ("entropy of proportions") and term1 as H(aggregate marginal); the code
  computes ΔH_t = mean per-sample entropy − Σ_c w_t(c)·h_ref(c) (BBSE-reweighted
  per-class reference entropy). Rewrote Sec V-C with the correct equation, defined
  h_ref(c)=E[H(p)|y=c], stated the score convention (larger=worse), noted the
  reference-supported support set, and clarified the only change BBSE makes is the
  weights on a fixed per-class entropy table (uncorrected uses the static source prior).
- **Deleted the bogus Jensen 'upper envelope' paragraph** (no aggregate-vs-per-sample
  gap exists; both figure and residual are per-sample means). Fixed the Sec V-A
  forward-reference accordingly.
- **Regime-2 spike** now stated as an empirical observation with the explicit
  condition (broken flows more uncertain than any reattributed class's reference
  entropy), not a deduction.
- Softened RW 'breaks all of them together' -> 'violates the shared assumption,
  biasing them simultaneously; errors spike together (shown empirically)'.
### Adversarial-reject critic — top risks + dispositions
- Risk1 (monitor wins only where injected): applied the honest robustness-vs-clean-
  accuracy TRADE-OFF reframe (insurance with a measurable premium) in Sec VI-B. NOTE:
  the critic's 'show a real spike' mitigation = the Natural Volume Surges analysis the
  USER removed; honoring the user, we reframe rather than re-add it.
- Risk2 (proprietary 2nd dataset undisclosed; single backbone): the closed-world set is
  now an explicit column (user request) but remains anonymized/uncheckable — OPEN: a
  second public backbone (QUIC teleportation / FlowPic) would strengthen external validity.
- Risk3 (Week-1 sensor-inflated headline): caveated; ordering-not-multipliers framing
  applied. OPEN consideration: lead abstract with Week-16 clean numbers.
- Lesser (novelty oversold, neg-transfer bs256 mild, repair thin, open-set decorative):
  partially addressed via scoping; flagged.

## Iteration 4 (2026-06-21) — consistency + figure relabels + HEADLINE-NUMBER issue
- Applied: 179/180 class-count note in hero caption; intro "not X" tic; figure "ours"
  labels removed across most monitor figures (figure agent edited 8 scripts, regen'd 6
  PNGs from cache, numbers unchanged); refreshed the flat OOD-sweep copy.

### *** CRITICAL OPEN ISSUE: anomaly-detection headline numbers ***
The figure agent + rigor critic together established that the published Week-1 anomaly
headline (AUROC 0.987 / FAR@95 4.6%; the abstract's "5.4x/14x") was computed with:
  (a) reference_week = 0 (the documented BIASED "easy" week, not the Week-1 source), and
  (b) on the FlowPic CNN backbone / a now-reorganized 10% inference dir.
A quick ref-week-1 regen gave very different numbers (BBSE viral ~0.81 / 67% FAR) but was
triply-confounded (ref week + backbone + full-flow vs 10% + 6 corrupted weeks), so NOT
syncable. fig_auroc_anomaly.png was restored to the committed (consistent) version to
avoid a table/figure mismatch.
RESOLUTION IN PROGRESS: SLURM job 598577 (slurm_files/run_anomaly_mm.slurm) re-runs BOTH
the Week-1 (reference_week=1) and Week-16 viral experiments on the MULTIMODAL
inference_auditfix 10% set (the same set the rest of the paper uses). Watcher bzutxnfko
will report the authoritative AUROC/FAR. **The headline FAR/AUROC and the abstract
multipliers will likely change; do not treat 4.6% as final until 598577 lands.**
This is the #1 thing for the user to review.

### DANN red cells: SLURM 598444-598447 still training (~hours); watcher bjlqw4ue2.

## Iteration 5 (2026-06-21) — AUTHORITATIVE anomaly-number sync (backbone + ref-week resolved)
Backbone agent + SLURM 598577/598568 resolved the headline-number issue:
- Week-1 HEADLINE was already on the MULTIMODAL backbone (good), but on reference_week=0
  (biased easy week). Re-ran reference_week=1 on the multimodal 10% set:
  BBSE AUROC 0.982 / FAR@95 11.1% ; uncorrected 0.956 / 23.1% ; MFWDD 0.665 / 67.6%
  (RLLS 0.985/10.2%, EM/MLLS 0.979/10.2%, soft-BBSE 0.983/13.0%, MLLS+BCTS 0.932/32.4%).
  => Synced Table II, abstract (11.1%, ~2x/6x), Results prose, conclusion, Fig 14
  (now 'ref Week 1', multimodal, no 'ours').
- Week-16 replication figure/table was the lone CNN-backbone artifact; re-ran on
  multimodal: BBSE 0.695/0.807, RLLS 0.695/0.799, uncorrected 0.626/0.867, MFWDD
  0.586/0.814, SLD-EM 0.603. Honestly weaker margin; synced Fig 12 (now _mm) + prose.
- NET EFFECT: the result HOLDS (BBSE/RLLS best, big MFWDD gap) but the headline is
  honestly less inflated (4.6%->11.1%; 5.4x/14x -> ~2x/6x). All anomaly figures now on
  the multimodal backbone, consistent with the rest of the paper.
