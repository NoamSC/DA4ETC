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
