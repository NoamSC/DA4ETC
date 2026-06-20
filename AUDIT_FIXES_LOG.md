# Audit Fixes Log

Running log of every correction made in response to `BUG_AUDIT_REPORT.md` (the multi-critic reliability audit). Updated as each item is fixed. IDs match the audit report.

**Legend:** ✅ done · 🔄 in progress · 📝 documented/deferred (with reason)

_Last updated: 2026-06-18, autonomous audit-fix pass._

---

## BLOCKERS (compile-breaking)

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| B1 | `fig:concept` — `figs/fig_continuous_vs_discrete_drift.png` missing (lead concept schematic) | Generated a clean 2-panel matplotlib schematic (continuous-drift vs discrete-teleportation) at that path. Script `scripts/viz/make_concept_and_entropy_figs.py`. | ✅ |
| B2 | `fig:entropy` — `figs/entopy_is_indicative (1).png` missing + misspelled/space path | Generated real-data entropy figure `figs/entropy_is_indicative.png` (correct vs incorrect predictive entropy, Week-16 forward; means 0.35 vs 1.54 nats); repointed `main.tex` and rewrote caption with the real numbers. | ✅ |
| B3 | `fig:app_overview` — active `\includegraphics{figs/fig_isolation_overview_5panel.png}` missing, marked NOT USED, never `\ref`'d | Deleted the entire float in `main.tex`. | ✅ |

## MAJOR

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| M1 | Proprietary name "Allot" in committed repo artifacts (179 tracked paths + status doc + slice_manifest absolute paths). Paper PDF source is clean. | Scrubbed all human-readable "Allot" → "Closed-World (proprietary)" in `UDA_BENCHMARK_STATUS.md`. **Deferred:** the 179-path `git mv` scrub (exps/figs/manifests/code/slurm) — too risky while active SLURM jobs 592120–592123 reference those paths; documented as a PRE-RELEASE task. | ✅ (safe part) / 📝 (path scrub deferred) |
| M2 | Hero figure labeled **microsoft-defender** as a "teleportation ≈ Week 26" exemplar, but metrics.json says `teleported=false, drift_week=22` (a gradual decline) — undercuts the thesis. | Swapped hero panel (b) to **skype** (`teleported=true`, drift 28), which has an existing t-SNE; relabeled "(b) skype (teleportation ≈ Week 28)"; rewrote the main caption (eset≈wk17; docker & skype≈wk28). | ✅ |
| M3 | Stats-rigor bootstrap clustered by `(week,seed)`=180 but the anomaly label is seed-invariant (36 real units) → manufactured CI non-overlap; correcting to week-level makes BBSE/uncorrected CIs **overlap**, falsifying the "does not overlap" claim. | Changed cluster id to **week-level** (`scripts/analysis/stats_rigor.py`) and re-ran (n_clusters 36): point estimates (0.690/0.627/0.577) and threshold sweep **unchanged**, CIs now **overlap** (BBSE 0.690 [0.633,0.745] vs uncorr 0.627 [0.585,0.669]). Rewrote the stats paragraph + Table `tab:statrigor` (caption + CI rows) to lead with point-estimate ordering + threshold-sweep dominance and **honestly state the CIs overlap** (no interval-separation claim). | ✅ |
| M4 | Isolation figure caption said "52-week stream" but panels/numbers are forward-only 36 weeks (and "52-week" violates the reference-week rule). | Caption → "Week-16, forward-only stream, weeks 17–52". | ✅ |
| M5 | Week-16 forward **uncorrected** tracking ρ is −0.983 (from cache) but paper said −0.987 in 4 places; the figure legend already shows −0.983. | Changed −0.987 → −0.983 at body (mirror prose), `fig:mirror` caption, the symmetric-control s→1 anchor, and the design comment. BBSE/RLLS/oracle/SLD-EM unchanged. | ✅ |
| M6 | Stale retracted DANN row (Δ=−0.069) + stale TLS numbers in committed `sota_benchmark_section.md`. | Regenerated tables from `regen_benchmark_table.py` (forward-only, `results/inference_auditfix/`); dropped the −0.069 row; DANN now +0.006 diagonal; added authoritative-source header. | ✅ |
| M7 | Intro cited `seddigh2019framework` (a success-reporting static framework) as evidence of "generalization failures". | Replaced with `\cite{shamsimukhametov2024ech}` (documents real ECH generalization collapse); re-homed `seddigh2019framework` as a background ETC-ML reference (no orphan bibitem). | ✅ |

## MINOR

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| m1 | Abstract: "negative transfer **on stable classes**" not measured per-class. | Softened to "net negative transfer over a frozen baseline—warping the stable majority it should preserve". | ✅ |
| m2 | "Week-1 used **only** to validate Week-10" contradicts the Week-1 headline FAR result. | Softened "only" to acknowledge Week-1 also anchors the primary anomaly benchmark (Week-16 replication kept explicit). | ✅ |
| m3 | OOD headline ranking computed on different per-detector populations (n=4380 vs 3509), undisclosed. | Added footnote disclosing the n_eval split (also explains the MFWDD 0.710 vs 0.723 difference, covering N5). | ✅ |
| m4 | Deprecated scan script still executed disowned Steps 2/3 and rewrote stale "BBSE stays flat" JSONs. | Guarded Steps 2/3 behind opt-in `--legacy` (default off); deleted stale `monitor_trace.json`/`injection_sweep.json`. | ✅ |
| m5 | Stats-rigor mislabeled "full-stream forward" (it is forward-only). | Changed to "forward-only" in the M3 rewrite. | ✅ |
| m6 | Repair stability bound \|Δ\|≤0.001 violated by microsoft-defender (0.00130). | Resolved by the m8 code fix (self-referential guard removed) → new max \|Δ\|=0.0006, so the ≤0.001 claim now holds; no paper softening needed. | ✅ |
| m7 | Stale TLS benchmark numbers in `sota_benchmark_section.md`. | Same regeneration as M6. | ✅ |
| m8 | Self-referential stability guard for microsoft-defender in `few_shot_repair_loop.py`. | `row_stable = stable_classes[stable_classes != c]` per row; re-ran (teleported recall unchanged; max stable \|Δ\|=0.0006). | ✅ |
| m9 | Six fragile figure paths with spaces/"(1)" suffix. | Repointed `fig_app_degradation_grid (1)` → space-free; added `\graphicspath` (m10) so staged flat copies resolve. | ✅ |
| m10 | `figs/` not resolvable when compiling from `paper/` (no `\graphicspath`). | Added `\graphicspath{{figs/}{../figs/}}` after `graphicx`. | ✅ |
| m11 | `fig:app_trainfreq` — active include of missing cut-candidate file. | Deleted the float. | ✅ |
| m12 | Four Related-Work citations had no SUMMARIES.md vetting note. | Verified all four against the real papers (incl. HOLMES CCS'24 PDF) — **all accurate**; added SUMMARIES.md entries. No `main.tex` change. | ✅ |
| m13 | `alexandari2020mlls` cited for a dissociation it does not make. | Dropped the cite at that clause; presented as our own finding (its other two correct uses kept). | ✅ |

## NITS

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| N1 | "At s=1 (0.967 vs 0.903)" used the clean/no-injection baseline, not the s=1.0 sweep point. | Changed to the actual s=1.0 values (0.921 vs 0.897, verified against `labelshift_sweep_results_estimators_fwd.json`); also corrected the crossover claim "by s≈3" → "between s=3 and s=5" (data: uncorrected still leads at s=3, ours leads by s=5). | ✅ |
| N2 | Two Week-16 Source-only baselines (0.798 vs 0.795) unexplained. | Added a clause to the DANN-table caption (diagonal averages weeks 17–52, excludes in-dist Week-16). | ✅ |
| N3 | Razor-thin CI non-overlap. | Resolved by M3 (CIs now correctly reported as overlapping). | ✅ |
| N4 | MLLS+BCTS prose 0.828 vs table 0.827. | Prose → 0.827. | ✅ |
| N5 | MFWDD clean AUROC 0.710 vs 0.723 (two streams). | Covered by the m3 footnote. | ✅ |
| N6 | Panel-D caption "vs 35–70%" lumped two baselines. | Split into "naive 35–46% / MFWDD 44–70%". | ✅ |
| N8 | "all collapse toward chance" overstated (land at 0.66–0.68). | Changed to "drop sharply (to ≈0.66–0.68)". | ✅ |
| N12 | Dead `ks_sweep` expression in `plot_repair_before_after.py`. | Removed. | ✅ |
| N7 | SUMMARIES.md citation key `shamsimukhametov2022ech` vs `…2024ech` in the paper. | Standardized the SUMMARIES.md backtick key to `shamsimukhametov2024ech`. | ✅ |
| N9 | Stale `main.tex` comment with **wrong** proprietary TTA numbers (TENT −0.213 / CoTTA −0.278). | Corrected the comment to the verified values (TENT −0.211 / CoTTA −0.173 / AdaBN −0.030). | ✅ |
| N10, N11 | Stale code comment drift_week (N10); orphan appendix `\label`s (N11). | Non-rendering / no compile warning; left as-is. | 📝 |

## Cross-cutting cleanups (beyond specific IDs)

- **`\TODO` markers:** redefined `\TODO` to a no-op so the ~20 red "[TODO:…]" placeholders no longer render in the submission PDF (notes preserved in source).
- **Figure staging:** copied all subdir figures (`figs/ref16/`, `figs/isolation_w16/panels/`, `figs/repair/`, `figs/natural_falsealarm/`, `figs/drift/`) to the flat `figs/` paths the paper references.
- **Structural re-check after edits:** braces 736/736 balanced, `$` even (512), no dangling `\ref`, no duplicate `\label`, all `\cite`→`\bibitem`, no orphan bibitems, no citations in abstract, **no missing active `\includegraphics`**.

## Independent verification of this fix-pass

A separate read-only critic re-checked every changed number against its source file and re-ran the structural checks: **10/10 PASS, no new errors introduced.** Confirmed: ρ=−0.983 (BBSE −0.988/RLLS −0.992/oracle −0.991 intact); hero skype swap valid (`teleported=true`); M3 table matches the n_clusters=36 rerun (CIs overlap); N1 s=1 values + crossover correct; entropy means 1.54/0.35; em_bcts 0.827; deleted floats leave no dangling refs; citations all resolve with no orphans; abstract citation-free; all active figures exist; `grep -rni allot paper/` empty.

## Round 2 — fresh-eyes whole-paper read (caught 3 items the sliced audit missed)

A full end-to-end reviewer read of the *corrected* paper surfaced three real defects (two were fallout from the Round-1 edits):
- **R2-1 (high):** microsoft-defender (the `teleported=false` control) still appeared in the **repair** section (prose + figure caption) as one of four flagged teleporter recoveries — the same defect M2 fixed in the hero, left in a second spot. **Fix:** reframed it explicitly as "a non-teleported partial-degradation control" in both the prose and the `fig:repair` caption (now a *stronger*, honest result: the prototype update also helps the control without harm).
- **R2-2 (medium):** the N1 fix (s=1 → 0.921/0.897) was applied to the body but **not** to the `fig:sweep` caption, which still read "(0.967 vs 0.903)". **Fix:** caption → "(0.921 vs 0.897)", crossover → "between s=3 and s=5".
- **R2-3 (medium):** appendix captions rendered draft scaffolding ("PARKED", "NOT USED", "SUPERSEDED", "Candidate if we want…", "was body Fig 10"), and the proprietary-placeholder `\fbox` became **empty** once `\TODO` was neutralized. **Fix:** reworded all six appendix captions to clean descriptive prose; replaced the empty box with a visible italic "[figure pending disclosure clearance]" placeholder.

Re-checked after Round 2: no rendered scaffolding words remain; braces 736/736; no dangling refs/cites; no missing figures.

## Tally
- **3/3 blockers** fixed · **7/7 major** fixed (M1 path-scrub deferred to pre-release) · **13/13 minor** fixed · **8 nits** fixed, **2 nits** (N10/N11) left as harmless non-rendering items.
- Net: every audit item that affects the compiled paper or a reported number is resolved and independently verified.

## Headline re-derivation (independent)

The paper's single most important result was re-derived from its source file `figs/auroc_anomaly_results.json` (Week-1 stream) and matches **exactly**: BBSE viral AUROC 0.9868→**0.987**, uncorrected 0.9587→**0.959**, MFWDD 0.6768→**0.677**, clean all 1.000; FAR@95 BBSE 0.0463→**4.6%**, uncorrected 0.25→**25.0%**, MFWDD 0.648→**64.8%**; FAR@90 2.8%/12.0%/59.3%; so 5.4× (25/4.6) and 14× (64.8/4.6) are correct.

## Round 3 — proprietary closed-world results completed (SLURM jobs landed)

The closed-world (anonymized) negative-transfer benchmark, scoped "preliminary / still running" in earlier drafts, is now **complete and folded into the body**. Jobs that finished: DANN training (both slices), DANN inference (593081), bs256 re-runs for all four adapters, 5-seed multiseed (seeds 1–4), and the one failed-and-resubmitted shard AdaBN bs256/seed42 (594734).

- **Tooling:** extended `scripts/analysis/closedworld_negtransfer_table.py` with `--suffix-tag`/`--seed` (DANN special-cased to its `_dann` exp dir); added `scripts/analysis/closedworld_assemble_quarter.py` to assemble the 3-block table → `results/analysis/closedworld_negtransfer_quarter.json`.
- **Bug fixed in inference loader:** `run_allot_inference.py` now strips training-only `domain_classifier.*` keys from DANN checkpoints (strict load otherwise), without which DANN inference crashed on `load_state_dict`.
- **Verified numbers** (clean source, forward-only, 36 windows, 60 stable classes): bs64 AdaBN −0.030 / TENT −0.211 / CoTTA −0.173; bs256 AdaBN −0.014 / TENT −0.114 / CoTTA −0.107 (5-seed std ≤0.001); **DANN +0.011** (neutral, stable-class recall 0.699 vs 0.696). Source-only forward mean acc 0.664, Macro-F1 0.524.
- **Paper edit (CW-1):** rewrote the body "External validation" paragraph (main.tex ~630): retitled from "(preliminary)"; **fixed the "Macro-$F_1=0.664$" mislabel** (0.664 is mean accuracy; Macro-F1 is 0.524); added the bs256 fair-operating-point deltas, the 5-seed reproducibility (std ≤0.001), and the DANN-neutral control; dropped the "DANN/multiseed still running" caveat; kept the 10%-sample caveat and "single-dataset external corroboration, not a headline result" framing.
- **Paper edit (CW-2):** corrected the stale "LEFT OUT of body" advisor comment (~767) to reflect that the prose summary is in body now, with the expanded table + t-SNE still disclosure-gated and consistent.
- Re-checked after Round 3: `grep -rni allot paper/` empty; braces balanced (761/761); `$` even (538); no stale "still running"/"preliminary"/"DANN-on-proprietary" refs.

## Round 4 — hero t-SNE figure: replaced weak skype panel
- **Why:** the Fig.~\ref{fig:hero} panel (b) `skype` did not visually read as a discrete
  teleportation. Quantified it from the cached embedding (per-week centroid track):
  skype's biggest week-to-week jump (27.5) is *smaller* than its own within-week spread
  (38.8), i.e. jump/spread = 0.7 — genuinely a non-teleporter. (docker 5.6, eset-edtd 1.4,
  apple-location 0.5, vmware 0.6 with net 1st->last drift only 5.1.)
- **Fix:** swapped panel (b) to `microsoft-defender` (class 98), jump/spread = 1.5 with one
  dominant jump wk25->26 — two cleanly separated colour clusters across empty latent space.
  Caption sub-label and the main caption's per-app week list updated
  (eset-edtd ~Wk17; microsoft-defender ~Wk26; docker-registry ~Wk28).
- **Provenance bonus:** the prior root renders silently mixed t-SNE fits (docker + defender
  were stale May-30 renders from a different cache; the baseline filename `..._w16_week16...`
  came from an all-weeks fit), contradicting the caption's "shared fit, overlay-comparable"
  claim. Re-rendered defender, docker, and the by-class baseline directly from the single
  `results/tsne_cache/tsne_w16_16to52_top10_hi16.npz` embedding (cache hit, no t-SNE recompute;
  the live inference dir no longer holds the per-week npz, so `load_all` can't be re-run).
  All four panels (a,b,c,d) now share that one fit (md5-verified vs `figs/drift/`).
  Baseline include path updated to `figs/tsne_background_w16_16to52_week16_byclass.png`.
- skype is retained where it is still valid: the per-class *recall*-collapse example
  (Fig.~\ref{fig:classdrop}, ~wk28) and the few-shot repair "hardest case" (0.16->0.43) —
  those describe a recall step-drop, independent of the diffuse latent track.
- Re-checked: `grep -rni allot paper/` empty; braces 761/761; `$` even (540); all four hero
  figs exist on disk. Not committed.

## Round 4b — CORRECTION to Round 4 (defender was the wrong replacement)
- A forked audit (M2) + my own per-week npz check showed the Round-4 choice was wrong:
  `microsoft-defender` is **not a clean teleporter** (frozen-W16 NCM recall 0.93 dips to
  0.74 then *recovers* to 0.85, never <0.3; t-SNE jump/spread only 1.5) and it is the
  explicit `teleported=False` **control** in the few-shot repair figure — so using it as
  the hero teleportation exemplar was a self-contradiction.
- **Fix (panel b -> eset-edf, class 56):** the one focal class with a genuinely clean
  *latent* teleportation among the cached-embedding set — jump/spread **3.0** (2nd only to
  docker's 5.6), discrete jump w33->w34, recall 0.97->0.52. Distinct week from docker and
  eset-edtd; shares the "eset-" vendor with panel (d) but is a different app teleporting at
  a different week, which *reinforces* the asynchronous per-class thesis. Copied from the
  same shared `tsne_w16_16to52_top10_hi16` fit (md5-verified).
- **docker week corrected 28 -> 27** in the hero panel (c) sub-label and the main caption:
  per-week npz shows recall 1.00 (w26) -> 0.02 (w27) -> 0.00 — the cliff *event* is w27.
- Rejected alternatives (data-backed): `ms-push` recall-cliffs at w32 but its latent track
  is diffuse (jump/spread 0.7) -> would smear like skype; `ms-settings` recovers; `apple-
  location` has no cliff (a forked #8 item wanted it as a *recall*-drop example — also
  invalid for that reason).
- Re-checked: hero block = (a) baseline, (b) eset-edf ~w34, (c) docker ~w27, (d) eset-edtd
  ~w17; all four figs on disk; `grep -rni allot paper/` empty; braces 761/761; `$` even
  (540). Not committed.
- LEFT FOR OTHER FORKS (not edited here, to avoid collision): main.tex:478 (`fig:classdrop`
  recall caption) still says docker `wk28` -> should be `wk27` (skype `wk28` is correct
  there, recall bottoms at w28). Owned by the classdrop/#8 fork.

## Round 4c — eset-edtd case-study validation (#9) + docker-week prose reconcile
- Launched a read-only validation subagent against the RAW per-week npz (not the prose/JSON).
  Outcome — **core case-study claims hold:**
  - Flat-then-step latent departure confirmed: L2 from W16 centroid ~5.5 (w14--16) -> ~8.1
    (w17) -> 12.9 (w18), then plateau. A jump, not a ramp.
  - Recall cliff is real, not an empty-bin artifact: class 57 has 685--2287 samples/week;
    recall=0 at w18 rests on 1535 samples. NCM and real-model pred_labels agree (w17 ~0.65,
    w18 ~0.00).
  - Isolated per-class drift, NOT a sensor artifact: at w18 eset-edtd is the *only* class with
    a >=0.40 recall drop; global macro-recall declines smoothly. Distinct from the w10 exporter
    artifact. Side-confirmed docker's >=0.40 drop lands at **w27**.
- **Applied:** docker-registry week reconciled in the case-study prose (main.tex:555
  "$\approx$Week~28"->"~27"; :559 "eight weeks late"->"nine", since 36-27=9). Hero figure
  already w27. Helper `few_shot_repair_loop.py drift_week=28` left as-is on purpose (it is a
  *post-drift support-pool start*, one week after the cliff, not the event week).
- **FLAGGED FOR A HUMAN (not edited):**
  1. Overclaim nuance: post-cliff eset-edtd flows do NOT settle into one fixed destination
     (king-games 59%@w18 -> opera-autoupdate ~40%@w19--30 -> scattered@w44). The cloud
     definitively *leaves* its W16 region (L2 stays high) but has no single new home. Prose
     doesn't assert a fixed destination, so no false sentence -- but consider a one-clause
     honest note ("evacuates its Week-16 region" rather than implying a fixed target).
  2. L2 quotes (main.tex:527): paper says $\approx$5.3 / 8.3; validation measured ~5.5 / 8.1
     (w18=12.9 matches exactly). Small; verify on the ORIGINAL computation basis before
     editing -- did not silently change numbers.
  3. SHAP drivers (PPI |SHAP|=4.38), domain-classifier AUC=0.92, BYTES_REV 5578->3551
     (main.tex:543--553): NOT checkable from the inference npz (labels/softmax/600-d
     embeddings only) -- needs the raw CESNET parquet. Separate raw-feature audit.
  4. eset-edtd week label: onset w17 vs completion w18 vs latent jump w17->18 (raw L2). Kept
     w17 (onset) -- it is already unified across the case-study title/caption/prose. Decide
     definitively if changing.

## Still needs a human / Overleaf check
- **LaTeX compile:** `pdflatex` is not installed locally; the missing-figure fixes were verified by file-existence, not a compile. Run `pdflatex`/`latexmk` from the repo root (or `paper/`, now that `\graphicspath` is set) to confirm a clean build.
- **M1 full scrub:** the 179-path "allot" rename must be done before any code/data release (deferred while SLURM jobs ran — now that they are done, this can proceed when convenient).
