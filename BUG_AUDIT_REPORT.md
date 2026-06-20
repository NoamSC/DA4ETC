# Paper Reliability Audit

Synthesis of an independent multi-critic audit of `paper/main.tex`, its result files, and supporting scripts/status docs. Each item below survived an independent skeptic's verdict (dismissed items are summarized at the bottom). Items are ordered within each tier by how much they threaten the paper's credibility.

---

## BLOCKERS (must fix before submission)

These break the LaTeX build. With any one of them active, `pdflatex` aborts and no PDF is produced (pdflatex is not installed locally, so this could not be confirmed by compile — see Confidence note).

### B1. Missing lead concept figure — `fig:concept` (compile-breaking)
- **Location:** `paper/main.tex:203` (active `\includegraphics`); `\label` at :212, `\ref` at :263.
- **Issue:** Active `\includegraphics{figs/fig_continuous_vs_discrete_drift.png}` references a file that does not exist on disk; no generator script exists. This is the paper's opening Sec. I "Continuous Drift Fallacy" concept figure, referenced from the Introduction.
- **Expected vs observed:** File present so the lead figure renders → file absent → File-not-found compile error (or blank box).
- **Fix:** Generate the schematic (matplotlib drawing the smooth-drift-vs-teleportation contrast in the caption) and place at `figs/fig_continuous_vs_discrete_drift.png`; or comment out line 203 and convert the `\ref{fig:concept}` at :263 to text until the asset exists. Generating it is preferred since it is the lead concept figure.

### B2. Missing entropy figure — `fig:entropy` (compile-breaking)
- **Location:** `paper/main.tex:775` (active `\includegraphics`); `\ref`s at :762, :770, :838.
- **Issue:** Active `\includegraphics{figs/entopy_is_indicative (1).png}` references a nonexistent file (nothing matching `*entop*`/`*indicative*` anywhere in the repo). The basename is also misspelled ("entopy") and the path has a space and a "(1)" suffix. The methodology prose (Sec. method "Entropy as an Error Proxy", Sec. residual) depends on this figure.
- **Expected vs observed:** File present → absent → compile error.
- **Fix:** Add the actual entropy-distribution figure under a clean basename (e.g. `figs/entropy_is_indicative.png`), update line 775, or remove the float at 774–779 and the three `\ref{fig:entropy}` citations. Also resolve the `\TODO{caption later.}` placeholders nearby.

### B3. Missing/invented 5-panel figure — `fig:app_overview` (compile-breaking)
- **Location:** `paper/main.tex:1433–1438` (active float, label at :1437).
- **Issue:** Active `\includegraphics{figs/fig_isolation_overview_5panel.png}` references a file that does not exist and has no generator (only the individual panels A–D exist). The caption itself is marked "NOT USED" with `\TODO{confirm filename.}`. The label is never `\ref`'d.
- **Expected vs observed:** File present, or float removed → file absent, float active → compile error.
- **Fix:** Delete the entire `\begin{figure}…\end{figure}` block at 1433–1438. It is unused, marked NOT USED, and its file does not exist — removing it costs nothing.

> Note: B2 and B3 share the same root failure mode (active include of a nonexistent file) with B4 below; they are separated only because B4 is a documented cut-candidate of lower stakes.

---

## MAJOR (inaccuracies / unsupported claims)

Ordered by threat to credibility: double-blind exposure first, then a self-undermining hero figure, then number/claim integrity issues.

### M1. Double-blind exposure: proprietary name "Allot" in committed repo artifacts
- **Merged from two critic findings:** `allot-name-leak-committed-tree` (exps/figs/manifest paths) and `status-doc-names-allot` (status markdown). Both are the same global-rule-(1) violation across different committed artifacts.
- **Location:**
  - `UDA_BENCHMARK_STATUS.md` (tracked, committed `671eab61`): names "Allot" 6× incl. a results-table header "## Allot — private (…closed-world)" with per-method accuracy rows (working-tree lines 121, 171, 240, 274, 285, 381; present in HEAD too).
  - Tracked paths: `exps/allot_multimodal/{early_eq,quarter_eq}/{config.json,label_mapping.json,slice_manifest.json}`, `figs/allot_tsne/tsne_allot_*.png`.
  - `slice_manifest.json` embeds full real paths: `/home/anatbr/dataset/Allot/allot_hourly_chunks_parquets/domain_1/chunk_2024-09-14_07-00.parquet` — leaking both name and on-disk location.
- **Scope:** `git ls-files | grep -i allot` → 179 tracked paths (also `INFERENCE_ON_ALLOT_HANDOFF.md`, `UDA_STATUS_ALLOT.md`, `data_utils/allot_*.py`, `scripts/.../*allot*.py`, `slurm_files/run_allot_*.slurm`, multiple `exps/allot_*`).
- **Expected vs observed:** No committed artifact contains "allot"/"Allot" or the real dataset path → 179 tracked paths + a tabulated private results doc do. `paper/main.tex` itself is clean (anonymized as "the independent, proprietary closed-world dataset (anonymized for review)"; the Allot rows are commented out).
- **Why major not blocker:** The paper PDF source is clean; the leak only matters **if the repo is shared as supplementary material** — in which case it escalates to a blocker.
- **Fix:** Before any code/data release, scrub all 179 tracked paths: `git mv` the trees to a neutral token (`exps/closedworld_cw/`, `figs/closedworld_tsne/`, `data_utils/closedworld_*.py`, `slurm_files/run_closedworld_*.slurm`), rename the handoff/status markdowns, and replace every "Allot" in `UDA_BENCHMARK_STATUS.md` with a neutral codename. Strip the absolute `/home/anatbr/dataset/Allot/...` paths from every tracked `slice_manifest.json` (redact or untrack via `.gitignore` + `git rm --cached`). Verify `git grep -i allot` returns empty. If full history is shared, rewrite history (filter-repo).

### M2. Hero figure presents a NON-teleported class as a teleportation exemplar
- **Location:** `paper/main.tex:222–223` (panel (b)) and :234–235 (caption "teleport across empty space").
- **Issue:** The headline "Continuous Drift Fallacy" figure labels **microsoft-defender** as "(b) … (teleportation ≈ Week 26)", but `results/repair/few_shot_repair_w16_v01/metrics.json` lists microsoft-defender with `teleported: false`, `drift_week: 22`, `base_recall_c: 0.553` (a gradual decline, not a cliff). Genuine teleported exemplars in the same file (eset-edtd, docker-registry) have `teleported: true` and `base_recall_c: 0.0`. Using a gradually-degrading class as the teleportation example undercuts the paper's central thesis. Also: panel (d) eset-edtd is captioned "≈ Week 17" but its `drift_week` is 18.
- **Expected vs observed:** A "teleportation" panel should be a class with `teleported=true` and matching drift week → microsoft-defender is `teleported=false`, `drift_week=22`, captioned "Week 26".
- **Fix:** Stop using microsoft-defender as a teleportation exemplar. Either (a) replace panel (b) with a genuinely teleported class (`teleported=true`, `base_recall_c=0.0`) from the per-class pipeline, or (b) relabel panel (b) as a gradual/partial-degradation contrast, correct "Week 26"→"Week 22", and soften the caption so it no longer claims all four panels teleport. Separately fix panel (d) "Week 17"→"Week 18".

### M3. Stats-rigor non-overlap claim is an artifact of pseudo-replication (week,seed clustering)
- **Location:** `scripts/analysis/stats_rigor.py:297–299` (cluster id), with the affected claim at `paper/main.tex:1010`.
- **Issue:** The bootstrap treats (week,seed) as 180 independent clusters, but the anomaly label is keyed only on `r['week']` (label_for_threshold :212–213) and the label-determining `week_clean_f1` is seed-invariant (:153–156, comment :288). All 5 seeds of a week carry the **same** label and resample the same week's data; only the window-sampling RNG differs. So the label-driven detection task has **36 independent units, not 180**. The skeptic **escalated this from minor to major** after recomputation: correcting to week-level (36) clustering inflates the BBSE CI from [0.657, 0.722] to ≈[0.634, 0.745], and the intervals then **OVERLAP** uncorrected's ([0.585, 0.669]). The paper's stated robustness claim at :1010 ("the BBSE confidence interval does not overlap the uncorrected one") holds only by the razor-thin 0.001 endpoint margin the pseudo-replication manufactures, and is **falsified** under correct clustering.
- **Expected vs observed:** Cluster by week (36) for the label-driven CI → currently clusters by (week,seed) (180), overstating precision and producing a non-overlap that does not survive correction.
- **What survives:** Point estimates (0.690 / 0.627 / 0.577), ordering, and the threshold-sweep dominance (0.713/0.690/0.748 vs 0.614/0.627/0.654) are all unaffected — the qualitative result holds.
- **Fix:** Change the cluster id at stats_rigor.py:297–299 from `(r['week'], r['seed'])` to `r['week']` (seeds fold in as nested windows), re-derive CIs (BBSE 0.690 [0.634, 0.745], uncorrected 0.627 [0.585, 0.669] — overlapping). Then **rewrite/remove main.tex:1010**: lead with the point-estimate separation and the threshold-sweep dominance (robust) and state the 95% CIs overlap under week-level clustering, so the separation is suggestive rather than CI-disjoint. Update Table `tab:statrigor` (1016–1033) and any caption advertising "(week,seed) clusters". If keeping (week,seed), at minimum add the caveat that seeds are nested (identical labels/data) so the CI is a lower bound on width and the non-overlap is not robust.

### M4. Isolation figure caption says "52-week stream" but figure/numbers are forward-only 36 weeks
- **Location:** `paper/main.tex:1103` (Fig. `isopanels` caption); result file `figs/isolation_w16/isolation_metrics_fwd.json`.
- **Issue:** Caption reads "Isolation ablation (Week-16, 52-week stream)", but the three included panels (`*_fwd.png`, md5-identical to the forward-only run's panels) and every cited number come from the **forward-only 36-week run**. `isolation_metrics_fwd.json` has `n_test_weeks=36` and clean AUROC ours/naive = 0.878/0.958 — the exact numbers in the caption and Sec. VI (main.tex:1082). The genuine 52-week run (`isolation_metrics.json`) has `n_test_weeks=52` and clean AUROC 0.903/0.967, which do **not** match the figure. "52-week stream" also violates the reference-week rule (Week-16 = clean forward reference). The generating script's own suptitle for this run reads "forward-only, weeks >16".
- **Expected vs observed:** Caption "Week-16, forward-only (weeks 17–52, 36 weeks)" → caption says "52-week stream".
- **Fix:** Change "Week-16, 52-week stream" → "Week-16, forward-only stream, weeks 17–52" at main.tex:1103. Keep the forward panels (the cited 0.958/0.878 numbers come from them).

### M5. Uncorrected Week-16 forward tracking ρ is −0.983, not −0.987 (number-vs-file mismatch in a primary figure)
- **Location:** `paper/main.tex:862` (body), :882 (Fig. `mirror` caption), :918 (severity-1× anchor), comment :66.
- **Issue:** The paper reports the Week-16 forward uncorrected entropy-gap tracking correlation as ρ=−0.987, but the value computed from the underlying cache (`results/inference/week_16_inference/metrics_cache_ref16_grid9.npz`, forward weeks 17–52) is −0.9828 → **−0.983**. The figure-generating script (`plot_mirror_w16_forward.py`) formats with `{:.3f}`, so the **figure legend already shows −0.983**, contradicting its own caption/body. All other values verified correct: BBSE −0.988, RLLS −0.992, oracle −0.991, SLD-EM +0.030 (inversion confirmed). The :918 severity-1× anchor inherits the same wrong value (it is sourced from this same cache; `mirror_extreme_results.json` is 0 bytes).
- **Expected vs observed:** ρ=−0.983 → paper states −0.987 in 4 places.
- **Why major:** Number-vs-result-file mismatch (global rule 5) in a primary-result figure that internally contradicts its own legend. It is rounding-level (−0.987 vs −0.983) and changes no conclusion.
- **Fix:** Change −0.987 → −0.983 at main.tex:862, :882, :918 (the "−0.987 → −0.938 (10×) → −0.845 (100×)" chain), and comment :66. Leave BBSE/RLLS/oracle/SLD-EM unchanged. No figure regeneration needed (legend already shows −0.983).

### M6. Stale, retracted DANN result (−0.069) in committed draft section
- **Location:** `paper/sota_benchmark_section.md:66` (Table 1 DANN row); also :89, :98, :105–108.
- **Issue:** This tracked draft still reports the **pre-audit, retracted** DANN row "DANN Δ=−0.069" ("global alignment doesn't generalize forward", with the F1b "largest far-week drop" narrative). That result was explicitly retracted for a BatchNorm bug (`UDA_BENCHMARK_STATUS.md:15–36`) and replaced everywhere else by DANN diagonal Δ=**+0.006** ("no longer harmful, merely useless", `main.tex:714`). The md also carries a stale week-16 DANN −0.012 (:89).
- **Expected vs observed:** All committed artifacts report corrected DANN diagonal Δ=+0.006 → md asserts −0.069 (and −0.012), contradicting the retraction and main.tex.
- **Why major not blocker:** The actual submission artifact (main.tex) is correct; this is a contradicting tracked support doc.
- **Fix:** Regenerate `sota_benchmark_section.md` from the audit-fixed forward-only numbers (`scripts/analysis/regen_benchmark_table.py`), or delete it since main.tex supersedes it. Drop the −0.069 row/clause (:66, :98), rewrite the F1b narrative (:105–108), fix the :89 week-16 value to +0.006.

### M7. Intro cites a success-reporting paper as evidence of "generalization failures"
- **Location:** `paper/main.tex:259–261`.
- **Issue:** The intro lists "outright generalization failures \cite{seddigh2019framework}", but per `SUMMARIES.md:57–63`, seddigh2019framework (MLTAT) is a static single-snapshot, 10 Gbps framework reporting ~90% accuracy with **no drift, DA, or generalization-failure result of any kind** ("Useful only as background on feature engineering"). The paper that actually documents an ETC generalization collapse (CNN 99.2%→38.2%) is `shamsimukhametov2021ann`, which is **not in the bibliography** (0 hits). This is a mis-citation in the paper's central "Continuous Drift Fallacy" framing argument.
- **Expected vs observed:** A citation documenting a real ETC generalization failure → a success-reporting static-framework paper.
- **Fix:** Replace `\cite{seddigh2019framework}` at :261 with `\cite{shamsimukhametov2024ech}` (already in bib; documents ECH collapse F-score 38.4% and cross-country failure Germany 38.4%/USA 49.2%), or add a `\bibitem` for `shamsimukhametov2021ann`. Move seddigh2019framework to a feature-engineering/background context. (Also reconcile the `shamsimukhametov2022ech`↔`shamsimukhametov2024ech` key mismatch — see N7.)

---

## MINOR (defects worth fixing; no claim/number breakage in the paper body)

### m1. Abstract: "negative transfer **on stable classes**" not measured per-class
- **Location:** `paper/main.tex:188` (abstract).
- **Issue:** The negative-transfer tables (nt_tls1/nt_tls16/nt_quic) report only aggregate Mean acc / Macro-F1 / In-dist / Far columns; no per-class (stable-class) harm decomposition exists. The "warps the stable majority" mechanism (:332) is argued, not measured. The body even notes the aggregate harm is largest on the **far/drifted** weeks (:594–596), in mild tension with "on stable classes."
- **Fix:** Soften to "induces aggregate negative transfer (we argue the mechanism is warping of stable-class decision boundaries under a global update)" or "induces net negative transfer over the frozen baseline." Or add a per-class stable-class Δ-recall breakdown and cite it.

### m2. "Week-1 used *only* to validate Week-10" contradicts the headline Week-1 FAR result
- **Location:** `paper/main.tex:455` vs :924/:1258 (and abstract :188).
- **Issue:** Line 455 says "Week-1 is used \emph{only} to expose and validate the documented Week-10 event," but the headline anomaly-detection result (4.6% FAR, 0.987 AUROC, 5.4×/14× reduction) is computed on the **Week-1 stream** (`figs/auroc_anomaly_results.json` inference_dir = `results/inference/week_1_inference`) and presented as the primary result in the abstract, Sec. V-C, and Conclusion. Borderline because the paper justifies it (Week-1 carries the degraded positives, :924–925) and adds a Week-16 replication (:936–947).
- **Fix:** Soften "only" at :455 to acknowledge Week-1 additionally supplies the healthy-vs-degraded positives for the primary anomaly benchmark, keeping the Week-16-replication safeguard explicit. No numbers change.

### m3. OOD headline ranking computed on different per-detector populations, undisclosed
- **Location:** `scripts/analysis/ood_baseline_detector.py:274–287`; `paper/main.tex:1040–1042`.
- **Issue:** The five clean-regime AUROCs (naive 0.958, MSP 0.928, ours 0.878, energy 0.861, MFWDD 0.710) are presented as one flat ranking, but ours/naive use n_eval=4380/n_pos=1678 (full window) while energy/MSP/MFWDD use n_eval=3509/n_pos=1135 (10% embedding subsample, NaN where region <15 points). That is 871 class-weeks and 543 positives apart, undisclosed in the paper. Each number is individually correct; the script docstring acknowledges the subsample but the paper does not.
- **Fix:** Either compute all five on a shared finite-energy mask, or add a footnote at :1040–1042 (and Fig caption :1054–1057) stating energy/MSP/MFWDD are over n_eval=3509 vs ours/naive over n_eval=4380. Footnote is lower-risk (no number changes).

### m4. Deprecated scan script still emits disowned "BBSE stays flat" framing on disk
- **Location:** `scripts/analysis/natural_falsealarm_scan.py:193–309`; outputs `results/analysis/natural_falsealarm/{monitor_trace.json,injection_sweep.json}`.
- **Issue:** The docstring deprecates Steps 2/3 as "conceptually wrong"/"NOT supported by the data," but main() still **executes** them and rewrites the JSONs carrying the disowned framing every run (injection_sweep at frac=0.40: uncorrected 0.294 vs bbse 0.038, driven by amplifying a ~0.4%-natural-share app to 40%). **Mitigating:** main.tex cites none of these files (only the corrected `fig_natural_perclass_residual.png` at :1143); the overclaim is not in the paper. This is a reproducibility/hygiene defect.
- **Fix:** Guard Steps 2/3 behind an opt-in `--legacy` flag (default off) or delete them; delete the stale `monitor_trace.json`/`injection_sweep.json` so the disowned numbers cannot be cited. No paper change needed.

### m5. Stats-rigor described as "full-stream forward" — self-contradictory mislabel
- **Location:** `paper/main.tex:1004`.
- **Issue:** Calls it "a separate full-stream forward experiment," but the result file (`stats_rigor_ref16_fwd.json`, `config.forward=true`, weeks 17–52 only) is the forward-only restricted stream. The script docstring explicitly contrasts "full 52-week stream" vs "--forward (weeks >16)" — so "full-stream forward" is self-contradictory. The table caption (:1017) correctly says "Week-16 forward viral stream."
- **Fix:** Change "full-stream forward experiment" → "forward-only experiment" (or "forward viral-stream experiment") at :1004.

### m6. Repair stability bound |Δ|≤0.001 violated by microsoft-defender
- **Location:** `paper/main.tex:1217–1218` and caption :1235; `results/repair/few_shot_repair_w16_v01/metrics.json`.
- **Issue:** Text/caption claim stable-majority Macro-F1 unchanged with |Δ|≤0.001, but microsoft-defender shifts stable Macro-F1 by +0.00130 at k=50 (−0.00195 at k=1), exceeding the bound. The figure code computes |Δ|max=0.00130 over the four reported classes. The three **teleported** classes satisfy |Δ|≤0.001 (max 0.00022).
- **Fix:** Scope the bound to the teleported classes ("|Δ|≤0.001 across the teleported classes"), or relax to "|Δ|<0.002". Apply to both body (:1217–1218) and caption (:1235); avoid rounding stab_dmax 0.00130 to "0.001" in figure labels.

### m7. Stale TLS benchmark numbers in committed draft section
- **Location:** `paper/sota_benchmark_section.md:60–65` (Table 1), :83–88 (Table 2b).
- **Issue:** Pre-audit TLS week-1/week-16 numbers disagree with `regen_benchmark_table.py` and main.tex: Source-only 0.613 (actual 0.608), TENT bs64 −0.130 (−0.126), CoTTA bs64 −0.104 (−0.094), CoTTA bs256 −0.041 (−0.028); **week-16 AdaBN +0.007 — a sign error (forward-only is −0.011)**; week-16 Source-only 0.759 (actual 0.798) with the whole Δ column stale. main.tex (the actual paper) carries the correct numbers, so no published number is wrong — this is a stale-doc hazard.
- **Fix:** Regenerate Tables 1/2/2b in the md from `regen_benchmark_table.py` (forward-only, `results/inference_auditfix/`), or add a deprecation header pointing to main.tex as authoritative. (Same root cause as M6 — fix both in one regeneration pass.)

### m8. Self-referential stability guard for microsoft-defender
- **Location:** `scripts/analysis/few_shot_repair_loop.py:153–156, 184–186, 216–218`.
- **Issue:** microsoft-defender (class 98, teleported=False) is in `stable_classes`, so its repair row scores the "stability guard" over a set that still contains class 98 — the guard for that row is self-referential, not an independent negative-transfer measure. The three teleported targets are correctly excluded. Materiality is negligible (one class's F1 change in a 176-class macro average).
- **Fix:** Exclude the currently-repaired class from the stability set per row: `row_stable = stable_classes[stable_classes != c]` for both baseline and repaired evaluation. Does not materially move the |Δ|≤0.001 claim; cleanliness only.

### m9. Six fragile figure paths with spaces/"(1)" suffix
- **Location:** `paper/main.tex:498, 775, 819, 889, 1372, 1381`.
- **Issue:** Six active `\includegraphics` paths contain spaces and a " (1)" suffix (fragile/non-portable under graphicx/Overleaf). Four resolve (space-free copies already exist on disk); **two are missing entirely** (:775, :1381 — overlap with B2 and m11). 
- **Fix:** Point the four existing ones at the space-free copies (`fig_combined_penalties.png`, `fig_estimation_accuracy.png`, `fig_app_degradation_grid.png`, `fig_mirror_effect.png`); supply/rename the two missing assets or comment them. Avoid spaces in all `figs/` filenames repo-wide.

### m10. `figs/` not resolvable when compiling from `paper/`
- **Location:** `paper/main.tex` (all `\includegraphics`; no `\graphicspath`).
- **Issue:** main.tex lives in `paper/` but every figure path is `figs/<name>.png` with no `\graphicspath`; `figs/` is at the repo root (no `paper/figs`, no symlink). A build launched from `paper/` finds no figures; only a repo-root CWD works.
- **Fix:** Add `\graphicspath{{../figs/}{figs/}}` after `\usepackage{graphicx}` (line 135) so the build is CWD-agnostic. Or create a `paper/figs` symlink / document repo-root compilation.

### m11. Missing appendix figure `fig:app_trainfreq` (compile-breaking but documented cut-candidate)
- **Location:** `paper/main.tex:1381` (active `\includegraphics`).
- **Issue:** Active `\includegraphics{figs/model_is_good_on_stuff_it_saw (1).png}` references a nonexistent file; the caption itself says "CUT-CANDIDATE (removed from body)" and the TODO recommends dropping it. Label never `\ref`'d. Breaks compile despite documented intent to cut. (The very next float at :1391–1400 already handles this case correctly with a commented include + `\fbox` placeholder.)
- **Fix:** Comment out line 1381 (or the whole float at 1380–1389), mirroring the :1391–1400 convention. Listed under MINOR (not BLOCKER) because it is an acknowledged cut-candidate with a trivial one-line fix.

### m12. Four Related-Work citations have no canonical SUMMARIES.md note
- **Location:** `paper/main.tex:357–359, 398–404`.
- **Issue:** `jiang2022fgnet`, `bahramali2023netclr`, `deng2023ares`, `deng2024holmes` have no SUMMARIES.md entry, so their characterizations (esp. the detailed HOLMES claims: supervised, spatial decision regions, per-class region updates on drift) cannot be validated against the project's vetted notes. No characterization is shown to be wrong — a provenance/process gap.
- **Fix:** Add four SUMMARIES.md entries (each with the `\bibitem` backtick key) and verify the inline claims, especially the HOLMES description at :398–404, against the CCS 2024 paper; or add a one-line exception note that fingerprinting-adjacent citations are validated directly from PDFs.

### m13. `alexandari2020mlls` cited for a dissociation it does not make
- **Location:** `paper/main.tex:1164–1170`.
- **Issue:** "Estimation accuracy and detection robustness are therefore not the same objective \cite{alexandari2020mlls}" attaches an estimation-accuracy/calibration reference (MLLS+BCTS hard-to-beat) to the paper's **own** empirical estimation-vs-detection dissociation, which that paper does not study. No SUMMARIES.md note exists for it. (Its uses at :369 and :1267 are correct.)
- **Fix:** Drop `\cite{alexandari2020mlls}` from the "therefore … not the same objective" conclusion at :1169–1170 and present the dissociation as this work's own finding; if citing the EM/MLLS-accuracy antecedent, move the cite onto that clause.

---

## NITS (cosmetic / low-impact; fix opportunistically)

- **N1. "At s=1 (0.967 vs 0.903)" mislabels the clean baseline** — `main.tex:1158` and caption :1179. 0.967/0.903 are `auroc_clean` (no-injection), not the severity=1.0 sweep point (0.921/0.897) in `labelshift_sweep_results_estimators_fwd.json`. **Fix:** relabel as clean/no-injection baseline, or use the actual s=1.0 values 0.921/0.897.
- **N2. Two different Week-16 Source-only baselines (0.798 vs 0.795)** — `main.tex:669` (tab:nt_tls16) vs :713 (tab:nt_dann). **Merged with `wk16-source-acc-inconsistency`.** Both are correct: the diagonal table averages over weeks 17–52 (week 16 has no diagonal model), excluding the high-accuracy in-dist week. **Fix:** add a one-line footnote to tab:nt_dann; do NOT collapse to one number (would corrupt the +0.006 delta).
- **N3. Razor-thin CI non-overlap (0.001 margin)** — `main.tex:1010–1011`. Subsumed by M3; once M3's week-level clustering is applied, this claim is rewritten anyway. **Fix:** see M3.
- **N4. MLLS+BCTS prose 0.828 vs table 0.827** — `main.tex:941` vs :994. Result file (`auroc_anomaly_results_ref16.json`, em_bcts viral) = 0.82746 → 0.827. **Fix:** change :941 to 0.827.
- **N5. MFWDD clean AUROC 0.710 (OOD sec) vs 0.723 (isolation sec)** — `main.tex:1042` vs :1082. Two distinct streams (n_eval 3509 vs 3548), each correct. **Fix:** add a clause noting they are separate Week-16 forward streams.
- **N6. Isolation Panel-D caption "vs 35–70%" lumps two baselines** — `main.tex:1106` vs body :1092 (naive 35–46% / MFWDD 44–70%). **Fix:** use the body's separated ranges; also resolve the adjacent `\TODO{caption later.}`.
- **N7. ECH citation key mismatch `shamsimukhametov2024ech` (main.tex) vs `shamsimukhametov2022ech` (SUMMARIES.md:74)** — same correct paper. **Fix:** standardize on 2024ech (true pub year) and update SUMMARIES.md. (Tie in with M7.)
- **N8. OOD "all collapse toward chance" overstates** — `main.tex:1045`. Contaminated AUROCs land at 0.66–0.68, not near 0.5. (Note: this understates competitors, not self-serving.) **Fix:** "drop sharply (to ~0.66–0.68)".
- **N9. Stale advisor comment with different proprietary numbers** — `main.tex:751` (comment: TENT −0.213 / CoTTA −0.278) vs body :627 (−0.211 / −0.173). Comment only; not rendered. **Fix:** delete or update the comment block at :749–753.
- **N10. eset-edtd drift_week 18 (repair code) vs "≈wk17" (paper)** — `few_shot_repair_loop.py:78` vs `main.tex:232/472/513`. Conservative (wk18 support is strictly post-drift; no leakage). **Fix:** align to 17, or fix the misleading code comment at :75 noting the one-week buffer.
- **N11. Four orphan appendix labels** — `main.tex:1388, 1421, 1437, 1445` (`fig:app_trainfreq/_3det_1320/_overview/_iso_w1`) never `\ref`'d. Cosmetic (no warnings). **Fix:** drop the unused `\label`s or delete the cut-candidate floats (ties in with B3/m11).
- **N12. Dead `ks_sweep` expression** — `scripts/viz/plot_repair_before_after.py:36` (`d["k_list"] and [...]` truthiness trick, never read). **Fix:** delete the accumulation line and its tuple-unpack init at :29.

---

## Not-a-bug / dismissed (skeptic marked is_real=false)

- **`symmetric-uncorr-rho-baseline`** (proposed −0.987→−0.983 for the *symmetric control* s=1 anchor): the symmetric run has **no** s=1 point (severities default to [10,100]); the −0.987 anchor is the legitimate clean forward-mirror s→1 limit. The −0.983 "run" value exists only in a hand-typed (and internally inconsistent) status note; the result JSON is empty. No paper change needed. *(Distinct from M5, which is the genuine −0.987→−0.983 error in the Week-16 forward **uncorrected tracking** value.)*
- **`quic-tent-bs64-delta-rounding`** (−0.065 vs −0.066): different statistics — single representative run (tab:nt_quic, −0.065) vs 5-seed mean (tab:nt_seedvar, −0.066 ± 0.0061). Consistent within std; forcing equality would misreport the seed-mean.
- **`conclusion-actively-harmful-overclaim`**: the sentence at :1248 is self-qualifying ("global adaptation is actively harmful — gradient methods … while the no-gradient AdaBN control does no harm"). The critic isolated the lead clause; the full sentence is accurate.
- **`natural-surge-honest-ok`** / **`natural-falsealarm-perclass-verified-clean`**: verification records — the per-class natural-surge analysis is honestly framed, threshold non-cherry-picked, all numbers match `perclass_residual.json`, discloses 2/4 benign false alarms and claims no BBSE advantage. No defect.
- **`allot-tsne-tracked-figs`** (the 26 `figs/allot_tsne` PNGs as a *paper* double-blind violation): downgraded to not-a-paper-defect because "allot" never reaches the compiled PDF (the t-SNE slot uses the anonymized `fig_tsne_closedworld_anon.png`). **However**, the underlying committed-name leak IS real and is captured under **M1** as a repo-wide release-hygiene issue — not dropped, re-scoped.
- **`bootstrap-mechanics-correct`**: audit confirmation — the cluster percentile bootstrap (n_boot=2000) is non-degenerate and reproduces all reported numbers. (Independent of M3, which concerns the *choice of cluster unit*, not the bootstrap mechanics.)
- **`table-numbers-match`** / **`dann-honestly-pending`**: confirmations — the closed-world table reproduces (Source-only 0.664, AdaBN −0.030, TENT −0.211, CoTTA −0.173, 21/113 cliffs), emitted artifacts are anonymized, and the paper correctly scopes DANN/multi-seed as "still running." No defect.
- **`shap-numbers-and-ranking-verified-sound`** / **`class-idx-and-counts-verified`** / **`drift-week-label-inconsistency`**: confirmations — eset-edtd SHAP ranking (SIZE 4.38 ≫ DIR 1.50 > IPT 0.49) is seed-stable, class idx 57 / 180 classes / latent jump 5.3→8.3→12.9 / AUC 0.92 all match files; the wk17/wk18 labels are already reconciled in main.tex:521–522 (jump completes at wk18) and drift_week=18 is just the post-onset support-pool start.

---

## Confidence & coverage

**Re-derived and matched (high confidence):**
- Stats-rigor (M3, m5, N3): recomputed from `stats_rigor_ref16_fwd.json` and `stats_rigor.py`; the week-vs-(week,seed) clustering inflation and CI overlap were reproduced (BBSE [0.634,0.745] vs uncorrected [0.585,0.669]). This is the single most consequential finding — it flips a stated CI-non-overlap claim.
- Mirror-effect ρ (M5): recomputed pearsonr over `metrics_cache_ref16_grid9.npz` forward weeks 17–52 → −0.9828; BBSE/RLLS/oracle/SLD-EM all confirmed correct.
- Isolation panels (M4): md5-matched the included `*_fwd.png` to the forward-only run; AUROC 0.878/0.958 ↔ `isolation_metrics_fwd.json` (n_test_weeks=36).
- Benchmark numbers (M6, m7, N2): `regen_benchmark_table.py` output matches main.tex; the md draft is stale (incl. the AdaBN sign error).
- Closed-world / SHAP / eset case study: fully reproduced (dismissed confirmations above).
- Double-blind (M1): `git ls-files | grep -i allot` → 179 paths; `slice_manifest.json` absolute-path leak confirmed.
- Citation cross-checks (M7, m12, m13, N7): verified against `SUMMARIES.md`.

**Solid claims (no defect found):** the natural-surge per-class analysis, the SHAP feature-importance ranking, the closed-world negative-transfer table, the bootstrap mechanics, and the DANN "still running" scoping are all sound and honestly framed.

**Still needs a human / Overleaf check:**
- **LaTeX compile (B1, B2, B3, m11, m9, m10): pdflatex is not installed locally, so the build was NOT compiled.** The missing-file and `\graphicspath` findings are inferred from file-existence checks and path analysis, not from a failed compile log. A single `pdflatex`/`latexmk` run from the repo root will confirm B1–B3 abort the build and surface any further missing assets; a run from `paper/` will confirm m10.
- **Figure regeneration decisions (M2, M4):** swapping the hero panel (b) vs relabeling is an authorial call; verify the chosen replacement class's `teleported`/`drift_week` against `metrics.json`.
- **Repo-release decision (M1):** whether the repo ships as supplementary material determines if M1 is a blocker; the scrub + history-rewrite should be verified with `git grep -i allot` returning empty.
