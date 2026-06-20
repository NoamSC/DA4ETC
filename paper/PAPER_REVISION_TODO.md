# Paper revision status (overnight run, 2026-06-21)

All of the user's "general" + local `[Todo:]` comments. [x]=done, [~]=in-progress (figure agent).

## GENERAL COMMENTS
- [x] G1  Abstract simplified (global-adaptation sentence) + op-loop/repair sentence added.
- [x] G2  ECH "knowing *when* to retrain is not enough" economics added to operational section.
- [x] G3  Illustration `fig:concept` removed entirely + all refs.
- [x] G4  `decomp_w1`+`decomp_w16` merged into one figure 5(a)/5(b) (`fig:decomp`).
- [x] G5  eset-edtd VERIFIED — **anchor was wrong**: ESET rename announced 28 Mar 2022 (~ISO wk13, NOT 27 Apr/wk17); "backend/endpoint/proxy changes" CONTRADICTED (rename only; edtd.eset.com endpoints unchanged; only server-IP change was 2025). Case study softened: date corrected, mechanism claim dropped, reframed as temporally-adjacent context, SHAP moved to appendix. **NEEDS USER REVIEW** — the anchor is now much weaker.
- [x] G6  Tables I–IV (+ multi-seed V) merged into one `tab:nt_merged`.
- [x] G7  "weaponize" → "leverage" everywhere (section title + body + comment).
- [x] G8  "arm" → "branch"/"regime" throughout Sec V.
- [x] G9  SLD-EM validated — **REAL finding, not a bug** (MLLS maximizes soft-posterior likelihood, absorbs degradation entropy into its prior → residual collapses). No change needed.
- [x] G10 Sec VI intro paragraph added (4-step roadmap).
- [x] G11 "Threat model" → "Two benign stressors", rewritten, numbers dropped.
- [x] G12 Solved by table merge — paper now 14pp (was 16), no large empty areas.
- [~] G13 Fig 14 "ref week 0"→"ref 1": figure agent regenerating with --reference_week 1 (may shift Table VI numbers → will sync).
- [x] G14 MFWDD validated — **FAIR, not a bug** (AUROC 1.00 under stable volume; collapses only under benign spikes, the thesis). No change.
- [x] G15 Table VII replaced with ROC figure `fig:auroc_w16` (fig_auroc_anomaly_ref16.png).
- [x] G16 "Statistical rigor" paragraph rewritten/shortened.
- [~] G17 "ours" relabel: tables/text done; figure agent relabeling figure legends → "BBSE-corrected residual".
- [x] G18 OOD-baselines + Isolation rewritten around the fwd_ood sweep figure; real label-shift ratios folded in (median 5×, p90 30×, p95 100×+, lower bound on network swings).
- [x] G19 Surge ratios VERIFIED correct (within-class temporal ratios; per-service sampling cancels, <2%). Footnote added.
- [x] G20 Fig 17 isopanels replaced in body by the fwd_ood sweep figure; A/B/C/D panels → appendix `fig:app_isopanels`.
- [x] G21 Citations: missing UDA refs added @ the UDA/CTTA foil sentence; Regime-2 cite set aligned to named estimators; jiang2022fgnet moved to per-class sentence.
- [x] G22 Reference integrity: 9 bibitems fixed (cesnetdrift2025 title+authors, ganin pages, jancicka authors, mfwdd/holmes pages, ziegler/lee years, li2017lwf year, shahraki title+authors). No fabricated refs.

## LOCAL [Todo:] inline comments
- [x] ALL ~50 inline `[Todo:]`/`[TODO:]` markers resolved/removed (verified: 0 remain; none leak into the PDF).
- [x] All `\TODO{caption later}` captions written, except the intentional proprietary-tSNE disclosure toggle.

## VALIDATION (final pass)
- [x] Compiles clean (14pp), no undefined refs/citations, no missing figures.
- [x] No "weaponiz", no "Allot", no stray "[Todo"/"[TODO" in source/PDF.
- [~] Figure agent (ref-1 regen + ours-relabel) finishing → recompile + sync numbers.
- [ ] Final critic agent re-reads the PDF.
- [ ] Regenerate final main.pdf + refresh paper_figs/ bundle.
