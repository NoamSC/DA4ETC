# Hand-off to main agent — figures + audit findings (no edits made)

All claims verified against the raw per-week npz (frozen W16 NCM recall), not the JSON.

## 1. Hero figure (main.tex fig:hero) — fix M2 violation (HIGH)
Panel (b) is currently `microsoft-defender`, which is `teleported=false` (gradual decline,
drift_week 22) — this re-introduces exactly what M2 flags. Replace panel (b) with a genuine
teleporter. docker-registry is the cleanest (0.98 one-week cliff at w27, jump/spread 5.6) but is
already panel (c); so either promote docker to the showcase slot and put a different true
teleporter in (c), or use eset-edtd (already (d)) plus firefox-settings/gitlab/apple-weather for
week-diversity. Do NOT use skype or apple-location (neither is a clean teleporter).

## 2. Hero caption week numbers — correct against data (HIGH)
- docker-registry: "≈ Week 28" → "≈ Week 27"  (recall 1.00→0.02 at w27; matches t-SNE jump w26→27).
- eset-edtd:       "≈ Week 17" → "≈ Week 18"  (drop completes at w18; M2 agrees).

## 3. fig_class_drop_w16_selected_grid caption (line 478) — do NOT swap to apple-location
apple-location has no cliff (it rises post-source, max single-week drop 0.16 at w50). To fix the
"docker & skype both ≈wk28" redundancy, replace skype with a real different-week teleporter —
firefox-settings (≈w43), gitlab (≈w40), or apple-weather (≈w30) — and regenerate the grid figure
(scripts/viz/plot_class_drop_w16_selected.py: add the chosen class id, drop 140). skype is itself
a gradual decliner, not a teleport, so removing it from a "sudden drop" grid is justified anyway.

## 4. Audit items needing the main agent
- M3: change stats_rigor.py:297 cluster id (week,seed)→week, re-derive CIs, rewrite main.tex:1010
  ("CIs overlap under week-level clustering; separation is suggestive, carried by point estimates +
  threshold-sweep dominance").
- m3: add a footnote at main.tex:1040–1042 / Fig caption disclosing energy/MSP/MFWDD use n_eval=3509
  vs ours/naive 4380 (or recompute all five on a shared finite-energy mask).
- M4: main.tex:1103 caption "Week-16, 52-week stream" → "Week-16, forward-only stream, weeks 17–52"
  (figure/numbers are the 36-week forward run; the real 52-week run gives different AUROCs).
- m4: guard natural_falsealarm_scan.py Steps 2/3 behind --legacy (default off) and delete the stale
  monitor_trace.json / injection_sweep.json so the retracted "BBSE stays flat" numbers can't be cited.
- M6: regenerate sota_benchmark_section.md from regen_benchmark_table.py (or delete it); kill the
  retracted DANN −0.069/−0.012 rows, set DANN diagonal to +0.006.

## 5. eset-edtd case-study validation (#9)
Main agent should launch this (forks can't spawn subagents). Seed it with: verify w18+ sample counts
for class 57 aren't an empty-bin artifact; identify the teleport destination class (single attractor
vs scatter); rule out a sensor cause at w18; reconcile caption week (w17 onset vs w18 completion).

## Supporting data (frozen W16 NCM recall per week, from npz)
- docker-registry (49): w16–26 ≈1.00, w27 0.02, w28+ 0.00  → cliff AT w27.
- skype (140):          0.72(w16)→0.60(w23)→0.42(w27)→0.11(w28), stays low → gradual, no jump.
- eset-edtd (57):       0.98(w16)→0.66(w17)→0.00(w18+), permanent → cliff completes w18.
- apple-location (19):  0.71(w16)→0.89(w30)→~0.55(w39), max drop 0.16 at w50 → NO cliff.

## Coordination note
Items 1–2 and M4 touch main.tex captions other forks (M2/M4/M5) may also edit — serialize to avoid
conflicts. This fork made no file edits.
