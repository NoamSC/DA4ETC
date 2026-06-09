# Binary Anomaly Detection & AUROC Experiment — Results

Fills the paper's Sec. V-C placeholders:
- Abstract `[TODO: ... reducing the false-alarm rate from XX% to YY% at matched detection]`
- `[PLACEHOLDER: Figure X — AUROC detection curves ... vs uncorrected entropy and an MFWDD-style detector]`

**Code:** [scripts/analysis/auroc_anomaly_detection.py](scripts/analysis/auroc_anomaly_detection.py)
**Reproduce:** `python scripts/analysis/auroc_anomaly_detection.py` (ref Week 1, seed 42)
**Artifacts:** [figs/fig_auroc_anomaly.png](figs/fig_auroc_anomaly.png), [figs/fig_auroc_score_dist.png](figs/fig_auroc_score_dist.png), [figs/auroc_anomaly_results.json](figs/auroc_anomaly_results.json)

---

## Headline (drop into the abstract)

> Under benign viral-spike volatility, at a matched **95 % detection rate** the BBSE-corrected
> residual cuts the **false-alarm rate to 4.6 %**, versus **25.0 %** for raw uncorrected entropy
> and **64.8 %** for an MFWDD-style global detector — a **5.4×** and **14×** reduction
> (AUROC 0.987 vs 0.959 vs 0.677).

At a stricter 90 % detection rate the gap is even sharper: FAR **2.8 %** (BBSE) vs **12.0 %**
(uncorrected) vs **59.3 %** (MFWDD).

---

## Why correlation alone was not enough (the point of this experiment)

On *clean* data every detector is excellent — uncorrected entropy, the BBSE residual, and the
MFWDD-style global score all reach **AUROC = 1.000** (left panel of `fig_auroc_anomaly.png`).
This mirrors the near-identical correlations in Fig. 9 (ρ ≈ −0.99 for both gaps) and confirms the
paper's claim that *the correction does not, and is not intended to, improve tracking on clean data*.

The advantage is **purely operational** and appears only under volume volatility: a detector can stay
highly correlated yet still fire spurious threshold crossings. The AUROC/false-alarm axis is what
separates the methods.

## Experimental design

- **Task.** A monitor sees only unlabelled softmax/predictions for a weekly window and must raise an
  alarm iff the model has *genuinely* degraded (covariate-shift-driven F1 collapse), without firing on
  benign traffic-volume volatility.
- **Ground truth.** `anomaly(week) = 1` iff the week's true (full-data) Macro F1 < 0.60. On
  CESNET-TLS-Year22 (ref = Week 1) this cleanly splits **weeks 1–9 healthy** (F1 0.72–0.76) from
  **weeks 10–52 degraded** (F1 0.34–0.57). A viral volume spike leaves the model weights and per-class
  conditionals untouched, so it never changes a week's degradation status — the label is fixed by the
  source week, the spike is pure nuisance.
- **Benign event = viral spike.** One randomly chosen app surges to a random fraction *f ∈ [0.30, 0.85]*
  of an 8 000-flow window, remainder at the natural mix. **Benign check:** on healthy weeks the spiked
  app's own per-class F1 stays at **0.83**, confirming no real degradation — every healthy-week alarm is
  a false alarm.
- **Two regimes.** *Clean* (natural mix) and *Viral* (benign spikes). 936 windows total
  (52 weeks × 6 clean + 12 viral).
- **Detectors** (all unsupervised, from the window only):
  - **Uncorrected entropy gap** — per-sample entropy minus a fixed train-prior baseline (Fig. 9's
    uncorrected line).
  - **MFWDD-style global drift** — feature-importance-weighted per-channel 1-Wasserstein distance of
    the window's softmax channels vs. the reference week (Σ w = 1); a global, label-shift-sensitive
    trigger, faithful to MFWDD's mechanism.
  - **BBSE-corrected residual (ours)** — per-sample entropy minus the BBSE-reweighted expected entropy
    (Fig. 9's corrected line).

## Results

| Regime | Detector | AUROC | FAR @ 90 % TPR | FAR @ 95 % TPR |
|---|---|---|---|---|
| Clean | Uncorrected entropy gap | 1.000 | 0.0 % | 0.0 % |
| Clean | MFWDD-style global | 1.000 | 0.0 % | 0.0 % |
| Clean | **BBSE-corrected (ours)** | 1.000 | 0.0 % | 0.0 % |
| **Viral** | Uncorrected entropy gap | 0.959 | 12.0 % | 25.0 % |
| **Viral** | MFWDD-style global | 0.677 | 59.3 % | 64.8 % |
| **Viral** | **BBSE-corrected (ours)** | **0.987** | **2.8 %** | **4.6 %** |

The MFWDD-style global detector is the **worst** under benign label shift — exactly the paper's thesis
that a global, label-shift-sensitive trigger collapses when the class mix moves without real degradation.

## The mechanism (for the `fig_auroc_score_dist.png` discussion)

A viral spike on a **high-uncertainty** app inflates the uncorrected per-sample entropy on a *healthy*
week up to ≈ 1.05 — squarely inside the genuine-degradation band (clean degraded weeks: 0.94–1.07).
The BBSE correction recognises that app as over-represented, raises the expected entropy accordingly,
and pulls the residual back to ≈ 0.16–0.41, well below any threshold that still detects real degradation.
Low-uncertainty spikes inflate neither detector; the false-alarm risk is concentrated exactly where BBSE
helps.

## Suggested figure captions

- **Fig. (AUROC):** *Binary anomaly detection — genuine covariate degradation vs. benign volume
  volatility.* On clean data (a) all detectors are perfect; under benign viral spikes (b) the
  BBSE-corrected residual (AUROC 0.987) dominates raw entropy (0.959) and the MFWDD-style global
  detector (0.677). Markers = operating point at 95 % matched detection.
- **Fig. (score dist):** *Why the uncorrected gap false-alarms.* A healthy-week viral spike lands in the
  degradation band for the uncorrected gap (orange crosses the 95 %-detection threshold), but the BBSE
  correction keeps it below threshold.

## Honest caveats / notes

- The advantage requires the **right benign perturbation**. A *symmetric* LogUniform label shift (as in
  the existing `fig_mirror_extreme.png`) barely moves either gap — on CESNET the covariate collapse
  (F1 0.76 → 0.34) dwarfs symmetric label-shift noise, so both gaps stay ~equally good (ρ −0.979 vs
  −0.969). The decisive case is an **asymmetric single-app viral spike** that concentrates on a
  high-uncertainty class, which is the realistic "viral traffic spike" the paper already describes.
- The viral app is drawn **uniformly at random** among apps with ≥ 20 flows in the week (not cherry-picked
  to high-entropy apps), so the reported FAR is an unbiased average over which app goes viral.
- Numbers are reproducible (seed 42). They shift by ≈ 1–2 pp with the pool size (`--clean_seeds`,
  `--viral_windows`) but the ordering BBSE ≫ uncorrected ≫ MFWDD is stable.
