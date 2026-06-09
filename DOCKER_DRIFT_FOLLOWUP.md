# docker-registry drift (W27→W28 2022) — follow-up notes

Status: **done & validated.** Hypothesis (sharp discrete concept jump for
`docker-registry`, class 49, at the Week 27/28 boundary) is **confirmed**.
This file is the quick "where things are + what to do next" reference; the full
writeup is [DOCKER_DRIFT_FINDINGS.md](DOCKER_DRIFT_FINDINGS.md).

## TL;DR result
At **Week 28, 2022**, `docker-registry` flows (same endpoint
`registry-1.docker.io`) abruptly re-segment: server→client packets become
**smaller and more numerous** (`ppi_rev_size_mean` 628→360 B, `ppi_size_min`
23→6 B, `PACKETS_REV` +31%), and the **TLS JA3 fingerprint mix flips** (client
TLS-stack/version change). SHAP says the encoder feels the jump almost entirely
through the **packet-size + direction sequences**, not timing.
- Raw drift effect size: Cohen's *d* up to ~2.0 (p ≪ 1e-300).
- Pre-vs-post separability: domain-classifier **AUC = 0.91**.
- Encoder SHAP top drivers: `PPI_SIZE(seq)` (2.49) ≫ `PPI_DIR(seq)` (1.41) ≫ rest.

## How to re-run
```bash
P=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python   # has shap 0.44.1
$P scripts/drift_validation/profile_docker_drift.py      # Task 1 (CPU, ~1 min)
CUDA_VISIBLE_DEVICES=0 $P scripts/drift_validation/validate_drift_shap.py  # Task 2 (GPU)
```

## Artifacts
| Path | What |
|---|---|
| `scripts/drift_validation/drift_features.py` | shared extractor (docker-registry, model-exact inputs) |
| `scripts/drift_validation/profile_docker_drift.py` | Task 1 raw profiling |
| `scripts/drift_validation/validate_drift_shap.py` | Task 2 SHAP (encoder + domain classifier) |
| `results/docker_drift/task1_feature_stats.csv` | per-feature mean/median/var, t/MWU, %Δ, Cohen d |
| `results/docker_drift/task1_summary.txt`, `task2_summary.txt` | text summaries |
| `results/docker_drift/task2_latent_jump.csv` | per-week embedding distance from W1 baseline |
| `results/docker_drift/task2_shap_ranking.csv`, `task2_encoder_shap_ranking.csv` | SHAP rankings |
| `figs/docker_drift/task1_weekly_trajectory.png` | shows flat→step→flat at W28 |
| `figs/docker_drift/task1_tls_ja3_shift.png` | JA3 fingerprint flip |
| `figs/docker_drift/task2_shap_beeswarm.png` / `_bar.png` | pre-vs-post drivers |
| `figs/docker_drift/task2_latent_jump.png`, `task2_encoder_shap_bar.png` | latent drift + encoder SHAP |

## Caveat to keep in mind
The encoder's **absolute** centroid-distance step at the boundary is modest
(9.34→10.07) — the Week-1 model is partially invariant. The drift is
unambiguous in the raw distributions and domain-classifier AUC; don't rely on
centroid distance alone.

## Suggested follow-ups
- [ ] **Confirm root cause externally:** map the two JA3 fingerprints to a TLS
  client/library + version (e.g. Docker/containerd or Go TLS bump around
  Jul 2022) to nail "protocol-version difference."
- [ ] **Generality:** repeat the W27→28 split for a few other classes — is this
  docker-specific or a dataset-wide collection/parsing change at W28?
  (`drift_features.extract_week` only needs the `APP=` filter swapped.)
- [ ] **Downstream impact:** check the Week-1 model's per-class accuracy on
  `docker-registry` W25→30 (results already exist under
  `results/inference/week_1_inference/WEEK-2022-2{5..9}.npz`,
  `WEEK-2022-30.npz`) to quantify whether the framing jump actually hurts
  classification or is absorbed.
- [ ] **TTA relevance:** if accuracy drops at W28, this is a clean test case for
  the TENT/CoTTA runs (`scripts/inference/run_inference.py --method {tent,cotta}`).
- [ ] **Sharper localization:** the current split is weekly; a per-day run across
  2022-07-04…07-17 would pin the change to a calendar day.

## Notes
- Script lives under `scripts/drift_validation/` (repo `scripts/<stage>/`
  convention), not repo root.
- `shap` was newly installed into the `ml2` env; nothing else changed there.
