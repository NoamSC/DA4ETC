# Concept-drift validation: `docker-registry` (class 49), Week 27→28 of 2022

Empirical validation of a hypothesized **sharp, discrete concept jump** in the
encrypted-traffic embeddings of the `docker-registry` application around the
Week 27 / 28 (2022) boundary in CESNET-TLS-Year22.

- **Dataset:** `/home/anatbr/dataset/CESNET-TLS-Year22_v2/WEEK-2022-{25..30}/{train,test}.parquet`
- **Class:** `docker-registry` (alphabetical label index **49**), SNI is constant
  `registry-1.docker.io` across all weeks (same service — the drift is in *how*
  it talks, not *what* it is).
- **Encoder:** Week-1 `Multimodal_CESNET`
  (`exps/cesnet_multimodal_each_week_train_v01/week_1/weights/best_model.pth`),
  inputs = PPI `(3,30)` = [IPT, DIR, SIZE] + 44 flow-stats → 600-d embedding.
- **Scripts:** `scripts/drift_validation/{drift_features,profile_docker_drift,validate_drift_shap}.py`
- **Outputs:** `results/docker_drift/*`, `figs/docker_drift/*`

---

## Task 1 — Raw-data feature profiling (PRE W25–27 vs POST W28–30)

~4 000 flows/week. Ranked by effect size (Cohen's *d*); every top feature is
significant at p ≪ 1e-300 (Welch t + Mann-Whitney). The per-week trajectory
(`task1_weekly_trajectory.png`) shows each is **flat for W25–27, steps at W28,
then flat again** — i.e. a discrete jump, not gradual drift.

| Feature | PRE mean | POST mean | %Δ | Cohen *d* | Interpretation |
|---|---:|---:|---:|---:|---|
| `ppi_size_min` | 23.0 | 5.9 | **−74%** | −2.07 | tiny (~ACK/keep-alive) packets appear |
| `ppi_n_rev` | 7.07 | 9.06 | +28% | +2.01 | more server→client packets |
| `ppi_rev_size_mean` | 628 | 360 | **−43%** | −1.94 | server packets much smaller |
| `bytes_per_pkt_rev` | 558 | 420 | −25% | −1.90 | reverse payload less full |
| `PACKETS_REV` | 12.7 | 16.6 | +31% | +1.76 | more reverse packets |
| `total_packets` | 26.5 | 32.9 | +24% | +1.43 | more segments per flow |
| `ppi_size_mean` | 524 | 371 | −29% | −1.42 | smaller packets overall |
| `PPI_LEN` | 12.0 | 14.2 | +18% | +1.39 | longer packet sequences |
| `ppi_size_std` | 535 | 455 | −15% | −1.22 | size distribution compresses |

**TLS handshake (proxy for protocol/version):** the JA3 fingerprint **mix
flips** at the boundary — the dominant PRE fingerprint `8269c5d3…` falls 48% → 36%
while `e8c56fa4…` rises 28% → 40% (`task1_tls_ja3_shift.png`). SNI is unchanged.
This is the signature of a **client TLS-stack / version change** (e.g. a Docker
client or registry TLS-library update) rather than a new service.

## Task 2 — SHAP attribution on the latent space (`validate_drift_shap.py`)

**(A) Latent jump is real but partly absorbed by the model.** Mean L2 distance
of each week's embeddings from the Week-1 `docker-registry` centroid rises
pre-avg **9.34 → post-avg 10.07** (`task2_latent_jump.png`). The absolute
centroid step is modest because the Week-1 encoder is partially invariant — the
sharper signal is in the *distributional* separability below.

**(A2) Encoder SHAP (GradientExplainer, target = distance-from-Week-1).** What
moves the embedding is overwhelmingly the **packet-size sequence** and
**direction sequence**, an order of magnitude above any flow-statistic:

```
PPI_SIZE(seq)   mean|SHAP| = 2.49   ← dominant
PPI_DIR(seq)    mean|SHAP| = 1.41
PHIST_DST_SIZES_3        = 0.12
PPI_IPT(seq)             = 0.11      (timing barely matters)
```

**(B) Pre-vs-post domain classifier + TreeExplainer — hold-out AUC = 0.91.**
Pre and post periods are almost fully separable from raw features alone
(strong drift). Top-10 drivers (`task2_shap_beeswarm.png`, `task2_shap_bar.png`):

```
ppi_rev_size_mean  |SHAP|=1.04  high→PRE   (628 → 360)
ppi_size_min       |SHAP|=0.90  high→PRE   ( 23 →   6)
ppi_n_rev          |SHAP|=0.51  high→POST  (7.1 → 9.1)
ppi_size_max       |SHAP|=0.29  high→PRE
bytes_per_pkt_rev  |SHAP|=0.28  high→PRE   (558 → 420)
DURATION           |SHAP|=0.14  high→POST  (0.76→0.96)
pkt_download_ratio |SHAP|=0.13  high→POST
ppi_size_std       |SHAP|=0.12  high→PRE
ppi_iat_std        |SHAP|=0.11  high→PRE
ppi_size_mean      |SHAP|=0.07  high→POST
```

Both SHAP views agree with Task 1: **packet sizes and the directional packet
pattern**, not inter-arrival timing, drive the jump.

---

## Conclusion — what caused the acute latent jump

Starting at **Week 28, 2022**, `docker-registry` flows (same endpoint,
`registry-1.docker.io`) abruptly switched to a **finer-grained
segmentation / framing pattern on the server→client direction**:

1. **Packet-size re-framing (primary cause).** Reverse-direction packets shrank
   ~43% (628→360 B mean) and the minimum packet size collapsed 23→6 B (tiny
   ACK/keep-alive packets now appear) — the response payload is delivered in
   **more, smaller segments** instead of fewer large ones. This is the single
   biggest mover of both the raw distribution and the encoder's embedding
   (`PPI_SIZE(seq)` dominates the encoder SHAP).
2. **More reverse packets / longer flows.** `PACKETS_REV` +31%, `ppi_n_rev`
   +28%, `PPI_LEN` +18%, total packets +24% — consistent with the same data
   split across more frames, shifting the directional pattern (`PPI_DIR(seq)`
   is the #2 encoder-SHAP driver).
3. **TLS client/version change.** The JA3 fingerprint composition flips at the
   boundary (dominant fingerprint 48%→36%, a second one 28%→40%), indicating a
   **client TLS-stack or protocol-version update** — the most likely root cause
   of the new framing behavior.
4. **Not a timing change.** Inter-arrival statistics (`PPI_IPT`, `ppi_iat_*`)
   contribute negligibly to the embedding shift.

In short: the jump is a **server-side response re-segmentation + TLS-client
version change** — *packet truncation/framing and protocol-version differences*,
not timing — that the Week-1 multimodal model "sees" almost entirely through the
**packet-size and direction sequences**.
