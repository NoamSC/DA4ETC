# Literature notes

Thesis-focused summaries of the PDFs in this directory, for writing the paper
*"The Illusion of Continuous Drift: Disentangling Discrete Covariate Shift from
Label Shift in Encrypted Traffic for Unsupervised Monitoring."*

Each entry was produced by reading the source PDF directly (one paper per agent,
no cross-contamination). Numbers are quoted from the papers. The citation key in
`backticks` on each Citation line maps to the paper's `\bibitem`. Entries are
grouped by role; each ends with an explicit relevance note (Core / Supporting /
Background) so you can tell load-bearing references from context.

- **Core** — dataset we use, or a method we build on / argue against directly.
- **Supporting** — independent evidence for our discreteness / per-class claim.
- **Background** — context, baselines, or operational-paradigm references.

---

## 1. Datasets (Core)

### CESNET-TLS-Year22: A Year-Spanning TLS Backbone Traffic Dataset (s41597-024-03927-4.pdf)
**Citation:** Karel Hynek, Jan Luxemburk, Jaroslav Pešek, Tomáš Čejka, Pavel Šiška — *Scientific Data* (Nature) 11:1156, 2024. `hynek2024cesnet`
**TL;DR:** A data descriptor for CESNET-TLS-Year22, a full-year (2022) TLS flow dataset captured on a national ISP backbone, designed for time-aware traffic classification and data-drift research.
**Problem:** Public traffic-classification datasets lack long time spans and date-time info, preventing time-consistent evaluation and study of how ML models degrade as traffic evolves.
**Method/Dataset:** Bidirectional TLS flows collected at 100 Gbps probes of CESNET3 (~half a million users) via ipfixprobe/ipfixcol2/NEMEA, filtered to TCP dst-port 443, dynamically sampled per service then uniformly 1:10. Each flow has packet sequences (first 30 packets), packet/IPT histograms, flow statistics, and the SNI domain for ground-truth labels.
**Key facts:** 180 web-service classes in 24 categories; 53 weekly snapshots (WEEK-2022-00..52); covers 45% of network TLS traffic; PPI max length 30; 8 log-scaled histogram bins. Per-week model tested on T+1..T+8: avg T+1 accuracy 96.3%, dropping to 90.5% at T+8.
**Relevance (Core):** Primary dataset. Independently documents the discrete drift events central to our thesis: a step-change at week 6 (peering-link change) and the **week-10 ipfixprobe artifact** (new exporter skips retransmitted packets → steep drop), with authors recommending splitting the year into weeks 1–9 and 11–52 — directly motivating our two-reference-week protocol.

### CESNET-QUIC22: One-Month Backbone QUIC Traffic Dataset (luxemburk2023quic.pdf)
**Citation:** Jan Luxemburk, Karel Hynek, Tomáš Čejka, Andrej Lukačovič, Pavel Šiška — *Data in Brief* 46 (2023), art. 108888. `luxemburk2023quic`
**TL;DR:** A data paper releasing one month of real ISP-backbone QUIC traffic (153M flows, 102 service classes + 3 background) as enriched flow records for encrypted-traffic classification.
**Problem:** No large public QUIC datasets existed; QUIC's encrypted handshake makes classification hard, and lab data cannot capture realistic ISP-scale behavior.
**Method/Dataset:** Captured from CESNET2 backbone (100 Gbps, ~half a million users) via ipfixprobe with QUIC/PSTATS/PHISTS plugins, then filtered, dynamically sampled per service, anonymized (Crypto-PAn). Each flow has packet-metadata sequences (first 30 packets), flow statistics, size/IPT histograms, and handshake fields (SNI, QUIC version, user agent).
**Key facts:** One month (31.10–27.11.2022), 27 TB raw; 153M flows; 102 service labels + 3 background; 17 categories; selected services cover 84% of CESNET QUIC traffic; PPI length 30; 8 log histogram bins. Four weeks W-2022-44..47.
**Relevance (Core):** Source dataset for our secondary benchmark (week-44 source, 102 classes, 4 eval weeks). Authors explicitly note the span enables research on QUIC distribution drift and classifier degradation, aligned with our framing — though the paper reports no drift/UDA benchmark itself.

### Fine-grained TLS Service Classification with Reject Option (luxemburk2023tls.pdf)
**Citation:** Jan Luxemburk, Tomáš Čejka — *Computer Networks* 220, art. 109467, 2023 (preprint arXiv:2202.11984v2). `luxemburk2023tls`
**TL;DR:** Introduces the CESNET-TLS22 dataset (140M flows, ~200 fine-grained service labels) and a multimodal 1D-CNN classifier with a reject option for unknown services.
**Problem:** Encrypted TLS classification needs large realistic fine-grained datasets, and deployed classifiers must reject services not seen in training rather than forcing a label.
**Method:** A 1.1M-parameter multimodal NN with two chains — PSTATS packet-metadata sequences (first 30 packets) via Conv1D and flow statistics via linear layers — fused into a shared representation. Three reject scores compared: softmax baseline, energy-based (LogSumExp), and a per-sample gradient-based method.
**Key results:** Full 191-class task: 97.04% accuracy, 98.65% superclass accuracy, 94.20% F1. Gradient-based reject (top-100): 91.94% TPR@5%FPR, 92.13% AUROC; NN beat LightGBM by +4.09% accuracy.
**Relevance (Core/Supporting):** The dataset/backbone precursor to our work (same CESNET-family multimodal model). Supports the discrete-drift premise: a 5%-FPR threshold calibrated on week 1 drifts to ~6.5% FPR on week 2, with authors stating TLS service behavior "is not static" and needs periodic re-calibration. No per-class teleportation, label-shift, or UDA analysis.

---

## 2. Encrypted-traffic classification models & feasibility (Background)

### ET-BERT: Pre-trained Transformer for Encrypted-Traffic Classification (lin2022etbert.pdf)
**Citation:** Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, Jing Yu — *WWW '22* (ACM Web Conf.), 2022. `lin2022etbert`
**TL;DR:** Pre-trains a BERT-style Transformer on large-scale unlabeled encrypted traffic via two self-supervised BURST tasks, then fine-tunes on small labeled sets to set SOTA across five tasks.
**Problem:** Existing classifiers depend on plaintext, hand-crafted features, or large labeled sets, making them brittle to imbalance and unable to generalize to new encryption (e.g. TLS 1.3).
**Method:** "Datagram2Token" converts packets into BURST-level tokens; pre-training uses Masked BURST Model and Same-origin BURST Prediction on unlabeled traffic, then packet- or flow-level fine-tuning. 12 layers, 12 heads, H=768.
**Key results:** ISCX-VPN-Service F1 98.9% (+5.2%), Cross-Platform-Android 92.5% (+5.4%), CSTNET-TLS 1.3 97.4% (+10.0%); 91.55% F1 with only 10% data; removing pre-training drops F1 by 37.57%.
**Relevance (Background):** A strong supervised closed-world backbone for encrypted-traffic classification. Does not address temporal drift, label shift, UDA/TTA, or unsupervised monitoring; its "generalization" is cross-dataset, not asynchronous temporal drift.

### MLTAT: A High-Speed ML Framework for Encrypted Traffic Classification (seddigh2019framework.pdf)
**Citation:** Nabil Seddigh, Biswajit Nandy, Don Bennett, Yonglin Ren, Serge Dolgikh, Colin Zeidler, Juhandre Knoetze, Naveen Sai Muthyala (Solana Networks) — *15th CNSM*, 2019. `seddigh2019framework`
**TL;DR:** An engineering paper presenting MLTAT, a 10 Gbps+ ML platform that classifies encrypted traffic using a restricted 30-feature set, plus co-training to generate labeled ground truth.
**Problem:** Encryption defeats DPI, while academic ML solutions use 100–200 features that can't run at line rate and rely on scarce, non-reproducible labels.
**Method:** MLTAT (Spark/HDFS/HBase, C parsing) integrates supervised classifiers (LR, SVM, C4.5, AdaBoost, MLP, NB) combined via bagging/voting over ~30 fast flow features. Labels built via single-app capture and multi-phase co-training.
**Key results:** 2170 PCAPs, 354,808 flows, 8–9 classes. C4.5 per-class precision/recall e.g. DNS 99.82%/98.17%, P2P 96.57%/87.16%; overall ~90% accuracy. Co-training 93.14%/91.61%/88.77% at 20%/10%/5% labels.
**Relevance (Background):** Static single-snapshot system; no drift, DA, or covariate-vs-label-shift notion. Useful only as background on feature engineering and ground-truth difficulty.

### Are Neural Networks the Best Way for Encrypted Traffic Classification? (shamsimukhametov2021ann.pdf)
**Citation:** D. Shamsimukhametov, M. Liubogoshchev, E. Khorov, I. F. Akyildiz — *2021 Int. Conf. Engineering and Telecommunication (En&T)*, IEEE, 2021. `shamsimukhametov2021ann`
**TL;DR:** A simple SNI-substring rule on the unencrypted TLS handshake beats payload-based CNNs, showing NN "success" rides on exposed unencrypted bytes.
**Problem:** Payload-based NNs report near-100% accuracy on "encrypted" traffic — the paper checks whether that is genuine or an artifact of unencrypted handshake bytes.
**Method:** A 12,350-flow, 3-class dataset (web/YouTube/Netflix) labeled via TLS-key-decrypted MIME inspection; baseline CNN (first 784 L4 payload bytes → 28×28 image) vs a learning-free SNI-substring matcher; CNN re-evaluated on fully/partly/un-encrypted payloads.
**Key results:** SNI rule: 100% across all metrics; CNN 99.2% accuracy. With fully-encrypted payload, CNN collapses to 38.2%; partly-encrypted 75.3%.
**Relevance (Background):** Reinforces that payload-NN classifiers depend on non-robust signal and degrade sharply when it disappears — skepticism toward complex global NN/UDA pipelines. No temporal drift, per-class, or label-shift analysis. (Note: this is the En&T 2021 paper, distinct from the ECH/IEEE-Access one below.)

### Early Traffic Classification with Encrypted ClientHello: A Multi-Country Study (shamsimukhametov2022ech.pdf)
**Citation:** Danil Shamsimukhametov, Anton Kurapov, Mikhail Liubogoshchev, Evgeny Khorov — *IEEE Access* 12, **2024** (DOI 10.1109/ACCESS.2024.3469730). `shamsimukhametov2022ech`
**TL;DR:** Under Encrypted ClientHello (ECH), pure packet/TLS-metadata classifiers collapse, but a hybrid Random Forest (hRFTC) combining recomposed handshake payload with flow statistics restores up to 94.6% F-score.
**Problem:** ECH encrypts the SNI and ClientHello metadata, breaking SNI/metadata-based early traffic classification. Does eTC survive growing variety/geography, and do flow features help under ECH?
**Method:** A 600k+ flow dataset across North America/Europe/Asia (19 classes), simulated ECH, and hRFTC — a Random Forest over recomposed CH/SH payload features + flow PS/IPT statistics, extended to QUIC. Compared vs packet-/flow-/hybrid baselines using packets up to the first downlink app packet.
**Key results:** Pure packet-based F-score falls to 38.4% under ECH; hRFTC reaches 94.6%. Cross-country generalization is poor (Germany 38.4%, USA 49.2%), so models need per-location retraining. Flow PS features carry >50% importance.
**Relevance (Background/Supporting):** Independently documents heterogeneous, structured shift (per-locale degradation requiring location-specific retraining) — evidence that encrypted-traffic shift is not a single global drift a frozen model absorbs. No BBSE, entropy residual, or UDA/TTA. ⚠️ Bib note: this is a **2024** paper despite the `2022ech` key.

---

## 3. Drift in encrypted traffic & evaluation bias (Supporting)

### MFWDD: Model-based Feature Weight Drift Detection on TLS/QUIC (MFWDD.pdf)
**Citation:** Lukáš Jančička, Dominik Soukup, Josef Koumar, Filip Němec, Tomáš Čejka — *2024 20th CNSM*, IFIP, 2024. `mfwdd2024`
**TL;DR:** Detects concept drift via a classifier-feature-importance-weighted aggregate of per-feature distribution-shift severities, triggering retraining without labels.
**Problem:** ML traffic classifiers degrade post-deployment; existing detectors are false-positive-prone and ignore which features matter to the model.
**Method:** Per-feature severity via KS test or normalized Wasserstein, combined into S = (1/n)·Σ w_i·s_i using classifier feature importances. Drift fires when S ≥ threshold (~0.5) → retrain. XGBoost on PPI/SPLT and NetTiSA features.
**Key results:** On CESNET-TLS-Year22 (week-1 reference, l=7), MFWDD-guided retraining beats the static baseline; recurring drifts every few months driven by 4th-packet features (SIZE_4/DIR_4), plus a March drift. Also evaluated on CESNET-QUIC22.
**Relevance (Supporting):** Aligns on the symptom — per-feature, recurring/asynchronous drift in encrypted TLS/QUIC. Remedy is supervised retraining off an input-feature detector; no covariate-vs-label split, no BBSE/entropy residual, no negative-transfer study.

### Drift-Based Dataset Stability Benchmark (2512.23762v1.pdf)
**Citation:** Dominik Soukup, Richard Plný, Daniel Vašata, Tomáš Čejka (CTU Prague & CESNET) — arXiv:2512.23762v1 [cs.LG], 2025. `cesnetdrift2025`
**TL;DR:** A reproducible dataset-stability benchmark built around MFWDD to track and explain model degradation over a one-year encrypted-traffic dataset.
**Problem:** Classifiers degrade post-deployment; existing dataset metrics don't support long-term evaluation or pinpoint which features/classes cause drift.
**Method:** Per-feature two-sample tests (KS supervised, normalized Wasserstein unsupervised) aggregated into S = Σ w_i·s_i weighted by feature importances. A workflow evaluates a frozen reference vs a drift-triggered retraining model window-by-window, splitting the dataset by per-class drift behavior.
**Key results:** On CESNET-TLS-Year22, retraining raised F1 0.546 → 0.857. **83 of 159 classes drifted at least once, 54 never drifted**; removing the 28 most-drifted classes raised retraining F1 to ~91% (+5%). Most-drifted features: DIR_4, SIZE_4, SIZE_3.
**Relevance (Supporting):** Strong independent evidence for discreteness/per-class drift — on the *same* CESNET data, drift is unevenly distributed and isolating the worst classes stabilizes the rest. No label-vs-covariate split, no BBSE, no negative-transfer study.

### Per-Feature/Per-Class Distribution Shift & the "Weekend Phenomenon" (jancicka2024noms.pdf)
**Citation:** Lukáš Jančička, Josef Koumar, Dominik Soukup, Tomáš Čejka — *IEEE NOMS 2024*, QoDaNeT workshop. `jancicka2024noms`
**TL;DR:** Statistical analysis of CESNET-TLS-Year22 shows input-feature distributions drift in a recurring, per-class, per-feature way (a seasonal "Weekend phenomenon"), not as a single global shift.
**Problem:** Encrypted-traffic classifiers rely on statistical features whose distributions drift, degrading accuracy; what kind of drift exists in real backbone TLS traffic?
**Method:** On the one-year dataset (507M flows, 180 classes, 54 features), compute per-class per-feature mean/std time series, apply autocorrelation and seasonal decomposition, and use KS tests to compare weekday vs weekend distributions class-by-class.
**Key results:** Autocorrelation significance at lag 7 (weekly seasonality); 38% of all feature-class permutations reject the weekday/weekend null at 5%. Drift is heterogeneous: some classes (docker-registry, kaspersky, doh…) very stable; others (avast, spotify, slack, dropbox, youtube…) differ on most features. Authors note change "can happen just on one class."
**Relevance (Supporting):** Directly corroborates our discrete/per-class claim ("can happen just on one class"). Drift they characterize is recurring/seasonal volatility (closer to benign cyclic shift) rather than asynchronous teleportations; no BBSE/entropy residual or UDA/TTA. Cite as independent confirmation.

### Data Drift in DL: Lessons Learned from Encrypted Traffic Classification (malekghaini2022data.pdf)
**Citation:** Navid Malekghaini, Elham Akbari, Mohammad A. Salahuddin, Noura Limam, Raouf Boutaba (U. Waterloo); Bertrand Mathieu, Stéphanie Moteau, Stéphane Tuffin (Orange Labs) — *IFIP Networking 2022*. `malekghaini2022data`
**TL;DR:** Two SOTA DL classifiers degrade substantially over two years of real ISP traffic, the decay driven by protocol/feature evolution (e.g. SPDY→HTTP/2); architecture adaptations partially recover accuracy.
**Problem:** DL classifiers trained on one snapshot decay in production as protocols and patterns change, but this is mostly studied on controlled/synthetic data.
**Method:** Evaluate the UW Tripartite model and a UCDavis CNN trained on 07-2019, tested frozen on later real datasets (09-2020…06-2021 + QUIC), decomposing models into header/flow-time-series/auxiliary parts and analyzing ALPN distributions; then propose architecture adaptations.
**Key results:** Avg accuracy drop: TLS-header part 40.75%, flow-time-series part 33.02%. ALPN shift: HTTP/2 +53.5%, SPDY −83.3% (2019→2021). Adaptations raised QUIC flow-series accuracy 86.7%→95.6%.
**Relevance (Supporting/contrast):** Direct evidence that drift harms frozen encrypted-traffic classifiers — but frames drift as **global, gradual covariate/protocol shift** and notes class distribution is largely stable, the *opposite* of our per-class/label-shift framing. No UDA/TTA, no per-class analysis, no BBSE. Good "drift exists" citation; contrast its global view with ours.

### Drift-oriented Self-evolving Encrypted Traffic Classification (chen2025drift.pdf)
**Citation:** Zihan Chen, Guang Cheng, Jinhui Li, Tian Qin, Yuyang Zhou, Xing Luan — venue not stated in PDF (IEEE template); arXiv:2501.04246v1, 2025. `chen2025drift`
**TL;DR:** A self-evolving classifier detects per-category concept drift via windowed multi-threshold confidence accumulation and continually fine-tunes on unlabeled high-confidence "silver samples," extending model lifetime from ~2 to ~8 months.
**Problem:** Constant app updates cause feature concept drift that degrades static classifiers; retraining is costly because labeled post-drift samples are scarce.
**Method:** Laida (3-σ) criterion flags samples with softmax confidence >0.997 as label-free "silver samples"; a per-category windowed multi-threshold counter decides when single-class or full-class drift warrants fine-tuning an LS-LSTM backbone.
**Key results:** Private dataset, 6 apps, 372 GB, Sept 2023–Jul 2024. Abstract claims +9% F1 and >8-month lifetime. Initial F1 0.9189 vs fine-tuned 0.9619/0.9822; on final long-span samples all degrade to ~0.59. Silver-sample share rises 41.66%→68.20%.
**Relevance (Supporting):** Aligns on per-category/asynchronous drift (per-class drift counters reset individually; update cycles "not aligned"), supporting our discrete view. Frames drift as continuous "concept drift" (not teleportations), pursues fine-tuning not monitoring, no UDA/BBSE. Long-span F1 collapse (~0.59) indirectly evidences the hidden-degradation problem our residual targets.

### TESSERACT: Eliminating Spatio-Temporal Experimental Bias in Malware Classification (pendlebury2019tesseract.pdf)
**Citation:** Feargus Pendlebury, Fabio Pierazzi, Roberto Jordaney, Johannes Kinder, Lorenzo Cavallaro — *USENIX Security 2019* (arXiv:1807.07838v4). `pendlebury2019tesseract`
**TL;DR:** Reported high malware-classification F1 is inflated by temporal and spatial experimental bias; enforcing realistic time/distribution constraints and a time-aware metric (AUT) reveals much lower true performance under concept drift.
**Problem:** Evaluations leak future knowledge (temporal bias) and use unrealistic malware/goodware ratios (spatial bias), masking real-world time decay.
**Method:** Three constraints (temporal training consistency, temporal window consistency, realistic ~10% malware ratio); a new metric AUT (Area Under Time); and a training-ratio tuning algorithm. Open-source framework on 129,728 AndroZoo apps (2014–2016).
**Key results:** Realistic settings drop performance up to ~50%; F1 falls from inflated 0.91/0.97 to 0.58/0.45/0.32/0.30. AUT(F1,24m): two classical algorithms 0.58/0.32 vs DL 0.64 — a robustness reversal hidden by biased setups.
**Relevance (Supporting, methodology):** Supports our evaluation-protocol stance — biased splits inflate results — motivating our frozen-source, temporally-ordered weekly benchmark and source-week discipline. AUT is conceptually adjacent to our degradation tracker. Malware domain (binary); treats drift as continuous global time decay, not per-class teleportations.

### Robustness Illusion in Encrypted-Traffic Classification (unpublished NDSS'26 submission — PDF not retained)
**Citation:** Anonymous double-blind submission, NDSS Symposium 2026. `anon2026papertigers` — ⚠️ *unpublished; do not cite as published, and confirm it is citable given its anonymity (and any author overlap with us) before relying on it.*
**Key findings (paraphrased — nothing reproduced verbatim from the submission):**
- Argues that the near-perfect accuracy routinely reported for neural encrypted-traffic classifiers is largely a measurement artifact: the bulk of prior evaluations test on a single time period, client population, and network vantage point, so high scores reflect collection conditions rather than genuine deployment robustness.
- Builds a drift-stress evaluation that layers increasing degrees of temporal, client-population, and network-vantage shift, and runs a broad sweep of recent classifiers over many configurations with repeated trials.
- Headline result: in-distribution, most models appear excellent, but under temporal shift performance collapses across essentially all of them — byte/payload-centric models degrade the worst — and combined time+client+network shift is close to catastrophic; crucially, larger-capacity models are not more robust.
**Relevance (Supporting):** Independent motivation for our "illusion of robustness / brittle temporal generalization" framing. It treats drift as a coarse global shift along time/client/network axes, though — it does not decompose drift into discrete per-class covariate teleportations vs. label shift, does not study negative transfer of global UDA/TTA, and proposes no BBSE-corrected residual. So it motivates the problem, not our mechanism or correction.

---

## 4. Label-shift estimation & the BBSE family (Core)

### EM-based Prior-Probability Adjustment for Label Shift (SLD) (saerens2002sld.pdf)
**Citation:** Marco Saerens, Patrice Latinne, Christine Decaestecker (ULB) — *Neural Computation* 14(1):21–41, 2002. `saerens2002sld`
**TL;DR:** A simple EM procedure re-estimates a classifier's new-domain class priors from its own unlabeled-data outputs and rescales posteriors, recovering Bayes-optimal decisions under label shift without refitting.
**Problem:** When training priors differ from deployment priors, posteriors become invalid; the new priors are unknown and must be estimated from unlabeled data.
**Method:** Assuming within-class densities p(x|ω) are unchanged, Bayes' rule gives a closed-form posterior correction once new priors are known; when unknown, EM iterates (rescale posteriors by current prior estimate, re-average to update priors). A likelihood-ratio χ² test decides whether priors actually changed.
**Key results:** On Ringnorm, EM estimates priors better than the confusion-matrix method (true 90%→85.6% vs 80.9%); adjustment raises accuracy toward the true-prior oracle. LR test flagged shift 10/10 when priors differed, 0/10 when not. On three UCI medical sets, adjustment always increased accuracy (Diabetes 67.4%→76.3%).
**Relevance (Core, foundational):** The foundational MLLS/SLD estimator underlying BBSE-style correction. Formalizes the exact separation we exploit — benign prior (volume) shift, under fixed p(x|ω), estimable from outputs alone. Its "only priors change" assumption is the regime we contrast against discrete per-class covariate teleportations; its LR test motivates distinguishing benign label shift from genuine degradation.

### Black Box Shift Estimation (BBSE) for Label Shift (1802.03916v3.pdf)
**Citation:** Zachary C. Lipton, Yu-Xiang Wang, Alexander J. Smola — *ICML 2018* (PMLR 80). `lipton2018bbse`
**TL;DR:** Detect, quantify, and correct label shift (p(y) changes, p(x|y) fixed) using only a black-box predictor's confusion matrix and its outputs on unlabeled target data.
**Problem:** Under label shift a classifier can silently become inaccurate when p(y) drifts; detect and re-weight without target labels.
**Method:** Estimate weights w_l = q(y_l)/p(y_l) by solving a linear system with the source confusion matrix C(f) (assumed invertible) and the predictor's mean target output: ŵ = Ĉ⁻¹ μ̂_ŷ. Paired with a detection test (BBSD) on f(x) and importance-weighted ERM (BBSC); consistency + error bounds (~‖w‖²log n / n).
**Key results:** MNIST: BBSD controls Type-I error at δ=0, beats kernel two-sample test at δ=0.5. Dirichlet/tweak-one shift: matches/beats KMM and scales far better (n=32000 vs ~8000). CIFAR-10: BBSC raises accuracy ~0.4→~0.95 under strong shift.
**Relevance (Core):** Direct foundation of our **BBSE-corrected entropy residual**. The factorization q(y,x)=q(y)p(x|y) is exactly the benign label-shift volatility we subtract out. Its "sporadic shift"/streaming notes anticipate our discrete asynchronous per-class setting, though it assumes a single global label shift.

### Regularized Learning under Label Shift (RLLS) (1903.09734v1.pdf)
**Citation:** Kamyar Azizzadenesheli, Anqi Liu, Fanny Yang, Animashree Anandkumar — *ICLR 2019*. `azizzadenesheli2019rlls`
**TL;DR:** A regularized importance-weight estimator for label shift with a dimension-independent generalization bound, improving on BBSE for small samples.
**Problem:** BBSE's confusion-matrix-inverse weights are unstable for small samples / near-singular matrices, with error linear in k (classes).
**Method:** Estimate weight-shift θ = w−1 via L2-regularized least squares against the empirical confusion matrix, then ŵ = 1 + λθ̂ (λ depends on target sample size); train on reweighted source. Proves a dimension-independent bound improving BBSE by a factor of k.
**Key results:** MNIST & CIFAR-10 (k=10), Tweak-One/Minority/Dirichlet shift. Large shift + large samples: order-of-magnitude smaller weight MSE, up to 20% higher accuracy/F1; low target samples: ≥10% gain via partial regularized weights.
**Relevance (Core):** Shares the BBSE/confusion-matrix foundation; motivates discounting benign label-shift volatility via importance weights rather than treating it as covariate drift. Assumes pure global label shift — complementary to our discrete framing.

### A Unified View of Label Shift Estimation (MLLS) (2003.07554v3.pdf)
**Citation:** Saurabh Garg, Yifan Wu, Sivaraman Balakrishnan, Zachary C. Lipton — *NeurIPS 2020*. `garg2020mlls`
**TL;DR:** Unifies BBSE and MLLS as calibration-based label-shift estimators; MLLS dominates empirically due to finer (BCTS) calibration, not its objective.
**Problem:** The two dominant weight estimators — moment-matching BBSE and EM-based MLLS — were poorly understood relative to each other.
**Method:** Casts both as generalized distribution matching over a space Z; proves canonical calibration + invertible confusion matrix suffice for MLLS consistency; shows BBSE = MLLS under coarse calibration; decomposes MLLS error into miscalibration + estimation terms (error ∝ 1/σ_f).
**Key results:** Synthetic GMM, MNIST, CIFAR-10 (Dirichlet shift): MLLS+BCTS uniformly dominates BBSE/RLLS/MLLS-CM, 2–10× lower MSE depending on shift magnitude. MLLS-CM performs like BBSE.
**Relevance (Core):** Methodological backbone for BBSE — explains its statistical inefficiency (coarse calibration loses information), justifying calibration choices when separating benign label shift from genuine degradation.

### Bayesian Quantification with Black-Box Estimators (ziegler2023bayesian.pdf)
**Citation:** Albert Ziegler, Paweł Czyż — *TMLR*, 2023 (arXiv:2302.09159). `ziegler2023bayesian`
**TL;DR:** A Bayesian generative model that reinterprets BBSE, invariant-ratio, and adjusted classify-and-count as approximate posterior inference, adding calibrated uncertainty and avoiding negative-probability estimates.
**Problem:** Estimating unknown class prevalence P_test(Y) under prior-probability (label) shift using a possibly-biased black-box classifier, with quantified uncertainty.
**Method:** Casts quantification as posterior inference, replacing the intractable P(X|Y) with a discretized map f:X→C. Builds an N-independent sufficient statistic for fast NUTS MCMC; proves MAP asymptotic consistency; reports flat/weak-prior MAP estimates.
**Key results:** Simulated categorical data: MAP matches BBSE/IR and is superior when K≠L. Wisconsin Breast Cancer: predicted prevalence varies materially across methods (EM 28% vs CC 34%), motivating uncertainty; also single-cell RNA-seq and Gaussian mixtures with valid credible intervals.
**Relevance (Core, foundational):** In the BBSE/label-shift family our residual builds on; formalizes the same prior-shift assumption. Its uncertainty-aware BBSE generalization could strengthen our correction. Assumes global label shift; no discrete drift or negative-transfer content.

### Doubly Flexible Estimation under Label Shift (lee2023doubly.pdf)
**Citation:** Seong-ho Lee, Yanyuan Ma, Jiwei Zhao — arXiv:2307.04250 (2023); *JASA* 120(549), 2024. `lee2023doubly`
**TL;DR:** A semiparametric estimator of a target-population characteristic under label shift that stays valid even when BOTH the label density-ratio model and the conditional outcome model are misspecified.
**Problem:** With labeled source (X,Y) and unlabeled target (X) under label shift, estimating ρ(y)=q_Y/p_Y is near-infeasible (Y unobserved in target) and E_p(·|x) suffers the curse of dimensionality.
**Method:** Directly estimates θ from an estimating equation, bypassing ρ(y) via a working model; "doubly flexible" allows both nuisance models misspecified, reducing to 1-D nonparametric regressions (Nadaraya-Watson) and a Fredholm integral equation solved by Landweber iteration. Handles classification and regression.
**Key results:** 1000-replicate simulations: naive estimator biased (Bias .184), doubly-flexible unbiased (Bias −.012, MSE .0173, CI .943). On MIMIC-III (n=16,691), proposed estimators match and cover the oracle while the naive one fails.
**Relevance (Background / future work):** Pure label-shift estimation (no traffic, no deep nets, regression focus). A candidate future-work upgrade to our label-shift correction — more robust to misspecification — though its low-dimensional nonparametric regressions make scaling to 180-class traffic non-trivial.

---

## 5. Online label shift (Background — contrast / future work)

### Online Label Shift Adaptation with Provable Dynamic Regret (bai2022online.pdf)
**Citation:** Yong Bai, Yu-Jie Zhang, Peng Zhao, Masashi Sugiyama, Zhi-Hua Zhou — *NeurIPS 2022*. `bai2022online`
**TL;DR:** Formalizes online label shift — label distribution drifts over time on unlabeled streaming data while class-conditionals stay fixed — and proposes online ensemble algorithms (ATLAS) with provable dynamic-regret guarantees.
**Problem:** A model trained offline is deployed on an unlabeled stream where D_t(y) changes but D(x|y) is constant; lack of supervision + non-stationarity make tracking shifting priors hard.
**Method:** Build an unbiased risk estimator via risk-rewriting + BBSE (confusion-matrix inversion of online predictions), then run OGD (UOGD). ATLAS adds a meta/base ensemble over a step-size pool for unknown shift intensity; ATLAS-ADA adds hint functions.
**Key results:** ATLAS achieves O(V_T^{1/3} T^{2/3}) dynamic regret (minimax optimal). Synthetic (T=10k): ATLAS 4.27/5.75/4.04 vs UOGD 6.17/6.37/5.46. Real (SHL, ArXiv, EuroSAT, MNIST…): outperforms under rapid shift (ArXiv Ber 21.11 vs FIX 30.63).
**Relevance (Background, contrast):** Uses the same BBSE confusion-matrix inversion our residual relies on. But assumes shift is purely in D_t(y) with globally fixed D(x|y) — the opposite of our discrete per-class covariate teleportations. Strong reference for benign label-shift volatility and a natural future-work direction (online estimation); no covariate drift or negative transfer.

### Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms (baby2023online.pdf)
**Citation:** Dheeraj Baby, Saurabh Garg, Thomson Yen, Sivaraman Balakrishnan, Zachary C. Lipton, Yu-Xiang Wang — *NeurIPS 2023* (arXiv:2305.19570). `baby2023online`
**TL;DR:** Adapts a frozen classifier to continuously drifting class marginals via a reduction to online regression, achieving minimax-optimal dynamic regret without knowing the drift extent.
**Problem:** Class marginals Q(y) drift while Q(x|y) stays fixed; prior OCO methods assume convex losses (excluding tree/black-box models) or only control static regret.
**Method:** Reduces unsupervised and supervised online label shift to online regression tracking drifting proportions, then re-weights/re-samples the frozen classifier via the confusion matrix (FLH-FTL). Side-steps convexity; minimax-optimal dynamic regret without prior knowledge of V_T.
**Key results:** Synthetic/MNIST/CIFAR/EuroSAT/Fashion/ArXiv under sinusoidal & Bernoulli shifts: FLH-FTL lowest error and marginal-MSE among many baselines (CIFAR Bernoulli error 10 vs Base 16; MNIST marginal MSE 0.15 vs 0.27). Supervised CT-RS reaches 17.12 vs oracle 16.32 at 5–15× lower cost. Random-Forest MNIST 13 vs 18.
**Relevance (Background, contrast):** The pure online label-shift contrast — assumes Q(x|y) invariant, only the marginal drifts, exactly the benign volume volatility our residual ignores. Lists covariate shift as future work. Its confusion-matrix re-weighting is the mechanism we adapt; motivates an online extension of our estimator.

---

## 6. Global UDA / TTA baselines (Core — methods we argue against)

### Domain-Adversarial Training of Neural Networks (DANN) (1505.07818v4.pdf)
**Citation:** Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky — *JMLR* 17 (2016), pp. 1–35. `ganin2016dann`
**TL;DR:** Learns domain-invariant features via a gradient-reversal layer that makes source/target representations indistinguishable while staying discriminative for the label task.
**Problem:** Source-trained classifiers degrade under target shift; only unlabeled target data is available (unsupervised DA).
**Method:** Feed-forward net jointly trains a label predictor and a domain classifier over shared features; a gradient-reversal layer maximizes domain-classifier loss for the feature extractor. End-to-end SGD; invariance weight λ annealed 0→1.
**Key results:** Office: 0.730 (A→W), 0.964 (D→W), 0.992 (W→D). Digits: MNIST→MNIST-M 0.7666, Syn→SVHN 0.9109, SVHN→MNIST 0.7385.
**Relevance (Core):** The canonical global feature-alignment UDA baseline we argue against. Assumes a single global covariate shift and forces marginal-feature invariance; under discrete per-class teleportations this conflates classes and can cause negative transfer. Its theory assumes shared p(y), making it vulnerable to label shift.

### Deep CORAL: Correlation Alignment for Deep Domain Adaptation (1607.01719v1.pdf)
**Citation:** Baochen Sun, Kate Saenko — arXiv:1607.01719v1 [cs.CV], 2016. `sun2016deepcoral`
**TL;DR:** A differentiable "CORAL loss" aligning source/target second-order feature statistics (covariances) end-to-end inside a deep CNN for unsupervised DA.
**Problem:** Deep nets degrade under domain shift with unlabeled targets; prior CORAL aligns covariances with a linear transform but is not end-to-end.
**Method:** Add CORAL loss = (1/4d²)·‖C_S − C_T‖²_F (squared Frobenius distance of covariances) to the classification loss. Applied to AlexNet fc8 with shared weights.
**Key results:** Office (31 classes, 6 shifts): D-CORAL avg 72.1, beating DAN 71.3, DDC 70.6, CNN 70.1; wins 3/6 shifts.
**Relevance (Core, baseline):** A global feature-alignment UDA baseline we argue against — aligns all classes' statistics jointly, the assumption that fails under discrete per-class teleportations. No label-shift handling; image object recognition only.

### Tent: Fully Test-Time Adaptation by Entropy Minimization (wang2020tent.pdf)
**Citation:** Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell — *ICLR 2021*. `wang2020tent`
**TL;DR:** Adapts a frozen source model at test time, with no source data or labels, by minimizing prediction entropy while updating only normalization statistics and affine scale/shift parameters.
**Problem:** Models lose accuracy under shift, but deployments often lack source data and labels at test time, so adaptation must rely on the model and unlabeled target batches only.
**Method:** Minimize the Shannon entropy of batched predictions per test batch; estimate normalization statistics from target data and update only affine γ, β (<1% of parameters) via one gradient per batch.
**Key results:** ImageNet-C (severe): 44.0% error vs 50.2% / 49.9% baselines. CIFAR-100-C: 37.3% vs source 67.2%. SVHN→MNIST: 8.2% vs 18.2%. VisDA-C: source 56.1%→45.6% error.
**Relevance (Core):** A primary global TTA baseline our paper critiques. A single shared batch-wise update assuming global covariate shift; under discrete per-class teleportations and benign label-shift volatility it is vulnerable to negative transfer and offers no covariate-vs-label-shift separation — motivating our BBSE-corrected entropy residual.

### CoTTA: Continual Test-Time Domain Adaptation (wang2022cotta.pdf)
**Citation:** Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai — *CVPR 2022* (arXiv:2203.13591). `wang2022cotta`
**TL;DR:** Adapts a frozen source model online to a continually changing target stream using weight- and augmentation-averaged pseudo-labels plus stochastic weight restoration to fight error accumulation and forgetting.
**Problem:** Existing TTA (e.g. TENT) assumes a static target; under continual change, pseudo-labels grow noisy (error accumulation) and long-term self-training causes forgetting.
**Method:** A mean-teacher generates weight-averaged pseudo-labels; augmentation-averaged predictions used when source confidence is low; a small fraction of weights is stochastically restored to source each step (Bernoulli p=0.01).
**Key results:** CIFAR10-C mean error 16.2% vs TENT-continual 20.7%, source 43.5%; CIFAR100-C 32.5% vs 60.9%; ImageNet-C 63.0% vs 66.5%; Cityscapes→ACDC 58.6 mIoU. TENT "quickly deteriorates after three corruption types."
**Relevance (Core):** A primary TTA baseline we argue against. Assumes global, gradually changing covariate shift — the continuous-drift assumption we challenge. Its own evidence that TENT collapses supports our negative-transfer claim; CoTTA targets global shift, not discrete teleportations or label-shift volatility, with no BBSE correction.

### Adaptive Batch Normalization (AdaBN) for Domain Adaptation (li2016adabn.pdf)
**Citation:** Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, Xiaodi Hou — "Revisiting Batch Normalization for Practical Domain Adaptation," arXiv:1603.04779 / ICLR 2017 workshop. `li2016adabn`
**TL;DR:** Replacing source-domain BatchNorm statistics with statistics recomputed on the target domain gives parameter-free, fine-tuning-free domain adaptation — the "bnstats" family.
**Problem:** DNNs overfit to source-domain bias and transfer poorly; existing UDA adds adaptation layers, optimization, and parameters.
**Method:** Hypothesizes label knowledge lives in weights, domain knowledge in BN statistics. AdaBN recomputes per-neuron mean/variance from the target domain and uses them in every BN layer at test time, keeping weights and BN scale/shift. Zero-parameter, complementary to CORAL.
**Key results:** Office-31 single-source avg 76.7 (vs 72.1 baseline); +CORAL 77.2. Multi-source 83.6/84.4 vs 82.1. Cloud detection mIOU 38.95%→64.50%. Symmetric-KL feature divergence drops 0.0716→0.0227.
**Relevance (Core, baseline):** The canonical "bnstats" baseline in our TTA family — adapts a single global set of BN statistics to the whole target, assuming one global covariate shift. Under discrete per-class teleportations and label-shift volatility, recomputing global BN moments can absorb benign label shift and misadapt, motivating negative-transfer concerns.

---

## 7. Open-set & continual learning (Background — operational paradigm)

### OpenMax: Open-Set Rejection of Unknown Classes for Deep Networks (bendale2016openmax.pdf)
**Citation:** Abhijit Bendale, Terrance E. Boult (UC Colorado Springs) — *CVPR 2016* (arXiv:1511.06233). `bendale2016openmax`
**TL;DR:** Replaces SoftMax with an EVT-calibrated layer estimating the probability an input belongs to an "unknown unknown" class, enabling rejection of unseen, fooling, and some adversarial inputs.
**Problem:** SoftMax forces probabilities over known classes, so closed-set nets confidently misclassify unknowns; thresholding SoftMax uncertainty is insufficient.
**Method:** Per class, fit a Weibull distribution to distances between correctly-classified training activations and the class mean activation vector; at test time recalibrate top-α activation scores by the Weibull CDF, redistribute mass to a synthetic unknown class, and reject below ε.
**Key results:** ILSVRC 2012 (AlexNet), 80,000 images (incl. 15K open-set + 15K fooling): OpenMax improves open-set F-measure ~4.3% over thresholded SoftMax and 12.3% over base; correctly classifies 9,847 more than the base net.
**Relevance (Background, operational):** Embodies the reject/flag-unknown paradigm our monitoring framing relies on, using activation-space (not output-space) distance as the failure signal — like our residual treating representation deviation as the signal rather than trusting softmax. No temporal/per-class drift, label-shift, or UDA content.

### iCaRL: Incremental Classifier and Representation Learning (rebuffi2017icarl.pdf)
**Citation:** Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert — *CVPR 2017* (arXiv:1611.07725). `rebuffi2017icarl`
**TL;DR:** Learns classifiers and representation jointly in a class-incremental setting, combining nearest-mean-of-exemplars, herding-based exemplar selection, and distillation + rehearsal to resist catastrophic forgetting.
**Problem:** Systems must keep learning new classes from a stream with bounded memory, but naive finetuning forgets; prior incremental methods needed a fixed representation.
**Method:** A CNN with per-class sigmoid outputs trained on new data + stored exemplars, loss = classification (new) + distillation (reproduce old outputs). Classification via nearest-mean-of-exemplars over L2-normalized features; budget K split m=K/t per class, exemplars chosen by herding.
**Key results:** iCIFAR-100: beats LwF.MC, fixed-representation, finetuning across 2/5/10/20/50-class batches (joint-training upper bound 68.6%). iILSVRC: leads at 100- and 1000-class. Confusion matrices: iCaRL near-uniform; finetuning collapses onto the last batch.
**Relevance (Background, operational):** A continual/class-incremental retraining paradigm — the practical alternative to UDA/TTA once degradation is detected. Per-class exemplar budgeting and asynchronous new-class handling echo our per-class view; finetuning's collapse onto recent classes is itself negative transfer. No BBSE/label-shift or entropy-residual content.

### Elastic Weight Consolidation (EWC) (kirkpatrick2017ewc.pdf)
**Citation:** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, et al. (DeepMind / Imperial) — *PNAS* 114(13):3521–3526, 2017 (arXiv:1612.00796). `kirkpatrick2017ewc`
**TL;DR:** Overcomes catastrophic forgetting by adding a quadratic penalty that slows learning on weights important to previous tasks, letting one fixed-capacity network learn tasks sequentially.
**Problem:** Networks trained sequentially abruptly forget old tasks; replaying all past data doesn't scale.
**Method:** Anchors each weight to its old value with stiffness = diagonal Fisher information (Laplace posterior approximation): L = L_B + Σ (λ/2)F_i(θ_i−θ*_{A,i})². First-order, PSD, scales to large models.
**Key results:** Permuted MNIST: EWC maintains accuracy across 10 tasks while SGD degrades; L2 blocks new learning. Fisher-overlap shows similar tasks share early-layer weights, dissimilar tasks recruit separate capacity. 10 sequential Atari games: EWC learns multiple where the no-penalty agent stays at ~0.
**Relevance (Background, operational):** Canonical method for the retraining/continual alternative to our monitoring approach. Its finding that dissimilar tasks recruit *separate* weights echoes our discrete/per-class claim and motivates why a global penalty (like global UDA/TTA) over-constrains when shifts are localized.

### Learning without Forgetting (LwF) (li2017lwf.pdf)
**Citation:** Zhizhong Li, Derek Hoiem — *IEEE TPAMI*, 2017 (arXiv:1606.09282). `li2017lwf`
**TL;DR:** Adds new tasks to a CNN using only new-task data, preserving old-task performance via a knowledge-distillation loss keeping responses on old tasks close to the original model's.
**Problem:** Storing/retraining on all old-task data is infeasible; naive fine-tuning forgets.
**Method:** Record the original network's outputs on new-task images for old tasks, then jointly optimize with a new-task logistic loss + temperature-scaled (T=2) distillation loss constraining old-task outputs. No old-task data needed.
**Key results:** AlexNet ImageNet→VOC: LwF new=76.1 (vs fine-tuning −0.3), old=56.2 (+0.9 over fine-tuning). Dissimilar task (Places365→CUB): old-task loss 3.8% (LwF) vs 8.4% (fine-tuning). Exception: ImageNet→MNIST.
**Relevance (Background, operational):** A way to update a deployed classifier on new data without revisiting old data while limiting forgetting. Contrasts with our unsupervised monitoring (LwF needs labeled new-task data + a discrete decision to add a task). No discrete per-class drift or label-shift content.

### TOPIC: Few-Shot Class-Incremental Learning (tao2020fscil.pdf)
**Citation:** Xiaoyu Tao, Xiaopeng Hong, Xinyuan Chang, Songlin Dong, Xing Wei, Yihong Gong — *CVPR 2020* (arXiv:2004.10956). `tao2020fscil`
**TL;DR:** Defines few-shot class-incremental learning (FSCIL) and proposes TOPIC, using a neural-gas network to preserve feature-manifold topology while adding new classes from few samples.
**Problem:** A deployed CNN must keep recognizing old classes while learning new ones from a handful of examples; naive finetuning forgets old and overfits new.
**Method:** Models the feature space as a neural-gas graph (competitive Hebbian learning); an anchor loss stabilizes topology (anti-forgetting), a min-max loss adapts it for few-shot discrimination. C-way K-shot sessions; drops standard distillation.
**Key results:** Beats iCaRL/EEIL/NCM on CIFAR100/miniImageNet/CUB200. Final-session CIFAR100 24.17%/29.37% (up to +13.52%); CUB200 26.28% vs EEIL 22.11%.
**Relevance (Background, operational):** Supports the paradigm of adding new classes cheaply over time, but addresses incremental *label-set growth*, not covariate/label drift. No discrete teleportation, BBSE, or negative-transfer content.

---

## 8. Interpretability (Background)

### SHAP: A Unified Game-Theoretic Framework for Feature Attribution (lundberg2017shap.pdf)
**Citation:** Scott M. Lundberg, Su-In Lee — *NIPS 2017*. `lundberg2017shap`
**TL;DR:** Unifies six feature-attribution methods under "additive feature attribution" and shows Shapley values are the unique solution satisfying local accuracy, missingness, and consistency.
**Problem:** Complex models are accurate but hard to interpret, and the many explanation methods lack a unifying theory.
**Method:** Defines additive feature attributions as linear functions of feature-presence variables, proves a unique solution (SHAP = Shapley values of a conditional expectation), and gives estimators: model-agnostic Kernel SHAP and model-specific Linear/Max/Deep SHAP.
**Key results:** Kernel SHAP needs fewer evaluations than Shapley sampling/LIME for comparable accuracy. MTurk studies (30 & 52 people): SHAP agrees more with human attributions than LIME/DeepLIFT. MNIST 8→3 masking: larger log-odds change than DeepLIFT/LIME.
**Relevance (Background, instrument):** A feature-attribution tool, not a drift/label-shift method. Relevant only if our work uses SHAP to attribute degradation to specific features or per-class signals — an interpretability instrument for explaining which features drive the discrete teleportations, not evidence for the thesis.
