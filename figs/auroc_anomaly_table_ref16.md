## Week-16 viral-spike AUROC / FAR@95 (forward weeks 17-52, 5 seeds [1, 2, 3, 4, 42])

Reference: WEEK-2022-16 (vanilla frozen source). f1_threshold=0.6 -> 26 degraded(+) / 10 healthy(-). 10%-sample protocol.


### CLEAN regime

| Detector | AUROC (mean±std) | FAR@90 (mean±std) | FAR@95 (mean±std) |
|---|---|---|---|
| Uncorrected entropy gap | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| MFWDD-style global drift | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| BBSE-corrected residual (ours) | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| BBSE-soft-corrected residual | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| RLLS-corrected residual | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| SLD-EM-corrected residual | 0.984 ± 0.004 | 0.057 ± 0.013 | 0.100 ± 0.011 |
| MLLS+BCTS-corrected residual | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### VIRAL regime

| Detector | AUROC (mean±std) | FAR@90 (mean±std) | FAR@95 (mean±std) |
|---|---|---|---|
| Uncorrected entropy gap | 0.708 ± 0.020 | 0.665 ± 0.041 | 0.837 ± 0.057 |
| MFWDD-style global drift | 0.650 ± 0.034 | 0.653 ± 0.064 | 0.732 ± 0.062 |
| BBSE-corrected residual (ours) | 0.834 ± 0.007 | 0.380 ± 0.047 | 0.675 ± 0.089 |
| BBSE-soft-corrected residual | 0.836 ± 0.003 | 0.408 ± 0.068 | 0.725 ± 0.079 |
| RLLS-corrected residual | 0.849 ± 0.015 | 0.325 ± 0.022 | 0.490 ± 0.027 |
| SLD-EM-corrected residual | 0.759 ± 0.021 | 0.705 ± 0.036 | 0.882 ± 0.019 |
| MLLS+BCTS-corrected residual | 0.828 ± 0.017 | 0.432 ± 0.064 | 0.680 ± 0.079 |
