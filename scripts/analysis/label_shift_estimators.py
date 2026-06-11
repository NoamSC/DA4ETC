"""Shared label-shift estimator helpers for the paper's estimator-variant
analyses (Figs 7/8/10/11 + Sec VI isolation panels).

All estimators here assume the label-shift invariant P(X|Y)=const and estimate
only the target marginal P(Y).  BCTS calibration uses the tested reference
implementation from the `abstention` library (Alexandari et al., ICML 2020).
"""

import numpy as np

SOFT_CLIP = 1e-7   # clip softmax before log for calibration / EM stability


def normalize_probs(softmax):
    t = np.clip(softmax.astype(np.float64), SOFT_CLIP, 1.0)
    t /= t.sum(axis=1, keepdims=True)
    return t


def fit_bcts_calibrator(ref_soft, ref_true, num_classes, n_fit=100_000, seed=42):
    """Bias-corrected temperature scaling fitted on reference-week softmax.

    Returns a callable f(probs)->calibrated probs (abstention TempScaling with
    bias_positions='all', i.e. BCTS of Alexandari et al. 2020).
    """
    from abstention.calibration import TempScaling
    rng = np.random.default_rng(seed)
    n = min(n_fit, len(ref_true))
    sub = rng.choice(len(ref_true), size=n, replace=False)
    v = normalize_probs(ref_soft[sub])
    onehot = np.eye(num_classes, dtype=np.float64)[ref_true[sub]]
    return TempScaling(bias_positions='all')(
        valid_preacts=v, valid_labels=onehot, posterior_supplied=True)


def calibrate(softmax, bcts):
    return bcts(normalize_probs(softmax))


def build_soft_confusion_pinv(ref_soft, ref_true, num_classes, rcond=1e-2):
    """Soft confusion C_s[k,c] = E[softmax_c | true=k] on the reference week,
    returned as pinv(C_s^T) with the same truncation as the paper's hard BBSE
    (an unregularized solve is singular at 180 classes)."""
    C = np.zeros((num_classes, num_classes))
    for k in range(num_classes):
        m = ref_true == k
        if m.sum() > 0:
            C[k] = ref_soft[m].mean(axis=0)
    return np.linalg.pinv(C.T, rcond=rcond)


def bbse_soft_estimate(softmax, Cs_T_pinv):
    """Soft-BBSE estimate of the target label marginal from window softmax."""
    q = softmax.astype(np.float64).mean(axis=0)
    p = np.clip(Cs_T_pinv @ q, 0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(len(q), 1.0 / len(q))
