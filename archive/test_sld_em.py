#!/usr/bin/env python
"""
Sanity tests for sld_em_estimation.

Key insight: EM needs stochastic softmax outputs (different samples from the
same class must produce different softmax values). Without within-class variation,
the mean softmax is biased and EM has no fixed point at the true distribution.

The correct simulation: logits = signal * one_hot(true) + log(train_prior) + noise
  - log(train_prior): bakes training calibration into the model outputs
  - noise: represents within-class feature variation (essential for EM to work)
"""

import numpy as np
from plot_paper_figures import sld_em_estimation

rng = np.random.default_rng(42)


def make_softmax(true_labels, train_prior, num_classes, signal=3.0, noise_std=1.5):
    """
    Simulate softmax from a model trained on train_prior.
    logit[k] = signal * delta(k, true) + log(train_prior[k]) + noise[k]

    noise_std controls within-class variation — essential for EM calibration.
    Without noise, all class-c samples are identical and the mean softmax is biased.
    """
    n = len(true_labels)
    log_prior = np.log(np.clip(train_prior, 1e-12, 1.0))
    logits = np.tile(log_prior, (n, 1)) + rng.normal(0, noise_std, (n, num_classes))
    logits[np.arange(n), true_labels] += signal
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def l1(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run_test(name, train_prior, test_prior, num_classes, n=20_000, signal=3.0, noise_std=1.5):
    train_prior = np.array(train_prior, dtype=float); train_prior /= train_prior.sum()
    test_prior  = np.array(test_prior,  dtype=float); test_prior  /= test_prior.sum()

    true_labels = rng.choice(num_classes, size=n, p=test_prior)
    softmax     = make_softmax(true_labels, train_prior, num_classes, signal, noise_std)

    em_est    = sld_em_estimation(train_prior, softmax)
    naive_est = np.bincount(softmax.argmax(axis=1), minlength=num_classes).astype(float)
    naive_est /= naive_est.sum()

    l1_em     = l1(em_est,     test_prior)
    l1_naive  = l1(naive_est,  test_prior)
    l1_static = l1(train_prior, test_prior)

    print(f"\n{'─'*60}")
    print(f" {name}")
    print(f"{'─'*60}")
    print(f"  train prior  : {np.round(train_prior, 3)}")
    print(f"  test prior   : {np.round(test_prior,  3)}")
    print(f"  EM estimate  : {np.round(em_est,      3)}")
    print(f"  naive est    : {np.round(naive_est,   3)}")
    print(f"  L1 — EM: {l1_em:.4f}   naive: {l1_naive:.4f}   static: {l1_static:.4f}")

    em_vs_naive  = "EM < naive  ✓" if l1_em < l1_naive  else "EM ≥ naive  ✗"
    em_vs_static = "EM < static ✓" if l1_em < l1_static else "EM ≥ static  (no shift expected)"
    print(f"  {em_vs_naive}   {em_vs_static}")
    return l1_em < l1_naive, l1_em < l1_static


K = 5
tests = [
    # When train==test, EM should return near train prior (beats static = 0 shift trivially)
    ("No shift (train == test)",
     [0.5, 0.25, 0.1, 0.1, 0.05], [0.5, 0.25, 0.1, 0.1, 0.05]),

    # Moderate shift — EM and naive both good, EM should beat static prior
    ("Moderate shift",
     [0.5, 0.25, 0.1, 0.1, 0.05], [0.05, 0.1, 0.1, 0.25, 0.5]),

    # Class disappears — EM should correctly push to 0
    ("Extreme shift (class 0 disappears)",
     [0.5, 0.25, 0.1, 0.1, 0.05], [0.0, 0.3, 0.3, 0.2, 0.2]),

    # Highly imbalanced + weak model: naive is biased, EM should clearly win
    ("Imbalanced prior + weak model (signal=1.5)",
     [0.9, 0.1], [0.1, 0.9]),
]

print("SLD-EM sanity tests")
print("(softmax: log-prior baseline + signal at true class + within-class noise)")

passed_naive = passed_static = 0
for i, (name, train, test) in enumerate(tests):
    sig = 1.5 if i == 3 else 3.0   # weaker model for the imbalanced test
    K_  = len(train)
    b1, b2 = run_test(name, train, test, K_, signal=sig)
    passed_naive  += b1
    passed_static += b2

print(f"\n{'='*60}")
print(f"  EM < naive  : {passed_naive}/{len(tests)}")
print(f"  EM < static : {passed_static}/{len(tests)}")
print(f"{'='*60}")
