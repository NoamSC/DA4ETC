#!/usr/bin/env python
"""Future-only (target week >= source week 16) DANN vs vanilla accuracy,
single panel (no delta subplot). Variant of plot_dann_diagnostics.py Fig 6."""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
EXP = os.path.join(ROOT, "exps/cesnet_tls_dann_fwd_w16_v01")
VANILLA_DIR = os.path.join(ROOT, "results/inference_auditfix/week_16_vanilla_bs64")
OUT_DIR = os.path.join(ROOT, "figs/dann")
SOURCE_WEEK = 16
# Future only: source week and all later weeks.
WEEKS = [w for w in range(0, 53) if w >= SOURCE_WEEK]


def week_dir(w):
    return os.path.join(EXP, f"WEEK-2022-16_val_WEEK-2022-{w:02d}")


def dann_final_acc(w):
    p = os.path.join(week_dir(w), "plots", "training_history.pth")
    if not os.path.exists(p):
        return np.nan
    h = torch.load(p, map_location="cpu", weights_only=False)
    va = h.get("val_accuracies")
    return va[-1] / 100.0 if va else np.nan


def vanilla_acc(w):
    p = os.path.join(VANILLA_DIR, f"WEEK-2022-{w:02d}.npz")
    if not os.path.exists(p):
        return np.nan
    d = np.load(p)
    return float(np.mean(d["true_labels"] == d["pred_labels"]))


dann = [dann_final_acc(w) for w in WEEKS]
van = [vanilla_acc(w) for w in WEEKS]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(WEEKS, dann, color="tab:purple", marker="o", ms=4, lw=1.4,
        label="DANN (final-epoch val acc)")
ax.plot(WEEKS, van, color="tab:gray", marker="s", ms=4, lw=1.4,
        label="Vanilla week-16 (frozen)")
ax.axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5, label="source week 16")
ax.set_xlabel("Target week (forward / future only)")
ax.set_ylabel("Accuracy (fraction)")
ax.set_title("DANN target-week accuracy vs vanilla week-16 frozen baseline (future only)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(OUT_DIR, "dann_vs_vanilla_acc_vs_week_future.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out)
