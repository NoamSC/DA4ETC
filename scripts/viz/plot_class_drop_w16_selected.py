#!/usr/bin/env python
"""Per-class recall@1 (R1) over weeks for a hand-picked set of apps, frozen
week-16 source model, forward only (weeks 16..52). R1 = recall = (# correctly
predicted class-c samples) / (# true class-c samples) per week."""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
SRC = "week_16"
INF_DIR = os.path.join(ROOT, f"results/inference_auditfix/{SRC}_vanilla_bs64")
DATASET_ROOT = "/home/anatbr/dataset/CESNET-TLS-Year22_v2"
OUT_DIR = os.path.join(ROOT, "figs")
SOURCE_WEEK = 16
MIN_COUNT = 30  # mask weeks where a class has fewer samples than this

# (index, expected name) as given by the user
SELECTED = [
    (57, "eset-edtd"),
    (102, "microsoft-settings"),
    (49, "docker-registry"),
    (101, "microsoft-push"),
    (56, "eset-edf"),
    (98, "microsoft-defender"),
    (167, "vmware-vcsa"),
    (140, "skype"),
    (97, "microsoft-authentication"),
    (19, "apple-location"),
]

# verify index -> name mapping
with open(os.path.join(DATASET_ROOT, "label_mapping.json")) as f:
    app_names = sorted(json.load(f).keys())
idx2name = {i: n for i, n in enumerate(app_names)}
for c, expected in SELECTED:
    actual = idx2name.get(c, "<oob>")
    flag = "" if actual == expected else f"  <-- MISMATCH (file says '{actual}')"
    print(f"cls {c:>3}: expected '{expected}', mapping '{actual}'{flag}")

# load per-week, forward only
files = sorted(glob.glob(os.path.join(INF_DIR, "WEEK-2022-*.npz")))
pairs = [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f) for f in files]
pairs = [(w, f) for w, f in pairs if w >= SOURCE_WEEK]
weeks = [w for w, _ in pairs]

n_classes = len(app_names)
recall = np.full((n_classes, len(weeks)), np.nan)
count = np.zeros((n_classes, len(weeks)), dtype=int)
for j, (w, f) in enumerate(pairs):
    d = np.load(f)
    t, p = d["true_labels"], d["pred_labels"]
    for c, _ in SELECTED:
        m = t == c
        n = int(m.sum())
        count[c, j] = n
        if n >= MIN_COUNT:
            recall[c, j] = float((p[m] == c).mean())

print(f"\nFrozen source {SRC}; forward weeks {weeks[0]}..{weeks[-1]}")
print(f"{'cls':>4} {'name':<26} {'first':>5} {'min':>5} {'last':>5} {'valid_wks':>9}")
for c, _ in SELECTED:
    r = recall[c]
    v = ~np.isnan(r)
    if v.any():
        print(f"{c:>4} {idx2name[c][:26]:<26} {r[v][0]:.2f} {np.nanmin(r):.2f} "
              f"{r[v][-1]:.2f} {int(v.sum()):>9}")
    else:
        print(f"{c:>4} {idx2name[c][:26]:<26} (no weeks with >= {MIN_COUNT} samples)")

# ---- Fig A: small multiples ------------------------------------------------
ncol = 5
nrow = int(np.ceil(len(SELECTED) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow),
                         sharex=True, sharey=True, squeeze=False)
for ax, (c, _) in zip(axes.ravel(), SELECTED):
    ax.plot(weeks, recall[c], color="tab:red", marker="o", ms=3, lw=1.4)
    ax.set_title(f"{idx2name[c]}  (cls {c})", fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
for ax in axes[:, 0]:
    ax.set_ylabel("R1 (recall)")
for ax in axes[-1]:
    ax.set_xlabel("Week")
for ax in axes.ravel()[len(SELECTED):]:
    ax.set_visible(False)
fig.suptitle(f"Per-class recall@1 over time (frozen {SRC} source, forward)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outA = os.path.join(OUT_DIR, "fig_class_drop_w16_selected_grid.png")
fig.savefig(outA, dpi=150)
plt.close(fig)
print("wrote", outA)

# ---- Fig B: overlay --------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.tab10
for i, (c, _) in enumerate(SELECTED):
    ax.plot(weeks, recall[c], marker="o", ms=3, lw=1.5, color=cmap(i % 10),
            label=f"{idx2name[c]} (cls {c})")
ax.axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5, label="source week 16")
ax.set_xlabel("Week")
ax.set_ylabel("R1 (per-class recall)")
ax.set_ylim(-0.03, 1.03)
ax.set_title(f"Per-class recall@1 over time (frozen {SRC} source, forward)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
outB = os.path.join(OUT_DIR, "fig_class_drop_w16_selected_overlay.png")
fig.savefig(outB, dpi=150)
plt.close(fig)
print("wrote", outB)
