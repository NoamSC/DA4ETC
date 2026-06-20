#!/usr/bin/env python
"""Per-class BEFORE vs AFTER few-shot prototype repair (Week-16 frozen source).

Replaces the harder-to-read recovery-vs-stability k-curve with a direct
before/after view: each teleported class's recall@1 post-drift (BEFORE, pre-repair)
next to its recall after nearest-class-mean prototype replacement (AFTER, k=50),
plus the stable-majority Macro-F1 before/after to show no negative transfer.

Reads results/repair/few_shot_repair_w16_v01/metrics.json (no recompute).
Numbers are on the 10%-sample frozen outputs (relative before/after only;
not comparable to full-data monitor figures).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MET = ROOT / "results/repair/few_shot_repair_w16_v01/metrics.json"
OUT = ROOT / "figs/repair/fig_perclass_before_after_w16.png"

d = json.load(open(MET))
K_AFTER = "50"
cls_order = ["eset-edtd", "docker-registry", "microsoft-defender", "skype"]

labels, before, after, after_err = [], [], [], []
for name in cls_order:
    c = d["classes"][name]
    labels.append(name)
    before.append(c["base_recall_c"])
    after.append(c["per_k"][K_AFTER]["recall_mean"])
    after_err.append(c["per_k"][K_AFTER]["recall_std"])

# stable majority (Macro-F1): before vs after, averaged over the teleported runs
stab_before = np.mean([d["classes"][n]["base_stable_macroF1"] for n in cls_order])
stab_after = np.mean([d["classes"][n]["per_k"][K_AFTER]["stable_f1_mean"] for n in cls_order])
stab_dmax = max(abs(d["classes"][n]["per_k"][K_AFTER]["stable_f1_mean"]
                    - d["classes"][n]["base_stable_macroF1"]) for n in cls_order)

labels.append("stable majority\n(Macro-F1)")
before.append(stab_before)
after.append(stab_after)
after_err.append(0.0)

n = len(labels)
x = np.arange(n)
w = 0.38

fig, ax = plt.subplots(figsize=(9.2, 5.0))
b1 = ax.bar(x - w / 2, before, w, label="before repair (post-drift)",
            color="#bdbdbd", edgecolor="#555")
b2 = ax.bar(x + w / 2, after, w, yerr=after_err, capsize=4,
            label="after repair (k=50, NCM prototype)", color="#2c7fb8",
            edgecolor="#10456b")

# overlay the k-sweep as dots on each flagged class's AFTER bar
for i, name in enumerate(cls_order):
    c = d["classes"][name]
    for k in ["1", "5", "10", "50"]:
        ax.plot(x[i] + w / 2, c["per_k"][k]["recall_mean"], "o", ms=3.5,
                color="#08306b", alpha=0.55, zorder=5)

# delta arrows / labels for flagged classes
for i in range(len(cls_order)):
    dy = after[i] - before[i]
    ax.annotate(f"+{dy:.2f}", (x[i] + w / 2, after[i] + after_err[i] + 0.03),
                ha="center", va="bottom", fontsize=9, color="#10456b", fontweight="bold")
# stable-majority delta
ax.annotate(f"Δ={stab_after - stab_before:+.3f}\n(|Δ|max={stab_dmax:.3f})",
            (x[-1], max(stab_before, stab_after) + 0.05), ha="center", va="bottom",
            fontsize=8.5, color="#444")

# CoTTA negative-transfer reference line
cotta = d["cotta_fwd_neg_transfer_macroF1"]
ax.axhline(stab_before + cotta, ls="--", lw=1.2, color="#d7301f", alpha=0.8)
ax.annotate(f"global CoTTA forward neg-transfer ({cotta:+.2f} Macro-F1)",
            (n - 0.5, stab_before + cotta - 0.005), ha="right", va="top",
            fontsize=8, color="#d7301f")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("recall@1 (flagged classes) / Macro-F1 (stable majority)")
ax.set_ylim(0, 1.02)
ax.set_title("Few-shot prototype repair: per-class recall recovers, stable majority untouched\n"
             "(Week-16 frozen source, forward-only; 10%-sample frozen outputs)",
             fontsize=10.5)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
print("BEFORE:", {l: round(b, 3) for l, b in zip(labels, before)})
print("AFTER :", {l: round(a, 3) for l, a in zip(labels, after)})
