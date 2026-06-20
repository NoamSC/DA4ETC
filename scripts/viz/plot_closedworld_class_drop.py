#!/usr/bin/env python
"""Per-class recall@1 over time on the proprietary closed-world dataset, hunting
the discrete 'teleportation' pattern: a class the frozen source classifies well
for many windows, then a sharp, sustained collapse (per-class discrete drift) —
the closed-world analogue of fig_class_sudden_drop on CESNET.

Anonymized: the dataset is labelled ONLY as "proprietary closed-world (anonymized
for review)". Class identities are shown as opaque indices, never real appIds.

Frozen source = the clean-regime quarter slice (Week-16 equivalent). Evaluated
forward only (window >= source window) on the vanilla (no-adaptation) outputs.
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
INF_DIR = os.path.join(ROOT, "exps/allot_multimodal/quarter_eq/inference")  # vanilla
SRC_WINDOW = 13
OUT_DIR = os.path.join(ROOT, "figs")
MIN_COUNT = 20      # ignore windows where a class has < this many samples
PRE_LEVEL = 0.70    # "looks good" threshold for the stable prefix
MIN_PRE = 4         # need a real stable run before the drop
MIN_POST = 3        # drop must persist
N_PLOT = 6


def _win(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])


files = sorted(glob.glob(os.path.join(INF_DIR, "window_*.npz")))
files = [f for f in files if _win(f) >= SRC_WINDOW]
windows = [_win(f) for f in files]

# Discover class space from all forward windows.
classes = set()
data = []
for f in files:
    z = np.load(f, allow_pickle=True)
    t, p = z["true_labels"].astype(int), z["pred_labels"].astype(int)
    data.append((t, p))
    classes.update(np.unique(t).tolist())
classes = sorted(classes)
cidx = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)

recall = np.full((n_classes, len(windows)), np.nan)
count = np.zeros((n_classes, len(windows)), dtype=int)
for j, (t, p) in enumerate(data):
    for c in np.unique(t):
        m = t == c
        n = int(m.sum())
        count[cidx[c], j] = n
        if n >= MIN_COUNT:
            recall[cidx[c], j] = float((p[m] == c).mean())


def score_class(ci):
    r = recall[ci]
    valid = ~np.isnan(r)
    if valid.sum() < (MIN_PRE + MIN_POST):
        return None
    idxs = np.where(valid)[0]
    best = None
    for bi in range(MIN_PRE, len(idxs) - MIN_POST + 1):
        pre = r[idxs[:bi]].mean()
        post = r[idxs[bi:]].mean()
        if pre < PRE_LEVEL:
            continue
        drop = pre - post
        consec = -np.diff(r[valid])
        sudden = np.nanmax(consec) if consec.size else 0.0
        sc = drop * (0.5 + 0.5 * sudden) * (post < 0.6)
        rec = dict(score=sc, drop=drop, pre=pre, post=post, sudden=sudden,
                   break_win=windows[idxs[bi]])
        if best is None or sc > best["score"]:
            best = rec
    return best


cand = []
for ci in range(n_classes):
    s = score_class(ci)
    if s and s["score"] > 0:
        s["ci"] = ci
        s["cls_label"] = f"class {ci}"  # opaque index, anonymized
        cand.append(s)
cand.sort(key=lambda x: x["score"], reverse=True)

print(f"Forward windows {windows[0]}..{windows[-1]}; {n_classes} classes; "
      f"{len(cand)} match the stable->sudden-drop teleportation pattern.")
print(f"{'rank':>4} {'cls':>4} {'pre':>5} {'post':>5} {'drop':>5} {'sudden':>6} {'break':>6}")
for i, s in enumerate(cand[:15]):
    print(f"{i:>4} {s['ci']:>4} {s['pre']:.2f} {s['post']:.2f} {s['drop']:.2f} "
          f"{s['sudden']:.2f} w{s['break_win']:>4}")

top = cand[:N_PLOT]
if not top:
    print("No teleportation-pattern classes found; emitting overall-accuracy fallback only.")

# ---- Fig: small-multiples of the top discrete-drop classes ----
if top:
    ncol = 3
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes.ravel(), top):
        ci = s["ci"]
        ax.plot(windows, recall[ci], color="tab:red", marker="o", ms=3, lw=1.4)
        ax.axvline(s["break_win"], ls=":", color="gray", lw=1.2)
        ax.set_title(f"{s['cls_label']}: pre={s['pre']:.2f}→post={s['post']:.2f}", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.set_ylabel("R1 (recall)")
    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("forward window")
    fig.suptitle("Proprietary closed-world (anonymized): per-class recall@1 over time —\n"
                 "discrete per-class 'teleportations' (stable, then sudden collapse) from a "
                 "frozen source model", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outA = os.path.join(OUT_DIR, "fig_closedworld_class_sudden_drop_grid.png")
    fig.savefig(outA, dpi=150)
    plt.close(fig)
    print("wrote", outA)

    # ---- overlay ----
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = plt.cm.tab10
    for i, s in enumerate(top):
        ax.plot(windows, recall[s["ci"]], marker="o", ms=3, lw=1.5, color=cmap(i % 10),
                label=f"{s['cls_label']} (drop w{s['break_win']})")
    ax.axhline(PRE_LEVEL, ls="--", color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("forward window")
    ax.set_ylabel("R1 (per-class recall)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Proprietary closed-world (anonymized): discrete, asynchronous per-class "
                 "drift —\nstable high recall, then sudden collapse at different windows")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    outB = os.path.join(OUT_DIR, "fig_closedworld_class_sudden_drop_overlay.png")
    fig.savefig(outB, dpi=150)
    plt.close(fig)
    print("wrote", outB)
