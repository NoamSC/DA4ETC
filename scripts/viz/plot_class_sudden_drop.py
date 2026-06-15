#!/usr/bin/env python
"""Per-class recall@1 (R1) over weeks, hunting the 'looks good then sudden drop'
teleportation pattern: a class the frozen source model classifies well for many
weeks, then a sharp, sustained collapse (discrete per-class drift).

Frozen source model = week-1 (canonical 52-week monitoring source), evaluated on
every week's test split. Per-class R1 = recall = (# correctly predicted samples
of class c) / (# true class-c samples) in that week.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/anatbr/students/noamshakedc/da4etc"
SRC = "week_1"  # frozen source model
INF_DIR = os.path.join(ROOT, f"results/inference_auditfix/{SRC}_vanilla_bs64")
DATASET_ROOT = "/home/anatbr/dataset/CESNET-TLS-Year22_v2"
OUT_DIR = os.path.join(ROOT, "figs")
MIN_COUNT = 30      # ignore weeks where a class has fewer than this many samples
PRE_LEVEL = 0.70    # "looks good" threshold for the stable prefix
MIN_PRE_WEEKS = 5   # need a real stable run before the drop
MIN_POST_WEEKS = 3  # drop must persist
N_PLOT = 6          # number of top classes to show

# ---- class index -> app name ----------------------------------------------
with open(os.path.join(DATASET_ROOT, "label_mapping.json")) as f:
    app_names = sorted(json.load(f).keys())
idx2name = {i: n for i, n in enumerate(app_names)}

# ---- load per-week files ---------------------------------------------------
files = sorted(glob.glob(os.path.join(INF_DIR, "WEEK-2022-*.npz")))
weeks = [int(os.path.basename(f).split("-")[-1].split(".")[0]) for f in files]
# forward only from source week 1
keep = [(w, f) for w, f in zip(weeks, files) if w >= 1]
weeks = [w for w, _ in keep]
files = [f for _, f in keep]

n_classes = len(app_names)
recall = np.full((n_classes, len(weeks)), np.nan)   # [class, week_idx]
count = np.zeros((n_classes, len(weeks)), dtype=int)
for j, f in enumerate(files):
    d = np.load(f)
    t, p = d["true_labels"], d["pred_labels"]
    for c in np.unique(t):
        m = t == c
        n = int(m.sum())
        count[c, j] = n
        if n >= MIN_COUNT:
            recall[c, j] = float((p[m] == c).mean())


# ---- score the 'stable then sudden drop' pattern ---------------------------
def score_class(c):
    r = recall[c]
    valid = ~np.isnan(r)
    if valid.sum() < (MIN_PRE_WEEKS + MIN_POST_WEEKS):
        return None
    best = None
    idxs = np.where(valid)[0]
    for bi in range(MIN_PRE_WEEKS, len(idxs) - MIN_POST_WEEKS + 1):
        pre_i = idxs[:bi]
        post_i = idxs[bi:]
        pre = r[pre_i].mean()
        post = r[post_i].mean()
        if pre < PRE_LEVEL:
            continue
        drop = pre - post
        # suddenness: largest single consecutive-week fall right around the break
        consec = -np.diff(r[valid])
        suddenness = np.nanmax(consec) if consec.size else 0.0
        # final level should stay low (sustained collapse, not a dip)
        sustained = pre - r[post_i].min()
        sc = drop * (0.5 + 0.5 * suddenness) * (post < 0.6)
        rec = dict(score=sc, drop=drop, pre=pre, post=post,
                   suddenness=suddenness, break_week=weeks[idxs[bi]])
        if best is None or sc > best["score"]:
            best = rec
    return best


cand = []
for c in range(n_classes):
    s = score_class(c)
    if s and s["score"] > 0:
        s["cls"] = c
        s["name"] = idx2name[c]
        cand.append(s)
cand.sort(key=lambda x: x["score"], reverse=True)

print(f"Frozen source: {SRC}; forward weeks {weeks[0]}..{weeks[-1]}; "
      f"{len(cand)} classes match the stable->drop pattern.")
print(f"{'rank':>4} {'cls':>4} {'name':<32} {'pre':>5} {'post':>5} "
      f"{'drop':>5} {'sudden':>6} {'break':>5}")
for i, s in enumerate(cand[:15]):
    print(f"{i:>4} {s['cls']:>4} {s['name'][:32]:<32} {s['pre']:.2f} {s['post']:.2f} "
          f"{s['drop']:.2f} {s['suddenness']:.2f} wk{s['break_week']:>3}")

top = cand[:N_PLOT]

# ---- Fig A: small-multiples of the top drop classes ------------------------
ncol = 3
nrow = int(np.ceil(len(top) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow),
                         sharex=True, squeeze=False)
for ax, s in zip(axes.ravel(), top):
    c = s["cls"]
    r = recall[c]
    ax.plot(weeks, r, color="tab:red", marker="o", ms=3, lw=1.4)
    ax.axvline(s["break_week"], ls=":", color="gray", lw=1.2)
    ax.set_title(f"{s['name']}  (cls {c})\npre={s['pre']:.2f} -> post={s['post']:.2f}",
                 fontsize=9)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.set_ylabel("R1 (recall)")
for ax in axes.ravel()[len(top):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("Week")
fig.suptitle(f"Per-class recall@1 over time — 'looks good then sudden drop' "
             f"(frozen {SRC} model)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
outA = os.path.join(OUT_DIR, "fig_class_sudden_drop_grid.png")
fig.savefig(outA, dpi=150)
plt.close(fig)
print("wrote", outA)

# ---- Fig B: overlay of the top classes on one axis -------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))
cmap = plt.cm.tab10
for i, s in enumerate(top):
    c = s["cls"]
    ax.plot(weeks, recall[c], marker="o", ms=3, lw=1.5, color=cmap(i % 10),
            label=f"{s['name']} (drop wk{s['break_week']})")
ax.axhline(PRE_LEVEL, ls="--", color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("Week")
ax.set_ylabel("R1 (per-class recall)")
ax.set_ylim(-0.03, 1.03)
ax.set_title(f"Discrete per-class teleportations: stable high recall, then sudden "
             f"collapse (frozen {SRC} model)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
outB = os.path.join(OUT_DIR, "fig_class_sudden_drop_overlay.png")
fig.savefig(outB, dpi=150)
plt.close(fig)
print("wrote", outB)
