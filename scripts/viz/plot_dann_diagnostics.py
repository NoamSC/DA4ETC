#!/usr/bin/env python
"""Diagnostic visualizations confirming the BatchNorm-fixed DANN re-run works.

Source week = WEEK-2022-16 (TLS-Year22). 52 target models (val_week N != 16).
Expected good behavior: domain-classifier accuracy starts high (real covariate
shift visible) then trends toward ~50% as the gradient-reversal layer aligns the
domains; DANN (domain BCE) loss rises toward 2*ln(2)=1.386. Separation magnitude
should scale with |week - 16| (temporal distance).

Pure analysis of already-produced artifacts. No SLURM, no training.
"""
import os
import re
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/anatbr/students/noamshakedc/da4etc"
EXP_DIR = os.path.join(REPO, "exps/cesnet_tls_dann_fwd_w16_v01")
LOG_DIR = os.path.join(REPO, "logs")
VANILLA_DIR = os.path.join(REPO, "results/inference_auditfix/week_16_vanilla_bs64")
OUT_DIR = os.path.join(REPO, "figs/dann")
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE_WEEK = 16
ALL_WEEKS = [w for w in range(0, 53) if w != SOURCE_WEEK]
SAMPLE_WEEKS = [0, 10, 17, 25, 40, 52]
MAX_CONFUSION = 2.0 * np.log(2.0)  # 1.386

# consistent color per sample week
_cmap = plt.get_cmap("tab10")
WEEK_COLOR = {w: _cmap(i) for i, w in enumerate(SAMPLE_WEEKS)}

DOM_RE = re.compile(r"Domain Classifier Accuracy=([\d.]+)%")


def week_dir(w):
    return os.path.join(EXP_DIR, f"WEEK-2022-16_val_WEEK-2022-{w:02d}")


def load_history(w):
    p = os.path.join(week_dir(w), "plots", "training_history.pth")
    if not os.path.exists(p):
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] failed to load history for week {w}: {e}")
        return None


def dann_loss_series(h):
    if h is None:
        return None
    to = h.get("train_other_losses")
    if not isinstance(to, dict) or "DANN Loss" not in to:
        return None
    return list(to["DANN Loss"])


def load_domain_acc(w):
    """Pick the log file with the most 'Domain Classifier Accuracy=' lines."""
    pat = os.path.join(LOG_DIR, f"*dann_fwd_w16_to_w{w:02d}_*.out")
    files = glob.glob(pat)
    best, best_vals = None, []
    for f in files:
        try:
            with open(f, "r", errors="ignore") as fh:
                vals = [float(m) for m in DOM_RE.findall(fh.read())]
        except Exception:
            vals = []
        if len(vals) > len(best_vals):
            best, best_vals = f, vals
    return best, best_vals


def vanilla_acc(w):
    p = os.path.join(VANILLA_DIR, f"WEEK-2022-{w:02d}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return float((d["true_labels"] == d["pred_labels"]).mean())


# ---------------------------------------------------------------------------
# Gather data
# ---------------------------------------------------------------------------
print("Loading per-week artifacts...")
hist = {w: load_history(w) for w in ALL_WEEKS}
dom = {}
dom_logfile = {}
for w in ALL_WEEKS:
    f, vals = load_domain_acc(w)
    dom[w] = vals
    dom_logfile[w] = f

# sanity check: domain-acc length vs DANN-loss length for sample weeks
for w in SAMPLE_WEEKS:
    dl = dann_loss_series(hist[w])
    ndl = len(dl) if dl else 0
    nda = len(dom[w])
    if ndl and nda and abs(ndl - nda) > 3:
        print(f"  [note] week {w}: DANN-loss epochs={ndl} vs domain-acc lines={nda} (mismatch)")

# ---------------------------------------------------------------------------
# Fig 1: domain accuracy over epochs (sample weeks)
# ---------------------------------------------------------------------------
def plot_domain_acc_epochs(ax):
    for w in SAMPLE_WEEKS:
        vals = dom[w]
        if not vals:
            continue
        ax.plot(range(1, len(vals) + 1), vals, color=WEEK_COLOR[w],
                marker="o", ms=2.5, lw=1.4, label=f"week {w} (|d|={abs(w-16)})")
    ax.axhline(50, ls="--", color="k", lw=1, alpha=0.7, label="50% chance / aligned")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Domain classifier accuracy (%)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


fig, ax = plt.subplots(figsize=(8, 5))
plot_domain_acc_epochs(ax)
ax.set_title("DANN domain accuracy over epochs\n(high -> falling toward 50% = adversary aligning the domains = GOOD)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_domain_acc_over_epochs.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: DANN loss over epochs (sample weeks)
# ---------------------------------------------------------------------------
def plot_dann_loss_epochs(ax):
    for w in SAMPLE_WEEKS:
        dl = dann_loss_series(hist[w])
        if not dl:
            continue
        ax.plot(range(1, len(dl) + 1), dl, color=WEEK_COLOR[w],
                marker="o", ms=2.5, lw=1.4, label=f"week {w}")
    ax.axhline(MAX_CONFUSION, ls="--", color="k", lw=1, alpha=0.7,
               label=f"2·ln2 ≈ {MAX_CONFUSION:.3f} (max-confusion)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DANN (domain BCE) loss")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


fig, ax = plt.subplots(figsize=(8, 5))
plot_dann_loss_epochs(ax)
ax.set_title("DANN (domain BCE) loss over epochs\n(rising toward 2·ln2 = adversary increasingly confused = GOOD)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_loss_over_epochs.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3: train & val total loss over epochs (two panels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for ax, key, ttl in zip(axes, ["train_losses", "val_losses"], ["Train total loss", "Val total loss"]):
    for w in SAMPLE_WEEKS:
        h = hist[w]
        if h is None or key not in h:
            continue
        v = h[key]
        ax.plot(range(1, len(v) + 1), v, color=WEEK_COLOR[w], lw=1.4, marker="o", ms=2, label=f"week {w}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ttl)
    ax.set_title(ttl)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
fig.suptitle("DANN train vs val total loss over epochs (sample weeks)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_train_val_loss_over_epochs.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4: train & val accuracy over epochs (solid=train, dashed=val)
# ---------------------------------------------------------------------------
def plot_train_val_acc(ax):
    for w in SAMPLE_WEEKS:
        h = hist[w]
        if h is None:
            continue
        ta, va = h.get("train_accuracies"), h.get("val_accuracies")
        if ta:
            ax.plot(range(1, len(ta) + 1), ta, color=WEEK_COLOR[w], ls="-", lw=1.4, label=f"week {w} train")
        if va:
            ax.plot(range(1, len(va) + 1), va, color=WEEK_COLOR[w], ls="--", lw=1.4, label=f"week {w} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=3)


fig, ax = plt.subplots(figsize=(9, 5.5))
plot_train_val_acc(ax)
ax.set_title("DANN train (solid) vs val (dashed) accuracy over epochs")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_train_val_acc_over_epochs.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5: domain accuracy vs week (final + max over epochs), all 52 weeks
# ---------------------------------------------------------------------------
final_dom = {w: (dom[w][-1] if dom[w] else np.nan) for w in ALL_WEEKS}
max_dom = {w: (max(dom[w]) if dom[w] else np.nan) for w in ALL_WEEKS}


def plot_domain_acc_vs_week(ax):
    ws = ALL_WEEKS
    fd = [final_dom[w] for w in ws]
    md = [max_dom[w] for w in ws]
    ax.plot(ws, md, color="tab:red", marker="o", ms=3, lw=1.2, label="max over epochs")
    ax.plot(ws, fd, color="tab:blue", marker="s", ms=3, lw=1.2, label="final epoch")
    ax.axhline(50, ls="--", color="k", lw=1, alpha=0.7, label="50% (aligned)")
    ax.axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5, label="source week 16")
    ax.set_xlabel("Target week")
    ax.set_ylabel("Domain classifier accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


fig, ax = plt.subplots(figsize=(11, 5))
plot_domain_acc_vs_week(ax)
ax.set_title("Domain separation across target weeks (KEY plot)\nmax >> 50% = real covariate shift the adversary then reduces; grows with |week-16|")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_domain_acc_vs_week.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6: DANN vs vanilla accuracy vs week (+ delta panel)
# ---------------------------------------------------------------------------
dann_acc = {}
for w in ALL_WEEKS:
    h = hist[w]
    if h and h.get("val_accuracies"):
        dann_acc[w] = h["val_accuracies"][-1] / 100.0  # fraction
    else:
        dann_acc[w] = np.nan
van_acc = {w: vanilla_acc(w) for w in ALL_WEEKS}


def plot_dann_vs_vanilla(ax_main, ax_delta):
    ws = ALL_WEEKS
    da = [dann_acc[w] for w in ws]
    va = [van_acc[w] if van_acc[w] is not None else np.nan for w in ws]
    ax_main.plot(ws, da, color="tab:purple", marker="o", ms=3, lw=1.3, label="DANN (final-epoch val acc)")
    ax_main.plot(ws, va, color="tab:gray", marker="s", ms=3, lw=1.3, label="Vanilla week-16 (frozen)")
    ax_main.axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5, label="source week 16")
    ax_main.set_ylabel("Accuracy (fraction)")
    ax_main.grid(alpha=0.3)
    ax_main.legend(fontsize=8)
    delta = [(dann_acc[w] - van_acc[w]) if van_acc[w] is not None and not np.isnan(dann_acc[w]) else np.nan for w in ws]
    ax_delta.bar(ws, delta, color=["tab:green" if (d is not None and not np.isnan(d) and d >= 0) else "tab:red" for d in delta])
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5)
    ax_delta.set_xlabel("Target week")
    ax_delta.set_ylabel("Δ (DANN − vanilla)")
    ax_delta.grid(alpha=0.3)


fig, (axm, axd) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
plot_dann_vs_vanilla(axm, axd)
axm.set_title("DANN target-week accuracy vs vanilla week-16 frozen baseline")
fig.text(0.5, 0.005,
         "Caveat: DANN uses val_data_frac=0.1 of target test split (transductive); "
         "vanilla uses data_sample_frac=0.1 seed=42. Populations ~equal; trend comparison valid.",
         ha="center", fontsize=7, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(os.path.join(OUT_DIR, "dann_vs_vanilla_acc_vs_week.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Extra: final domain acc vs |week-16| scatter with trend line
# ---------------------------------------------------------------------------
def plot_distance_scatter(ax):
    dist = np.array([abs(w - SOURCE_WEEK) for w in ALL_WEEKS], dtype=float)
    fd = np.array([final_dom[w] for w in ALL_WEEKS], dtype=float)
    md = np.array([max_dom[w] for w in ALL_WEEKS], dtype=float)
    ax.scatter(dist, md, color="tab:red", s=22, alpha=0.7, label="max domain acc")
    ax.scatter(dist, fd, color="tab:blue", s=22, alpha=0.7, label="final domain acc")
    # trend line on max (more informative of separation magnitude)
    mask = ~np.isnan(md)
    if mask.sum() >= 2:
        m, b = np.polyfit(dist[mask], md[mask], 1)
        xs = np.linspace(dist.min(), dist.max(), 50)
        ax.plot(xs, m * xs + b, color="tab:red", ls="--", lw=1.5,
                label=f"max trend (slope={m:.2f}/wk)")
    ax.axhline(50, ls=":", color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Temporal distance |week − 16|")
    ax.set_ylabel("Domain classifier accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


fig, ax = plt.subplots(figsize=(8, 5))
plot_distance_scatter(ax)
ax.set_title("Domain separation grows with temporal distance from source week 16")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dann_domain_acc_vs_distance.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 7: dashboard 2x3
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
plot_domain_acc_epochs(axes[0, 0]); axes[0, 0].set_title("Domain acc over epochs (sample weeks)")
plot_dann_loss_epochs(axes[0, 1]); axes[0, 1].set_title("DANN loss over epochs")
plot_train_val_acc(axes[0, 2]); axes[0, 2].set_title("Train (solid) / Val (dashed) accuracy")
plot_domain_acc_vs_week(axes[1, 0]); axes[1, 0].set_title("Domain acc vs target week (final & max)")
# panel for dann vs vanilla: just main curve here
ws = ALL_WEEKS
axes[1, 1].plot(ws, [dann_acc[w] for w in ws], color="tab:purple", marker="o", ms=3, lw=1.2, label="DANN")
axes[1, 1].plot(ws, [van_acc[w] if van_acc[w] is not None else np.nan for w in ws],
                color="tab:gray", marker="s", ms=3, lw=1.2, label="Vanilla w16")
axes[1, 1].axvline(SOURCE_WEEK, ls=":", color="green", lw=1.5)
axes[1, 1].set_xlabel("Target week"); axes[1, 1].set_ylabel("Accuracy (fraction)")
axes[1, 1].set_title("DANN vs vanilla accuracy vs week"); axes[1, 1].grid(alpha=0.3); axes[1, 1].legend(fontsize=8)
plot_distance_scatter(axes[1, 2]); axes[1, 2].set_title("Domain separation vs |week−16|")
fig.suptitle("DANN BatchNorm-fix diagnostic dashboard (source = WEEK-2022-16)", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUT_DIR, "dann_diagnostic_dashboard.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DANN DIAGNOSTIC SUMMARY")
print("=" * 70)
print("\nPer sample week:")
print(f"{'week':>5} {'|d|':>4} {'final_dom%':>10} {'max_dom%':>9} {'final_DANN':>11} {'final_val_acc%':>14}")
for w in SAMPLE_WEEKS:
    dl = dann_loss_series(hist[w])
    fdann = dl[-1] if dl else float("nan")
    h = hist[w]
    fval = h["val_accuracies"][-1] if (h and h.get("val_accuracies")) else float("nan")
    print(f"{w:>5} {abs(w-16):>4} {final_dom[w]:>10.2f} {max_dom[w]:>9.2f} {fdann:>11.3f} {fval:>14.2f}")

# across all weeks
fd_arr = np.array([final_dom[w] for w in ALL_WEEKS], dtype=float)
md_arr = np.array([max_dom[w] for w in ALL_WEEKS], dtype=float)
dist_arr = np.array([abs(w - 16) for w in ALL_WEEKS], dtype=float)
fmask = ~np.isnan(fd_arr); mmask = ~np.isnan(md_arr)
print("\nAcross all 52 target weeks:")
print(f"  final domain acc:  mean={np.nanmean(fd_arr):.2f}%  min={np.nanmin(fd_arr):.2f}%  max={np.nanmax(fd_arr):.2f}%  (weeks with data: {fmask.sum()})")
print(f"  max   domain acc:  mean={np.nanmean(md_arr):.2f}%  min={np.nanmin(md_arr):.2f}%  max={np.nanmax(md_arr):.2f}%  (weeks with data: {mmask.sum()})")


def corr(x, y, mask):
    if mask.sum() >= 2:
        return float(np.corrcoef(x[mask], y[mask])[0, 1])
    return float("nan")


print(f"  corr(|week-16|, final domain acc) = {corr(dist_arr, fd_arr, fmask):.3f}")
print(f"  corr(|week-16|, max   domain acc) = {corr(dist_arr, md_arr, mmask):.3f}")

deltas = []
for w in ALL_WEEKS:
    if van_acc[w] is not None and not np.isnan(dann_acc[w]):
        deltas.append(dann_acc[w] - van_acc[w])
deltas = np.array(deltas)
print(f"  mean DANN−vanilla accuracy delta = {deltas.mean():+.4f} (n={len(deltas)}; "
      f"min={deltas.min():+.4f}, max={deltas.max():+.4f})")

# weeks missing domain-acc data
missing_dom = [w for w in ALL_WEEKS if not dom[w]]
missing_hist = [w for w in ALL_WEEKS if hist[w] is None]
if missing_dom:
    print(f"\n  [weeks missing domain-acc log lines]: {missing_dom}")
if missing_hist:
    print(f"  [weeks missing training_history]: {missing_hist}")

print("\nPNG files written to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        print("  ", os.path.join(OUT_DIR, f))
