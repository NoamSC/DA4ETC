import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Ellipse

REPO = "/home/anatbr/students/noamshakedc/da4etc"
FIGS = os.path.join(REPO, "figs")

# ---------------------------------------------------------------------------
# FIGURE 1 - concept schematic: continuous vs discrete drift
# ---------------------------------------------------------------------------
MUTED = {
    "blue": "#4878a8",
    "orange": "#d98c4a",
    "green": "#6aa56a",
    "gray": "#9a9a9a",
    "dark": "#333333",
}


def blob(ax, x, y, color, alpha=0.85, w=0.9, h=0.7, lw=0.0, edge=None, ls="-"):
    e = Ellipse((x, y), w, h, facecolor=color, alpha=alpha, edgecolor=edge or color,
                linewidth=lw, linestyle=ls)
    ax.add_patch(e)
    return e


def make_concept_fig():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=220)

    for ax in (axL, axR):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("feature dim 1", fontsize=8, color=MUTED["gray"])
        ax.set_ylabel("feature dim 2", fontsize=8, color=MUTED["gray"])
        for spine in ax.spines.values():
            spine.set_edgecolor(MUTED["gray"])
            spine.set_linewidth(0.8)

    # ---- LEFT: continuous drift ----
    axL.set_title("Continuous drift\n(the UDA assumption)", fontsize=10,
                  fontweight="bold", color=MUTED["dark"])
    # a class smoothly sliding across feature space, fading old positions
    xs = np.linspace(2.0, 8.0, 6)
    ys = 3.0 + 3.5 * np.sin(np.linspace(0.2, 1.3, 6))
    for i, (x, y) in enumerate(zip(xs, ys)):
        a = 0.25 + 0.55 * (i / (len(xs) - 1))
        blob(axL, x, y, MUTED["blue"], alpha=a, w=0.95, h=0.8)
    # curving trajectory arrow
    for i in range(len(xs) - 1):
        ar = FancyArrowPatch((xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                             arrowstyle="-|>", mutation_scale=9,
                             color=MUTED["dark"], lw=1.0, alpha=0.65,
                             connectionstyle="arc3,rad=0.15")
        axL.add_patch(ar)
    axL.text(xs[0], ys[0] - 1.0, "wk 16", fontsize=8, ha="center", color=MUTED["dark"])
    axL.text(xs[-1], ys[-1] + 0.9, "wk 30", fontsize=8, ha="center", color=MUTED["dark"])
    axL.text(5.0, 0.9, "smooth, gradual\n(global adaptation can track it)",
             fontsize=8, ha="center", color=MUTED["blue"], style="italic")

    # ---- RIGHT: discrete per-class teleportation ----
    axR.set_title("Discrete per-class teleportation\n(what we observe)",
                  fontsize=10, fontweight="bold", color=MUTED["dark"])
    # two stable classes
    blob(axR, 2.4, 7.4, MUTED["green"], alpha=0.85, w=1.0, h=0.85)
    axR.text(2.4, 6.6, "class B (stable)", fontsize=7.5, ha="center", color=MUTED["green"])
    blob(axR, 2.8, 2.6, MUTED["gray"], alpha=0.85, w=1.0, h=0.85)
    axR.text(2.8, 1.8, "class C (stable)", fontsize=7.5, ha="center", color=MUTED["gray"])

    # teleporting class A: stable blob at left, jump to distant region
    blob(axR, 2.0, 5.0, MUTED["orange"], alpha=0.85, w=1.0, h=0.85)
    axR.text(2.0, 4.2, "class A\nwk 16 ... wk t-1\n(stable)", fontsize=7.5, ha="center",
             color=MUTED["orange"])
    # destination blob (new empty region)
    blob(axR, 8.3, 6.0, MUTED["orange"], alpha=0.85, w=1.0, h=0.85, lw=1.2,
         edge=MUTED["orange"], ls="--")
    axR.text(8.3, 7.0, "wk t: JUMP", fontsize=8, ha="center", fontweight="bold",
             color=MUTED["orange"])
    # big dashed-gap arrow across the empty region
    ar = FancyArrowPatch((2.9, 5.1), (7.4, 5.9), arrowstyle="-|>",
                         mutation_scale=13, color=MUTED["orange"], lw=1.6,
                         linestyle=(0, (4, 3)),
                         connectionstyle="arc3,rad=-0.2")
    axR.add_patch(ar)
    axR.text(5.1, 3.7, "single discrete jump\n(asynchronous, one class)",
             fontsize=8, ha="center", color=MUTED["orange"], style="italic")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig_continuous_vs_discrete_drift.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# FIGURE 2 - entropy is indicative (REAL data)
# ---------------------------------------------------------------------------
def make_entropy_fig():
    # forward weeks 17-52
    weeks = range(17, 53)
    H_correct = []
    H_incorrect = []
    eps = 1e-12
    for w in weeks:
        fp = os.path.join(REPO, f"results/inference_auditfix/week_16_vanilla_bs64/WEEK-2022-{w:02d}.npz")
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        sm = d["softmax"].astype(np.float64)
        true = d["true_labels"]
        pred = d["pred_labels"]
        H = -np.sum(sm * np.log(sm + eps), axis=1)
        correct = pred == true
        H_correct.append(H[correct])
        H_incorrect.append(H[~correct])

    Hc = np.concatenate(H_correct)
    Hi = np.concatenate(H_incorrect)
    mean_c = float(Hc.mean())
    mean_i = float(Hi.mean())
    n_c = int(Hc.size)
    n_i = int(Hi.size)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    bins = np.linspace(0, max(Hc.max(), Hi.max()), 80)
    ax.hist(Hc, bins=bins, density=True, alpha=0.55, color="#4878a8",
            label=f"correct (n={n_c:,})")
    ax.hist(Hi, bins=bins, density=True, alpha=0.55, color="#c44e52",
            label=f"incorrect (n={n_i:,})")

    ax.axvline(mean_c, color="#2f5577", lw=1.6, ls="--")
    ax.axvline(mean_i, color="#8a2f33", lw=1.6, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(mean_c, ymax * 0.92, f"mean={mean_c:.2f}", color="#2f5577",
            fontsize=9, ha="right", va="top", rotation=90)
    ax.text(mean_i, ymax * 0.92, f"mean={mean_i:.2f}", color="#8a2f33",
            fontsize=9, ha="left", va="top", rotation=90)

    ax.set_xlabel("Predictive entropy (nats)", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title("Misclassified flows have higher predictive entropy\n"
                 "(Week-16 source, forward weeks 17-52)", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, "entropy_is_indicative.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, mean_c, mean_i, n_c, n_i


if __name__ == "__main__":
    p1 = make_concept_fig()
    print("FIG1:", p1)
    p2, mc, mi, nc, ni = make_entropy_fig()
    print("FIG2:", p2)
    print(f"correct mean entropy = {mc:.4f} (n={nc})")
    print(f"incorrect mean entropy = {mi:.4f} (n={ni})")
