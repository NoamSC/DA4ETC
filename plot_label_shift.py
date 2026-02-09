"""Plot label distribution shift over weeks in CESNET-TLS-Year22_v2."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_label_distribution(parquet_path):
    """Read only the APP column and return normalized label counts."""
    df = pd.read_parquet(parquet_path, columns=["APP"])
    counts = df["APP"].value_counts(normalize=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Plot label shift across weeks")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/anatbr/dataset/CESNET-TLS-Year22_v2",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Number of top labels to show individually (rest grouped as 'other')",
    )
    parser.add_argument("--output", type=str, default="label_shift.png")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    # Collect all week directories sorted by week number
    week_dirs = sorted(dataset_root.glob("WEEK-2022-*"), key=lambda p: int(p.name.split("-")[-1]))

    weeks = []
    distributions = []
    for week_dir in week_dirs:
        test_path = week_dir / "test.parquet"
        if not test_path.exists():
            continue
        week_num = int(week_dir.name.split("-")[-1])
        dist = get_label_distribution(test_path)
        weeks.append(week_num)
        distributions.append(dist)
        print(f"Week {week_num}: {len(dist)} unique labels, {dist.iloc[0]:.3f} top-1 share ({dist.index[0]})")

    # Build a unified DataFrame: rows=weeks, cols=app labels, values=proportions
    dist_df = pd.DataFrame(distributions, index=weeks).fillna(0)
    dist_df.index.name = "week"

    # Identify top-K labels by average proportion across all weeks
    avg_proportions = dist_df.mean().sort_values(ascending=False)
    top_labels = avg_proportions.head(args.top_k).index.tolist()
    other = dist_df.drop(columns=top_labels).sum(axis=1)

    # --- Compute pairwise JS divergence matrix ---
    from scipy.spatial.distance import jensenshannon

    n_weeks = len(weeks)
    js_matrix = np.zeros((n_weeks, n_weeks))
    for i in range(n_weeks):
        for j in range(i + 1, n_weeks):
            d = jensenshannon(dist_df.iloc[i].values, dist_df.iloc[j].values)
            js_matrix[i, j] = d
            js_matrix[j, i] = d

    # --- Plot 1: Stacked area chart ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 20), gridspec_kw={"height_ratios": [1, 1, 1.2]})

    ax = axes[0]
    plot_df = dist_df[top_labels].copy()
    plot_df["other"] = other
    ax.stackplot(plot_df.index, plot_df.values.T, labels=plot_df.columns, alpha=0.85)
    ax.set_xlabel("Week")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Label Distribution Over Time (top {args.top_k} + other)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.set_ylim(0, 1)

    # --- Plot 2: JS divergence from week 0 ---
    js_divs = js_matrix[0]

    ax = axes[1]
    ax.plot(weeks, js_divs, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("JS Divergence from Week 0")
    ax.set_title("Label Shift: Jensen-Shannon Divergence vs. Week 0")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Pairwise JS divergence heatmap ---
    ax = axes[2]
    im = ax.imshow(js_matrix, cmap="YlOrRd", aspect="equal", origin="lower")
    ax.set_xticks(range(0, n_weeks, 5))
    ax.set_xticklabels([weeks[i] for i in range(0, n_weeks, 5)])
    ax.set_yticks(range(0, n_weeks, 5))
    ax.set_yticklabels([weeks[i] for i in range(0, n_weeks, 5)])
    ax.set_xlabel("Week")
    ax.set_ylabel("Week")
    ax.set_title("Pairwise Jensen-Shannon Divergence (all weeks vs all weeks)")
    plt.colorbar(im, ax=ax, label="JS Divergence")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()