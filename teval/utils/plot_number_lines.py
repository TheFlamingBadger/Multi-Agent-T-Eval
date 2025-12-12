"""Plot min, max, and average on stacked number lines for CSV data.

The input CSV is expected to have two categorical columns followed by one or more
numeric columns (e.g., model, strategy, average F1 score). The script builds two
horizontal number-line plots:

1. One line per unique value in the first column.
2. One line per unique value in the second column.

Each line shows the min-max range (as a horizontal bar) and the average (as a
marker) across all numeric values associated with that category.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked number lines showing min, max, and average for each unique "
            "value in the first two columns of a CSV file."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file (e.g., results.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (e.g., plots.png). Shows the plot interactively regardless.",
    )
    return parser.parse_args()


def _collect_values(rows: Iterable[List[str]], idx: int) -> Dict[str, List[float]]:
    """Collect numeric values keyed by the category in column idx."""

    collected: Dict[str, List[float]] = {}
    for row in rows:
        if len(row) < 3:
            # Need at least two categorical columns + one numeric column.
            continue
        key = row[idx]
        try:
            numeric_values = [float(value) for value in row[2:]]
        except ValueError:
            # Skip rows with non-numeric value in the numeric portion.
            continue
        collected.setdefault(key, []).extend(numeric_values)
    return collected


def _compute_stats(collected: Dict[str, List[float]]) -> List[Tuple[str, float, float, float]]:
    """Return (label, min, mean, max) for each category, preserving insertion order."""

    stats: List[Tuple[str, float, float, float]] = []
    for label, values in collected.items():
        if not values:
            continue
        min_v = min(values)
        max_v = max(values)
        mean_v = sum(values) / len(values)
        stats.append((label, min_v, mean_v, max_v))
    return stats


def _plot_number_lines(ax, stats: List[Tuple[str, float, float, float]], legend_loc: str = "upper right"):
    """Draw a stacked number-line plot for the provided stats."""

    y_positions = list(range(len(stats)))
    for y, (label, min_v, mean_v, max_v) in zip(y_positions, stats):
        ax.hlines(y, xmin=min_v, xmax=max_v, color="tab:blue", linewidth=2)
        ax.plot(min_v, y, marker="|", color="tab:red", markersize=14, label="min" if y == 0 else None)
        ax.plot(max_v, y, marker="|", color="tab:green", markersize=14, label="max" if y == 0 else None)
        ax.plot(mean_v, y, marker="o", color="tab:orange", markersize=10, label="avg" if y == 0 else None)
        ax.text(max_v, y + 0.15, label, fontsize=11, va="bottom")

    ax.set_xlabel("Average F1 Score")
    ax.set_ylim(-0.5, len(stats) - 0.5)
    ax.get_yaxis().set_visible(False)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
    ax.legend(loc=legend_loc)


def main():
    args = parse_args()

    with args.csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Collect per-category numeric values for first and second columns separately.
    first_col_values = _collect_values(rows, idx=0)
    second_col_values = _collect_values(rows, idx=1)

    first_stats = _compute_stats(first_col_values)
    second_stats = _compute_stats(second_col_values)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    _plot_number_lines(axes[0], first_stats, legend_loc="upper right")
    _plot_number_lines(axes[1], second_stats, legend_loc="upper right")

    # Ensure top plot shows x-axis tick labels as well.
    axes[0].tick_params(labelbottom=True)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved plot to {args.output}")

    plt.show()


if __name__ == "__main__":
    main()
