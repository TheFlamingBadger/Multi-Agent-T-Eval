"""Radar diagram visualizer for one or more orchestrator evaluation result files."""

import argparse
import matplotlib as mpl
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mmengine

if __package__ in (None, ""):
    # Allow running as `python teval/utils/visualize_radar_models.py` from repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from teval.utils.convert_results import compute_scores, derive_model_name
else:
    from .convert_results import compute_scores, derive_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a radar diagram from one or more evaluation summary JSON files (model_-1.json)."
    )
    parser.add_argument(
        "result_paths",
        nargs="+",
        help="One or more paths to model summary JSON files.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default=None,
        help="Comma-separated names to use for each result file (must match the number of provided paths).",
    )
    parser.add_argument(
        "--export",
        "--output",
        dest="export",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g., output.png). Shows the plot interactively regardless.",
    )
    parser.add_argument(
        "--colours",
        type=str,
        default=None,
        help="Comma-separated colours to use for each result file (must match the number of provided paths).",
    )
    return parser.parse_args()


def _load_scores(result_path: Path) -> Tuple[List[str], List[Optional[float]], Optional[float], str]:
    """Load summary JSON and compute per-category scores."""
    data = mmengine.load(result_path)
    final_scores, category_scores = compute_scores(data)
    categories = [name for name, _ in category_scores]
    scores = [score for _, score in category_scores]
    overall = final_scores[0] if final_scores else None
    label = derive_model_name(result_path.name)
    return categories, scores, overall, label


def _prepare_series(paths: Iterable[Path], model_names: Optional[Sequence[str]] = None):
    reference_categories: Optional[List[str]] = None
    series: List[Tuple[str, List[Optional[float]], Optional[float]]] = []

    for idx, path in enumerate(paths):
        categories, scores, overall, label = _load_scores(path)
        if model_names is not None:
            label = model_names[idx]
        if reference_categories is None:
            reference_categories = categories
        elif categories != reference_categories:
            raise ValueError(
                f"Category mismatch for {path}: expected {reference_categories}, got {categories}"
            )
        series.append((label, scores, overall))

    if reference_categories is None:
        raise ValueError("No categories found in provided result files.")

    return reference_categories, series


def plot_radar(
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]], Optional[float]]],
    colours: Optional[Sequence[str]] = None,
):
    """Plot a radar chart for one or more model score series."""
    font_size = 14
    plt.rcParams.update({"font.size": font_size})

    palette = mpl.rcParamsDefault.get("axes.prop_cycle", plt.rcParams["axes.prop_cycle"]).by_key().get("color", [])
    if not palette:
        palette = list(plt.cm.get_cmap("tab10").colors)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for idx, (label, scores, overall) in enumerate(series):
        values = [score if score is not None else 0.0 for score in scores]
        values += values[:1]
        display_label = label
        if overall is not None:
            display_label = f"{label} (overall {overall:.2f})"
        if colours is not None:
            color = colours[idx]
        else:
            color = palette[idx % len(palette)]
        ax.plot(angles, values, linewidth=2, label=display_label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=font_size)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0, 1, 6)], fontsize=font_size)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        prop={"size": font_size},
        borderaxespad=1.0,
        ncol=1,
    )

    fig.subplots_adjust(top=0.92, bottom=0.22)
    return fig, ax


def main():
    args = parse_args()
    paths = [Path(p).expanduser().resolve() for p in args.result_paths]
    model_names = None
    if args.model_names:
        model_names = [name.strip() for name in args.model_names.split(",") if name.strip()]
        if len(model_names) != len(paths):
            raise ValueError(
                f"Expected {len(paths)} model names for {len(paths)} files, got {len(model_names)}"
            )

    colours = None
    if args.colours:
        colours = [colour.strip() for colour in args.colours.split(",") if colour.strip()]
        if len(colours) != len(paths):
            raise ValueError(
                f"Expected {len(paths)} colours for {len(paths)} files, got {len(colours)}"
            )

    categories, series = _prepare_series(paths, model_names)
    fig, _ = plot_radar(categories, series, colours)

    if args.export:
        export_path = Path(args.export).expanduser()
        fig.savefig(export_path, bbox_inches="tight")
        print(f"Saved radar diagram to {export_path}")

    plt.show()


if __name__ == "__main__":
    main()
