"""Visualize routing rubric scores as stacked column charts."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    # Allow running as `python teval/utils/visualize_router_rubric.py` from repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from teval.utils.convert_results import (
        build_category_file_map,
        derive_model_name,
        resolve_category_files,
    )
except ModuleNotFoundError:
    try:  # pragma: no cover - fallback for relative import when __package__ is set
        from .convert_results import (
            build_category_file_map,
            derive_model_name,
            resolve_category_files,
        )
    except ModuleNotFoundError:

        def build_category_file_map(model_name: str) -> Dict[str, List[str]]:
            return {
                "Instruct": [f"instruct_{model_name}.json"],
                "Plan": [f"plan_str_{model_name}.json", f"plan_json_{model_name}.json"],
                "Reason": [
                    f"reason_str_{model_name}.json",
                    f"reason_retrieve_understand_json_{model_name}.json",
                ],
                "Retrieve": [
                    f"retrieve_str_{model_name}.json",
                    f"reason_retrieve_understand_json_{model_name}.json",
                ],
                "Understand": [
                    f"understand_str_{model_name}.json",
                    f"reason_retrieve_understand_json_{model_name}.json",
                ],
                "Review": [f"review_str_{model_name}.json"],
            }

        def resolve_category_files(base_dir: str, filename: str) -> List[str]:
            exact = Path(base_dir) / filename
            if exact.exists():
                return [exact.as_posix()]
            stem = exact.stem
            ext = exact.suffix
            pattern = Path(base_dir) / f"{stem}_*{ext}"
            return [p.as_posix() for p in sorted(pattern.parent.glob(pattern.name))]

        def derive_model_name(result_path: str) -> str:
            stem = Path(result_path).stem
            return stem[:-3] if stem.endswith("_-1") else stem


if __package__ in (None, ""):
    from teval.utils.format_load import format_load
else:
    from .format_load import format_load


DEFAULT_CATEGORY_ORDER = [
    "Instruct",
    "Plan",
    "Reason",
    "Retrieve",
    "Understand",
    "Review",
]
CASE_ID = Tuple[str, str, str]  # (dataset, category, entry_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render stacked column charts for routing rubric scores across benchmark subsets."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory containing routing outputs (e.g., work_dirs/<model>_routing).",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g., stacked_rubric.png).",
    )
    parser.add_argument(
        "--separate-files",
        action="store_true",
        help="Save each axis as its own PNG with the axis name appended to --export.",
    )
    parser.add_argument(
        "--split-by",
        choices=["subset", "none"],
        default="subset",
        help="Optional grouping for counts; default keeps capability subsets separate.",
    )
    parser.add_argument(
        "--stack-type",
        choices=["subset", "score"],
        default=None,
        help=(
            "Stacking mode. 'subset' stacks by capability subset (current behavior); "
            "'score' stacks by SLM-LLM score delta buckets; omit for unstacked bars."
        ),
    )
    parser.add_argument(
        "--slm-path",
        type=Path,
        help="SLM direct evaluation directory (required for --stack-type score).",
    )
    parser.add_argument(
        "--llm-path",
        type=Path,
        help="LLM direct evaluation directory (required for --stack-type score).",
    )
    parser.add_argument(
        "--max-total",
        dest="max_total",
        type=int,
        default=9,
        help="Maximum total score to display on the total chart (default 9).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive display (useful for headless environments).",
    )
    return parser.parse_args()


def _safe_load_json(path: Path):
    try:
        import mmengine  # type: ignore
    except ModuleNotFoundError:
        mmengine = None

    if mmengine is not None:
        try:
            return mmengine.load(path)
        except Exception:
            pass

    with path.open() as f:
        return json.load(f)


def _iter_entries(data: object) -> Iterable[Tuple[str, dict]]:
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            for key, value in data.items():
                yield str(key), value
        else:
            yield ("", data)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield str(idx), item


def _parse_score_payload(payload: object) -> Optional[Dict[str, int]]:
    raw: Optional[dict] = None
    if isinstance(payload, dict):
        raw = payload
    elif isinstance(payload, str):
        try:
            parsed = format_load(payload)
            raw = parsed if isinstance(parsed, dict) else None
        except Exception:
            raw = None

    if not isinstance(raw, dict):
        return None

    numeric: Dict[str, int] = {}
    for key, value in raw.items():
        if key == "route":
            continue
        try:
            numeric[key] = int(value)
        except (TypeError, ValueError):
            continue

    if "total" not in numeric and numeric:
        numeric["total"] = sum(v for k, v in numeric.items() if k != "total")

    return numeric or None


def _extract_scores(entry: dict) -> Optional[Dict[str, int]]:
    trace = entry.get("orchestration_trace")
    if not isinstance(trace, dict):
        return None

    steps = trace.get("steps")
    if not isinstance(steps, list):
        return None

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") != "routing_llm_call":
            continue
        scores = _parse_score_payload(step.get("response"))
        if not scores:
            continue
        if scores.get("total") == 12:
            continue
        if any(key != "total" and value > 4 for key, value in scores.items()):
            continue
        return scores
    return None


def _load_entries_from_path(path: Path) -> Iterable[dict]:
    if path.is_file():
        data = _safe_load_json(path)
        yield from _iter_entries(data)
    elif path.is_dir():
        for child in sorted(path.glob("*.json")):
            if not child.is_file():
                continue
            data = _safe_load_json(child)
            yield from _iter_entries(data)


def _discover_summary_file(run_dir: Path) -> Optional[Path]:
    summaries = sorted(run_dir.glob("*_-1.json"))
    return summaries[0] if summaries else None


def _discover_sources(run_dir: Path, model_name: str) -> List[Tuple[str, Path]]:
    category_map = build_category_file_map(model_name)
    sources: List[Tuple[str, Path]] = []
    seen: Set[Path] = set()

    for category, filenames in category_map.items():
        for filename in filenames:
            for resolved in resolve_category_files(run_dir, filename):
                candidate = Path(resolved)
                if candidate.exists() and candidate not in seen:
                    sources.append((category, candidate))
                    seen.add(candidate)

            stem = Path(filename).stem
            for candidate in run_dir.glob(f"{stem}*"):
                if candidate in seen:
                    continue
                if candidate.is_dir():
                    sources.append((category, candidate))
                    seen.add(candidate)

    return sources


def _order_categories(sources: Sequence[Tuple[str, Path]]) -> List[str]:
    discovered = [category for category, _ in sources]
    ordered: List[str] = []
    for name in DEFAULT_CATEGORY_ORDER:
        if name in discovered and name not in ordered:
            ordered.append(name)
    for name in discovered:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _infer_display_tokens(directory: Path) -> Tuple[str, str]:
    """
    Return (base_name, prefix) where base_name strips known orchestrator suffixes
    and prefix trims everything after the first underscore.
    """
    name = directory.name
    for marker in ("_direct", "_routing"):
        idx = name.find(marker)
        if idx != -1:
            name = name[:idx]
            break
    prefix = name.split("_", 1)[0] if "_" in name else name
    return name, prefix


def _normalize_aliases(*aliases: str) -> Tuple[str, ...]:
    """Return ordered unique aliases while dropping empties."""
    seen = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.append(alias)
    return tuple(seen)


def _extract_dataset(stem: str, aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if not alias:
            continue
        token = f"_{alias}_"
        if token in stem:
            return stem.split(token)[0]
        token = f"_{alias}"
        if token in stem:
            return stem.split(token)[0]
    return None


@dataclass
class RubricRecord:
    category: str
    entry_id: str
    scores: Dict[str, int]
    dataset: str


def collect_rubric_scores(run_dir: Path) -> Tuple[List[RubricRecord], List[str], str]:
    summary_file = _discover_summary_file(run_dir)
    model_name = derive_model_name(summary_file.name) if summary_file else run_dir.name

    sources = _discover_sources(run_dir, model_name)
    if not sources:
        raise FileNotFoundError(
            f"No routing result files found in {run_dir}. "
            "Ensure the directory contains per-category JSON outputs."
        )

    category_order = _order_categories(sources)
    inferred_name, inferred_prefix = _infer_display_tokens(run_dir)
    name_aliases = _normalize_aliases(model_name, inferred_name, inferred_prefix)
    records: List[RubricRecord] = []
    for category, path in sources:
        for entry in _load_entries_from_path(path):
            entry_id, payload = (
                entry if isinstance(entry, tuple) else ("", entry)
            )  # backwards compat
            scores = _extract_scores(payload)
            if scores is None:
                continue
            dataset = _extract_dataset(path.stem, name_aliases) or path.stem
            records.append(
                RubricRecord(
                    category=category, entry_id=entry_id, scores=scores, dataset=dataset
                )
            )

    return records, category_order, model_name


def _infer_axes(records: Sequence[RubricRecord]) -> List[str]:
    seen: List[str] = []
    for record in records:
        for key in record.scores.keys():
            if key not in seen:
                seen.append(key)
    # Always move total to the end if present.
    if "total" in seen:
        seen = [k for k in seen if k != "total"] + ["total"]
    return seen


def _compute_ranges(
    records: Sequence[RubricRecord],
    max_total: int,
) -> Tuple[Dict[str, List[int]], List[str]]:
    axes = _infer_axes(records)
    ranges: Dict[str, List[int]] = {}
    for axis in axes:
        observed_max = max((rec.scores.get(axis, 0) for rec in records), default=0)
        upper = max(max_total, observed_max) if axis == "total" else observed_max
        ranges[axis] = list(range(upper + 1))
    return ranges, axes


def _build_unstacked_counts(
    records: Sequence[RubricRecord],
    ranges: Dict[str, List[int]],
) -> Dict[str, Dict[str, List[int]]]:
    counts: Dict[str, Dict[str, List[int]]] = {
        axis: {"All": [0] * len(score_range)} for axis, score_range in ranges.items()
    }
    for record in records:
        for axis, score_range in ranges.items():
            value = record.scores.get(axis)
            if isinstance(value, int) and 0 <= value < len(score_range):
                counts[axis]["All"][value] += 1
    return counts


def _build_group_counts(
    records: Sequence[RubricRecord],
    ranges: Dict[str, List[int]],
    categories: Sequence[str],
    group_fn=lambda record: record.category,
) -> Dict[str, Dict[str, List[int]]]:
    counts: Dict[str, Dict[str, List[int]]] = {
        axis: {category: [0] * len(score_range) for category in categories}
        for axis, score_range in ranges.items()
    }

    for record in records:
        group = group_fn(record)
        if group not in categories:
            continue
        for axis, score_range in ranges.items():
            value = record.scores.get(axis)
            if not isinstance(value, int):
                continue
            if 0 <= value < len(score_range):
                counts[axis][group][value] += 1

    return counts


def _collect_eval_scores(
    base_dir: Path, name_aliases: Sequence[str]
) -> Dict[Tuple[str, str, str], float]:
    summary_file = _discover_summary_file(base_dir)
    model_name = derive_model_name(summary_file.name) if summary_file else base_dir.name
    name_aliases = _normalize_aliases(*name_aliases, model_name)
    sources = _discover_sources(base_dir, model_name)
    scores: Dict[Tuple[str, str, str], float] = {}

    for category, path in sources:
        dataset = _extract_dataset(path.stem, name_aliases) or path.stem
        for entry_id, entry in _load_entries_from_path(path):
            value = None
            if isinstance(entry, tuple):  # backwards compat
                entry_id, entry = entry
            if isinstance(entry, dict):
                value = entry.get("evaluation_result")
            if isinstance(value, (int, float)):
                scores[(dataset, category, str(entry_id))] = float(value)
    return scores


SCORE_BUCKETS: Dict[str, Tuple[str, str]] = {
    "llm_better": (
        "SLM - LLM < -0.5",
        (240 / 255, 57 / 255, 83 / 255),
    ),  # (240, 57, 83)
    "llm_marginal": (
        "-0.5 ≤ SLM - LLM < 0",
        (255 / 255, 185 / 255, 27 / 255),
    ),  # (255, 185, 27)
    "slm_not_worse": (
        "SLM - LLM ≥ 0",
        (30 / 255, 212 / 255, 163 / 255),
    ),  # (30, 212, 163)
}


def _bucket_for_diff(diff: float) -> str:
    if diff < -0.5:
        return "llm_better"
    if diff < 0:
        return "llm_marginal"
    return "slm_not_worse"


def _build_score_bucket_counts(
    records: Sequence[RubricRecord],
    ranges: Dict[str, List[int]],
    slm_scores: Dict[Tuple[str, str, str], float],
    llm_scores: Dict[Tuple[str, str, str], float],
) -> Tuple[Dict[str, Dict[str, List[int]]], int]:
    counts: Dict[str, Dict[str, List[int]]] = {
        axis: {bucket: [0] * len(score_range) for bucket in SCORE_BUCKETS}
        for axis, score_range in ranges.items()
    }
    missing = 0
    for record in records:
        key = (record.dataset, record.category, record.entry_id)
        slm_val = slm_scores.get(key)
        llm_val = llm_scores.get(key)
        if slm_val is None or llm_val is None:
            missing += 1
            continue
        bucket = _bucket_for_diff(slm_val - llm_val)
        for axis, score_range in ranges.items():
            value = record.scores.get(axis)
            if isinstance(value, int) and 0 <= value < len(score_range):
                counts[axis][bucket][value] += 1
    return counts, missing


def plot_columns(
    counts: Dict[str, Dict[str, List[int]]],
    ranges: Dict[str, List[int]],
    axes: Sequence[str],
    series_order: Sequence[str],
    *,
    stacked: bool,
    colors: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    num_axes = len(axes)
    ncols = 2 if num_axes > 1 else 1
    nrows = int(np.ceil(num_axes / ncols))
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize or (12, 4 * nrows))
    axes_flat = axes_arr.flatten() if hasattr(axes_arr, "flatten") else [axes_arr]
    cmap = plt.get_cmap("tab20")

    for idx, axis_key in enumerate(axes):
        ax = axes_flat[idx]
        score_range = ranges[axis_key]
        x = list(score_range)
        axis_label = axis_key.replace("_", " ").title()
        if stacked:
            bottoms = np.zeros(len(score_range))
            for series_idx, label in enumerate(series_order):
                series = counts[axis_key].get(label) or [0] * len(score_range)
                color = (colors or {}).get(label, cmap(series_idx % cmap.N))
                ax.bar(
                    x,
                    series,
                    bottom=bottoms,
                    label=label,
                    color=color,
                    edgecolor="black",
                )
                bottoms += np.array(series)
        else:
            n_series = max(1, len(series_order))
            bar_width = 0.8 / n_series
            offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_series)
            for series_idx, label in enumerate(series_order):
                series = counts[axis_key].get(label) or [0] * len(score_range)
                color = (colors or {}).get(label, cmap(series_idx % cmap.N))
                positions = [val + offsets[series_idx] for val in x]
                ax.bar(
                    positions,
                    series,
                    width=bar_width,
                    label=label,
                    color=color,
                    edgecolor="black",
                )
            ax.set_xticks(x)

        ax.set_xticks(x)
        ax.set_xlabel(f"Rubric Score ({axis_label})", fontsize=15)
        ax.set_ylabel("Test Case Count", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", framealpha=0.9, fontsize=14)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig


def _save_separate_figures(
    base_path: Path,
    counts: Dict[str, Dict[str, List[int]]],
    ranges: Dict[str, List[int]],
    axes: Sequence[str],
    series_order: Sequence[str],
    *,
    stacked: bool,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    # Match aspect ratio of the combined figure: per-axis width derived from the
    # 2-column layout used when multiple axes exist; height stays at 4 units.
    combined_ncols = 2 if len(axes) > 1 else 1
    per_axis_figsize = (12 / combined_ncols, 4)
    for axis_key in axes:
        sub_counts = {axis_key: counts[axis_key]}
        sub_ranges = {axis_key: ranges[axis_key]}
        fig = plot_columns(
            sub_counts,
            sub_ranges,
            [axis_key],
            series_order,
            stacked=stacked,
            colors=colors,
            figsize=per_axis_figsize,
        )
        export_path = base_path.with_name(
            f"{base_path.stem}_{axis_key}{base_path.suffix}"
        )
        fig.savefig(export_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {axis_key} chart to {export_path}")


def _compute_distribution_stats(
    counts: Dict[str, Dict[str, List[int]]],
    ranges: Dict[str, List[int]],
    axes: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for axis in axes:
        score_range = ranges[axis]
        series_counts = counts.get(axis, {})
        total_counts = [0] * len(score_range)
        for series in series_counts.values():
            for idx, val in enumerate(series):
                total_counts[idx] += val
        total = sum(total_counts)
        if total == 0:
            stats[axis] = {"mean": None, "std": None, "skew": None}
            continue
        scores = np.array(score_range, dtype=float)
        freq = np.array(total_counts, dtype=float)
        mean = float(np.sum(scores * freq) / total)
        variance = float(np.sum(freq * (scores - mean) ** 2) / total)
        std = float(np.sqrt(variance))
        if std == 0:
            skew = 0.0
        else:
            skew = float(np.sum(freq * ((scores - mean) / std) ** 3) / total)
        stats[axis] = {"mean": mean, "std": std, "skew": skew}
    return stats


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    records, categories, model_name = collect_rubric_scores(run_dir)
    if not records:
        raise RuntimeError(f"No routing rubric scores found under {run_dir}")

    ranges, axes = _compute_ranges(records, args.max_total)
    series_order: Sequence[str] = ["All"]
    stacked = False
    colors: Optional[Dict[str, str]] = None

    if args.stack_type == "subset":
        group_fn = (
            (lambda record: "All")
            if args.split_by == "none"
            else (lambda record: record.category)
        )
        if args.split_by == "none":
            categories = ["All"]
        counts = _build_group_counts(records, ranges, categories, group_fn=group_fn)
        series_order = categories
        stacked = True
    elif args.stack_type == "score":
        if not args.slm_path or not args.llm_path:
            raise ValueError(
                "--slm-path and --llm-path are required when --stack-type score is used."
            )
        slm_dir = args.slm_path.expanduser().resolve()
        llm_dir = args.llm_path.expanduser().resolve()
        for directory in (slm_dir, llm_dir):
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")
        slm_name, slm_prefix = _infer_display_tokens(slm_dir)
        llm_name, llm_prefix = _infer_display_tokens(llm_dir)
        slm_aliases = _normalize_aliases(slm_name, slm_prefix)
        llm_aliases = _normalize_aliases(llm_name, llm_prefix)
        slm_scores = _collect_eval_scores(slm_dir, slm_aliases)
        llm_scores = _collect_eval_scores(llm_dir, llm_aliases)
        counts, missing = _build_score_bucket_counts(
            records, ranges, slm_scores, llm_scores
        )
        if missing:
            print(
                f"Warning: skipped {missing} routing entries without matching SLM/LLM scores."
            )
        series_order = list(SCORE_BUCKETS.keys())
        colors = {bucket: color for bucket, (_, color) in SCORE_BUCKETS.items()}
        stacked = True
    else:
        counts = _build_unstacked_counts(records, ranges)

    stats = _compute_distribution_stats(counts, ranges, axes)
    print("Rubric score distribution stats (aggregate across series):")
    for axis in axes:
        axis_label = axis.replace("_", " ").title()
        axis_stats = stats.get(axis, {})
        mean = axis_stats.get("mean")
        std = axis_stats.get("std")
        skew = axis_stats.get("skew")

        def fmt(val: Optional[float]) -> str:
            return f"{val:.3f}" if isinstance(val, float) else "N/A"

        print(
            f"  {axis_label:<12}  mean: {fmt(mean)}  std: {fmt(std)}  skew: {fmt(skew)}"
        )

    fig = plot_columns(
        counts,
        ranges,
        axes,
        series_order,
        stacked=stacked,
        colors=colors,
    )

    if args.separate_files and not args.export:
        raise ValueError(
            "--separate-files requires --export to specify the base file path."
        )

    if args.export:
        export_path = args.export.expanduser().resolve()
        fig.savefig(export_path, bbox_inches="tight")
        print(f"Saved stacked charts to {export_path}")
        if args.separate_files:
            _save_separate_figures(
                export_path,
                counts,
                ranges,
                axes,
                series_order,
                stacked=stacked,
                colors=colors,
            )

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
