"""Visualize routing rubric scores as stacked column charts."""

import argparse
import json
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


AXES = (
    ("complexity", "Complexity"),
    ("ambiguity", "Ambiguity"),
    ("constraint_sensitivity", "Constraint Sensitivity"),
    ("total", "Total"),
)
DEFAULT_CATEGORY_ORDER = ["Instruct", "Plan", "Reason", "Retrieve", "Understand", "Review"]


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
        "--title",
        type=str,
        default=None,
        help="Optional title. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g., stacked_rubric.png).",
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


def _iter_entries(data: object) -> Iterable[dict]:
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            yield from (v for v in data.values() if isinstance(v, dict))
        else:
            yield data
    elif isinstance(data, list):
        yield from (item for item in data if isinstance(item, dict))


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

    try:
        complexity = int(raw.get("complexity"))
        ambiguity = int(raw.get("ambiguity"))
        constraint = int(raw.get("constraint_sensitivity"))
    except (TypeError, ValueError):
        return None

    total_val = raw.get("total")
    try:
        total = int(total_val) if total_val is not None else complexity + ambiguity + constraint
    except (TypeError, ValueError):
        total = complexity + ambiguity + constraint

    return {
        "complexity": complexity,
        "ambiguity": ambiguity,
        "constraint_sensitivity": constraint,
        "total": total,
    }


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
        if scores:
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
            if isinstance(data, dict) and not any(isinstance(v, dict) for v in data.values()):
                if data:
                    yield data
            else:
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


def collect_rubric_scores(run_dir: Path) -> Tuple[List[Tuple[str, Dict[str, int]]], List[str], str]:
    summary_file = _discover_summary_file(run_dir)
    model_name = derive_model_name(summary_file.name) if summary_file else run_dir.name

    sources = _discover_sources(run_dir, model_name)
    if not sources:
        raise FileNotFoundError(
            f"No routing result files found in {run_dir}. "
            "Ensure the directory contains per-category JSON outputs."
        )

    category_order = _order_categories(sources)
    records: List[Tuple[str, Dict[str, int]]] = []
    for category, path in sources:
        for entry in _load_entries_from_path(path):
            scores = _extract_scores(entry)
            if scores is None:
                continue
            records.append((category, scores))

    return records, category_order, model_name


def build_counts(
    records: Sequence[Tuple[str, Dict[str, int]]],
    categories: Sequence[str],
    max_total: int,
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, List[int]]]:
    observed_total = max((score["total"] for _, score in records), default=0)
    total_upper = max(max_total, observed_total)

    ranges = {
        "complexity": list(range(4)),
        "ambiguity": list(range(4)),
        "constraint_sensitivity": list(range(4)),
        "total": list(range(total_upper + 1)),
    }
    counts: Dict[str, Dict[str, List[int]]] = {
        axis: {category: [0] * len(score_range) for category in categories}
        for axis, score_range in ranges.items()
    }

    for category, score in records:
        for axis, score_range in ranges.items():
            value = score.get(axis)
            if not isinstance(value, int):
                continue
            if 0 <= value < len(score_range):
                counts[axis][category][value] += 1

    return counts, ranges


def plot_stacked_columns(
    counts: Dict[str, Dict[str, List[int]]],
    ranges: Dict[str, List[int]],
    categories: Sequence[str],
    *,
    title: str,
):
    fig, axes_arr = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes_arr.flatten()
    cmap = plt.get_cmap("tab20")

    for idx, (axis_key, axis_label) in enumerate(AXES):
        ax = axes_flat[idx]
        score_range = ranges[axis_key]
        x = list(score_range)
        bottoms = np.zeros(len(score_range))

        for cat_idx, category in enumerate(categories):
            series = counts[axis_key].get(category) or [0] * len(score_range)
            ax.bar(
                x,
                series,
                bottom=bottoms,
                label=category,
                color=cmap(cat_idx % cmap.N),
                edgecolor="black",
            )
            bottoms += np.array(series)

        ax.set_xticks(x)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(axis_label)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    axes_flat[0].legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), title="Subset")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    records, categories, model_name = collect_rubric_scores(run_dir)
    if not records:
        raise RuntimeError(f"No routing rubric scores found under {run_dir}")

    counts, ranges = build_counts(records, categories, args.max_total)
    chart_title = args.title or f"Routing Rubric Distribution ({model_name})"
    fig = plot_stacked_columns(counts, ranges, categories, title=chart_title)

    if args.export:
        export_path = args.export.expanduser().resolve()
        fig.savefig(export_path, bbox_inches="tight")
        print(f"Saved stacked charts to {export_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
