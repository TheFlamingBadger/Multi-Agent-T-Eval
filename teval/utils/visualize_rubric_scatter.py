"""Scatter plot of rubric total scores vs. model evaluation results."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    # Allow running as `python teval/utils/visualize_rubric_scatter.py` from repo root.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from teval.utils.convert_results import (
        build_category_file_map,
        derive_model_name,
        resolve_category_files,
    )
except ModuleNotFoundError:
    try:  # pragma: no cover - relative import fallback
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


EntryKey = Tuple[str, str]  # (category, entry_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter plot of rubric total scores vs evaluation_result."
    )
    parser.add_argument(
        "--direct-dir",
        required=True,
        type=Path,
        help="Directory containing direct evaluation outputs (e.g., work_dirs/<model>_direct).",
    )
    parser.add_argument(
        "--rubric-dir",
        required=True,
        type=Path,
        help="Directory containing rubric routing outputs.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Optional path to save the figure (e.g., rubric_scatter.png).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Defaults to derived model name.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive display (headless environments).",
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
        for key, val in data.items():
            if isinstance(val, dict):
                yield str(key), val
    elif isinstance(data, list):
        for idx, val in enumerate(data):
            if isinstance(val, dict):
                yield str(idx), val


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
                if candidate.is_dir() or candidate.suffix == ".json":
                    sources.append((category, candidate))
                    seen.add(candidate)

    return sources


def _category_order(sources: Sequence[Tuple[str, Path]]) -> List[str]:
    ordered: List[str] = []
    for category, _ in sources:
        if category not in ordered:
            ordered.append(category)
    return ordered


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
        cpx = int(raw.get("complexity"))
        amb = int(raw.get("ambiguity"))
        con = int(raw.get("constraint_sensitivity"))
    except (TypeError, ValueError):
        return None

    total_val = raw.get("total")
    try:
        total = int(total_val) if total_val is not None else cpx + amb + con
    except (TypeError, ValueError):
        total = cpx + amb + con

    return {
        "complexity": cpx,
        "ambiguity": amb,
        "constraint_sensitivity": con,
        "total": total,
    }


def _extract_total_from_entry(entry: dict) -> Optional[int]:
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
            return scores.get("total")
    return None


def _load_entries_from_path(path: Path) -> Iterable[Tuple[str, dict]]:
    if path.is_file():
        data = _safe_load_json(path)
        yield from _iter_entries(data)
    elif path.is_dir():
        for child in sorted(path.glob("*.json")):
            data = _safe_load_json(child)
            yield from _iter_entries(data)


def collect_direct_scores(run_dir: Path) -> Dict[EntryKey, float]:
    summary = _discover_summary_file(run_dir)
    model_name = derive_model_name(summary.name) if summary else run_dir.name
    sources = _discover_sources(run_dir, model_name)

    scores: Dict[EntryKey, float] = {}
    for category, path in sources:
        for entry_id, entry in _load_entries_from_path(path):
            val = entry.get("evaluation_result")
            if isinstance(val, (int, float)):
                scores[(category, entry_id)] = float(val)
    return scores


def collect_rubric_totals(run_dir: Path) -> Dict[EntryKey, int]:
    summary = _discover_summary_file(run_dir)
    model_name = derive_model_name(summary.name) if summary else run_dir.name
    sources = _discover_sources(run_dir, model_name)

    totals: Dict[EntryKey, int] = {}
    for category, path in sources:
        for entry_id, entry in _load_entries_from_path(path):
            total = _extract_total_from_entry(entry)
            if isinstance(total, int):
                totals[(category, entry_id)] = total
    return totals


def _prepare_points(
    direct_scores: Dict[EntryKey, float],
    rubric_totals: Dict[EntryKey, int],
) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []
    for key, score in direct_scores.items():
        total = rubric_totals.get(key)
        if total is None:
            continue
        points.append((total, score))
    return points


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return y_pred, float(r2), rmse


def plot_scatter(points: List[Tuple[int, float]], title: str, export: Optional[Path], show: bool) -> None:
    if not points:
        raise RuntimeError("No overlapping entries between direct and rubric data.")

    x_vals = np.array([p[0] for p in points], dtype=float)
    y_vals = np.array([p[1] for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Box-and-whisker at each total score.
    grouped: Dict[int, List[float]] = {}
    for total, score in points:
        grouped.setdefault(total, []).append(score)
    totals_sorted = sorted(grouped.keys())
    data = [grouped[t] for t in totals_sorted]
    ax.boxplot(
        data,
        positions=totals_sorted,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#9ecae1", color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(markeredgecolor="#08519c", markerfacecolor="#08519c", markersize=4),
    )

    # Best-fit line computed over individual points.
    y_pred, r2, rmse = _linear_fit(x_vals, y_vals)
    sort_idx = np.argsort(x_vals)
    ax.plot(x_vals[sort_idx], y_pred[sort_idx], color="red", linewidth=2, label="Best fit")

    ax.set_xlabel("granite_4_4b Estimated Question Difficulty (Rubric Score)")
    ax.set_ylabel("azure_gpt4o Test Case Score")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1.05)

    text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}"
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5),
    )

    fig.tight_layout()

    if export:
        export_path = export.expanduser().resolve()
        fig.savefig(export_path, bbox_inches="tight")
        print(f"Saved scatter plot to {export_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    direct_dir = args.direct_dir.expanduser().resolve()
    rubric_dir = args.rubric_dir.expanduser().resolve()
    if not direct_dir.exists():
        raise FileNotFoundError(f"Direct directory not found: {direct_dir}")
    if not rubric_dir.exists():
        raise FileNotFoundError(f"Rubric directory not found: {rubric_dir}")

    direct_scores = collect_direct_scores(direct_dir)
    rubric_totals = collect_rubric_totals(rubric_dir)
    points = _prepare_points(direct_scores, rubric_totals)

    title = args.title
    if not title:
        direct_summary = _discover_summary_file(direct_dir)
        rubric_summary = _discover_summary_file(rubric_dir)
        name_parts = []
        for summary in (direct_summary, rubric_summary):
            if summary:
                name_parts.append(derive_model_name(summary.name))
        title = " vs ".join(name_parts) if name_parts else "Rubric vs Evaluation"

    plot_scatter(points, title, args.export, not args.no_show)


if __name__ == "__main__":
    main()
