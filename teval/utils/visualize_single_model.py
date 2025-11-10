"""Sankey visualizer for single-model orchestrator evaluation results."""

import argparse
import os
from typing import Any, Dict, List, Tuple

import mmengine

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "plotly is required for visualize_single_model.py. "
        "Install it with `pip install plotly kaleido`."
    ) from exc

from teval.utils.convert_results import (
    derive_model_name,
    build_category_file_map,
    resolve_category_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize single-model orchestrator evaluation results as a Sankey diagram."
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to the summary JSON (model_-1.json) or a specific per-task result file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Single-Model Orchestrator Results",
        help="Optional diagram title.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="When set, save the figure instead of opening an interactive window.",
    )
    parser.add_argument(
        "--download-format",
        choices=["png", "html"],
        default="png",
        help="Output format when --download is set (default: png).",
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default=None,
        help="Optional explicit path for the exported figure.",
    )
    return parser.parse_args()


def _iter_samples(result_path: str):
    data = mmengine.load(result_path)
    if isinstance(data, dict):
        return data.items()
    return enumerate(data)


def _outcome(entry: Dict[str, Any]) -> str:
    score = entry.get("evaluation_result")
    if isinstance(score, (int, float)):
        return "success" if score >= 0.5 else "failure"
    return "unknown"


def aggregate_counts(result_path: str, label: str | None = None) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total": 0,
        "strategies": {},
        "file_path": result_path,
        "label": label or os.path.basename(result_path),
    }

    for _, entry in _iter_samples(result_path):
        if not isinstance(entry, dict):
            continue
        trace = entry.get("orchestration_trace")
        if not isinstance(trace, dict):
            continue
        strategy = trace.get("strategy") or "unknown_strategy"
        outcome = _outcome(entry)

        bucket = stats["strategies"].setdefault(
            strategy,
            {"total": 0, "success": 0, "failure": 0, "unknown": 0},
        )
        bucket["total"] += 1
        bucket[outcome] += 1
        stats["total"] += 1

    return stats


def discover_result_files(summary_path: str) -> List[str]:
    base_dir = os.path.dirname(summary_path)
    model_name = derive_model_name(summary_path)
    category_map = build_category_file_map(model_name)

    discovered: List[str] = []
    seen = set()
    for filenames in category_map.values():
        for filename in filenames:
            for file_path in resolve_category_files(base_dir, filename):
                if file_path not in seen and os.path.exists(file_path):
                    seen.add(file_path)
                    discovered.append(file_path)
    return discovered


def aggregate_file_list(file_paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    file_stats: List[Dict[str, Any]] = []
    global_stats: Dict[str, Any] = {"total": 0, "strategies": {}}

    for file_path in file_paths:
        stats = aggregate_counts(file_path, label=os.path.basename(file_path))
        if stats["total"] == 0:
            continue
        file_stats.append(stats)
        global_stats["total"] += stats["total"]
        for strategy, bucket in stats["strategies"].items():
            agg = global_stats["strategies"].setdefault(
                strategy, {"total": 0, "success": 0, "failure": 0, "unknown": 0}
            )
            agg["total"] += bucket["total"]
            agg["success"] += bucket["success"]
            agg["failure"] += bucket["failure"]
            agg["unknown"] += bucket["unknown"]

    return file_stats, global_stats


def build_sankey_data(global_stats: Dict[str, Any]) -> Tuple[List[str], List[int], List[int], List[int]]:
    node_labels: List[str] = []
    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []

    def add_node(label: str) -> int:
        node_labels.append(label)
        return len(node_labels) - 1

    def add_flow(source: int, target: int, value: int) -> None:
        if value <= 0:
            return
        sources.append(source)
        targets.append(target)
        values.append(value)

    all_idx = add_node("All Samples")
    outcome_success_idx = add_node("Evaluation Success")
    outcome_failure_idx = add_node("Evaluation Failure")
    outcome_unknown_idx = add_node("Evaluation Unknown")

    for strategy, bucket in global_stats.get("strategies", {}).items():
        strat_idx = add_node(f"Strategy: {strategy}")
        add_flow(all_idx, strat_idx, bucket["total"])
        add_flow(strat_idx, outcome_success_idx, bucket["success"])
        add_flow(strat_idx, outcome_failure_idx, bucket["failure"])
        add_flow(strat_idx, outcome_unknown_idx, bucket["unknown"])

    return node_labels, sources, targets, values


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def print_stats(global_stats: Dict[str, Any], file_stats: List[Dict[str, Any]]) -> None:
    total = global_stats["total"]
    print(f"Total samples: {total}")
    if total == 0:
        print("No compatible traces found in the provided files.")
        return

    for strategy, bucket in global_stats["strategies"].items():
        print(
            f"Strategy '{strategy}': {bucket['total']} samples | "
            f"success {bucket['success']} ({_pct(bucket['success'], bucket['total'])}), "
            f"failure {bucket['failure']} ({_pct(bucket['failure'], bucket['total'])}), "
            f"unknown {bucket['unknown']} ({_pct(bucket['unknown'], bucket['total'])})"
        )

    for stats in file_stats:
        print(f"\n[{stats['label']}] ({stats['total']} samples)")
        for strategy, bucket in stats["strategies"].items():
            print(
                f"  - {strategy}: success {bucket['success']} ({_pct(bucket['success'], bucket['total'])}), "
                f"failure {bucket['failure']} ({_pct(bucket['failure'], bucket['total'])}), "
                f"unknown {bucket['unknown']} ({_pct(bucket['unknown'], bucket['total'])})"
            )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.result_path):
        raise FileNotFoundError(f"Result file not found: {args.result_path}")

    result_path = args.result_path
    basename = os.path.basename(result_path)
    if basename.endswith("_-1.json"):
        discovered_files = discover_result_files(result_path)
        if not discovered_files:
            raise ValueError(
                "Unable to locate per-task result files based on the summary file. "
                "Ensure the per-task JSON files are present in the same directory."
            )
        file_paths = discovered_files
    else:
        file_paths = [result_path]

    file_stats, global_stats = aggregate_file_list(file_paths)
    if not file_stats or global_stats["total"] == 0:
        raise ValueError(
            "No compatible orchestration traces found. "
            "Verify that the results were produced by single-model orchestrators."
        )

    print_stats(global_stats, file_stats)
    node_labels, sources, targets, values = build_sankey_data(global_stats)

    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, pad=20, thickness=20),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    fig.update_layout(title_text=args.title, font=dict(size=12))

    if args.download:
        if args.download_path:
            output_path = args.download_path
        else:
            stem = os.path.splitext(os.path.basename(result_path))[0]
            output_path = os.path.join(
                os.path.dirname(result_path),
                f"{stem}_single_model.{args.download_format}",
            )
        if args.download_format == "html":
            if not output_path.lower().endswith(".html"):
                output_path = os.path.splitext(output_path)[0] + ".html"
            fig.write_html(output_path)
        else:
            if not output_path.lower().endswith(".png"):
                output_path = os.path.splitext(output_path)[0] + ".png"
            fig.write_image(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        fig.show()


if __name__ == "__main__":  # pragma: no cover
    main()
