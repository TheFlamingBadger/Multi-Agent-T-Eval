"""Visualize AgenticOrchestrator self-check dynamics with a Sankey diagram."""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import mmengine

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "plotly is required for visualize_agentic.py. "
        "Install it with `pip install plotly kaleido`."
    ) from exc

from teval.utils.convert_results import (
    derive_model_name,
    build_category_file_map,
    resolve_category_files,
)


FORMAT_LABELS = {
    "json": "JSON Samples",
    "str": "String Samples",
}
FORMAT_ORDER = ["str", "json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize AgenticOrchestrator runs as a Sankey diagram showing format, "
            "number of self-check calls, and success/failure outcomes."
        )
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help=(
            "Path to model_-1.json (the summary file) or a single per-task result JSON "
            "generated with the agentic orchestrator."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Defaults to the containing directory.",
    )
    return parser.parse_args()


def _infer_directory_title(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(directory)
    if base:
        return base
    fallback = os.path.basename(os.path.abspath(path))
    return fallback or directory or ""


def _response_format(entry: Dict[str, Any]) -> str:
    meta = entry.get("meta_data") or {}
    fmt = str(meta.get("response_format", "json")).lower()
    return "str" if fmt == "str" else "json"


def _iter_samples(result_path: str):
    data = mmengine.load(result_path)
    if isinstance(data, dict):
        return data.items()
    return enumerate(data)


def _is_success(entry: Dict[str, Any]) -> bool:
    score = entry.get("evaluation_result")
    if isinstance(score, (int, float)):
        return score >= 0.5
    return False


def _count_model_calls(trace: Dict[str, Any]) -> int:
    steps = trace.get("steps")
    if isinstance(steps, list) and steps:
        return len(steps)
    return int(trace.get("total_steps", 1)) or 1


def _init_format_bucket() -> Dict[str, Any]:
    return {
        "total": 0,
        "success": 0,
        "failure": 0,
        "call_counts": {},
    }


def aggregate_agentic_counts(result_path: str, label: Optional[str] = None) -> Dict[str, Any]:
    stats = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "formats": {},
        "call_outcomes": {},
        "file_path": result_path,
        "label": label or os.path.basename(result_path),
    }

    for _, entry in _iter_samples(result_path):
        if not isinstance(entry, dict):
            continue
        trace = entry.get("orchestration_trace")
        if not isinstance(trace, dict) or trace.get("strategy") != "agentic":
            continue

        response_format = _response_format(entry)
        format_bucket = stats["formats"].setdefault(response_format, _init_format_bucket())

        success = _is_success(entry)
        call_count = _count_model_calls(trace)

        stats["total"] += 1
        format_bucket["total"] += 1

        call_bucket = stats["call_outcomes"].setdefault(
            call_count, {"total": 0, "success": 0, "failure": 0}
        )
        call_bucket["total"] += 1

        format_bucket["call_counts"][call_count] = (
            format_bucket["call_counts"].get(call_count, 0) + 1
        )

        if success:
            stats["success"] += 1
            format_bucket["success"] += 1
            call_bucket["success"] += 1
        else:
            stats["failure"] += 1
            format_bucket["failure"] += 1
            call_bucket["failure"] += 1

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
    per_file: List[Dict[str, Any]] = []
    global_stats = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "formats": {},
        "call_outcomes": {},
    }

    for file_path in file_paths:
        stats = aggregate_agentic_counts(file_path, label=os.path.basename(file_path))
        if stats["total"] == 0:
            continue
        per_file.append(stats)
        global_stats["total"] += stats["total"]
        global_stats["success"] += stats["success"]
        global_stats["failure"] += stats["failure"]

        for fmt, fmt_stats in stats["formats"].items():
            bucket = global_stats["formats"].setdefault(fmt, _init_format_bucket())
            bucket["total"] += fmt_stats["total"]
            bucket["success"] += fmt_stats["success"]
            bucket["failure"] += fmt_stats["failure"]
            for call_count, count in fmt_stats["call_counts"].items():
                bucket["call_counts"][call_count] = bucket["call_counts"].get(call_count, 0) + count

        for call_count, call_stats in stats["call_outcomes"].items():
            call_bucket = global_stats["call_outcomes"].setdefault(
                call_count, {"total": 0, "success": 0, "failure": 0}
            )
            call_bucket["total"] += call_stats["total"]
            call_bucket["success"] += call_stats["success"]
            call_bucket["failure"] += call_stats["failure"]

    return per_file, global_stats


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def print_stats(global_stats: Dict[str, Any], file_stats: List[Dict[str, Any]]) -> None:
    total = global_stats.get("total", 0)
    print(f"Agentic samples: {total}")
    if total == 0:
        print("No agentic orchestrator traces found.")
        return

    success = global_stats.get("success", 0)
    failure = global_stats.get("failure", 0)
    print(f"  Success: {success} ({_pct(success, total)})")
    print(f"  Failure: {failure} ({_pct(failure, total)})")

    for fmt in FORMAT_ORDER:
        bucket = global_stats.get("formats", {}).get(fmt)
        if not bucket or bucket["total"] == 0:
            continue
        label = FORMAT_LABELS[fmt]
        fmt_total = bucket["total"]
        fmt_success = bucket["success"]
        fmt_failure = bucket["failure"]
        print(
            f"  {label}: {fmt_total} ({_pct(fmt_total, total)}) | "
            f"success {fmt_success} ({_pct(fmt_success, fmt_total)}), "
            f"failure {fmt_failure} ({_pct(fmt_failure, fmt_total)})"
        )

    call_outcomes = global_stats.get("call_outcomes", {})
    if call_outcomes:
        print("Call count distribution:")
        for call_count in sorted(call_outcomes):
            bucket = call_outcomes[call_count]
            print(
                f"  {call_count} call(s): {bucket['total']} "
                f"({bucket['success']} success, {bucket['failure']} failure)"
            )

    for stats in file_stats:
        file_total = stats["total"]
        if file_total == 0:
            continue
        print(f"\n[{stats['label']}] {file_total} samples")
        print(
            f"  Success: {stats['success']} ({_pct(stats['success'], file_total)}) | "
            f"Failure: {stats['failure']} ({_pct(stats['failure'], file_total)})"
        )
        for fmt in FORMAT_ORDER:
            bucket = stats.get("formats", {}).get(fmt)
            if not bucket or bucket["total"] == 0:
                continue
            label = FORMAT_LABELS[fmt]
            print(
                f"    {label}: {bucket['total']} samples "
                f"({bucket['success']} success, {bucket['failure']} failure)"
            )


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

    format_nodes: Dict[str, int] = {}
    formats_data = global_stats.get("formats", {})
    for fmt in FORMAT_ORDER:
        bucket = formats_data.get(fmt)
        if not bucket or bucket["total"] == 0:
            continue
        format_nodes[fmt] = add_node(FORMAT_LABELS[fmt])

    call_nodes: Dict[int, int] = {}
    for call_count in sorted(global_stats.get("call_outcomes", {})):
        call_nodes[call_count] = add_node(f"{call_count} call(s)")

    success_idx = add_node("Success")
    failure_idx = add_node("Failure")

    # Flows: format -> call count
    for fmt, fmt_node in format_nodes.items():
        bucket = formats_data.get(fmt) or {}
        for call_count, count in sorted(bucket.get("call_counts", {}).items()):
            call_node = call_nodes.get(call_count)
            if call_node is None:
                continue
            add_flow(fmt_node, call_node, count)

    # Flows: call count -> outcome
    for call_count, call_node in call_nodes.items():
        bucket = global_stats["call_outcomes"].get(call_count, {})
        add_flow(call_node, success_idx, bucket.get("success", 0))
        add_flow(call_node, failure_idx, bucket.get("failure", 0))

    return node_labels, sources, targets, values


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.result_path):
        raise FileNotFoundError(f"Result file not found: {args.result_path}")

    basename = os.path.basename(args.result_path)
    if basename.endswith("_-1.json"):
        discovered_files = discover_result_files(args.result_path)
        if not discovered_files:
            raise ValueError(
                "Unable to locate per-task result files in the same directory as the summary file."
            )
        file_paths = discovered_files
    else:
        file_paths = [args.result_path]

    file_stats, global_stats = aggregate_file_list(file_paths)
    if not file_stats or global_stats["total"] == 0:
        raise ValueError(
            "No agentic orchestrator traces found. Ensure the input file was produced with AgenticOrchestrator."
        )

    print_stats(global_stats, file_stats)
    node_labels, sources, targets, values = build_sankey_data(global_stats)

    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, pad=20, thickness=20),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    title_text = args.title or _infer_directory_title(args.result_path)
    fig.update_layout(title_text=title_text, font=dict(size=12))

    auto_html_path = os.path.splitext(args.result_path)[0] + "_agentic.html"
    fig.write_html(auto_html_path)
    print(f"Saved interactive HTML to {auto_html_path}")


if __name__ == "__main__":
    main()
