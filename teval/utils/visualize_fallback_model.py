import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import mmengine

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - plotly optional
    raise ImportError(
        "plotly is required for visualize_fallback_model.py. "
        "Install it with `pip install plotly kaleido`."
    ) from exc


from teval.utils.convert_results import (
    derive_model_name,
    build_category_file_map,
    resolve_category_files,
)


JSON_ERROR_HINTS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"property name enclosed in double quotes", re.IGNORECASE), "missing_key"),
    (re.compile(r"expecting '\]'", re.IGNORECASE), "missing_bracket"),
    (re.compile(r"expecting '\}'", re.IGNORECASE), "missing_brace"),
    (re.compile(r"expecting ',' delimiter", re.IGNORECASE), "missing_delimiter"),
    (re.compile(r"expecting ':' delimiter", re.IGNORECASE), "missing_delimiter"),
    (re.compile(r"extra data", re.IGNORECASE), "extra_data"),
    (re.compile(r"unterminated string", re.IGNORECASE), "unterminated_string"),
    (re.compile(r"invalid control character", re.IGNORECASE), "invalid_control_char"),
    (re.compile(r"invalid\s+\\escape", re.IGNORECASE), "invalid_escape"),
    (re.compile(r"invalid escape", re.IGNORECASE), "invalid_escape"),
    (re.compile(r"expecting value", re.IGNORECASE), "missing_value"),
]


def _normalize_error_label(parts: List[Optional[str]]) -> Optional[str]:
    cleaned: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text or text.lower() == "none":
            continue
        cleaned.append(re.sub(r"\s+", "_", text))
    if not cleaned:
        return None
    return "_".join(cleaned)


def _classify_jsondecode_error(message: Optional[str]) -> Optional[str]:
    if not message:
        return None
    lowered = message.lower()
    for pattern, label in JSON_ERROR_HINTS:
        if pattern.search(lowered):
            return label
    return None


def _describe_parse_error(parse_attempt: Dict[str, Any]) -> str:
    if not parse_attempt:
        return "unknown_error"

    issue = parse_attempt.get("issue")
    if issue:
        normalized_issue = _normalize_error_label([issue])
        if normalized_issue:
            return normalized_issue

    err_type = parse_attempt.get("error_type")
    err_message = parse_attempt.get("error_message") or parse_attempt.get("detail")

    if (err_type or "").lower() == "jsondecodeerror":
        detail = _classify_jsondecode_error(err_message)
        if detail:
            return detail

    normalized = _normalize_error_label([err_type, err_message])
    if normalized:
        return normalized

    normalized = _normalize_error_label([err_type])
    if normalized:
        return normalized

    if err_message:
        normalized = _normalize_error_label([err_message])
        if normalized:
            return normalized

    return "unknown_error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize FallbackModelOrchestrator results as a Sankey diagram."
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to the model summary file (e.g., work_dirs/.../model_-1.json) "
        "or a single per-task result json.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional diagram title. Defaults to the parent directory name of the result path.",
    )
    return parser.parse_args()


def _infer_directory_title(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(directory)
    if base:
        return base
    fallback = os.path.basename(os.path.abspath(path))
    return fallback or directory or ""


def _iter_samples(result_path: str):
    data = mmengine.load(result_path)
    if isinstance(data, dict):
        return data.items()
    return enumerate(data)


def _is_success(entry: Dict) -> bool:
    score = entry.get("evaluation_result")
    if isinstance(score, (int, float)):
        return score >= 0.5
    return False


def aggregate_counts(result_path: str, label: Optional[str] = None) -> Dict[str, Any]:
    total = 0
    primary_correct = 0
    primary_incorrect = 0

    parse_counts: Dict[str, int] = {}
    secondary_correct: Dict[str, int] = {}
    secondary_incorrect: Dict[str, int] = {}

    for _, entry in _iter_samples(result_path):
        if not isinstance(entry, dict):
            continue
        trace = entry.get("orchestration_trace")
        if not isinstance(trace, dict) or trace.get("strategy") != "fallback_model":
            continue

        total += 1
        fallback = trace.get("fallback_triggered", False)
        success = _is_success(entry)

        if not fallback:
            if success:
                primary_correct += 1
            else:
                primary_incorrect += 1
            continue

        parse_attempt = (
            trace.get("steps", [{}])[0].get("parse_attempt", {}) if trace.get("steps") else {}
        )
        err_type = _describe_parse_error(parse_attempt)
        parse_counts[err_type] = parse_counts.get(err_type, 0) + 1

        if success:
            secondary_correct[err_type] = secondary_correct.get(err_type, 0) + 1
        else:
            secondary_incorrect[err_type] = secondary_incorrect.get(err_type, 0) + 1

    return {
        "total": total,
        "primary_correct": primary_correct,
        "primary_incorrect": primary_incorrect,
        "parse_counts": parse_counts,
        "secondary_correct": secondary_correct,
        "secondary_incorrect": secondary_incorrect,
        "file_path": result_path,
        "label": label or os.path.basename(result_path),
    }


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
    global_stats = {
        "total": 0,
        "primary_correct": 0,
        "primary_incorrect": 0,
        "parse_counts": {},
        "secondary_correct": {},
        "secondary_incorrect": {},
    }

    for file_path in file_paths:
        stats = aggregate_counts(file_path, label=os.path.basename(file_path))
        if stats["total"] == 0:
            continue
        file_stats.append(stats)
        global_stats["total"] += stats["total"]
        global_stats["primary_correct"] += stats["primary_correct"]
        global_stats["primary_incorrect"] += stats["primary_incorrect"]
        for err_type, count in stats["parse_counts"].items():
            global_stats["parse_counts"][err_type] = (
                global_stats["parse_counts"].get(err_type, 0) + count
            )
            global_stats["secondary_correct"][err_type] = (
                global_stats["secondary_correct"].get(err_type, 0)
                + stats["secondary_correct"].get(err_type, 0)
            )
            global_stats["secondary_incorrect"][err_type] = (
                global_stats["secondary_incorrect"].get(err_type, 0)
                + stats["secondary_incorrect"].get(err_type, 0)
            )

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
    primary_success_idx = add_node("Primary Success")
    primary_failure_idx = add_node("Primary Failure")
    parse_counts = global_stats.get("parse_counts", {})
    parse_nodes: Dict[str, int] = {}
    for err_type, total in parse_counts.items():
        label = f"Primary Parse Error ({err_type})"
        parse_nodes[err_type] = add_node(label)
    overall_success_idx = add_node("Overall Success")
    overall_failure_idx = add_node("Overall Failure")

    primary_correct = global_stats.get("primary_correct", 0)
    primary_incorrect = global_stats.get("primary_incorrect", 0)

    add_flow(all_idx, primary_success_idx, primary_correct)
    add_flow(all_idx, primary_failure_idx, primary_incorrect)
    for err_type, total in parse_counts.items():
        add_flow(all_idx, parse_nodes[err_type], total)

    add_flow(primary_success_idx, overall_success_idx, primary_correct)
    add_flow(primary_failure_idx, overall_failure_idx, primary_incorrect)

    secondary_success_idx: Optional[int] = None
    secondary_failure_idx: Optional[int] = None
    secondary_unknown_idx: Optional[int] = None
    secondary_success_total = 0
    secondary_failure_total = 0
    secondary_unknown_total = 0

    for err_type, total in parse_counts.items():
        sec_success = global_stats.get("secondary_correct", {}).get(err_type, 0)
        sec_failure = global_stats.get("secondary_incorrect", {}).get(err_type, 0)

        parse_node_idx = parse_nodes[err_type]
        if sec_success:
            if secondary_success_idx is None:
                secondary_success_idx = add_node("Secondary Success")
            add_flow(parse_node_idx, secondary_success_idx, sec_success)
            secondary_success_total += sec_success

        if sec_failure:
            if secondary_failure_idx is None:
                secondary_failure_idx = add_node("Secondary Failure")
            add_flow(parse_node_idx, secondary_failure_idx, sec_failure)
            secondary_failure_total += sec_failure

        unaccounted = total - (sec_success + sec_failure)
        if unaccounted > 0:
            if secondary_unknown_idx is None:
                secondary_unknown_idx = add_node("Secondary Unknown")
            add_flow(parse_node_idx, secondary_unknown_idx, unaccounted)
            secondary_unknown_total += unaccounted

    if secondary_success_idx is not None and secondary_success_total > 0:
        add_flow(secondary_success_idx, overall_success_idx, secondary_success_total)
    if secondary_failure_idx is not None and secondary_failure_total > 0:
        add_flow(secondary_failure_idx, overall_failure_idx, secondary_failure_total)
    if secondary_unknown_idx is not None and secondary_unknown_total > 0:
        add_flow(secondary_unknown_idx, overall_failure_idx, secondary_unknown_total)

    return node_labels, sources, targets, values


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def print_stats(global_stats: Dict[str, Any], file_stats: List[Dict[str, Any]]) -> None:
    total = global_stats["total"]
    print(f"Total samples: {total}")
    if total == 0:
        print("No fallback_model traces found in the provided files.")
        return

    print(
        f"Primary correct: {global_stats['primary_correct']} "
        f"({_pct(global_stats['primary_correct'], total)})"
    )
    print(
        f"Primary incorrect: {global_stats['primary_incorrect']} "
        f"({_pct(global_stats['primary_incorrect'], total)})"
    )

    fallback_total = sum(global_stats["parse_counts"].values())
    print(f"Primary parse failures: {fallback_total} ({_pct(fallback_total, total)})")
    for err_type, count in global_stats["parse_counts"].items():
        sec_success = global_stats["secondary_correct"].get(err_type, 0)
        sec_fail = global_stats["secondary_incorrect"].get(err_type, 0)
        print(
            f"  - {err_type}: {count} | secondary correct: {sec_success}, secondary incorrect: {sec_fail}"
        )

    for stats in file_stats:
        if stats["total"] == 0:
            continue
        print(f"\n[{stats['label']}] ({stats['total']} samples)")
        print(
            f"  Primary correct: {stats['primary_correct']} "
            f"({_pct(stats['primary_correct'], stats['total'])})"
        )
        print(
            f"  Primary incorrect: {stats['primary_incorrect']} "
            f"({_pct(stats['primary_incorrect'], stats['total'])})"
        )
        fallback_total = sum(stats["parse_counts"].values())
        print(
            f"  Parse failures: {fallback_total} "
            f"({_pct(fallback_total, stats['total'])})"
        )
        for err_type, count in stats["parse_counts"].items():
            sec_success = stats["secondary_correct"].get(err_type, 0)
            sec_fail = stats["secondary_incorrect"].get(err_type, 0)
            print(
                f"    - {err_type}: {count} "
                f"(secondary correct: {sec_success}, secondary incorrect: {sec_fail})"
            )


def main():
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
            "No fallback_model traces found in the specified files. "
            "Verify that the orchestrator was FallbackModel for these results."
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

    auto_html_path = os.path.splitext(args.result_path)[0] + ".html"
    fig.write_html(auto_html_path)
    print(f"Saved interactive HTML to {auto_html_path}")


if __name__ == "__main__":
    main()
