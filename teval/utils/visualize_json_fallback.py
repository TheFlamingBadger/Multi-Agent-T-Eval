import argparse
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import mmengine

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - plotly optional
    raise ImportError(
        "plotly is required for visualize_json_fallback.py. "
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


def _shorten_dataset_label(label: str, existing: Set[str]) -> str:
    stem = os.path.splitext(os.path.basename(label))[0]
    parts = [part for part in stem.split("_") if part]
    if len(parts) >= 2:
        short = "_".join(parts[:2])
    else:
        short = stem

    candidate = short
    counter = 2
    while candidate in existing:
        candidate = f"{short}_{counter}"
        counter += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize JsonFallbackOrchestrator results as a Sankey diagram."
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
        default="Json Fallback Flow",
        help="Optional diagram title.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Save the visualization instead of displaying it interactively.",
    )
    parser.add_argument(
        "--download-format",
        type=str,
        choices=["png", "html"],
        default="png",
        help="Download format when --download is provided (default: png).",
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default=None,
        help="Optional explicit output path. When omitted, a default path based on "
        "the result file name is used.",
    )
    return parser.parse_args()


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
        if not isinstance(trace, dict) or trace.get("strategy") != "json_fallback":
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


def build_sankey_data(
    file_stats: List[Dict[str, Any]],
    global_stats: Dict[str, Any],
) -> Tuple[List[str], List[int], List[int], List[int]]:
    node_labels = ["All Samples"]
    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []

    dataset_indices: Dict[str, int] = {}
    used_dataset_labels: Set[str] = {"All Samples"}
    for stats in file_stats:
        node_idx = len(node_labels)
        dataset_indices[stats["label"]] = node_idx
        display_label = _shorten_dataset_label(stats["label"], used_dataset_labels)
        node_labels.append(display_label)
        used_dataset_labels.add(display_label)
        sources.append(0)
        targets.append(node_idx)
        values.append(stats["total"])

    primary_correct_idx = len(node_labels)
    node_labels.append("Primary Correct")
    primary_incorrect_idx = len(node_labels)
    node_labels.append("Primary Incorrect")

    parse_node_indices: Dict[str, int] = {}
    for stats in file_stats:
        dataset_idx = dataset_indices[stats["label"]]
        if stats["primary_correct"]:
            sources.append(dataset_idx)
            targets.append(primary_correct_idx)
            values.append(stats["primary_correct"])
        if stats["primary_incorrect"]:
            sources.append(dataset_idx)
            targets.append(primary_incorrect_idx)
            values.append(stats["primary_incorrect"])
        for err_type, count in stats["parse_counts"].items():
            if err_type not in parse_node_indices:
                node_idx = len(node_labels)
                node_labels.append(f"Parse Error: {err_type}")
                parse_node_indices[err_type] = node_idx
            sources.append(dataset_idx)
            targets.append(parse_node_indices[err_type])
            values.append(count)

    secondary_correct_idx: Optional[int] = None
    secondary_incorrect_idx: Optional[int] = None
    if parse_node_indices:
        secondary_correct_idx = len(node_labels)
        node_labels.append("Secondary Correct")
        secondary_incorrect_idx = len(node_labels)
        node_labels.append("Secondary Incorrect")

    for err_type, node_idx in parse_node_indices.items():
        sec_correct = global_stats["secondary_correct"].get(err_type, 0)
        sec_incorrect = global_stats["secondary_incorrect"].get(err_type, 0)
        if sec_correct and secondary_correct_idx is not None:
            sources.append(node_idx)
            targets.append(secondary_correct_idx)
            values.append(sec_correct)
        if sec_incorrect and secondary_incorrect_idx is not None:
            sources.append(node_idx)
            targets.append(secondary_incorrect_idx)
            values.append(sec_incorrect)

    return node_labels, sources, targets, values


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def print_stats(global_stats: Dict[str, Any], file_stats: List[Dict[str, Any]]) -> None:
    total = global_stats["total"]
    print(f"Total samples: {total}")
    if total == 0:
        print("No JsonFallback traces found in the provided files.")
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
            "No JsonFallback traces found in the specified files. "
            "Verify that the orchestrator was JsonFallback for these results."
        )

    print_stats(global_stats, file_stats)
    node_labels, sources, targets, values = build_sankey_data(file_stats, global_stats)

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
                f"{stem}_json_fallback.{args.download_format}",
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


if __name__ == "__main__":
    main()
