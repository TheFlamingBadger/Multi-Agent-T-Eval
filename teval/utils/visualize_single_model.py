"""Sankey visualizer for single-model orchestrator evaluation results."""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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

FORMAT_LABELS = {
    "json": "JSON Samples",
    "str": "String Samples",
}
FORMAT_ORDER = ["str", "json"]
PARSE_MISC_THRESHOLD = 20


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
    err_type = parse_attempt.get("error_type")
    err_message = parse_attempt.get("error_message")

    if (err_type or "").lower() == "jsondecodeerror":
        detail = _classify_jsondecode_error(err_message)
        if detail:
            return detail

    normalized = _normalize_error_label([err_type, err_message])
    if normalized:
        return normalized

    if err_type:
        normalized = _normalize_error_label([err_type])
        if normalized:
            return normalized

    if err_message:
        normalized = _normalize_error_label([err_message])
        if normalized:
            return normalized

    return "unknown_error"


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    stripped = text[3:]
    if stripped.startswith("json"):
        stripped = stripped[4:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _normalize_prediction(prediction: Any) -> str:
    if prediction is None:
        return ""
    if not isinstance(prediction, str):
        normalized = str(prediction)
    else:
        normalized = prediction
    normalized = normalized.strip()
    if normalized.startswith("```"):
        normalized = _strip_code_fence(normalized)
    return normalized.strip()


def _attempt_parse_prediction(prediction: Any, require_json: bool) -> Tuple[bool, Optional[str]]:
    normalized = _normalize_prediction(prediction)
    if not normalized:
        return False, "empty_prediction"
    if not require_json:
        return True, None
    try:
        json.loads(normalized)
    except Exception as exc:  # pragma: no cover
        label = _describe_parse_error(
            {"error_type": exc.__class__.__name__, "error_message": str(exc)}
        )
        return False, label
    return True, None


def _init_format_bucket() -> Dict[str, Any]:
    return {
        "total": 0,
        "primary_success": 0,
        "primary_failure": 0,
        "parse_counts": {},
        "parse_success": {},
        "parse_failure": {},
    }


def _merge_format_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["total"] += src.get("total", 0)
    dst["primary_success"] += src.get("primary_success", 0)
    dst["primary_failure"] += src.get("primary_failure", 0)
    for err_type, count in src.get("parse_counts", {}).items():
        dst["parse_counts"][err_type] = dst["parse_counts"].get(err_type, 0) + count
    for err_type, count in src.get("parse_success", {}).items():
        dst["parse_success"][err_type] = dst["parse_success"].get(err_type, 0) + count
    for err_type, count in src.get("parse_failure", {}).items():
        dst["parse_failure"][err_type] = dst["parse_failure"].get(err_type, 0) + count


def _response_format(entry: Dict[str, Any]) -> str:
    meta = entry.get("meta_data") or {}
    fmt = str(meta.get("response_format", "json")).lower()
    if fmt == "str":
        return "str"
    return "json"


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
        "primary_success": 0,
        "primary_failure": 0,
        "parse_counts": {},
        "parse_success": {},
        "parse_failure": {},
        "file_path": result_path,
        "label": label or os.path.basename(result_path),
        "formats": {},
    }

    for _, entry in _iter_samples(result_path):
        if not isinstance(entry, dict):
            continue
        trace = entry.get("orchestration_trace")
        if not isinstance(trace, dict):
            continue
        outcome = _outcome(entry)
        fmt = _response_format(entry)
        fmt_bucket = stats["formats"].setdefault(fmt, _init_format_bucket())
        stats["total"] += 1
        fmt_bucket["total"] += 1

        require_json = fmt != "str"
        parse_ok, err_label = _attempt_parse_prediction(entry.get("prediction"), require_json=require_json)
        if not parse_ok:
            label = err_label or "parse_error"
            stats["parse_counts"][label] = stats["parse_counts"].get(label, 0) + 1
            fmt_bucket["parse_counts"][label] = fmt_bucket["parse_counts"].get(label, 0) + 1
            if outcome == "success":
                stats["parse_success"][label] = stats["parse_success"].get(label, 0) + 1
                fmt_bucket["parse_success"][label] = fmt_bucket["parse_success"].get(label, 0) + 1
            else:
                stats["parse_failure"][label] = stats["parse_failure"].get(label, 0) + 1
                fmt_bucket["parse_failure"][label] = fmt_bucket["parse_failure"].get(label, 0) + 1
            continue

        if outcome == "success":
            stats["primary_success"] += 1
            fmt_bucket["primary_success"] += 1
        else:
            stats["primary_failure"] += 1
            fmt_bucket["primary_failure"] += 1

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
    global_stats: Dict[str, Any] = {
        "total": 0,
        "primary_success": 0,
        "primary_failure": 0,
        "parse_counts": {},
        "parse_success": {},
        "parse_failure": {},
        "formats": {},
    }

    for file_path in file_paths:
        stats = aggregate_counts(file_path, label=os.path.basename(file_path))
        if stats["total"] == 0:
            continue
        file_stats.append(stats)
        global_stats["total"] += stats["total"]
        global_stats["primary_success"] += stats["primary_success"]
        global_stats["primary_failure"] += stats["primary_failure"]
        for err_type, count in stats["parse_counts"].items():
            global_stats["parse_counts"][err_type] = (
                global_stats["parse_counts"].get(err_type, 0) + count
            )
            global_stats["parse_success"][err_type] = (
                global_stats["parse_success"].get(err_type, 0)
                + stats["parse_success"].get(err_type, 0)
            )
            global_stats["parse_failure"][err_type] = (
                global_stats["parse_failure"].get(err_type, 0)
                + stats["parse_failure"].get(err_type, 0)
            )
        for fmt, fmt_stats in stats.get("formats", {}).items():
            bucket = global_stats["formats"].setdefault(fmt, _init_format_bucket())
            _merge_format_stats(bucket, fmt_stats)

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

    format_nodes: Dict[str, int] = {}
    formats_data = global_stats.get("formats", {})
    for fmt in FORMAT_ORDER:
        bucket = formats_data.get(fmt)
        if not bucket:
            continue
        if bucket.get("total", 0) <= 0:
            continue
        node_idx = add_node(FORMAT_LABELS[fmt])
        format_nodes[fmt] = node_idx

    parse_counts = global_stats.get("parse_counts", {})
    major_labels = sorted(
        [label for label, count in parse_counts.items() if count >= PARSE_MISC_THRESHOLD]
    )
    major_label_set = set(major_labels)
    minor_labels = sorted(label for label in parse_counts if label not in major_label_set)

    parse_success_total = 0
    parse_success_success_total = 0
    parse_success_failure_total = 0
    no_parse_total = 0
    no_parse_success_total = 0
    no_parse_failure_total = 0
    for fmt, bucket in formats_data.items():
        parse_ok = bucket.get("primary_success", 0) + bucket.get("primary_failure", 0)
        if fmt == "str":
            no_parse_total += parse_ok
            no_parse_success_total += bucket.get("primary_success", 0)
            no_parse_failure_total += bucket.get("primary_failure", 0)
        else:
            parse_success_total += parse_ok
            parse_success_success_total += bucket.get("primary_success", 0)
            parse_success_failure_total += bucket.get("primary_failure", 0)

    parse_success_node_idx: Optional[int] = None
    if parse_success_total > 0:
        parse_success_node_idx = add_node("Parse Success")
    no_parse_node_idx: Optional[int] = None
    if no_parse_total > 0:
        no_parse_node_idx = add_node("No Parse")

    parse_error_nodes: Dict[str, int] = {}
    for label in major_labels:
        parse_error_nodes[label] = add_node(f"Parse Error ({label})")

    misc_total = sum(parse_counts.get(label, 0) for label in minor_labels)
    misc_node_idx: Optional[int] = None
    if misc_total > 0:
        misc_node_idx = add_node("Parse Errors (Misc. Aggr.)")

    overall_success_idx = add_node("Success")
    overall_failure_idx = add_node("Failure")

    for fmt, node_idx in format_nodes.items():
        bucket = formats_data[fmt]
        parse_success_count = bucket.get("primary_success", 0) + bucket.get("primary_failure", 0)
        if fmt == "str":
            if no_parse_node_idx is not None and parse_success_count > 0:
                add_flow(node_idx, no_parse_node_idx, parse_success_count)
        else:
            if parse_success_node_idx is not None and parse_success_count > 0:
                add_flow(node_idx, parse_success_node_idx, parse_success_count)

        for label in major_labels:
            count = bucket.get("parse_counts", {}).get(label, 0)
            if count > 0:
                add_flow(node_idx, parse_error_nodes[label], count)

        if misc_node_idx is not None:
            agg_count = sum(
                count
                for label, count in bucket.get("parse_counts", {}).items()
                if label not in major_label_set
            )
            if agg_count > 0:
                add_flow(node_idx, misc_node_idx, agg_count)

    if parse_success_node_idx is not None:
        add_flow(parse_success_node_idx, overall_success_idx, parse_success_success_total)
        add_flow(parse_success_node_idx, overall_failure_idx, parse_success_failure_total)

    if no_parse_node_idx is not None:
        add_flow(no_parse_node_idx, overall_success_idx, no_parse_success_total)
        add_flow(no_parse_node_idx, overall_failure_idx, no_parse_failure_total)

    for label in major_labels:
        node_idx = parse_error_nodes[label]
        succ = global_stats["parse_success"].get(label, 0)
        fail = global_stats["parse_failure"].get(label, 0)
        add_flow(node_idx, overall_success_idx, succ)
        remainder = parse_counts.get(label, 0) - (succ + fail)
        add_flow(node_idx, overall_failure_idx, fail + max(remainder, 0))

    if misc_node_idx is not None:
        misc_success = sum(global_stats["parse_success"].get(label, 0) for label in minor_labels)
        misc_failure = sum(global_stats["parse_failure"].get(label, 0) for label in minor_labels)
        misc_remainder = misc_total - (misc_success + misc_failure)
        add_flow(misc_node_idx, overall_success_idx, misc_success)
        add_flow(misc_node_idx, overall_failure_idx, misc_failure + max(misc_remainder, 0))

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

    for fmt in FORMAT_ORDER:
        bucket = global_stats.get("formats", {}).get(fmt)
        if not bucket or bucket.get("total", 0) == 0:
            continue
        label = FORMAT_LABELS[fmt]
        fmt_total = bucket["total"]
        print(f"{label}: {fmt_total} ({_pct(fmt_total, total)})")

    primary_success = global_stats.get("primary_success", 0)
    primary_failure = global_stats.get("primary_failure", 0)
    parse_total = sum(global_stats.get("parse_counts", {}).values())

    print(
        f"Model success: {primary_success} "
        f"({_pct(primary_success, total)})"
    )
    print(
        f"Model failure: {primary_failure} "
        f"({_pct(primary_failure, total)})"
    )
    print(
        f"Parse failures: {parse_total} "
        f"({_pct(parse_total, total)})"
    )
    for err_type, count in global_stats.get("parse_counts", {}).items():
        ps = global_stats["parse_success"].get(err_type, 0)
        pf = global_stats["parse_failure"].get(err_type, 0)
        print(
            f"  - {err_type}: {count} | success {ps}, failure {pf}"
        )
    for fmt in FORMAT_ORDER:
        bucket = global_stats.get("formats", {}).get(fmt)
        if not bucket or bucket.get("total", 0) == 0:
            continue
        label = FORMAT_LABELS[fmt]
        parse_total_fmt = sum(bucket.get("parse_counts", {}).values())
        print(
            f"{label} parse failures: {parse_total_fmt} "
            f"({_pct(parse_total_fmt, bucket['total'])})"
        )

    for stats in file_stats:
        print(f"\n[{stats['label']}] ({stats['total']} samples)")
        print(
            f"  Model success: {stats['primary_success']} "
            f"({_pct(stats['primary_success'], stats['total'])})"
        )
        print(
            f"  Model failure: {stats['primary_failure']} "
            f"({_pct(stats['primary_failure'], stats['total'])})"
        )
        parse_total_file = sum(stats["parse_counts"].values())
        print(
            f"  Parse failures: {parse_total_file} "
            f"({_pct(parse_total_file, stats['total'])})"
        )
        for err_type, count in stats["parse_counts"].items():
            ps = stats["parse_success"].get(err_type, 0)
            pf = stats["parse_failure"].get(err_type, 0)
            print(
                f"    - {err_type}: {count} | success {ps}, failure {pf}"
            )
        for fmt in FORMAT_ORDER:
            bucket = stats.get("formats", {}).get(fmt)
            if not bucket or bucket.get("total", 0) == 0:
                continue
            label = FORMAT_LABELS[fmt]
            parse_total_fmt = sum(bucket.get("parse_counts", {}).values())
            print(
                f"    {label}: {bucket['total']} samples, parse failures {parse_total_fmt}"
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
    title_text = args.title or _infer_directory_title(args.result_path)
    fig.update_layout(title_text=title_text, font=dict(size=12))

    auto_html_path = os.path.splitext(args.result_path)[0] + ".html"
    fig.write_html(auto_html_path)
    print(f"Saved interactive HTML to {auto_html_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
