import argparse
import os
from typing import Dict, List, Tuple

import mmengine

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - plotly optional
    raise ImportError(
        "plotly is required for visualize_json_fallback.py. "
        "Install it with `pip install plotly kaleido`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize JsonFallbackOrchestrator results as a Sankey diagram."
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to a per-task result json (e.g., work_dirs/.../instruct_xxx.json).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Json Fallback Flow",
        help="Optional diagram title.",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        help="Optional output path. If ends with .html saves interactive HTML, "
        "otherwise attempts static image export (requires kaleido).",
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


def aggregate_counts(result_path: str):
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
        err_type = parse_attempt.get("error_type") or parse_attempt.get("issue") or "unknown_error"
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
    }


def build_sankey_data(stats: Dict[str, any]) -> Tuple[List[str], List[int], List[int], List[int]]:
    node_labels = ["All Samples", "Primary Correct", "Primary Incorrect"]
    node_index = {
        "source": 0,
        "primary_correct": 1,
        "primary_incorrect": 2,
    }

    sources: List[int] = []
    targets: List[int] = []
    values: List[int] = []

    # Primary flows
    sources.append(node_index["source"])
    targets.append(node_index["primary_correct"])
    values.append(stats["primary_correct"])

    sources.append(node_index["source"])
    targets.append(node_index["primary_incorrect"])
    values.append(stats["primary_incorrect"])

    parse_nodes: Dict[str, int] = {}
    secondary_nodes_correct: Dict[str, int] = {}
    secondary_nodes_incorrect: Dict[str, int] = {}

    for err_type, count in stats["parse_counts"].items():
        node_idx = len(node_labels)
        node_labels.append(f"Parse Error: {err_type}")
        parse_nodes[err_type] = node_idx

        sources.append(node_index["source"])
        targets.append(node_idx)
        values.append(count)

        correct_idx = len(node_labels)
        node_labels.append(f"Secondary Correct ({err_type})")
        secondary_nodes_correct[err_type] = correct_idx

        incorrect_idx = len(node_labels)
        node_labels.append(f"Secondary Incorrect ({err_type})")
        secondary_nodes_incorrect[err_type] = incorrect_idx

        sec_correct = stats["secondary_correct"].get(err_type, 0)
        sec_incorrect = stats["secondary_incorrect"].get(err_type, 0)

        if sec_correct:
            sources.append(node_idx)
            targets.append(correct_idx)
            values.append(sec_correct)
        if sec_incorrect:
            sources.append(node_idx)
            targets.append(incorrect_idx)
            values.append(sec_incorrect)

    return node_labels, sources, targets, values


def print_stats(stats: Dict[str, any]) -> None:
    total = stats["total"]
    print(f"Total samples: {total}")
    if total == 0:
        return
    def pct(value):
        return f"{(value / total) * 100:.1f}%"

    print(f"Primary correct: {stats['primary_correct']} ({pct(stats['primary_correct'])})")
    print(f"Primary incorrect: {stats['primary_incorrect']} ({pct(stats['primary_incorrect'])})")

    fallback_total = sum(stats["parse_counts"].values())
    print(f"Primary parse failures: {fallback_total} ({pct(fallback_total)})")
    for err_type, count in stats["parse_counts"].items():
        sec_success = stats["secondary_correct"].get(err_type, 0)
        sec_fail = stats["secondary_incorrect"].get(err_type, 0)
        print(
            f"  - {err_type}: {count} | secondary correct: {sec_success}, secondary incorrect: {sec_fail}"
        )


def main():
    args = parse_args()
    if not os.path.exists(args.result_path):
        raise FileNotFoundError(f"Result file not found: {args.result_path}")

    stats = aggregate_counts(args.result_path)
    print_stats(stats)
    node_labels, sources, targets, values = build_sankey_data(stats)

    fig = go.Figure(
        go.Sankey(
            node=dict(label=node_labels, pad=20, thickness=20),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    fig.update_layout(title_text=args.title, font=dict(size=12))

    if args.download:
        output_path = args.download
        if output_path.lower().endswith(".html"):
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
