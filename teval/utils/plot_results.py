import argparse
import ast
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mmengine

if __package__ in (None, ''):
    # Allow running as `python teval/utils/plot_results.py` by adding repo root to sys.path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from teval.utils.convert_results import (
        build_category_file_map,
        compute_scores,
        derive_model_name,
        resolve_category_files,
    )
else:
    from .convert_results import (
        build_category_file_map,
        compute_scores,
        derive_model_name,
        resolve_category_files,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot evaluation scores for a result file.')
    parser.add_argument('--result_path', type=str, required=True, help='Path to the evaluation JSON result.')
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Optional title for the generated figure.',
    )
    parser.add_argument(
        '--download_only',
        action='store_true',
        help='Skip GUI display and write the figure as a PNG beside the result file.',
    )
    return parser.parse_args()


class ProgressReporter:
    """Lightweight textual progress feedback for long-running operations."""

    def __init__(self, total_steps: int):
        self.total_steps = max(total_steps, 1)
        self.completed = 0

    def update(self, message: str) -> None:
        self.completed += 1
        percent = (self.completed / self.total_steps) * 100
        print(f"[{self.completed}/{self.total_steps} | {percent:5.1f}%] {message}", flush=True)


def load_result(result_path: str) -> dict:
    if not os.path.exists(result_path):
        raise FileNotFoundError(f'Result file not found: {result_path}')
    return mmengine.load(result_path)


def prepare_category_data(result: dict) -> Tuple[List[str], List[Optional[float]], Optional[float]]:
    final_scores, category_scores = compute_scores(result)
    overall = final_scores[0] if final_scores else None
    categories = [name for name, _ in category_scores]
    scores = [score for _, score in category_scores]
    return categories, scores, overall


def plot_scores(
    categories: Iterable[str],
    scores: Iterable[Optional[float]],
    overall: Optional[float],
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_fig = True
    else:
        fig = ax.figure

    categories_list = list(categories)
    score_list = list(scores)

    plot_values = [score if score is not None else 0.0 for score in score_list]
    bar_colors = ['tab:blue' if score is not None else '0.75' for score in score_list]

    bars = ax.bar(categories_list, plot_values, color=bar_colors, edgecolor='black')

    for bar, score in zip(bars, score_list):
        height = bar.get_height()
        if score is None:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, 'N/A', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f'{score:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )

    if overall is not None:
        ax.axhline(overall, linestyle='--', linewidth=1.5, color='tab:orange', label=f'Average ({overall:.2f})')
        ax.legend()
    else:
        ax.axhline(0, linestyle='--', linewidth=1.0, color='tab:orange')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Average Score')
    ax.set_xlabel('Benchmark Category')
    ax.set_title(title or 'Evaluation Scores')
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.7)

    if created_fig:
        fig.tight_layout()
    return fig, ax


ERROR_CODE_PATTERN = re.compile(r"Error code:\s*(\d+)")
GENERIC_CODE_PATTERN = re.compile(r"'code':\s*('?)([^',}]+)\1")


def _normalize_error_label(parts: List[Optional[str]]) -> Optional[str]:
    cleaned = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text or text.lower() == 'none':
            continue
        cleaned.append(re.sub(r'\s+', '_', text))
    if not cleaned:
        return None
    return '_'.join(cleaned)


def _parse_error_payload(payload: object) -> Optional[str]:
    if isinstance(payload, dict):
        inner = payload.get('error', payload)
        if isinstance(inner, dict):
            label = _normalize_error_label([
                inner.get('code'),
                inner.get('type'),
            ])
            if label:
                return label
            return _normalize_error_label([
                inner.get('code'),
                inner.get('message'),
            ])
    return None


def _label_from_error_string(error_str: str) -> Optional[str]:
    code = None
    match = ERROR_CODE_PATTERN.search(error_str)
    if match:
        code = match.group(1)

    payload = None
    dash_idx = error_str.find('-')
    if dash_idx != -1:
        suffix = error_str[dash_idx + 1 :].strip()
        if suffix.startswith('{'):
            try:
                payload = ast.literal_eval(suffix)
            except (ValueError, SyntaxError):
                payload = None
    if payload is None:
        generic_match = GENERIC_CODE_PATTERN.search(error_str)
        if generic_match:
            candidate = generic_match.group(2)
            if candidate:
                code = candidate

    label = _normalize_error_label([code])

    payload_label = _parse_error_payload(payload)
    if payload_label:
        if label and not payload_label.startswith(label):
            return f"{label}_{payload_label}"
        return payload_label

    if label:
        return label

    trimmed = error_str.strip()
    if not trimmed:
        return None
    if len(trimmed) > 80:
        trimmed = trimmed[:77] + '...'
    return re.sub(r'\s+', '_', trimmed)


def _label_from_error_value(error_value: object) -> Optional[str]:
    if isinstance(error_value, dict):
        label = _normalize_error_label([
            error_value.get('code'),
            error_value.get('type'),
        ])
        if label:
            return label
        return _normalize_error_label([
            error_value.get('code'),
            error_value.get('message'),
        ])
    if isinstance(error_value, str):
        return _label_from_error_string(error_value)
    return None


def _extract_trace_error(trace: object) -> Optional[str]:
    stack = [trace]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            label = _label_from_error_value(current.get('error'))
            if label:
                return label
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append(item)
    return None


def extract_error_code(entry: Dict) -> Optional[str]:
    """Derive an error label from orchestration traces and evaluation metadata."""
    trace = entry.get('orchestration_trace')
    trace_label = _extract_trace_error(trace)
    if trace_label:
        return trace_label

    eval_result = entry.get('evaluation_result')
    if isinstance(eval_result, dict):
        label = _parse_error_payload(eval_result)
        if label:
            return label
        return 'non_numeric_result'

    meta = entry.get('meta_data')
    if isinstance(meta, dict):
        for key in ('error_code', 'error_type', 'error'):
            value = meta.get(key)
            if value:
                return _normalize_error_label([value])

    if eval_result is None:
        return 'missing_evaluation'

    if not isinstance(eval_result, (int, float)):
        return _normalize_error_label([f'unhandled_{type(eval_result).__name__}'])

    return None


def gather_category_errors_with_parse(
    result_path: str,
    categories: Iterable[str],
) -> Tuple[Dict[str, Counter], Dict[str, Counter], Dict[str, int]]:
    """Aggregate error counts and parse failure modes per benchmark category."""
    base_dir = os.path.dirname(result_path)
    model_name = derive_model_name(result_path)
    category_files = build_category_file_map(model_name)

    file_cache: Dict[str, Optional[dict]] = {}
    error_counts: Dict[str, Counter] = {category: Counter() for category in categories}
    parse_failure_counts: Dict[str, Counter] = {
        category: Counter() for category in categories
    }
    total_counts: Dict[str, int] = {category: 0 for category in categories}

    for category in categories:
        filenames = category_files.get(category, [])
        for filename in filenames:
            for file_path in resolve_category_files(base_dir, filename):
                if file_path not in file_cache:
                    if os.path.exists(file_path):
                        file_cache[file_path] = mmengine.load(file_path)
                    else:
                        file_cache[file_path] = None
                data = file_cache[file_path]
                if not data:
                    continue
                iterable = data.values() if isinstance(data, dict) else data
                entries = [entry for entry in iterable if isinstance(entry, dict)]
                sample_count = len(entries)
                total_counts[category] += sample_count
                for entry in entries:
                    failure_info = entry.get('parse_failure')
                    if isinstance(failure_info, dict):
                        mode = failure_info.get('mode') or 'parse_failure'
                        parse_failure_counts[category][mode] += 1
                    code = extract_error_code(entry)
                    if code:
                        error_counts[category][code] += 1

    return error_counts, parse_failure_counts, total_counts


def plot_error_counts(
    categories: Iterable[str],
    error_counts: Dict[str, Counter],
    parse_failure_counts: Dict[str, Counter],
    total_counts: Dict[str, int],
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_fig = True
    else:
        fig = ax.figure

    categories_list = list(categories)
    parsed_values: List[float] = []
    for category in categories_list:
        total = total_counts.get(category, 0)
        fail_total = sum(parse_failure_counts[category].values())
        parsed_values.append(max(total - fail_total, 0))

    ax.bar(
        categories_list,
        parsed_values,
        color='tab:green',
        edgecolor='black',
        label='Parsed',
    )

    bottom = parsed_values[:]

    all_parse_modes = sorted(
        {mode for counter in parse_failure_counts.values() for mode in counter}
    )
    for mode in all_parse_modes:
        values = [parse_failure_counts[category].get(mode, 0) for category in categories_list]
        ax.bar(
            categories_list,
            values,
            bottom=bottom,
            label=f'parse:{mode}',
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    all_error_codes = sorted(
        {code for counter in error_counts.values() for code in counter}
    )
    error_bottom = bottom[:]
    error_totals = {
        category: sum(counter.values()) for category, counter in error_counts.items()
    }

    for code in all_error_codes:
        values: List[float] = []
        for idx, category in enumerate(categories_list):
            leftover = max(total_counts.get(category, 0) - error_bottom[idx], 0.0)
            raw_total = error_totals.get(category, 0)
            raw_value = error_counts[category].get(code, 0)
            if raw_total > 0 and leftover > 0:
                scale = leftover / raw_total
                values.append(raw_value * scale)
            else:
                values.append(0.0)
        ax.bar(categories_list, values, bottom=error_bottom, label=code)
        error_bottom = [b + v for b, v in zip(error_bottom, values)]

    residual = []
    for idx, category in enumerate(categories_list):
        total = total_counts.get(category, 0)
        gap = max(total - error_bottom[idx], 0.0)
        residual.append(gap)
    if any(value > 1e-6 for value in residual):
        ax.bar(
            categories_list,
            residual,
            bottom=error_bottom,
            color='0.8',
            edgecolor='black',
            label='Unclassified',
        )
        error_bottom = [b + v for b, v in zip(error_bottom, residual)]

    if all_error_codes or all_parse_modes or any(parsed_values):
        ax.legend(title='Outcome', loc='upper right')

    ax.set_ylabel('Sample Count')
    ax.set_xlabel('Benchmark Category')
    ax.set_title(title or 'Errors by Category')
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.7)
    y_max = max(total_counts.get(category, 0) for category in categories_list) if categories_list else 0
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.05)

    if created_fig:
        fig.tight_layout()
    return fig, ax


def main() -> None:
    args = parse_args()

    reporter = ProgressReporter(total_steps=4)
    reporter.update('Loading result file')
    result = load_result(args.result_path)
    reporter.update('Computing category scores')
    categories, scores, overall = prepare_category_data(result)

    reporter.update('Aggregating error counts')
    error_counts, parse_failure_counts, total_counts = gather_category_errors_with_parse(
        args.result_path,
        categories,
    )

    reporter.update('Rendering figure')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    plot_scores(categories, scores, overall, title=args.title, ax=axes[0])
    plot_error_counts(
        categories,
        error_counts,
        parse_failure_counts,
        total_counts,
        title='Parsing & Error Breakdown',
        ax=axes[1],
    )

    if args.download_only:
        result_path = Path(args.result_path)
        output_path = result_path.with_suffix('.png')
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}", flush=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
