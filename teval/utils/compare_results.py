"""Compare two model summary JSON files and report performance and token deltas."""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mmengine

from .convert_results import (
    TOKEN_KEYS,
    build_category_file_map,
    combine_category_stats,
    compute_file_stats,
    compute_scores,
    derive_model_name,
    resolve_category_files,
)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two evaluation summary files (model_-1.json) and show score/token deltas."
    )
    parser.add_argument(
        "model1",
        type=Path,
        help="Path to the first model summary JSON (model_-1.json).",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="Path to the second model summary JSON (model_-1.json).",
    )
    return parser.parse_args()


def _load_scores(result_path: Path) -> Tuple[List[str], List[Optional[float]]]:
    result = mmengine.load(result_path)
    final_scores, category_scores = compute_scores(result)
    categories = [name for name, _ in category_scores]
    scores = [score for _, score in category_scores]
    overall = final_scores[0] if final_scores else None
    return ["Overall"] + categories, [overall] + scores


def _collect_token_stats(result_path: Path) -> Tuple[List[str], List[Optional[float]]]:
    base_dir = result_path.parent
    model_name = derive_model_name(result_path.name)
    category_files = build_category_file_map(model_name)

    category_stats: Dict[str, Dict[str, object]] = {}
    file_cache: Dict[str, Optional[Dict[str, object]]] = {}
    totals: Dict[str, float] = {key: 0.0 for key in TOKEN_KEYS}
    total_count = 0

    for category, filenames in category_files.items():
        stats_to_combine: List[Dict[str, object]] = []
        for filename in filenames:
            for file_path in resolve_category_files(str(base_dir), filename):
                if file_path not in file_cache:
                    file_cache[file_path] = compute_file_stats(file_path)
                entry_stats = file_cache[file_path]
                if entry_stats:
                    stats_to_combine.append(entry_stats)
        if stats_to_combine:
            category_stats[category] = combine_category_stats(stats_to_combine)
        else:
            category_stats[category] = {
                "avg_tokens": {key: None for key in TOKEN_KEYS},
                "token_count": 0,
            }

        token_avg_obj = category_stats[category].get("avg_tokens", {})
        token_avg = token_avg_obj if isinstance(token_avg_obj, dict) else {}
        token_samples = category_stats[category]["token_count"]
        if isinstance(token_samples, (int, float)) and token_samples:
            for key in TOKEN_KEYS:
                val = token_avg.get(key)
                if isinstance(val, (int, float)):
                    totals[key] += float(val) * float(token_samples)
            total_count += float(token_samples)

    overall_avg = {
        key: (totals[key] / total_count) if total_count else None for key in TOKEN_KEYS
    }
    categories = ["Overall"] + list(category_stats.keys())
    avg_totals = [overall_avg] + [
        category_stats[c]["avg_tokens"] for c in category_stats
    ]

    # Return average total tokens per row
    total_tokens_series = []
    for entry in avg_totals:
        if isinstance(entry, dict):
            total_tokens_series.append(entry.get("total_tokens"))
        else:
            total_tokens_series.append(None)
    return categories, total_tokens_series


def _format_delta(
    reference: Optional[float], current: Optional[float], suffix: str = ""
) -> str:
    if reference is None or current is None:
        return "(N/A)"
    delta = (reference - current) * 100
    text = f"{delta:+.1f}%{suffix}"
    color = GREEN if delta > 0 else RED if delta < 0 else ""
    reset = RESET if color else ""
    return f"{color}{text}{reset}"


def _format_percent_delta(reference: Optional[float], current: Optional[float]) -> str:
    if reference in (None, 0) or current is None:
        return "(N/A)"
    pct = (reference - current) / reference * 100
    if pct is None:
        return "(N/A)"
    color = GREEN if pct > 0 else RED if pct < 0 else ""
    reset = RESET if color else ""
    return f"{color}({pct:+.1f}%){reset}"


def _color_for_score(score_pct: float) -> str:
    """Map a 0-100 score to a gradient (0 red, 50 orange, 75 yellow, 100 green)."""
    anchors = [
        (0.0, (255, 0, 0)),  # red
        (50.0, (255, 165, 0)),  # orange
        (75.0, (255, 255, 0)),  # yellow
        (100.0, (0, 128, 0)),  # green
    ]
    clamped = max(0.0, min(100.0, score_pct))
    for (a_val, a_rgb), (b_val, b_rgb) in zip(anchors, anchors[1:]):
        if clamped <= b_val:
            t = (clamped - a_val) / (b_val - a_val) if b_val > a_val else 0.0
            r = int(a_rgb[0] + (b_rgb[0] - a_rgb[0]) * t)
            g = int(a_rgb[1] + (b_rgb[1] - a_rgb[1]) * t)
            b = int(a_rgb[2] + (b_rgb[2] - a_rgb[2]) * t)
            return f"\033[38;2;{r};{g};{b}m"
    r, g, b = anchors[-1][1]
    return f"\033[38;2;{r};{g};{b}m"


def _format_score(value: Optional[float]) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    pct = value * 100
    color = _color_for_score(pct)
    return f"{color}{pct:5.1f}%{RESET}"


def _visible_len(text: str) -> int:
    return len(ANSI_PATTERN.sub("", text))


def _pad(text: str, width: int, align: str = "right") -> str:
    vis = _visible_len(text)
    if vis >= width:
        return text
    pad_len = width - vis
    if align == "left":
        return text + " " * pad_len
    return " " * pad_len + text


def _render_table(
    categories: List[str],
    scores1: List[Optional[float]],
    scores2: List[Optional[float]],
    tokens1: List[Optional[float]],
    tokens2: List[Optional[float]],
    name1: str,
    name2: str,
):
    cat_width = max(12, max(len(cat) for cat in categories))
    score_width = max(12, len(name1) + 2, len(name2) + 2)
    token_width = max(14, len(name1) + 8, len(name2) + 8)
    delta_score_w = 10
    delta_token_w = 12

    header_parts = [
        _pad("Category", cat_width, "left"),
        _pad(name2, score_width),
        _pad(name1, score_width),
        _pad("Δ Score", delta_score_w),
        _pad(f"{name2} Tokens", token_width),
        _pad(f"{name1} Tokens", token_width),
        _pad("Δ Tokens%", delta_token_w),
    ]
    header = " ".join(header_parts)

    print(header)
    print("-" * _visible_len(header))

    def _row(idx: int):
        cat = _pad(categories[idx], cat_width, "left")
        s1 = _pad(_format_score(scores2[idx]), score_width)  # swapped: model2 first
        s2 = _pad(_format_score(scores1[idx]), score_width)
        d_score = _pad(_format_delta(scores1[idx], scores2[idx]), delta_score_w)
        t1 = _pad(
            f"{tokens2[idx]:.2f}" if isinstance(tokens2[idx], (int, float)) else "N/A",
            token_width,
        )
        t2 = _pad(
            f"{tokens1[idx]:.2f}" if isinstance(tokens1[idx], (int, float)) else "N/A",
            token_width,
        )
        d_tokens = _pad(
            _format_percent_delta(tokens1[idx], tokens2[idx]), delta_token_w
        )
        print(" ".join([cat, s1, s2, d_score, t1, t2, d_tokens]))

    for idx in range(1, len(categories)):
        _row(idx)

    print("-" * _visible_len(header))
    _row(0)


def main():
    args = parse_args()
    m1 = args.model1.resolve()
    m2 = args.model2.resolve()

    categories1, scores1 = _load_scores(m1)
    categories2, scores2 = _load_scores(m2)
    if categories1 != categories2:
        raise ValueError(
            f"Category mismatch between models: {categories1} vs {categories2}"
        )

    token_categories1, tokens1 = _collect_token_stats(m1)
    token_categories2, tokens2 = _collect_token_stats(m2)
    if token_categories1 != token_categories2:
        raise ValueError(
            f"Token category mismatch between models: {token_categories1} vs {token_categories2}"
        )

    name1 = derive_model_name(m1.name)
    name2 = derive_model_name(m2.name)

    _render_table(categories1, scores1, scores2, tokens1, tokens2, name1, name2)


if __name__ == "__main__":
    main()
