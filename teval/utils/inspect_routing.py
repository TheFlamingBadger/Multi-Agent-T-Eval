"""Inspect routing outcomes and correctness across SLM/LLM decisions."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mmengine

Route = str
Label = str

ROUTE_SLM = "SLM"
ROUTE_LLM = "LLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect routing logs against a base model summary to see how many "
            "examples were routed to each model and answered correctly."
        )
    )
    parser.add_argument(
        "routing_logs",
        type=Path,
        help="Path to routing evaluation logs (JSON).",
    )
    parser.add_argument(
        "base_summary",
        type=Path,
        help="Path to the base model summary JSON (model_-1.json).",
    )
    return parser.parse_args()


def _iter_entries(data: object) -> Iterable[Tuple[str, dict]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), value
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, dict):
                yield str(idx), value


def _detect_route(trace: object) -> Optional[Route]:
    if not isinstance(trace, dict):
        return None
    selection = trace.get("selection")
    if isinstance(selection, str):
        sel = selection.lower()
        if "large" in sel:
            return ROUTE_LLM
        if "small" in sel:
            return ROUTE_SLM

    steps = trace.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("type") == "routing_llm_call":
                resp = step.get("response")
                if isinstance(resp, str):
                    lower = resp.lower()
                    if "large" in lower:
                        return ROUTE_LLM
                    if "small" in lower:
                        return ROUTE_SLM
    return None


def _is_correct(entry: dict) -> Optional[bool]:
    eval_result = entry.get("evaluation_result")
    if isinstance(eval_result, (int, float)):
        return eval_result > 0
    if isinstance(eval_result, bool):
        return bool(eval_result)
    return None


def _build_correct_lookup(base_data: object) -> Dict[str, Optional[bool]]:
    lookup: Dict[str, Optional[bool]] = {}
    for key, entry in _iter_entries(base_data):
        lookup[key] = _is_correct(entry)
    return lookup


def tally_routes(
    routing_data: object, base_lookup: Dict[str, Optional[bool]]
) -> Dict[Route, Dict[Label, int]]:
    tally: Dict[Route, Dict[Label, int]] = {
        ROUTE_SLM: {"correct": 0, "incorrect": 0, "unknown": 0},
        ROUTE_LLM: {"correct": 0, "incorrect": 0, "unknown": 0},
    }

    for key, entry in _iter_entries(routing_data):
        route = _detect_route(entry.get("orchestration_trace"))
        if route not in (ROUTE_SLM, ROUTE_LLM):
            continue

        correctness = _is_correct(entry)
        if correctness is None:
            correctness = base_lookup.get(key)

        if correctness is True:
            tally[route]["correct"] += 1
        elif correctness is False:
            tally[route]["incorrect"] += 1
        else:
            tally[route]["unknown"] += 1

    return tally


def _row(label: str, counts: Dict[Label, int], width: int) -> str:
    total = sum(counts.values())
    def fmt(val: int) -> str:
        if total == 0:
            return f"{val} (0.0%)"
        pct = (val / total) * 100
        return f"{val} ({pct:4.1f}%)"

    parts = [
        f"{label:<14}",
        f"{fmt(counts['correct']):>{width}}",
        f"{fmt(counts['incorrect']):>{width}}",
        f"{fmt(counts['unknown']):>{width}}",
        f"{total:>6}",
    ]
    return "  ".join(parts)


def render_table(tally: Dict[Route, Dict[Label, int]]) -> None:
    width = 14
    header = [
        f"{'Route':<14}",
        f"{'Correct':>{width}}",
        f"{'Incorrect':>{width}}",
        f"{'Unknown':>{width}}",
        f"{'Total':>6}",
    ]
    print("  ".join(header))
    print("-" * (len("  ".join(header))))
    print(_row("Routed to SLM", tally[ROUTE_SLM], width))
    print(_row("Routed to LLM", tally[ROUTE_LLM], width))


def main():
    args = parse_args()
    routing_data = mmengine.load(args.routing_logs)
    base_data = mmengine.load(args.base_summary)
    base_lookup = _build_correct_lookup(base_data)

    tally = tally_routes(routing_data, base_lookup)
    render_table(tally)


if __name__ == "__main__":
    main()
