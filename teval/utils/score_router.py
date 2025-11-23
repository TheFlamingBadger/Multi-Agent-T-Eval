import argparse
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import mmengine

Route = str
CASE_ID = Tuple[str, str]  # (dataset_name, entry_id)

ROUTE_SLM = "SLM"
ROUTE_LLM = "LLM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare routing decisions against direct SLM correctness and compute "
            "precision/recall/F1."
        )
    )
    parser.add_argument(
        "--llm-path",
        required=True,
        type=Path,
        help="Directory containing the LLM direct evaluation logs.",
    )
    parser.add_argument(
        "--slm-path",
        required=True,
        type=Path,
        help="Directory containing the SLM direct evaluation logs.",
    )
    parser.add_argument(
        "--router-path",
        required=True,
        type=Path,
        help="Directory containing the routing evaluation logs (SLM model run with routing orchestrator).",
    )
    return parser.parse_args()


def _infer_display_name(directory: Path) -> str:
    name = directory.name
    for suffix in ("_direct", "_routing"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _iter_entries(data: object) -> Iterator[Tuple[str, dict]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), value
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, dict):
                yield str(idx), value


def _extract_dataset(stem: str, display_name: str) -> Optional[str]:
    token = f"_{display_name}_"
    if token in stem:
        return stem.split(token)[0]
    token = f"_{display_name}"
    if token in stem:
        return stem.split(token)[0]
    return None


def _collect_scores(directory: Path, display_name: str) -> Dict[CASE_ID, float]:
    scores: Dict[CASE_ID, float] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if stem.endswith("_-1"):
            continue
        dataset = _extract_dataset(stem, display_name)
        if not dataset:
            continue
        data = mmengine.load(path)
        for entry_id, entry in _iter_entries(data):
            val = entry.get("evaluation_result")
            if isinstance(val, (int, float)):
                scores[(dataset, entry_id)] = float(val)
    return scores


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


def _collect_routes(directory: Path, display_name: str) -> Dict[CASE_ID, Route]:
    routes: Dict[CASE_ID, Route] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if stem.endswith("_-1"):
            continue
        dataset = _extract_dataset(stem, display_name)
        if not dataset:
            continue
        data = mmengine.load(path)
        for entry_id, entry in _iter_entries(data):
            route = _detect_route(entry.get("orchestration_trace"))
            if route in (ROUTE_LLM, ROUTE_SLM):
                routes[(dataset, entry_id)] = route
    return routes


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "N/A"


def main() -> None:
    args = parse_args()

    llm_dir = args.llm_path
    slm_dir = args.slm_path
    routing_dir = args.router_path

    for directory in (llm_dir, slm_dir, routing_dir):
        if not directory.exists():
            raise FileNotFoundError(f"Missing directory: {directory}")

    llm_name = _infer_display_name(llm_dir)
    slm_name = _infer_display_name(slm_dir)
    routing_name = _infer_display_name(routing_dir)
    if routing_name != slm_name:
        raise ValueError(
            f"Routing directory '{routing_dir}' appears to target '{routing_name}', "
            f"but SLM directory '{slm_dir}' looks like '{slm_name}'. Ensure they match."
        )

    slm_scores = _collect_scores(slm_dir, slm_name)
    routes = _collect_routes(routing_dir, routing_name)
    llm_scores = _collect_scores(llm_dir, llm_name)

    compared_keys = sorted(set(routes) & set(slm_scores) & set(llm_scores))
    missing_route = sorted(set(slm_scores) - set(routes))
    missing_slm = sorted(set(routes) - set(slm_scores))
    missing_llm = sorted(set(routes) - set(llm_scores))

    tp = fp = fn = tn = 0
    for key in compared_keys:
        route = routes[key]
        slm_score = slm_scores.get(key)
        llm_score = llm_scores.get(key)
        if slm_score is None or llm_score is None:
            continue
        prefer_slm = slm_score >= llm_score
        if route == ROUTE_SLM:
            if prefer_slm:
                tp += 1
            else:
                fp += 1
        else:
            if prefer_slm:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    accuracy = (tp + tn) / total if total else None

    print("Routing vs SLM-direct comparison")
    print("--------------------------------")
    print(f"LLM direct dir : {llm_dir}")
    print(f"SLM direct dir : {slm_dir}")
    print(f"Routing dir    : {routing_dir}")
    print()
    print(f"Total cases with routing+scores      : {len(compared_keys)}")
    print(f"Missing routing entries for SLM cases: {len(missing_route)}")
    print(f"Missing SLM direct scores for routes : {len(missing_slm)}")
    print(f"Missing LLM direct scores for routes : {len(missing_llm)}")
    print()
    print("Confusion matrix counts")
    print(f"  TP (routed SLM, prefer SLM) : {tp}")
    print(f"  FP (routed SLM, prefer LLM) : {fp}")
    print(f"  FN (routed LLM, prefer SLM) : {fn}")
    print(f"  TN (routed LLM, prefer LLM) : {tn}")
    print()
    print(f"Precision: {_fmt(precision)}")
    print(f"Recall   : {_fmt(recall)}")
    print(f"F1       : {_fmt(f1)}")
    print(f"Accuracy : {_fmt(accuracy)}")


if __name__ == "__main__":
    main()
