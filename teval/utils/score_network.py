"""
Score network routing decisions against direct results from multiple endpoints.

For each sample, the "correct" endpoint is defined as the one with the highest
evaluation_result across the direct runs specified in the network config. Ties
are broken by lower cost (the optional `cost` field in the config), then by
alphabetical endpoint name.

Usage example:
    python -m teval.utils.score_network \\
        --network-config config/granite_qwen.py \\
        --router-path work_dirs/Qwen2.5-3B_network/
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import mmengine

CASE_ID = Tuple[str, str]  # (dataset_name, entry_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare network routing decisions against direct endpoint results "
            "and compute accuracy and coverage."
        )
    )
    parser.add_argument(
        "--network-config",
        required=True,
        type=Path,
        help="Path to the Python config exposing endpoints_dict with direct_results_path.",
    )
    parser.add_argument(
        "--router-path",
        required=True,
        type=Path,
        help="Directory containing the routing evaluation logs (network orchestrator run).",
    )
    return parser.parse_args()


def _iter_entries(data: object) -> Iterator[Tuple[str, dict]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), value
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, dict):
                yield str(idx), value


def _normalize_aliases(*aliases: str) -> List[str]:
    seen: List[str] = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.append(alias)
    return seen


def _extract_dataset(stem: str, aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        token = f"_{alias}_"
        if token in stem:
            return stem.split(token)[0]
        token = f"_{alias}"
        if token in stem:
            return stem.split(token)[0]
    return None


def _collect_scores(directory: Path, aliases: Sequence[str]) -> Dict[CASE_ID, float]:
    scores: Dict[CASE_ID, float] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if stem.endswith("_-1"):
            continue
        dataset = _extract_dataset(stem, aliases)
        if not dataset:
            continue
        data = mmengine.load(path)
        for entry_id, entry in _iter_entries(data):
            val = entry.get("evaluation_result")
            if isinstance(val, (int, float)):
                scores[(dataset, entry_id)] = float(val)
    return scores


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if isinstance(value, float) else "N/A"


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if (
        precision is None
        or recall is None
        or precision + recall == 0
    ):
        return None
    return 2 * precision * recall / (precision + recall)


def _load_endpoints(config_path: Path) -> List[dict]:
    spec = importlib.util.spec_from_file_location("network_config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    endpoints = getattr(module, "endpoints_dict", None)
    if not isinstance(endpoints, list):
        raise ValueError(f"'endpoints_dict' must be a list in {config_path}")
    normalized: List[dict] = []
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        name = ep.get("name")
        direct_path = ep.get("direct_results_path")
        cost = ep.get("cost", float("inf"))
        if not name or not isinstance(name, str):
            raise ValueError("Each endpoint must include a string 'name'.")
        if not direct_path or not isinstance(direct_path, str):
            raise ValueError(
                f"Endpoint '{name}' missing 'direct_results_path' (string path to direct outputs)."
            )
        normalized.append(
            {
                "name": name,
                "direct_dir": Path(direct_path),
                "cost": float(cost) if isinstance(cost, (int, float)) else float("inf"),
            }
        )
    return normalized


def _aliases_for_endpoint(endpoint: dict) -> List[str]:
    name = endpoint["name"]
    name_prefix = name.split("_", 1)[0] if "_" in name else name
    base_dir = endpoint["direct_dir"].name
    dir_base = base_dir
    for marker in ("_direct", "_routing"):
        idx = dir_base.find(marker)
        if idx != -1:
            dir_base = dir_base[:idx]
            break
    dir_prefix = dir_base.split("_", 1)[0] if "_" in dir_base else dir_base
    return _normalize_aliases(name, name_prefix, dir_base, dir_prefix)


def _detect_route(trace: object, endpoint_names: Sequence[str]) -> Optional[str]:
    if not isinstance(trace, dict):
        return None

    # Preferred: completion step contains the endpoint explicitly
    steps = trace.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                endpoint = step.get("endpoint")
                if isinstance(endpoint, str) and endpoint in endpoint_names:
                    return endpoint

    selection = trace.get("selection")
    if isinstance(selection, str):
        normalized = selection.strip().lower()
        for name in endpoint_names:
            if name.lower() == normalized:
                return name

    # Fallback: search routing_llm response text for any endpoint name
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict) and step.get("type") == "routing_llm_call":
                resp = step.get("response")
                if isinstance(resp, str):
                    lower_resp = resp.lower()
                    for name in endpoint_names:
                        if name.lower() in lower_resp:
                            return name
    return None


def _collect_routes(
    directory: Path, endpoint_names: Sequence[str]
) -> Dict[CASE_ID, str]:
    routes: Dict[CASE_ID, str] = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if stem.endswith("_-1"):
            continue
        data = mmengine.load(path)
        for entry_id, entry in _iter_entries(data):
            route = _detect_route(entry.get("orchestration_trace"), endpoint_names)
            dataset = stem.split("_", 1)[0] if "_" in stem else stem
            if route:
                routes[(dataset, entry_id)] = route
    return routes


def _determine_best_endpoint(
    case_id: CASE_ID,
    endpoint_scores: Dict[str, Dict[CASE_ID, float]],
    endpoint_costs: Dict[str, float],
) -> Optional[str]:
    best_endpoint = None
    best_score = None
    best_cost = None
    for endpoint, scores in endpoint_scores.items():
        score = scores.get(case_id)
        if score is None:
            continue
        cost = endpoint_costs.get(endpoint, float("inf"))
        if best_score is None or score > best_score:
            best_score = score
            best_cost = cost
            best_endpoint = endpoint
        elif score == best_score:
            if cost < (best_cost if best_cost is not None else float("inf")):
                best_score = score
                best_cost = cost
                best_endpoint = endpoint
            elif cost == best_cost and best_endpoint and endpoint < best_endpoint:
                best_endpoint = endpoint
    return best_endpoint


def main() -> None:
    args = parse_args()

    endpoints = _load_endpoints(args.network_config)
    endpoint_names = [ep["name"] for ep in endpoints]
    endpoint_costs = {ep["name"]: ep["cost"] for ep in endpoints}

    routes_dir = args.router_path
    if not routes_dir.exists():
        raise FileNotFoundError(f"Routing directory not found: {routes_dir}")

    # Collect scores for each endpoint
    endpoint_scores: Dict[str, Dict[CASE_ID, float]] = {}
    for ep in endpoints:
        aliases = _aliases_for_endpoint(ep)
        direct_dir = ep["direct_dir"]
        if not direct_dir.exists():
            raise FileNotFoundError(
                f"Direct results directory for '{ep['name']}' not found: {direct_dir}"
            )
        endpoint_scores[ep["name"]] = _collect_scores(direct_dir, aliases)

    routes = _collect_routes(routes_dir, endpoint_names)

    all_case_ids: Set[CASE_ID] = set()
    for scores in endpoint_scores.values():
        all_case_ids.update(scores.keys())

    best_by_case: Dict[CASE_ID, str] = {}
    for case_id in sorted(all_case_ids):
        best = _determine_best_endpoint(case_id, endpoint_scores, endpoint_costs)
        if best is not None:
            best_by_case[case_id] = best

    compared_keys = sorted(case for case in routes.keys() if case in best_by_case)
    missing_route = sorted(set(best_by_case) - set(routes.keys()))
    total_cases = len(best_by_case)
    total_routed_with_truth = len(compared_keys)

    per_endpoint_counts = {
        name: {"routed": 0, "correct": 0, "best": 0} for name in endpoint_names
    }

    for case_id, best in best_by_case.items():
        per_endpoint_counts[best]["best"] += 1
        routed = routes.get(case_id)
        if routed in per_endpoint_counts:
            per_endpoint_counts[routed]["routed"] += 1
        if routed == best and routed in per_endpoint_counts:
            per_endpoint_counts[routed]["correct"] += 1

    correct = sum(stats["correct"] for stats in per_endpoint_counts.values())

    # Compute confusion-style tallies to mirror score_router output format
    tp = correct  # routed == best
    fp = total_routed_with_truth - correct  # routed but not best
    fn = len(missing_route)  # missing routing for cases with direct scores
    tn = 0  # not meaningful in multi-endpoint, kept for format parity

    total_with_missing = total_routed_with_truth + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = _f1(precision, recall)
    accuracy = (tp + tn) / total_with_missing if total_with_missing else None
    routing_coverage = (
        total_routed_with_truth / total_cases if total_cases else None
    )

    print("Routing vs Network-direct comparison")
    print("------------------------------------")
    print(f"Network config : {args.network_config}")
    print(f"Routing dir    : {routes_dir}")
    print(f"Endpoints      : {', '.join(endpoint_names)}")
    print()
    print(f"Total cases with direct scores         : {total_cases}")
    print(f"Total cases with routing+scores        : {total_routed_with_truth}")
    print(f"Missing routing entries for direct cases: {fn}")
    print()
    print("Confusion matrix counts")
    print(f"  TP (routed best endpoint)   : {tp}")
    print(f"  FP (routed non-best)        : {fp}")
    print(f"  FN (direct score, no route) : {fn}")
    print(f"  TN (not used)               : {tn}")
    print()
    print(f"Precision (overall): {_fmt(precision)}")
    print(f"Recall    (overall): {_fmt(recall)}")
    print(f"F1        (overall): {_fmt(f1)}")
    print(f"Accuracy  (overall): {_fmt(accuracy)}")
    print(f"Coverage  (routing): {_fmt(routing_coverage)}")
    print()
    print("Per-endpoint routing metrics:")
    for name in endpoint_names:
        stats = per_endpoint_counts[name]
        routed = stats["routed"]
        corr = stats["correct"]
        best = stats["best"]
        precision_ep = corr / routed if routed else None
        recall_ep = corr / best if best else None
        f1_ep = _f1(precision_ep, recall_ep)
        call_rate_ep = (
            routed / total_routed_with_truth if total_routed_with_truth else None
        )
        print(
            f"  {name:<12} routed: {routed:>5}  best: {best:>5}  correct: {corr:>5}  "
            f"prec: {_fmt(precision_ep)}  rec: {_fmt(recall_ep)}  F1: {_fmt(f1_ep)}  "
            f"call_rate: {_fmt(call_rate_ep)}"
        )


if __name__ == "__main__":
    main()
