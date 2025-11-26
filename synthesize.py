import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import mmengine
from tqdm import tqdm

from teval.utils.meta_template import meta_template_dict
from teval.orchestrators.azure_openai import AzureOpenAIOrchestrator
from lagent.llms.huggingface import HFTransformerCasualLM, HFTransformerChat
from lagent.llms.openai import GPTAPI


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Route-only benchmark that reuses cached direct answers. "
            "Only the router is invoked; question answers are loaded from "
            "pre-computed direct results specified in a network config."
        )
    )
    parser.add_argument("--dataset_path", type=str, default="data/instruct_v1.json")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["api", "hf", "azure"],
        default="hf",
        help="Backend for the router model (answers are pulled from cache).",
    )
    parser.add_argument("--model_display_name", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out_name", type=str, default="tmp.json")
    parser.add_argument("--out_dir", type=str, default="work_dirs/")
    parser.add_argument("--model_path", type=str, help="Router model identifier or path")
    parser.add_argument(
        "--eval",
        type=str,
        choices=["instruct", "reason", "plan", "retrieve", "review", "understand", "rru"],
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=-1,
        help="Number of samples to route; -1 means all.",
    )
    parser.add_argument("--prompt_type", type=str, default="json", choices=["json", "str"])
    parser.add_argument("--meta_template", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--naive-prompt",
        dest="naive_prompt",
        action="store_true",
        help="Use the original routing prompt (without skill scores) for RoutingOrchestrator.",
    )
    parser.add_argument(
        "--rubric",
        dest="rubric_prompt",
        action="store_true",
        help="Use the rubric-based routing prompt.",
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default="routing",
        choices=["routing", "network"],
        help="Routing strategy: two-path routing or configurable endpoint network.",
    )
    parser.add_argument(
        "--azure_env_path",
        type=str,
        default=".env",
        help="Path to .env file with Azure OpenAI credentials (router only).",
    )
    parser.add_argument(
        "--network_config",
        type=str,
        default="config/granite_qwen.py",
        help="Path to a Python file exposing endpoints_dict with direct_results_path.",
        dest="network_config",
    )
    parser.add_argument(
        "--network-config",
        type=str,
        dest="network_config",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--small-endpoint",
        type=str,
        default=None,
        help="(Routing only) Endpoint name to use when router selects the small model.",
    )
    parser.add_argument(
        "--large-endpoint",
        type=str,
        default=None,
        help="(Routing only) Endpoint name to use when router selects the large model.",
    )
    args = parser.parse_args()
    return args


def load_dataset(dataset_path, out_dir, is_resume=False, tmp_folder_name="tmp"):
    dataset = mmengine.load(dataset_path)
    total_num = len(dataset)
    tested_num = 0
    if is_resume:
        cache_dir = os.path.join(out_dir, tmp_folder_name)
        os.makedirs(cache_dir, exist_ok=True)
        file_list = os.listdir(cache_dir)
        for filename in file_list:
            if filename.split(".")[0] in dataset:
                tested_num += 1
                file_id = filename.split(".")[0]
                dataset.pop(file_id)
            else:
                print(f"Warning: {filename} not in dataset, remove it from cache")
                os.remove(os.path.join(cache_dir, filename))
    return dataset, tested_num, total_num


def _strip_special_tokens(text: str) -> str:
    text = text.split("<eoa>")[0]
    text = text.split("<TOKENS_UNUSED_1>")[0]
    text = text.split("<|im_end|>")[0]
    text = text.split("\nuser")[0]
    text = text.split("\nassistant")[0]
    text = text.split("\nUSER")[0]
    text = text.split("[INST]")[0]
    text = text.split("<|user|>")[0]
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    text = text.strip("`").strip()
    return text


def _dataset_alias(dataset_path: Union[str, Path]) -> Tuple[str, str]:
    stem = Path(dataset_path).stem
    alias = re.sub(r"_v\d+$", "", stem)
    return alias, stem


def _load_network_config(config_path: str) -> List[dict]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Endpoint config not found: {config_path}")
    cfg_dir = path.parent
    spec = importlib.util.spec_from_file_location("synthesize_network_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    endpoints = getattr(module, "endpoints_dict", None)
    if not isinstance(endpoints, list):
        raise ValueError(f"'endpoints_dict' must be a list in {config_path}")

    normalized: List[dict] = []
    for idx, ep in enumerate(endpoints):
        if not isinstance(ep, dict):
            raise ValueError(f"Endpoint entry at index {idx} is not a dict: {ep!r}")
        name = ep.get("name")
        direct_raw = ep.get("direct_results_path")
        cost = ep.get("cost", float("inf"))
        if not name or not isinstance(name, str):
            raise ValueError(f"Endpoint at index {idx} is missing a string 'name'")
        if not direct_raw or not isinstance(direct_raw, str):
            raise ValueError(
                f"Endpoint '{name}' missing 'direct_results_path' (string path to cached direct outputs)."
            )
        direct_path = Path(direct_raw).expanduser()
        if not direct_path.is_absolute():
            candidate = (cfg_dir / direct_raw).expanduser()
            if candidate.exists():
                direct_path = candidate
        normalized.append(
            {
                "name": name,
                "description": ep.get("description", ""),
                "direct_results_path": direct_path,
                "router_system_prompt": ep.get("router_system_prompt"),
                "meta_template": ep.get("meta_template"),
                "use_chat_template": ep.get("use_chat_template", False),
                "cost": float(cost) if isinstance(cost, (int, float)) else float("inf"),
            }
        )
    if not normalized:
        raise ValueError(f"No endpoints found in {config_path}")
    return normalized


def _resolve_direct_file(
    base_dir: Path, dataset_alias: str, dataset_stem: str, endpoint_name: str
) -> Path:
    candidates = [
        base_dir / f"{dataset_alias}_{endpoint_name}_direct.json",
        base_dir / f"{dataset_stem}_{endpoint_name}_direct.json",
        base_dir / f"{dataset_alias}_{endpoint_name}.json",
        base_dir / f"{dataset_stem}_{endpoint_name}.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find cached direct results for endpoint '{endpoint_name}' in {base_dir}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _load_direct_results(
    endpoints: List[dict], dataset_alias: str, dataset_stem: str
) -> Tuple[Dict[str, dict], Dict[str, Path]]:
    cache: Dict[str, dict] = {}
    source_paths: Dict[str, Path] = {}
    for ep in endpoints:
        name = ep["name"]
        base_dir = ep["direct_results_path"]
        direct_file = _resolve_direct_file(base_dir, dataset_alias, dataset_stem, name)
        cache[name] = mmengine.load(direct_file)
        source_paths[name] = direct_file
    return cache, source_paths


def _normalize_history(history: Union[List[dict], Dict[str, str], str]) -> List[dict]:
    if isinstance(history, list):
        return history
    if isinstance(history, dict):
        return [history]
    return [{"role": "user", "content": str(history)}]


class RoutingSelector:
    def __init__(self, router_llm, use_naive_prompt=False, use_rubric_prompt=False):
        if use_naive_prompt and use_rubric_prompt:
            raise ValueError("Choose at most one routing prompt variant.")
        self.router_llm = router_llm
        if use_rubric_prompt:
            self.router_system_prompt = self._rubric_router_prompt()
            self.router_prompt_variant = "rubric"
        elif use_naive_prompt:
            self.router_system_prompt = self._naive_router_prompt()
            self.router_prompt_variant = "naive"
        else:
            self.router_system_prompt = self._default_router_prompt()
            self.router_prompt_variant = "default"

    def route_batch(self, histories: List[List[dict]]) -> List[Tuple[str, str, bool, float, List[dict]]]:
        messages_batch = [self._build_routing_messages(h) for h in histories]
        routing_start = perf_counter()
        responses = self.router_llm.chat(
            messages_batch, do_sample=False, temperature=0
        )
        routing_elapsed = perf_counter() - routing_start
        per_item_elapsed = routing_elapsed / max(len(histories), 1)
        results = []
        for raw, messages in zip(responses, messages_batch):
            choice, invalid_reason = self._extract_choice(raw)
            selection = choice or "large language model"
            results.append((selection, raw, invalid_reason is not None, per_item_elapsed, messages))
        return results

    def _build_routing_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        rendered_history = self._render_history(history)
        context_block = f"<conversation_context>\n{rendered_history}\n</conversation_context>"
        system_prompt = self.router_system_prompt
        if "<insert context/>" in system_prompt:
            system_content = system_prompt.replace("<insert context/>", context_block)
        else:
            system_content = f"{system_prompt}\n{context_block}"
        user_content = (
            "Score the conversation and return only the required JSON response."
            if self.router_prompt_variant == "rubric"
            else "Choose the model now and output only one allowed model name. No justification."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _render_history(self, history: List[Dict[str, str]]) -> str:
        lines = []
        for message in history:
            role = message.get("role", "unknown")
            content = str(message.get("content", "")).strip()
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _extract_choice(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(text, str):
            return None, "routing output was not a string"
        json_choice = self._extract_json_route(text)
        if json_choice:
            return json_choice, None
        matches = re.findall(
            r"\\b(small language model|large language model)\\b",
            text,
            flags=re.IGNORECASE,
        )
        unique_matches = {m.lower() for m in matches}
        if not unique_matches:
            return None, "no valid model name found"
        if len(unique_matches) > 1:
            first_choice = matches[0].lower()
            return first_choice, "tie detected between allowed models"
        choice = unique_matches.pop()
        return choice, None

    def _extract_json_route(self, text: str) -> Optional[str]:
        candidate = text.strip()
        fenced = re.match(
            r"```(?:json)?\\s*(.*?)\\s*```$", candidate, flags=re.IGNORECASE | re.DOTALL
        )
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            data = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        route = data.get("route")
        if not isinstance(route, str):
            return None
        normalized = re.sub(r"[\\s_]+", " ", route).strip().lower()
        if normalized in {"slm", "small language model"} or "small" in normalized:
            return "small language model"
        if normalized in {"llm", "large language model"} or "large" in normalized:
            return "large language model"
        return None

    def _default_router_prompt(self) -> str:
        return (
            "You are a routing assistant. Read the entire conversation and pick exactly one model for the final answer:"
            ' \"small language model\" or \"large language model\". '
            "Send to the large language model when the request needs multi-step reasoning, outside knowledge/citations, critique/review, long or multi-part context, ambiguous goals, or non-trivial code/math. "
            "Send to the small language model when the ask is short and concrete: direct instructions, simple formatting, extraction, rewriting, summarizing what is already in the prompt, or filling a template. "
            "If the task is not clearly in the hard cases above, default to the small language model to save cost. "
            "Respond with only the chosen model name. Do not explain, justify, or answer the user's question."
        )

    def _naive_router_prompt(self) -> str:
        return (
            "You are a routing assistant. Read the entire conversation and pick exactly one model to respond to the user's query"
            'Pick either: \"small language model\" or \"large language model\". '
            "The small language model is much cheaper but suited only to very simple tasks "
            "The large language model is expensive but much smarter, suited to tasks with long contexts or those requiring reasoning"
            "The conversation context is as follows: "
            "Respond with only the chosen model name. Do not explain, justify, or answer the user's question."
        )

    def _rubric_router_prompt(self) -> str:
        return (
            "You are a routing model responsible for choosing whether a query should be handled by a Small Language Model (SLM) or by a Large Language Model (LLM).\\n\\n"
            "Your goal is to reliably score the user query on three difficulty axes and then determine the correct model based on the rubric below.\\n\\n"
            "---\\n"
            "DIFFICULTY PRIORS (Important)"
            "Most user queries are NOT highly difficult. The majority fall into the"
            "0-2 range on each axis. Scores of 3 should be rare and used only for"
            "genuinely complex cases."
            "Use these priors when scoring:"
            "SCORING RUBRIC (0-3 each)\\n\\n"
            "1. Complexity (0-3)\\n"
            "   - 0: Simple, single-step request. No reasoning required.\\n"
            "   - 1: Mild reasoning. One or two steps; low cognitive load.\\n"
            "   - 2: Multi-step reasoning, transformation, or non-trivial logic.\\n"
            "   - 3: Deep or multi-hop reasoning, chain-of-thought needed, or tool-use complexity.\\n\\n"
            "2. Ambiguity (0-3)\\n"
            "   - 0: Request is clear, specific, and objective.\\n"
            "   - 1: Minor ambiguity or open-endedness.\\n"
            "   - 2: Requires precision, factual correctness, or domain knowledge.\\n"
            "   - 3: High ambiguity, specialized factual recall, or high error sensitivity.\\n\\n"
            "3. Constraint Sensitivity (0-3)\\n"
            "   - 0: Free-form response, no format constraints.\\n"
            "   - 1: Light structure (lists, short templates).\\n"
            "   - 2: Strict formatting or structured outputs (JSON, API args).\\n"
            "   - 3: Highly rigid schemas or multi-field arguments with correctness checks.\\n\\n"
            "---\\n"
            "DECISION RULE\\n\\n"
            "Compute:\\n"
            "  TOTAL_SCORE = Complexity + Ambiguity + ConstraintSensitivity\\n\\n"
            'If TOTAL_SCORE >= 8 -> route to \"llm\".\\n'
            'If TOTAL_SCORE <= 6 -> route to \"slm\".\\n\\n'
            "---\\n"
            "OUTPUT FORMAT\\n\\n"
            "Respond ONLY with a JSON object in the exact structure:\\n\\n"
            "{\\n"
            '  \"complexity\": <0-3>,\\n'
            '  \"ambiguity\": <0-3>,\\n'
            '  \"constraint_sensitivity\": <0-3>,\\n'
            '  \"total\": <sum>,\\n'
            '  \"route\": \"large_language_modeel\" | \"small_language_modeel\"\\n'
            "}\\n\\n"
            "---\\n"
            "CONTEXT\\n\\n"
            "<insert context/>\\n"
        )


class NetworkSelector:
    def __init__(self, router_llm, endpoints: List[dict]):
        self.router_llm = router_llm
        self.endpoints = endpoints
        self.endpoint_name_map = {ep["name"].lower(): ep["name"] for ep in endpoints}
        self.default_endpoint = endpoints[0]["name"]
        custom_prompt = endpoints[0].get("router_system_prompt")
        if custom_prompt:
            self.router_system_prompt = custom_prompt
        else:
            self.router_system_prompt = self._default_router_prompt(endpoints)

    def route_batch(self, histories: List[List[dict]]) -> List[Tuple[str, str, bool, float, List[dict]]]:
        messages_batch = [self._build_routing_messages(h) for h in histories]
        routing_start = perf_counter()
        responses = self.router_llm.chat(
            messages_batch, do_sample=False, temperature=0
        )
        routing_elapsed = perf_counter() - routing_start
        per_item_elapsed = routing_elapsed / max(len(histories), 1)
        results = []
        for raw, messages in zip(responses, messages_batch):
            choice, invalid_reason = self._extract_choice(raw)
            selection = choice or self.default_endpoint
            results.append((selection, raw, invalid_reason is not None, per_item_elapsed, messages))
        return results

    def _default_router_prompt(self, endpoints: List[Dict[str, str]]) -> str:
        parts = [
            "You are a routing assistant. Choose exactly one endpoint from the list below to answer the user.",
            "Respond with only the endpoint name (verbatim) and nothing else.",
            "Available endpoints:",
        ]
        for ep in endpoints:
            desc = ep.get("description") or "No description provided."
            parts.append(f"- {ep['name']}: {desc}")
        return "\\n".join(parts)

    def _build_routing_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        rendered_history = self._render_history(history)
        context_block = f\"<conversation_context>\\n{rendered_history}\\n</conversation_context>\"
        system_content = f\"{self.router_system_prompt}\\n{context_block}\"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Select the best endpoint and reply with only its name."},
        ]

    def _render_history(self, history: List[Dict[str, str]]) -> str:
        lines = []
        for message in history:
            role = message.get("role", "unknown")
            content = str(message.get("content", "")).strip()
            lines.append(f"{role.upper()}: {content}")
        return "\\n".join(lines)

    def _extract_choice(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(text, str):
            return None, "routing output was not a string"
        json_choice = self._extract_json_choice(text)
        if json_choice:
            return json_choice, None

        normalized_text = text.strip().lower()
        matches = []
        for lower_name, original in self.endpoint_name_map.items():
            if re.search(re.escape(lower_name), normalized_text):
                matches.append(original)

        unique_matches = {m for m in matches}
        if not unique_matches:
            return None, "no valid endpoint name found"
        if len(unique_matches) > 1:
            first_choice = matches[0]
            return first_choice, "tie detected between endpoints"
        choice = unique_matches.pop()
        return choice, None

    def _extract_json_choice(self, text: str) -> Optional[str]:
        candidate = text.strip()
        fenced = re.match(
            r"```(?:json)?\\s*(.*?)\\s*```$", candidate, flags=re.IGNORECASE | re.DOTALL
        )
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            data = json.loads(candidate)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        for key in ("route", "model", "endpoint"):
            route = data.get(key)
            if isinstance(route, str):
                normalized = route.strip().lower()
                if normalized in self.endpoint_name_map:
                    return self.endpoint_name_map[normalized]
        return None


def _select_routing_endpoints(
    endpoints: List[dict], small_override: Optional[str], large_override: Optional[str]
) -> Tuple[str, str]:
    names = [ep["name"] for ep in endpoints]
    if small_override:
        if small_override not in names:
            raise ValueError(f"--small-endpoint '{small_override}' not found in config endpoints: {names}")
        small = small_override
    else:
        small = min(endpoints, key=lambda ep: ep.get("cost", float("inf"))).get("name")
    if large_override:
        if large_override not in names:
            raise ValueError(f"--large-endpoint '{large_override}' not found in config endpoints: {names}")
        large = large_override
    else:
        large = max(endpoints, key=lambda ep: ep.get("cost", float("-inf"))).get("name")
    return small, large


def _resolve_endpoint_for_selection(
    selection: str, orchestrator: str, endpoints: List[dict], small_large: Tuple[str, str]
) -> str:
    if orchestrator == "network":
        names_lower = {ep["name"].lower(): ep["name"] for ep in endpoints}
        return names_lower.get(selection.lower(), endpoints[0]["name"])
    # routing orchestrator
    small, large = small_large
    normalized = selection.lower().strip()
    if "small" in normalized:
        return small
    if "large" in normalized:
        return large
    return large


def synthesize(
    dataset: dict,
    selector,
    endpoints: List[dict],
    direct_results: Dict[str, dict],
    source_paths: Dict[str, Path],
    orchestrator: str,
    out_dir: str,
    tmp_folder_name: str,
    test_num: int,
    batch_size: int,
    routing_small_large: Tuple[str, str],
):
    random_list = list(dataset.keys())[:test_num]
    batch_histories: List[List[dict]] = []
    batch_ids: List[str] = []
    for idx in tqdm(random_list):
        history = _normalize_history(dataset[idx]["origin_prompt"])
        batch_histories.append(history)
        batch_ids.append(idx)
        if len(batch_ids) == batch_size or idx == random_list[-1]:
            results = selector.route_batch(batch_histories)
            for ptr, (selection, routing_raw, invalid_flag, elapsed, routing_messages) in enumerate(results):
                data_ptr = batch_ids[ptr]
                endpoint = _resolve_endpoint_for_selection(
                    selection, orchestrator, endpoints, routing_small_large
                )
                endpoint_cache = direct_results.get(endpoint, {})
                cached = endpoint_cache.get(data_ptr) or endpoint_cache.get(str(data_ptr))
                if cached is None:
                    raise KeyError(
                        f"Entry '{data_ptr}' not found in cached direct results for endpoint '{endpoint}'."
                    )
                record = dict(cached)
                record["prediction"] = _strip_special_tokens(record.get("prediction", ""))
                trace = {
                    "strategy": orchestrator,
                    "selection": selection,
                    "resolved_endpoint": endpoint,
                    "invalid_routing_output": invalid_flag,
                    "routing_response": routing_raw,
                    "routing_elapsed_seconds": elapsed,
                    "cached_completion": True,
                    "cached_source_path": str(source_paths[endpoint]),
                    "steps": [
                        {
                            "type": "routing_llm_call",
                            "messages": routing_messages,
                            "response": routing_raw,
                            "elapsed_seconds": elapsed,
                        },
                        {
                            "type": "cached_completion",
                            "endpoint": endpoint,
                            "source_path": str(source_paths[endpoint]),
                        },
                    ],
                }
                record["orchestration_trace"] = trace
                record["inference_time_seconds"] = elapsed
                mmengine.dump(record, os.path.join(out_dir, tmp_folder_name, f"{data_ptr}.json"))
            batch_histories = []
            batch_ids = []

    results = dict()
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split(".")[0]
        results[file_id] = mmengine.load(os.path.join(out_dir, tmp_folder_name, filename))
    return results


if __name__ == "__main__":
    args = parse_args()
    if args.naive_prompt and args.rubric_prompt:
        raise ValueError("Choose at most one routing prompt variant.")
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_folder_name = os.path.splitext(args.out_name)[0]
    os.makedirs(os.path.join(args.out_dir, tmp_folder_name), exist_ok=True)

    dataset, tested_num, total_num = load_dataset(
        args.dataset_path, args.out_dir, args.resume, tmp_folder_name=tmp_folder_name
    )
    if args.test_num == -1:
        test_num = max(total_num - tested_num, 0)
    else:
        test_num = max(min(args.test_num - tested_num, total_num - tested_num), 0)

    dataset_alias, dataset_stem = _dataset_alias(args.dataset_path)
    endpoints = _load_network_config(args.network_config)
    direct_results, source_paths = _load_direct_results(endpoints, dataset_alias, dataset_stem)

    # Initialize router LLM
    if args.model_type == "azure":
        router_llm = AzureOpenAIOrchestrator(env_path=args.azure_env_path)
    elif args.model_type == "api":
        router_llm = GPTAPI(args.model_path)
    elif args.model_type == "hf":
        meta_template = meta_template_dict.get(args.meta_template)
        if "chatglm" in args.model_display_name:
            router_llm = HFTransformerChat(path=args.model_path, meta_template=meta_template)
        else:
            router_llm = HFTransformerCasualLM(
                path=args.model_path, meta_template=meta_template, max_new_tokens=512
            )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    if args.orchestrator == "network":
        selector = NetworkSelector(router_llm, endpoints)
    elif args.orchestrator == "routing":
        selector = RoutingSelector(
            router_llm,
            use_naive_prompt=args.naive_prompt,
            use_rubric_prompt=args.rubric_prompt,
        )
    else:
        raise ValueError("Only routing or network orchestrators are supported.")

    routing_small_large = _select_routing_endpoints(
        endpoints, args.small_endpoint, args.large_endpoint
    )

    print(f"Using {args.orchestrator} orchestrator (router-only synthesis)")
    print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
    output_file_path = os.path.join(args.out_dir, args.out_name)
    if test_num != 0:
        prediction = synthesize(
            dataset,
            selector,
            endpoints,
            direct_results,
            source_paths,
            args.orchestrator,
            args.out_dir,
            tmp_folder_name=tmp_folder_name,
            test_num=test_num,
            batch_size=args.batch_size,
            routing_small_large=routing_small_large,
        )
        mmengine.dump(prediction, output_file_path)

    if args.eval:
        import teval.evaluators as evaluator_factory

        if args.model_display_name == "":
            model_display_name = args.model_type
        else:
            model_display_name = args.model_display_name
        os.makedirs(args.out_dir, exist_ok=True)
        eval_mapping = dict(
            instruct="InstructEvaluator",
            plan="PlanningEvaluator",
            review="ReviewEvaluator",
            reason="ReasonRetrieveUnderstandEvaluator",
            retrieve="ReasonRetrieveUnderstandEvaluator",
            understand="ReasonRetrieveUnderstandEvaluator",
            rru="ReasonRetrieveUnderstandEvaluator",
        )
        if "_zh" in args.dataset_path:
            bert_score_model = "thenlper/gte-large-zh"
            json_path = os.path.join(
                args.out_dir, model_display_name + "_" + str(args.test_num) + "_zh.json"
            )
        else:
            bert_score_model = "all-mpnet-base-v2"
            json_path = os.path.join(
                args.out_dir, model_display_name + "_" + str(args.test_num) + ".json"
            )
        evaluator_class = getattr(evaluator_factory, eval_mapping[args.eval])
        evaluator = evaluator_class(
            output_file_path,
            default_prompt_type=args.prompt_type,
            eval_type=args.eval,
            bert_score_model=bert_score_model,
        )
        if os.path.exists(json_path):
            results = mmengine.load(json_path)
        else:
            results = dict()
        eval_results = evaluator.evaluate()
        print(eval_results)
        results[args.eval + "_" + args.prompt_type] = eval_results
        print(f"Writing Evaluation Results to {json_path}")
        mmengine.dump(results, json_path)
