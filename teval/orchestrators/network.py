import importlib.util
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

from lagent.llms.huggingface import HFTransformerCasualLM, HFTransformerChat
from lagent.llms.openai import GPTAPI

from teval.utils.meta_template import meta_template_dict

from .base import BaseOrchestrator
from .azure_openai import AzureOpenAIOrchestrator


class NetworkOrchestrator(BaseOrchestrator):
    """
    Router that selects among a set of configured endpoints defined in a
    Python config file (see config/granite_qwen.py for an example).

    The router model reads the conversation, chooses one endpoint name, and the
    orchestrator forwards the query to the matched endpoint LLM.
    """

    def __init__(
        self,
        router_llm,
        endpoint_config_path: str,
        router_system_prompt: Optional[str] = None,
        default_meta_template: str = "qwen",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            router_llm: LLM used to perform the routing decision.
            endpoint_config_path: Path to a Python file that exposes
                ``endpoints_dict`` (list of endpoint dicts).
            router_system_prompt: Optional override for the routing prompt. If
                omitted, a prompt is built from the endpoints list.
            default_meta_template: Fallback meta template name for HF endpoints
                when a specific template is not provided in the config.
            **kwargs: Forwarded to BaseOrchestrator.
        """
        super().__init__(router_llm, **kwargs)
        self.router_llm = router_llm
        self.default_meta_template = default_meta_template
        self.endpoints = self._load_endpoints(endpoint_config_path)
        if not self.endpoints:
            raise ValueError(
                f"No endpoints found in config file: {endpoint_config_path}"
            )

        self.endpoint_llms = self._build_endpoint_llms(self.endpoints)
        self.endpoint_name_map = {
            name.lower(): name for name in self.endpoint_llms.keys()
        }
        self.default_endpoint = next(iter(self.endpoint_llms.keys()))

        if router_system_prompt:
            self.router_system_prompt = router_system_prompt
        else:
            self.router_system_prompt = self._default_router_prompt(self.endpoints)

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        histories, was_single = self._normalize_input(message_histories)
        results: List[str] = []
        traces: List[Dict[str, Any]] = []

        for history in histories:
            routing_messages = self._build_routing_messages(history)
            routing_start = perf_counter()
            routing_raw = self.router_llm.chat(
                [routing_messages], do_sample=False, temperature=0
            )[0]
            routing_elapsed = perf_counter() - routing_start

            choice, invalid_reason = self._extract_choice(routing_raw)
            selected_endpoint = choice or self.default_endpoint
            selected_llm = self.endpoint_llms.get(selected_endpoint)

            if selected_llm is None:
                invalid_reason = (
                    invalid_reason
                    or f"Selected endpoint '{selected_endpoint}' not configured."
                )
                selected_endpoint = self.default_endpoint
                selected_llm = self.endpoint_llms[selected_endpoint]

            completion_start = perf_counter()
            completion_text = selected_llm.chat([history], **kwargs)[0]
            completion_elapsed = perf_counter() - completion_start
            underlying_trace = getattr(selected_llm, "last_trace", []) or []

            results.append(completion_text)

            routing_step = {
                "type": "routing_llm_call",
                "messages": routing_messages,
                "response": routing_raw,
                "elapsed_seconds": routing_elapsed,
            }
            completion_step: Dict[str, Any] = {
                "type": "endpoint_llm_call",
                "endpoint": selected_endpoint,
                "messages": history,
                "response": completion_text,
                "elapsed_seconds": completion_elapsed,
            }
            if underlying_trace:
                completion_step["underlying_trace"] = underlying_trace[0]

            traces.append(
                {
                    "strategy": "network_routing",
                    "selection": selected_endpoint,
                    "invalid_routing_output": invalid_reason is not None,
                    "invalid_reason": invalid_reason,
                    "total_elapsed_seconds": routing_elapsed + completion_elapsed,
                    "steps": [routing_step, completion_step],
                }
            )

        self._record_trace(traces)
        return self._denormalize_output(results, was_single)

    def _load_endpoints(self, config_path: str) -> List[Dict[str, Any]]:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Endpoint config not found: {config_path}")
        spec = importlib.util.spec_from_file_location("teval_network_config", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config module from {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        endpoints = getattr(module, "endpoints_dict", None)
        if endpoints is None:
            raise ValueError(
                f"'endpoints_dict' not found in endpoint config: {config_path}"
            )
        if not isinstance(endpoints, list):
            raise ValueError(
                f"'endpoints_dict' must be a list of endpoint dicts in {config_path}"
            )

        normalized: List[Dict[str, Any]] = []
        for idx, ep in enumerate(endpoints):
            if not isinstance(ep, dict):
                raise ValueError(f"Endpoint entry at index {idx} is not a dict: {ep!r}")
            name = ep.get("name")
            ep_type = ep.get("type")
            path_value = ep.get("path")
            if not name or not isinstance(name, str):
                raise ValueError(f"Endpoint at index {idx} is missing a string 'name'")
            if ep_type not in {"hf", "api", "azure"}:
                raise ValueError(
                    f"Endpoint '{name}' has unsupported type '{ep_type}'. "
                    "Use 'hf', 'api', or 'azure'."
                )
            if ep_type in {"hf", "api"} and (
                not path_value or not isinstance(path_value, str)
            ):
                raise ValueError(
                    f"Endpoint '{name}' is missing a string 'path' or model identifier"
                )

            resolved_path = None
            if path_value and isinstance(path_value, str):
                raw_path = Path(path_value).expanduser()
                cfg_relative = (path.parent / path_value).expanduser()
                if raw_path.exists():
                    resolved_path = str(raw_path.resolve())
                elif cfg_relative.exists():
                    resolved_path = str(cfg_relative.resolve())
                else:
                    # Keep as-is (likely a HuggingFace repo id or will error later)
                    resolved_path = path_value

            normalized.append(
                {
                    "name": name,
                    "description": ep.get("description", ""),
                    "type": ep_type,
                    "path": resolved_path,
                    "meta_template": ep.get("meta_template"),
                    "use_chat_template": bool(ep.get("use_chat_template", False)),
                    "max_new_tokens": ep.get("max_new_tokens", 512),
                    "model_kwargs": ep.get("model_kwargs") or {},
                    "env_path": ep.get("env_path") or ep.get("path"),
                }
            )
        return normalized

    def _build_endpoint_llms(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        llms: Dict[str, Any] = {}
        for ep in endpoints:
            name = ep["name"]
            if ep["type"] == "hf":
                template_name = ep.get("meta_template") or self.default_meta_template
                meta_template = meta_template_dict.get(template_name)
                if meta_template is None:
                    raise ValueError(
                        f"Unknown meta template '{template_name}' for endpoint '{name}'."
                    )
                model_kwargs = dict(ep.get("model_kwargs") or {})
                # Keep behavior consistent with other scripts: default to device_map="auto"
                model_kwargs.setdefault("device_map", "auto")
                max_new_tokens = ep.get("max_new_tokens", 512)
                path_str = ep["path"]
                if path_str is None:
                    raise ValueError(
                        f"Endpoint '{name}' is missing a path for HF model loading."
                    )
                path_obj = Path(path_str).expanduser()
                if (
                    path_obj.is_absolute() or path_str.startswith(".")
                ) and not path_obj.exists():
                    raise FileNotFoundError(
                        f"Local path for endpoint '{name}' not found: {path_str}"
                    )
                llm_cls = (
                    HFTransformerChat
                    if ep.get("use_chat_template")
                    else HFTransformerCasualLM
                )
                llms[name] = llm_cls(
                    path=path_str,
                    meta_template=meta_template,
                    max_new_tokens=max_new_tokens,
                    model_kwargs=model_kwargs,
                )
            elif ep["type"] == "api":
                llms[name] = GPTAPI(ep["path"], **(ep.get("model_kwargs") or {}))
            elif ep["type"] == "azure":
                llms[name] = AzureOpenAIOrchestrator(env_path=ep.get("env_path"))
            else:  # pragma: no cover - guarded by validation
                raise ValueError(f"Unsupported endpoint type: {ep['type']}")
        return llms

    def _default_router_prompt(self, endpoints: List[Dict[str, Any]]) -> str:
        parts = [
            "You are a routing assistant. Choose exactly one endpoint from the list below to answer the user.",
            "Respond with only the endpoint name (verbatim) and nothing else.",
            "Available endpoints:",
        ]
        for ep in endpoints:
            desc = ep.get("description") or "No description provided."
            parts.append(f"- {ep['name']}: {desc}")
        return "\n".join(parts)

    def _build_routing_messages(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        rendered_history = self._render_history(history)
        context_block = (
            f"<conversation_context>\n{rendered_history}\n</conversation_context>"
        )
        system_content = f"{self.router_system_prompt}\n{context_block}"
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": "Select the best endpoint and reply with only its name.",
            },
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
            r"```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.IGNORECASE | re.DOTALL
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

    def __repr__(self) -> str:
        endpoint_list = ", ".join(self.endpoint_llms.keys())
        return f"NetworkOrchestrator(router={self.router_llm}, endpoints=[{endpoint_list}])"
