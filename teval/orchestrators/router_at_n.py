import json
import re
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from .azure_openai import AzureOpenAIOrchestrator
from .routing import RoutingOrchestrator


class RouterAtNOrchestrator(RoutingOrchestrator):
    """
    Routing orchestrator that queries the router multiple times and only
    selects the small language model if every vote chooses it.
    """

    def __init__(
        self,
        router_llm,
        helper_env_path: Optional[str] = None,
        router_system_prompt: Optional[str] = None,
        use_naive_prompt: bool = False,
        use_rubric_prompt: bool = False,
        router_votes: int = 3,
        **kwargs,
    ) -> None:
        """
        Args:
            router_llm: Small language model used for routing and as the small
                completion path.
            helper_env_path: Optional path to Azure credentials (for large
                model path).
            router_system_prompt: Optional override for the routing system
                prompt.
            use_naive_prompt: If True, use the original minimal routing prompt
                (without scores).
            use_rubric_prompt: If True, use the rubric-based routing prompt.
            router_votes: Number of router queries to run. The request is sent
                to the small model only if all votes select it.
            **kwargs: Forwarded to BaseOrchestrator.
        """
        super().__init__(
            router_llm,
            helper_env_path=helper_env_path,
            router_system_prompt=router_system_prompt,
            use_naive_prompt=use_naive_prompt,
            use_rubric_prompt=use_rubric_prompt,
            **kwargs,
        )
        self.router_votes = max(1, int(router_votes))
        self.router_llm = router_llm
        self.small_completion_llm = router_llm
        self.large_completion_llm = AzureOpenAIOrchestrator(env_path=helper_env_path)

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs,
    ) -> List[str]:
        histories, was_single = self._normalize_input(message_histories)

        results: List[str] = []
        traces: List[Dict[str, object]] = []

        for history in histories:
            routing_messages = self._build_routing_messages(history)
            routing_steps: List[Dict[str, object]] = []
            all_small_votes = True
            small_vote_count = 0
            large_vote_count = 0
            any_invalid = False

            for vote_idx in range(self.router_votes):
                routing_start = perf_counter()
                routing_raw = self.router_llm.chat(
                    [routing_messages], do_sample=False, temperature=0
                )[0]
                routing_elapsed = perf_counter() - routing_start

                choice, invalid_reason = self._extract_choice(routing_raw)
                selected_path = choice or "large language model"
                is_invalid = invalid_reason is not None

                if not is_invalid and selected_path == "small language model":
                    small_vote_count += 1
                else:
                    large_vote_count += 1
                    all_small_votes = False
                any_invalid = any_invalid or is_invalid

                routing_steps.append(
                    {
                        "type": "routing_llm_call",
                        "iteration": vote_idx + 1,
                        "messages": routing_messages,
                        "response": routing_raw,
                        "selection": selected_path,
                        "invalid_routing_output": is_invalid,
                        "elapsed_seconds": routing_elapsed,
                    }
                )

            if all_small_votes:
                completion_start = perf_counter()
                completion_text = self.small_completion_llm.chat([history], **kwargs)[0]
                completion_elapsed = perf_counter() - completion_start
                underlying_trace = getattr(self.small_completion_llm, "last_trace", [])
                step_type = "small_llm_call"
                selected_path = "small language model"
            else:
                completion_start = perf_counter()
                completion_text = self.large_completion_llm.completion(
                    [history], **kwargs
                )[0]
                completion_elapsed = perf_counter() - completion_start
                underlying_trace = getattr(self.large_completion_llm, "last_trace", [])
                step_type = "large_llm_call"
                selected_path = "large language model"

            results.append(completion_text)

            completion_step = {
                "type": step_type,
                "messages": history,
                "response": completion_text,
                "elapsed_seconds": completion_elapsed,
            }
            if underlying_trace:
                completion_step["underlying_trace"] = underlying_trace[0]

            total_elapsed = completion_elapsed + sum(
                step["elapsed_seconds"] for step in routing_steps
            )

            traces.append(
                {
                    "strategy": "routing_at_n",
                    "selection": selected_path,
                    "invalid_routing_output": any_invalid,
                    "votes": {
                        "required_all_small": True,
                        "router_votes": self.router_votes,
                        "small_votes": small_vote_count,
                        "large_or_invalid_votes": large_vote_count,
                    },
                    "total_elapsed_seconds": total_elapsed,
                    "steps": routing_steps + [completion_step],
                }
            )

        self._record_trace(traces)
        return self._denormalize_output(results, was_single)

    def _extract_json_route(self, text: str) -> Optional[str]:
        # Copy of parent method, kept local to avoid depending on protected APIs.
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
        route = data.get("route")
        if not isinstance(route, str):
            return None
        normalized = re.sub(r"[\s_]+", " ", route).strip().lower()
        if normalized in {"slm", "small language model"} or "small" in normalized:
            return "small language model"
        if normalized in {"llm", "large language model"} or "large" in normalized:
            return "large language model"
        return None

    def __repr__(self) -> str:
        return f"RouterAtNOrchestrator(router_votes={self.router_votes})"
