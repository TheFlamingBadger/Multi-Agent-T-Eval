import json
import re
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from .azure_openai import AzureOpenAIOrchestrator
from .base import BaseOrchestrator


class RoutingOrchestrator(BaseOrchestrator):
    """
    Orchestrator that routes each request to either a small local model or a
    larger Azure-hosted model based on a lightweight routing step.
    """

    def __init__(
        self,
        router_llm,
        helper_env_path: Optional[str] = None,
        router_system_prompt: Optional[str] = None,
        use_naive_prompt: bool = False,
        use_rubric_prompt: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            router_llm: Small language model used for both routing and the
                small-model completion path.
            helper_env_path: Optional path to Azure credentials (for large
                model path).
            router_system_prompt: Optional override for the routing system
                prompt.
            use_naive_prompt: If True, use the original minimal routing prompt
                (without scores) to retain previous behavior.
            use_rubric_prompt: If True, use the rubric-based routing prompt.
            **kwargs: Forwarded to BaseOrchestrator.
        """
        super().__init__(router_llm, **kwargs)
        self.router_llm = router_llm
        self.small_completion_llm = router_llm
        self.large_completion_llm = AzureOpenAIOrchestrator(env_path=helper_env_path)
        if use_naive_prompt and use_rubric_prompt:
            raise ValueError("Only one routing prompt variant can be enabled.")
        self.router_prompt_variant = "default"
        if router_system_prompt:
            self.router_system_prompt = router_system_prompt
            self.router_prompt_variant = "custom"
        elif use_rubric_prompt:
            self.router_system_prompt = self._rubric_router_prompt()
            self.router_prompt_variant = "rubric"
        elif use_naive_prompt:
            self.router_system_prompt = self._naive_router_prompt()
            self.router_prompt_variant = "naive"
        else:
            self.router_system_prompt = self._default_router_prompt()

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
            routing_start = perf_counter()
            routing_raw = self.router_llm.chat(
                [routing_messages], do_sample=False, temperature=0
            )[0]
            routing_elapsed = perf_counter() - routing_start

            choice, invalid_reason = self._extract_choice(routing_raw)
            selected_path = choice or "large language model"
            is_invalid = invalid_reason is not None

            if selected_path == "small language model":
                completion_start = perf_counter()
                completion_text = self.small_completion_llm.chat([history], **kwargs)[0]
                completion_elapsed = perf_counter() - completion_start
                underlying_trace = getattr(self.small_completion_llm, "last_trace", [])
                step_type = "small_llm_call"
            else:
                completion_start = perf_counter()
                completion_text = self.large_completion_llm.completion(
                    [history], **kwargs
                )[0]
                completion_elapsed = perf_counter() - completion_start
                underlying_trace = getattr(self.large_completion_llm, "last_trace", [])
                step_type = "large_llm_call"

            results.append(completion_text)

            routing_step = {
                "type": "routing_llm_call",
                "messages": routing_messages,
                "response": routing_raw,
                "elapsed_seconds": routing_elapsed,
            }
            completion_step = {
                "type": step_type,
                "messages": history,
                "response": completion_text,
                "elapsed_seconds": completion_elapsed,
            }
            if underlying_trace:
                completion_step["underlying_trace"] = underlying_trace[0]

            traces.append(
                {
                    "strategy": "routing",
                    "selection": selected_path,
                    "invalid_routing_output": is_invalid,
                    "total_elapsed_seconds": routing_elapsed + completion_elapsed,
                    "steps": [routing_step, completion_step],
                }
            )

        self._record_trace(traces)
        return self._denormalize_output(results, was_single)

    def _default_router_prompt(self) -> str:
        return (
            "You are a routing assistant. Read the entire conversation and pick exactly one model for the final answer:"
            ' "small language model" or "large language model". '
            "Send to the large language model when the request needs multi-step reasoning, outside knowledge/citations, critique/review, long or multi-part context, ambiguous goals, or non-trivial code/math. "
            "Send to the small language model when the ask is short and concrete: direct instructions, simple formatting, extraction, rewriting, summarizing what is already in the prompt, or filling a template. "
            "If the task is not clearly in the hard cases above, default to the small language model to save cost. "
            "Respond with only the chosen model name. Do not explain, justify, or answer the user's question."
        )

    def _naive_router_prompt(self) -> str:
        return (
            "You are a routing assistant. Read the entire conversation and pick exactly one model to respond to the user's query"
            'Pick either: "small language model" or "large language model". '
            "The small language model is much cheaper but suited only to very simple tasks "
            "The large language model is expensive but much smarter, suited to tasks with long contexts or those requiring reasoning"
            "The conversation context is as follows: "
            "Respond with only the chosen model name. Do not explain, justify, or answer the user's question."
        )

    def _rubric_router_prompt(self) -> str:
        return (
            "You are a routing model responsible for choosing whether a query should be handled by a Small Language Model (SLM) or by a Large Language Model (LLM).\n\n"
            "Your goal is to reliably score the user query on three difficulty axes and then determine the correct model based on the rubric below.\n\n"
            "---\n"
            "SCORING RUBRIC (0-3 each)\n\n"
            "1. Complexity (0-3)\n"
            "   - 0: Simple, single-step request. No reasoning required.\n"
            "   - 1: Mild reasoning. One or two steps; low cognitive load.\n"
            "   - 2: Multi-step reasoning, transformation, or non-trivial logic.\n"
            "   - 3: Deep or multi-hop reasoning, chain-of-thought needed, or tool-use complexity.\n\n"
            "2. Ambiguity (0-3)\n"
            "   - 0: Request is clear, specific, and objective.\n"
            "   - 1: Minor ambiguity or open-endedness.\n"
            "   - 2: Requires precision, factual correctness, or domain knowledge.\n"
            "   - 3: High ambiguity, specialized factual recall, or high error sensitivity.\n\n"
            "3. Constraint Sensitivity (0-3)\n"
            "   - 0: Free-form response, no format constraints.\n"
            "   - 1: Light structure (lists, short templates).\n"
            "   - 2: Strict formatting or structured outputs (JSON, API args).\n"
            "   - 3: Highly rigid schemas or multi-field arguments with correctness checks.\n\n"
            "---\n"
            "DECISION RULE\n\n"
            "Compute:\n"
            "  TOTAL_SCORE = Complexity + Ambiguity + ConstraintSensitivity\n\n"
            'If TOTAL_SCORE >= 6 -> route to "llm".\n'
            'If TOTAL_SCORE <= 5 -> route to "slm".\n\n'
            "When uncertain, err toward the LLM.\n\n"
            "---\n"
            "OUTPUT FORMAT\n\n"
            "Respond ONLY with a JSON object in the exact structure:\n\n"
            "{\n"
            '  "complexity": <0-3>,\n'
            '  "ambiguity": <0-3>,\n'
            '  "constraint_sensitivity": <0-3>,\n'
            '  "total": <sum>,\n'
            '  "route": "large_language_modeel" | "small_language_modeel"\n'
            "}\n\n"
            "---\n"
            "CONTEXT\n\n"
            "<insert context/>\n"
        )

    def _build_routing_messages(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        rendered_history = self._render_history(history)
        context_block = (
            f"<conversation_context>\n{rendered_history}\n</conversation_context>"
        )
        system_prompt = self.router_system_prompt
        if "<insert context/>" in system_prompt:
            system_content = system_prompt.replace("<insert context/>", context_block)
        else:
            system_content = f"{system_prompt}\n{context_block}"
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    "Score the conversation and return only the required JSON response."
                    if self.router_prompt_variant == "rubric"
                    else "Choose the model now and output only one allowed model name. No justification."
                ),
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
        """
        Parse routing output to identify a single allowed label.

        Returns (choice, invalid_reason). If invalid_reason is not None, choice
        may be None.
        """
        if not isinstance(text, str):
            return None, "routing output was not a string"
        json_choice = self._extract_json_route(text)
        if json_choice:
            return json_choice, None
        matches = re.findall(
            r"\b(small language model|large language model)\b",
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
        return "RoutingOrchestrator(router=small_llm, large=AzureOpenAI)"
