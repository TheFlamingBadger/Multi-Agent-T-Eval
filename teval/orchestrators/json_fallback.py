from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

from .azure_openai import AzureOpenAIOrchestrator
from .base import BaseOrchestrator


class JsonFallbackOrchestrator(BaseOrchestrator):
    """
    Orchestrator that first queries a local primary model and only escalates to
    an Azure-hosted helper when the primary response cannot be parsed as JSON.

    The orchestration trace mirrors :class:`ReasoningAsToolOrchestrator`,
    capturing both attempts and recording the specific parse failure that
    triggered the fallback.
    """

    def __init__(
        self,
        primary_llm,
        helper_env_path: Optional[str] = None,
        strip_code_fence: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            primary_llm: Locally running small language model (provides ``chat``).
            helper_env_path: Optional path to the Azure OpenAI credential file.
            strip_code_fence: Whether to remove ```json fences before parsing.
            **kwargs: Forwarded to :class:`BaseOrchestrator`.
        """
        super().__init__(primary_llm, **kwargs)
        self.strip_code_fence = strip_code_fence
        self.helper = AzureOpenAIOrchestrator(env_path=helper_env_path)

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        histories, was_single = self._normalize_input(message_histories)

        primary_start = perf_counter()
        primary_responses = self.llm.chat(histories, **kwargs)
        primary_elapsed = perf_counter() - primary_start
        primary_trace = getattr(self.llm, "last_trace", []) or []

        results: List[str] = []
        traces: List[Dict[str, Any]] = []

        fallback_indices: List[int] = []
        fallback_histories: List[List[Dict[str, str]]] = []
        parse_attempts: List[Dict[str, Any]] = []

        for idx, (history, response_text) in enumerate(
            zip(histories, primary_responses)
        ):
            normalized_text, parse_result = self._attempt_parse(response_text)
            parse_attempts.append(parse_result)

            final_text = (
                normalized_text if parse_result["ok"] else str(response_text)
            )
            if parse_result["ok"]:
                results.append(final_text.strip())
            else:
                results.append("")  # placeholder, will fill after fallback
                fallback_indices.append(idx)
                fallback_histories.append(history)

            step_payload = {
                "type": "primary_llm_call",
                "messages": history,
                "response": response_text,
                "elapsed_seconds": primary_elapsed,
                "parse_attempt": parse_result,
            }
            trace_entry: Dict[str, Any] = {
                "strategy": "json_fallback",
                "config": {
                    "strip_code_fence": self.strip_code_fence,
                },
                "original_messages": history,
                "total_elapsed_seconds": primary_elapsed,
                "steps": [step_payload],
                "fallback_triggered": not parse_result["ok"],
            }
            if idx < len(primary_trace) and primary_trace[idx]:
                trace_entry["underlying_primary_trace"] = primary_trace[idx]

            traces.append(trace_entry)

        if fallback_indices:
            helper_start = perf_counter()
            helper_responses = self.helper.completion(fallback_histories, **kwargs)
            helper_elapsed = perf_counter() - helper_start
            helper_trace = getattr(self.helper, "last_trace", []) or []

            for local_idx, result_index in enumerate(fallback_indices):
                helper_response = helper_responses[local_idx]
                results[result_index] = (
                    helper_response
                    if isinstance(helper_response, str)
                    else str(helper_response)
                ).strip()

                step_payload = {
                    "type": "helper_llm_call",
                    "messages": fallback_histories[local_idx],
                    "response": helper_response,
                    "elapsed_seconds": helper_elapsed,
                    "fallback_reason": parse_attempts[result_index],
                }
                if local_idx < len(helper_trace) and helper_trace[local_idx]:
                    step_payload["underlying_helper_trace"] = helper_trace[local_idx]

                traces[result_index]["steps"].append(step_payload)
                traces[result_index]["helper_response"] = helper_response
                traces[result_index]["total_elapsed_seconds"] += helper_elapsed

        self._record_trace(traces)
        return self._denormalize_output(results, was_single)

    def _attempt_parse(self, response: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Try to parse the response as JSON; return normalized text and metadata.
        """
        result = {
            "ok": False,
            "error_type": None,
            "error_message": None,
            "normalized_text": None,
        }

        if not isinstance(response, str):
            normalized = str(response)
        else:
            normalized = response

        normalized = normalized.strip()
        if self.strip_code_fence and normalized.startswith("```"):
            normalized = self._strip_code_fence(normalized)

        try:
            json.loads(normalized)
        except Exception as exc:  # pragma: no cover - direct exception capture
            result["error_type"] = exc.__class__.__name__
            result["error_message"] = str(exc)
            result["normalized_text"] = normalized
            return normalized, result

        result["ok"] = True
        result["normalized_text"] = normalized
        return normalized, result

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Remove surrounding ``` fences (optionally tagged with 'json').
        """
        if not text.startswith("```"):
            return text
        stripped = text[3:]
        if stripped.startswith("json"):
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()
