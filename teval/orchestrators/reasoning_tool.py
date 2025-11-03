from __future__ import annotations

import copy
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

from .azure_openai import AzureOpenAIOrchestrator
from .base import BaseOrchestrator


class ReasoningAsToolOrchestrator(BaseOrchestrator):
    """
    Orchestrator that allows a primary model to delegate to an Azure helper model.

    The primary (typically lightweight) model receives a system prompt encouraging it
    to output either a direct answer (`FINAL_ANSWER: ...`) or request escalation by
    emitting `CALL_FOR_HELP` (optionally with justification). When escalation occurs,
    this orchestrator forwards the original message history to an Azure
    helper model and returns the helper's answer directly.

    Args:
        llm: Primary language model instance with a ``chat`` method.
        helper_env_path: Optional path to a .env file containing Azure credentials.
        system_prompt: Optional override for the injected system prompt.
        call_token: Token used by the primary model to request help (default: ``CALL_FOR_HELP``).
        final_token: Token used by the primary model to give final answers (default: ``FINAL_ANSWER``).
        **kwargs: Additional params forwarded to :class:`BaseOrchestrator`.
    """

    DEFAULT_CALL_TOKEN = "CALL_FOR_HELP"
    DEFAULT_FINAL_TOKEN = "FINAL_ANSWER"

    def __init__(
        self,
        llm,
        helper_env_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        call_token: str = DEFAULT_CALL_TOKEN,
        final_token: str = DEFAULT_FINAL_TOKEN,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, **kwargs)
        self.call_token = call_token
        self.final_token = final_token
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.helper = AzureOpenAIOrchestrator(env_path=helper_env_path)

        self._final_pattern = re.compile(
            rf"^{re.escape(self.final_token)}\s*:\s*(.+)$", re.IGNORECASE
        )
        self._call_pattern = re.compile(
            rf"^{re.escape(self.call_token)}(?:\s*:\s*(.*))?$", re.IGNORECASE
        )

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Run the primary model, optionally delegate to the helper, and return answers.
        """
        histories, was_single = self._normalize_input(message_histories)
        prompted_histories = [self._inject_prompt(history) for history in histories]

        start_primary = perf_counter()
        primary_responses = self.llm.chat(prompted_histories, **kwargs)
        primary_elapsed = perf_counter() - start_primary
        primary_trace = getattr(self.llm, "last_trace", []) or []

        results: List[str] = []
        traces: List[Dict[str, Any]] = []
        helper_requests: List[Tuple[int, List[Dict[str, str]], str]] = []

        for idx, (original_history, prompted_history, response_text) in enumerate(
            zip(histories, prompted_histories, primary_responses)
        ):
            parsed = self._parse_primary_response(response_text)
            trace_entry: Dict[str, Any] = {
                "strategy": "reasoning_as_tool",
                "config": {
                    "call_token": self.call_token,
                    "final_token": self.final_token,
                },
                "original_messages": original_history,
                "primary_response": response_text,
                "total_elapsed_seconds": primary_elapsed,
                "steps": [
                    {
                        "type": "primary_llm_call",
                        "messages": prompted_history,
                        "response": response_text,
                        "elapsed_seconds": primary_elapsed,
                        "parsed": parsed.to_dict(),
                    }
                ],
                "escalated": parsed.escalate,
            }

            if idx < len(primary_trace) and primary_trace[idx]:
                trace_entry["underlying_primary_trace"] = primary_trace[idx]

            if parsed.escalate:
                helper_requests.append((idx, original_history, parsed.helper_payload))
                results.append("")  # placeholder to be filled later
            else:
                final_text = parsed.final_answer or (
                    response_text if isinstance(response_text, str) else str(response_text)
                )
                results.append(final_text.strip())

            traces.append(trace_entry)

        if helper_requests:
            helper_indices, helper_histories, helper_payloads = zip(*helper_requests)

            helper_start = perf_counter()
            helper_responses = self.helper.completion(list(helper_histories), **kwargs)
            helper_elapsed = perf_counter() - helper_start
            helper_trace = getattr(self.helper, "last_trace", []) or []

            for local_idx, (result_index, history, helper_response) in enumerate(
                zip(helper_indices, helper_histories, helper_responses)
            ):
                results[result_index] = (
                    helper_response if isinstance(helper_response, str) else str(helper_response)
                ).strip()
                step_payload = {
                    "type": "helper_llm_call",
                    "messages": history,
                    "response": helper_response,
                    "elapsed_seconds": helper_elapsed,
                    "forwarded_context": "original_history",
                    "delegate_note": helper_payloads[local_idx],
                }
                if local_idx < len(helper_trace) and helper_trace[local_idx]:
                    step_payload["underlying_helper_trace"] = helper_trace[local_idx]
                traces[result_index]["steps"].append(step_payload)
                traces[result_index]["helper_response"] = helper_response
                traces[result_index]["total_elapsed_seconds"] += helper_elapsed

        self._record_trace(traces)
        return self._denormalize_output(results, was_single)

    def _inject_prompt(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Prepend or merge the Reasoning-as-a-Tool system prompt.
        """
        history_copy = copy.deepcopy(history)
        prompt_message = {"role": "system", "content": self.system_prompt}

        if history_copy and history_copy[0].get("role") == "system":
            merged = copy.deepcopy(history_copy[0])
            original = merged.get("content", "")
            merged["content"] = f"{self.system_prompt}\n\n{original}".strip()
            return [merged] + history_copy[1:]
        return [prompt_message] + history_copy

    def _parse_primary_response(self, response: Any) -> "ParsedPrimaryResponse":
        """
        Examine the primary model response and determine next actions.
        """
        if not isinstance(response, str):
            return ParsedPrimaryResponse(escalate=False, raw_response=str(response))

        final_match = None
        call_match = None
        for line in response.splitlines():
            if final_match is None:
                final_match = self._final_pattern.match(line.strip())
            if call_match is None:
                call_match = self._call_pattern.match(line.strip())

        if final_match:
            return ParsedPrimaryResponse(
                escalate=False,
                final_answer=final_match.group(1).strip(),
                raw_response=response,
            )

        if call_match:
            payload = call_match.group(1).strip() if call_match.group(1) else ""
            return ParsedPrimaryResponse(
                escalate=True,
                helper_payload=payload,
                raw_response=response,
            )

        return ParsedPrimaryResponse(escalate=False, raw_response=response)

    def _default_system_prompt(self) -> str:
        """
        Build the default system prompt instructing the primary model how to delegate.
        """
        return (
            "You are an efficient problem solver with access to a much stronger Azure-based helper model. "
            "Decide whether you can answer the user's request on your own.\n"
            f"- If you can solve it, respond with a single line `{self.final_token}: <your answer>`.\n"
            f"- If it requires advanced reasoning or you are uncertain, emit `{self.call_token}` "
            "on its own line (optionally with a short justification after a colon) to delegate the task.\n"
            "Do not provide any additional text after the final decision line."
        )


class ParsedPrimaryResponse:
    """
    Helper container for parsing the primary model response.
    """

    def __init__(
        self,
        escalate: bool,
        final_answer: Optional[str] = None,
        helper_payload: Optional[str] = None,
        raw_response: Optional[str] = None,
    ) -> None:
        self.escalate = escalate
        self.final_answer = final_answer
        self.helper_payload = helper_payload or ""
        self.raw_response = raw_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escalate": self.escalate,
            "final_answer": self.final_answer,
            "helper_payload": self.helper_payload,
            "raw_response": self.raw_response,
        }
