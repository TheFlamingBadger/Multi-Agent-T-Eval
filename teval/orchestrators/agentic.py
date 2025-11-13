import copy
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import BaseOrchestrator
from .reasoning_tool import ReasoningAsToolOrchestrator


class AgenticOrchestrator(BaseOrchestrator):
    """
    Orchestrator that enforces a self-check loop before finalizing a response.
    
    Workflow:
        1. Call the underlying LLM once to obtain an initial answer.
        2. Issue a follow-up call with an additional skeptical system prompt that
           reviews the previous answer. The model must either emit a special
           acceptance token or return a corrected answer.
        3. The review step repeats until the model emits the acceptance token or
           the maximum number of review cycles is reached.
    """

    MAX_REVIEW_CYCLES = 3
    ACCEPT_TOKEN = "<RETURN_ORIGINAL>"
    DEFAULT_REVIEW_SYSTEM_PROMPT = (
        "You are a meticulous and skeptical reviewer. "
        "Double-check assistant answers for factual accuracy, completeness, and safety. "
        "Only allow answers that are provably correct."
    )
    DEFAULT_REVIEW_USER_PROMPT = (
        "Evaluate the assistant reply you just gave. "
        "If it fully satisfies the request with no issues, respond with exactly {accept_token}. "
        "If anything is incorrect, incomplete, or unsafe, provide a corrected and complete answer now. "
        "Do not mention the acceptance token when giving corrections."
    )

    def __init__(
        self,
        llm,
        review_system_prompt: Optional[str] = None,
        review_user_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.review_system_prompt = review_system_prompt or self.DEFAULT_REVIEW_SYSTEM_PROMPT
        default_user_prompt = self.DEFAULT_REVIEW_USER_PROMPT.format(accept_token=self.ACCEPT_TOKEN)
        self.review_user_prompt = review_user_prompt or default_user_prompt

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs,
    ) -> List[str]:
        histories, was_single = self._normalize_input(message_histories)
        final_responses: List[str] = []
        traces: List[Dict[str, object]] = []

        for history in histories:
            response, trace = self._run_agentic_cycle(history, **kwargs)
            final_responses.append(response)
            traces.append(trace)

        self._record_trace(traces)
        return self._denormalize_output(final_responses, was_single)

    def _run_agentic_cycle(
        self,
        history: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, Dict[str, object]]:
        steps: List[Dict[str, object]] = []
        total_elapsed = 0.0

        # Initial generation (no extra prompts)
        initial_history = copy.deepcopy(history)
        initial_response, elapsed, call_trace = self._call_llm(initial_history, **kwargs)
        total_elapsed += elapsed
        steps.append(
            {
                "phase": "initial_generation",
                "messages": initial_history,
                "response": initial_response,
                "elapsed_seconds": elapsed,
                "underlying_trace": call_trace,
            }
        )

        candidate = initial_response
        accepted = False

        for iteration in range(1, self.MAX_REVIEW_CYCLES + 1):
            review_history = self._build_review_history(history, candidate)
            review_response, elapsed, call_trace = self._call_llm(review_history, **kwargs)
            total_elapsed += elapsed
            accepted = self._contains_accept_token(review_response)
            steps.append(
                {
                    "phase": "self_check",
                    "iteration": iteration,
                    "messages": review_history,
                    "response": review_response,
                    "elapsed_seconds": elapsed,
                    "accepted": accepted,
                    "underlying_trace": call_trace,
                }
            )

            if accepted:
                final_response = candidate
                break
            candidate = review_response
        else:
            final_response = candidate

        trace = {
            "strategy": self._trace_strategy(),
            "config": self._trace_config(),
            "original_messages": history,
            "total_elapsed_seconds": total_elapsed,
            "steps": steps,
            "accepted": accepted,
        }

        return final_response, trace

    def _build_review_history(
        self,
        original_history: List[Dict[str, str]],
        candidate_response: str,
    ) -> List[Dict[str, str]]:
        review_history = copy.deepcopy(original_history)
        review_history.insert(
            0,
            {
                "role": "system",
                "content": self.review_system_prompt,
            },
        )
        review_history.append(
            {
                "role": "assistant",
                "content": candidate_response,
            }
        )
        review_history.append(
            {
                "role": "user",
                "content": self.review_user_prompt,
            }
        )
        return review_history

    def _trace_strategy(self) -> str:
        return "agentic"

    def _trace_config(self) -> Dict[str, Any]:
        return {
            "max_review_cycles": self.MAX_REVIEW_CYCLES,
            "accept_token": self.ACCEPT_TOKEN,
            "review_system_prompt": self.review_system_prompt,
        }

    def _contains_accept_token(self, response: str) -> bool:
        stripped = response.strip()
        return stripped == self.ACCEPT_TOKEN or stripped.startswith(f"{self.ACCEPT_TOKEN}\n")

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, float, Optional[Dict[str, object]]]:
        start = perf_counter()
        response = self.llm.chat([messages], **kwargs)[0]
        elapsed = perf_counter() - start
        call_trace = self._extract_underlying_trace()
        return response, elapsed, call_trace

    def _extract_underlying_trace(self) -> Optional[Dict[str, object]]:
        raw_trace = getattr(self.llm, "last_trace", None)
        if isinstance(raw_trace, list) and raw_trace:
            return raw_trace[0]
        if isinstance(raw_trace, dict):
            return raw_trace
        return None


class AgenticReasoningToolOrchestrator(AgenticOrchestrator):
    """
    Agentic orchestrator that routes every model call through ReasoningAsTool.

    Each agentic generation (initial + review passes) gains access to the helper
    model for escalation via CALL_FOR_HELP, while the outer loop still enforces
    the self-check acceptance token logic.
    """

    def __init__(
        self,
        llm,
        helper_env_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        call_token: str = ReasoningAsToolOrchestrator.DEFAULT_CALL_TOKEN,
        final_token: str = ReasoningAsToolOrchestrator.DEFAULT_FINAL_TOKEN,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.reasoning_tool = ReasoningAsToolOrchestrator(
            llm,
            helper_env_path=helper_env_path,
            system_prompt=system_prompt,
            call_token=call_token,
            final_token=final_token,
        )
        self._reasoning_config = {
            "call_token": call_token,
            "final_token": final_token,
            "system_prompt": self.reasoning_tool.system_prompt,
        }

    def _trace_strategy(self) -> str:
        return "agentic_reasoning_tool"

    def _trace_config(self) -> Dict[str, Any]:
        config = super()._trace_config()
        config.update(
            {
                "reasoning_tool": {
                    "call_token": self._reasoning_config["call_token"],
                    "final_token": self._reasoning_config["final_token"],
                    "system_prompt": self._reasoning_config["system_prompt"],
                }
            }
        )
        return config

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, float, Optional[Dict[str, object]]]:
        start = perf_counter()
        response = self.reasoning_tool.completion([messages], **kwargs)[0]
        elapsed = perf_counter() - start
        call_trace = self._extract_reasoning_trace()
        return response, elapsed, call_trace

    def _extract_reasoning_trace(self) -> Optional[Dict[str, object]]:
        raw_trace = getattr(self.reasoning_tool, "last_trace", None)
        if isinstance(raw_trace, list) and raw_trace:
            return raw_trace[0]
        if isinstance(raw_trace, dict):
            return raw_trace
        return None
