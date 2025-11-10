from time import perf_counter
from typing import Any, Dict, List, Optional, Union

from .base import BaseOrchestrator


class ReActOrchestrator(BaseOrchestrator):
    """
    Minimal ReAct orchestrator that prepends a fixed system prompt.

    The underlying LLM handles all Reason + Act behavior. This orchestrator only
    injects the canonical ReAct instructions as the first system message.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a ReAct (Reason + Act) agent. Solve the user's task by iterating "
        "through Thought, Action, and Observation steps. When you need to call a tool, "
        "emit `Action: <tool_name>[<json_args>]`. After each tool call, write "
        "`Observation:` with the tool result. Conclude with `Final Answer:` followed by "
        "the solution for the user. Do not fabricate observations."
    )

    def __init__(
        self,
        llm,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, **kwargs)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Prepend the ReAct system prompt and delegate directly to the LLM.
        """
        histories, was_single = self._normalize_input(message_histories)
        augmented_histories = [self._prepend_system_prompt(history) for history in histories]

        call_start = perf_counter()
        responses = self.llm.chat(augmented_histories, **kwargs)
        elapsed = perf_counter() - call_start

        model_name = (
            getattr(self.llm, "model_name", None)
            or getattr(self.llm, "path", None)
            or self.llm.__class__.__name__
        )

        traces = []
        for history, response in zip(augmented_histories, responses):
            traces.append(
                {
                    "strategy": "react",
                    "model": model_name,
                    "total_elapsed_seconds": elapsed,
                    "steps": [
                        {
                            "type": "llm_call",
                            "messages": history,
                            "response": response,
                            "elapsed_seconds": elapsed,
                        }
                    ],
                }
            )

        self._record_trace(traces)
        return self._denormalize_output(responses, was_single)

    def _prepend_system_prompt(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return a new history with the ReAct prompt at the front."""
        prompt_message = {"role": "system", "content": self.system_prompt}
        # Copy the history to avoid mutating the caller's list.
        new_history = [prompt_message]
        new_history.extend(history)
        return new_history

