from __future__ import annotations

import copy
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import BaseOrchestrator


class ReActOrchestrator(BaseOrchestrator):
    """
    Orchestrator implementing the ReAct (Reason + Act) prompting framework.

    This orchestrator injects a ReAct system prompt, instructing the model to
    interleave explicit reasoning (`THOUGHT`) with tool usage (`ACTION`) and
    environment feedback (`OBSERVATION`). The model must conclude with a
    `FINAL_ACTION` line that contains the answer to return to the caller.

    The orchestrator captures the full reasoning trace while returning only the
    final action text from :meth:`completion`.

    Args:
        llm: Underlying language model that provides a ``chat`` method.
        thought_tag: Tag used to mark internal reasoning lines. Default: ``THOUGHT``.
        action_tag: Tag used to mark action lines. Default: ``ACTION``.
        observation_tag: Tag used for observation lines. Default: ``OBSERVATION``.
        final_action_tag: Tag used for the completion line. Default: ``FINAL_ACTION``.
        react_system_prompt: Optional custom ReAct system prompt. When omitted a
            default prompt is generated from the tag configuration.
        max_reasoning_steps: Optional advisory limit described to the model in
            the system prompt to discourage excessive looping.
        **kwargs: Forwarded to :class:`BaseOrchestrator`.
    """

    DEFAULT_THOUGHT_TAG = "THOUGHT"
    DEFAULT_ACTION_TAG = "ACTION"
    DEFAULT_OBSERVATION_TAG = "OBSERVATION"
    DEFAULT_FINAL_ACTION_TAG = "FINAL_ACTION"

    def __init__(
        self,
        llm,
        thought_tag: str = DEFAULT_THOUGHT_TAG,
        action_tag: str = DEFAULT_ACTION_TAG,
        observation_tag: str = DEFAULT_OBSERVATION_TAG,
        final_action_tag: str = DEFAULT_FINAL_ACTION_TAG,
        react_system_prompt: Optional[str] = None,
        max_reasoning_steps: Optional[int] = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, **kwargs)
        self.thought_tag = thought_tag
        self.action_tag = action_tag
        self.observation_tag = observation_tag
        self.final_action_tag = final_action_tag
        self.max_reasoning_steps = max_reasoning_steps
        self.react_system_prompt = react_system_prompt or self._build_default_prompt()
        # Precompile regex for parsing final action.
        final_tag_pattern = re.escape(self.final_action_tag)
        self._final_action_regex = re.compile(
            rf"^{final_tag_pattern}\s*:\s*(.+)$", re.IGNORECASE
        )

    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate completions while extracting only the final action for each item.

        The reasoning trace emitted by the model is preserved in ``last_trace``
        for downstream inspection.
        """
        histories, was_single = self._normalize_input(message_histories)
        react_histories = [self._inject_react_prompt(history) for history in histories]

        call_start = perf_counter()
        responses = self.llm.chat(react_histories, **kwargs)
        elapsed = perf_counter() - call_start
        underlying_trace = getattr(self.llm, "last_trace", []) or []

        final_actions: List[str] = []
        traces: List[Dict[str, Any]] = []

        for index, (original_history, react_history, raw_response) in enumerate(
            zip(histories, react_histories, responses)
        ):
            reasoning, final_action = self._parse_react_response(raw_response)
            structured_reasoning = self._extract_tagged_segments(reasoning)
            # Fallback to raw response if parsing failed.
            if final_action is not None:
                resolved_final = final_action
            else:
                resolved_final = raw_response if isinstance(raw_response, str) else str(raw_response)

            final_actions.append(resolved_final)

            trace_entry: Dict[str, Any] = {
                "strategy": "react",
                "config": {
                    "thought_tag": self.thought_tag,
                    "action_tag": self.action_tag,
                    "observation_tag": self.observation_tag,
                    "final_action_tag": self.final_action_tag,
                    "max_reasoning_steps": self.max_reasoning_steps,
                },
                "total_elapsed_seconds": elapsed,
                "original_messages": original_history,
                "steps": [
                    {
                        "type": "llm_call",
                        "messages": react_history,
                        "raw_response": raw_response,
                        "reasoning_trace": reasoning,
                        "reasoning_steps": structured_reasoning,
                        "final_action": resolved_final,
                        "elapsed_seconds": elapsed,
                    }
                ],
            }
            if index < len(underlying_trace) and underlying_trace[index]:
                trace_entry["underlying_trace"] = underlying_trace[index]
            traces.append(trace_entry)

        self._record_trace(traces)
        return self._denormalize_output(final_actions, was_single)

    def _build_default_prompt(self) -> str:
        """
        Construct the default ReAct system prompt using the configured tags.
        """
        advisory_clause = (
            f"You should solve most tasks within {self.max_reasoning_steps} reasoning steps."
            if self.max_reasoning_steps
            else "Use only the steps necessary to solve the task."
        )
        return (
            "You are an autonomous problem-solving assistant that follows the ReAct "
            "framework (Reason + Act). "
            f"{advisory_clause} Produce your output using the required XML-free tokens below:\n"
            f"1. Prefix every internal reasoning statement with `{self.thought_tag}: `.\n"
            f"2. When you choose an action, emit a single line `{self.action_tag}: <action and arguments>`.\n"
            f"3. Record environment feedback with `{self.observation_tag}: <observation>`.\n"
            f"4. After all reasoning, provide the user-facing result on a final line "
            f"`{self.final_action_tag}: <final answer for the user>`.\n"
            "Do not add any explanation, commentary, or additional text after the final action line."
        )

    def _inject_react_prompt(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Insert the ReAct system prompt at the beginning of the message history.

        The original history is left untouched via deep copy.
        """
        history_copy = copy.deepcopy(history)
        react_message = {"role": "system", "content": self.react_system_prompt}

        if history_copy and history_copy[0].get("role") == "system":
            combined = copy.deepcopy(history_copy[0])
            original_content = combined.get("content", "")
            combined["content"] = (
                f"{self.react_system_prompt}\n\n{original_content}".strip()
            )
            return [combined] + history_copy[1:]
        return [react_message] + history_copy

    def _parse_react_response(self, response: Any) -> Tuple[str, Optional[str]]:
        """
        Split a model response into reasoning trace and final action.
        """
        if not isinstance(response, str):
            return "", None

        lines = response.splitlines()
        final_action: Optional[str] = None
        final_index: Optional[int] = None

        for idx, line in enumerate(lines):
            match = self._final_action_regex.match(line.strip())
            if match:
                final_action = match.group(1).strip()
                final_index = idx
                break

        if final_index is None:
            reasoning_lines = lines
        else:
            reasoning_lines = lines[:final_index]

        reasoning = "\n".join(reasoning_lines).strip()
        return reasoning, final_action

    def _extract_tagged_segments(self, reasoning: str) -> List[Dict[str, str]]:
        """
        Break the reasoning text into structured segments keyed by tag.
        """
        if not reasoning:
            return []

        tag_pattern = "|".join(
            re.escape(tag)
            for tag in {
                self.thought_tag,
                self.action_tag,
                self.observation_tag,
            }
        )
        segment_regex = re.compile(
            rf"^\s*({tag_pattern})\s*:\s*(.+)$", re.IGNORECASE
        )

        segments: List[Dict[str, str]] = []
        for line in reasoning.splitlines():
            match = segment_regex.match(line.strip())
            if not match:
                continue
            tag, content = match.group(1).upper(), match.group(2).strip()
            segments.append(
                {
                    "tag": tag,
                    "content": content,
                }
            )
        return segments
