from __future__ import annotations

import copy
import json
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
        max_feedback_rounds: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm, **kwargs)
        self.thought_tag = thought_tag
        self.action_tag = action_tag
        self.observation_tag = observation_tag
        self.final_action_tag = final_action_tag
        self.max_reasoning_steps = max_reasoning_steps
        self.react_system_prompt = react_system_prompt or self._build_default_prompt()
        self.max_feedback_rounds = max(0, int(max_feedback_rounds))
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

        final_actions: List[str] = []
        traces: List[Dict[str, Any]] = []

        for index, original_history in enumerate(histories):
            react_history = self._inject_react_prompt(original_history)
            attempt_history = copy.deepcopy(react_history)
            expectations = self._infer_expectations(original_history)

            attempts_trace: List[Dict[str, Any]] = []
            accumulated_elapsed = 0.0
            resolved_final = ""
            resolved_reasoning = ""

            for attempt in range(self.max_feedback_rounds + 1):
                call_start = perf_counter()
                responses = self.llm.chat([attempt_history], **kwargs)
                call_elapsed = perf_counter() - call_start
                accumulated_elapsed += call_elapsed
                raw_response = responses[0]
                underlying_trace = getattr(self.llm, "last_trace", []) or []
                underlying = underlying_trace[0] if underlying_trace else None

                reasoning, final_action = self._parse_react_response(raw_response)
                structured_reasoning = self._extract_tagged_segments(reasoning)

                if final_action is not None:
                    candidate = final_action
                else:
                    candidate = raw_response if isinstance(raw_response, str) else str(raw_response)

                normalized_final, normalization_notes = self._normalize_final_action(candidate)
                validation = self._validate_final_action(
                    normalized_final, expectations
                )

                attempt_entry: Dict[str, Any] = {
                    "type": "llm_call",
                    "attempt_index": attempt + 1,
                    "messages": attempt_history,
                    "raw_response": raw_response,
                    "reasoning_trace": reasoning,
                    "reasoning_steps": structured_reasoning,
                    "final_action": normalized_final,
                    "normalization": normalization_notes,
                    "validation": validation,
                    "elapsed_seconds": call_elapsed,
                }
                if not validation["ok"] and validation.get("feedback"):
                    attempt_entry["feedback_prompt"] = validation["feedback"]
                if underlying:
                    attempt_entry["underlying_trace"] = underlying

                attempts_trace.append(attempt_entry)

                resolved_final = normalized_final
                resolved_reasoning = reasoning

                if validation["ok"]:
                    break

                if attempt < self.max_feedback_rounds:
                    feedback_prompt = validation.get("feedback")
                    if feedback_prompt:
                        attempt_history = attempt_history + [
                            {
                                "role": "assistant",
                                "content": raw_response if isinstance(raw_response, str) else str(raw_response),
                            },
                            {
                                "role": "user",
                                "content": feedback_prompt,
                            },
                        ]
                    else:
                        # No specific feedback available; stop looping.
                        break
                else:
                    break

            final_actions.append(resolved_final)

            trace_entry: Dict[str, Any] = {
                "strategy": "react",
                "config": {
                    "thought_tag": self.thought_tag,
                    "action_tag": self.action_tag,
                    "observation_tag": self.observation_tag,
                    "final_action_tag": self.final_action_tag,
                    "max_reasoning_steps": self.max_reasoning_steps,
                    "max_feedback_rounds": self.max_feedback_rounds,
                },
                "expectations": expectations,
                "total_elapsed_seconds": accumulated_elapsed,
                "original_messages": original_history,
                "steps": attempts_trace,
                "resolved": bool(attempts_trace and attempts_trace[-1]["validation"]["ok"]),
                "resolved_after_attempts": len(attempts_trace),
                "final_action": resolved_final,
                "final_reasoning": resolved_reasoning,
            }
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

    def _infer_expectations(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Heuristically infer formatting expectations from the original prompt."""
        text_chunks: List[str] = []
        for message in history:
            if message.get("role") in {"system", "user"}:
                text_chunks.append(str(message.get("content", "")))
        combined = " ".join(text_chunks).lower()

        requires_json = (
            "```json" in combined
            or "json format" in combined
            or "response in json" in combined
            or "output in json" in combined
        )

        expected_keys: List[str] = []
        if all(token in combined for token in ["goal", "api name", "input params"]):
            expected_keys.extend(["goal", "name", "args"])
        if all(token in combined for token in ["thought", "name", "args"]):
            for key in ["thought", "name", "args"]:
                if key not in expected_keys:
                    expected_keys.append(key)

        plan_item_keys: List[str] = []
        if "plan" in combined and "id" in combined and "name" in combined and "args" in combined:
            plan_item_keys = ["id", "name", "args"]

        expects_answer_choice = "answer:" in combined and "choosing from a, b, c, d, and e" in combined

        return {
            "requires_json": requires_json,
            "required_keys": expected_keys,
            "plan_item_keys": plan_item_keys,
            "expects_answer_choice": expects_answer_choice,
        }

    def _normalize_final_action(self, text: str) -> Tuple[str, List[str]]:
        """Strip formatting artefacts (e.g., code fences) from the final action."""
        notes: List[str] = []
        if text is None:
            return "", notes

        normalized = str(text).strip()
        fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
        fence_match = fence_pattern.match(normalized)
        if fence_match:
            normalized = fence_match.group(1).strip()
            notes.append("stripped_code_fence")

        return normalized, notes

    @staticmethod
    def _looks_like_json(text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith("{") or stripped.startswith("[")

    def _validate_final_action(
        self, final_text: str, expectations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run lightweight validation on the final action output and return feedback metadata.
        """
        validation = {
            "ok": True,
            "issue": None,
            "detail": None,
            "feedback": None,
        }

        stripped = final_text.strip()
        if not stripped:
            validation.update(
                ok=False,
                issue="empty_final_action",
                feedback=(
                    "Your previous reply did not include a final answer. Please provide only the final answer, "
                    "formatted exactly as requested."
                ),
            )
            return validation

        if expectations.get("expects_answer_choice"):
            if not re.fullmatch(r"Answer:\s*[A-E]", stripped):
                validation.update(
                    ok=False,
                    issue="string_choice_parse_error",
                    detail=stripped,
                    feedback=(
                        "Your final answer must be exactly `Answer: X` with X as one of A, B, C, D, or E. "
                        "Reply again using only that format."
                    ),
                )
                return validation

        requires_json = expectations.get("requires_json") or self._looks_like_json(stripped)
        if requires_json:
            try:
                parsed = json.loads(stripped)
            except Exception as exc:  # pragma: no cover - parser guidance
                validation.update(
                    ok=False,
                    issue="json_parse_error",
                    detail=str(exc),
                    feedback=(
                        "Your final answer must be valid JSON without code fences or commentary. "
                        f"The JSON parser reported: {exc}. Please regenerate the answer as valid JSON that matches the instructions."
                    ),
                )
                return validation

            if isinstance(parsed, list):
                required_item_keys = expectations.get("plan_item_keys") or []
                if required_item_keys:
                    for idx, item in enumerate(parsed):
                        if not isinstance(item, dict):
                            validation.update(
                                ok=False,
                                issue="invalid_plan_item",
                                detail=f"item {idx} is {type(item).__name__}",
                                feedback=(
                                    "Each plan step must be a JSON object with keys "
                                    f"{', '.join(required_item_keys)}. Please regenerate the plan accordingly."
                                ),
                            )
                            return validation
                        missing = [key for key in required_item_keys if key not in item]
                        if missing:
                            validation.update(
                                ok=False,
                                issue="invalid_plan_item",
                                detail=f"item {idx} missing keys {missing}",
                                feedback=(
                                    "Each plan step must include the keys "
                                    f"{', '.join(required_item_keys)}. Please fix the plan and reply again."
                                ),
                            )
                            return validation
            elif isinstance(parsed, dict):
                required_keys = expectations.get("required_keys") or []
                missing = [key for key in required_keys if key not in parsed]
                if missing:
                    validation.update(
                        ok=False,
                        issue="json_missing_key",
                        detail=f"missing keys {missing}",
                        feedback=(
                            f"The JSON answer must include the keys {', '.join(required_keys)}. "
                            "Please provide a JSON object containing all required keys and nothing else."
                        ),
                    )
                    return validation
            else:
                if expectations.get("requires_json"):
                    validation.update(
                        ok=False,
                        issue="not_dict_response",
                        detail=f"type {type(parsed).__name__}",
                        feedback=(
                            "The final answer must be a JSON object or list as specified. "
                            "Please respond with valid JSON matching the instructions."
                        ),
                    )
                    return validation

        return validation
