from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional, Tuple

from .registry import PromptFramework, register_prompt_framework


@register_prompt_framework("react")
class ReActFramework(PromptFramework):
    """Framework that injects a ReAct-style system prompt into the dataset."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a ReAct (Reason + Act) agent. Solve the user's task by iterating "
        "through Thought, Action, and Observation steps. When you need to call a tool, "
        "emit `Action: <tool_name>[<json_args>]`. After each tool call, write "
        "`Observation:` with the tool result. Conclude with `Final Answer:` followed by "
        "the solution for the user. Do not fabricate observations."
    )
    VALID_MODES = {"overwrite", "prepend"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = kwargs.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        mode = (kwargs.get("system_prompt_mode", "overwrite") or "overwrite").lower()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported system_prompt_mode '{mode}'. "
                f"Choose from {sorted(self.VALID_MODES)}."
            )
        self.system_prompt_mode = mode
        self.meta_template = kwargs.get("meta_template")

    def build_sample(
        self, sample: Dict
    ) -> Tuple[Iterable[Dict], Optional[str]]:
        origin_prompt = sample.get("origin_prompt")
        if not isinstance(origin_prompt, list):
            raise TypeError("ReActFramework expects 'origin_prompt' to be a list of messages.")
        messages = self._inject_system_prompt(copy.deepcopy(origin_prompt))
        return messages, self.meta_template

    def _inject_system_prompt(self, messages):
        if self.system_prompt_mode == "overwrite":
            for message in messages:
                if message.get("role") == "system":
                    message["content"] = self.system_prompt
                    break
            else:
                messages.insert(0, dict(role="system", content=self.system_prompt))
        elif self.system_prompt_mode == "prepend":
            messages.insert(0, dict(role="system", content=self.system_prompt))
        return messages
