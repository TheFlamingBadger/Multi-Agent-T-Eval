from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

from .registry import PromptFramework, register_prompt_framework


@register_prompt_framework("dual_plan")
class DualPlanFramework(PromptFramework):
    """Prompt framework that prepares planner and actor message flows."""

    DEFAULT_PLAN_ROLE = "system"
    DEFAULT_PLAN_TEMPLATE = "Here is the approved plan:\n{plan}"
    DEFAULT_PLAN_POSITION = "prepend"
    VALID_MODES = {"overwrite", "prepend"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planner_meta_template = kwargs.get("planner_meta_template")
        self.actor_meta_template = kwargs.get("actor_meta_template")
        self.planner_system_prompt = kwargs.get("planner_system_prompt")
        self.actor_system_prompt = kwargs.get("actor_system_prompt")
        base_mode = self._normalize_mode(kwargs.get("system_prompt_mode", "overwrite"), "")
        self.planner_system_prompt_mode = self._normalize_mode(
            kwargs.get("planner_system_prompt_mode", base_mode), "planner"
        )
        self.actor_system_prompt_mode = self._normalize_mode(
            kwargs.get("actor_system_prompt_mode", base_mode), "actor"
        )
        self.plan_role = kwargs.get("plan_message_role", self.DEFAULT_PLAN_ROLE)
        self.plan_template = kwargs.get("plan_message_template", self.DEFAULT_PLAN_TEMPLATE)
        self.plan_position = kwargs.get("plan_message_position", self.DEFAULT_PLAN_POSITION)

    def build_sample(self, sample: Dict) -> Tuple[Dict[str, Union[List[Dict], Dict]], Dict[str, Optional[str]]]:
        origin_prompt = sample.get("origin_prompt")
        if not isinstance(origin_prompt, list):
            raise TypeError(
                "DualPlanFramework expects 'origin_prompt' to be a list of message dicts."
            )
        planner_messages = self._apply_system_override(
            origin_prompt, self.planner_system_prompt, self.planner_system_prompt_mode
        )
        actor_messages = self._apply_system_override(
            origin_prompt, self.actor_system_prompt, self.actor_system_prompt_mode
        )
        payload = dict(
            planner_messages=planner_messages,
            actor_messages=actor_messages,
            plan_directive=dict(role=self.plan_role, template=self.plan_template, position=self.plan_position),
        )
        meta = dict(planner=self.planner_meta_template, actor=self.actor_meta_template)
        return payload, meta

    @classmethod
    def _apply_system_override(
        cls, messages: List[Dict], override: Optional[str], mode: str
    ) -> List[Dict]:
        cloned = copy.deepcopy(messages)
        if override is None:
            return cloned
        if mode == "overwrite":
            for msg in cloned:
                if msg.get("role") == "system":
                    msg["content"] = override
                    break
            else:
                cloned.insert(0, dict(role="system", content=override))
        elif mode == "prepend":
            cloned.insert(0, dict(role="system", content=override))
        else:
            raise ValueError(
                f"Unsupported system prompt mode '{mode}'. "
                f"Choose from {sorted(cls.VALID_MODES)}."
            )
        return cloned

    @classmethod
    def _normalize_mode(cls, mode_value: Optional[str], label: str) -> str:
        mode = (mode_value or "overwrite").lower()
        if mode not in cls.VALID_MODES:
            key = f"{label}_system_prompt_mode" if label else "system_prompt_mode"
            raise ValueError(
                f"Unsupported {key} '{mode_value}'. Choose from {sorted(cls.VALID_MODES)}."
            )
        return mode
