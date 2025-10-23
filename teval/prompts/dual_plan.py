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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.planner_meta_template = kwargs.get("planner_meta_template")
        self.actor_meta_template = kwargs.get("actor_meta_template")
        self.planner_system_prompt = kwargs.get("planner_system_prompt")
        self.actor_system_prompt = kwargs.get("actor_system_prompt")
        self.plan_role = kwargs.get("plan_message_role", self.DEFAULT_PLAN_ROLE)
        self.plan_template = kwargs.get("plan_message_template", self.DEFAULT_PLAN_TEMPLATE)
        self.plan_position = kwargs.get("plan_message_position", self.DEFAULT_PLAN_POSITION)

    def build_sample(self, sample: Dict) -> Tuple[Dict[str, Union[List[Dict], Dict]], Dict[str, Optional[str]]]:
        origin_prompt = sample.get("origin_prompt")
        if not isinstance(origin_prompt, list):
            raise TypeError(
                "DualPlanFramework expects 'origin_prompt' to be a list of message dicts."
            )
        planner_messages = self._apply_system_override(origin_prompt, self.planner_system_prompt)
        actor_messages = self._apply_system_override(origin_prompt, self.actor_system_prompt)
        payload = dict(
            planner_messages=planner_messages,
            actor_messages=actor_messages,
            plan_directive=dict(role=self.plan_role, template=self.plan_template, position=self.plan_position),
        )
        meta = dict(planner=self.planner_meta_template, actor=self.actor_meta_template)
        return payload, meta

    @staticmethod
    def _apply_system_override(messages: List[Dict], override: Optional[str]) -> List[Dict]:
        cloned = copy.deepcopy(messages)
        if override is None:
            return cloned
        for msg in cloned:
            if msg.get("role") == "system":
                msg["content"] = override
                break
        else:
            cloned.insert(0, dict(role="system", content=override))
        return cloned
