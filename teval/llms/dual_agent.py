from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from lagent.llms.base_llm import BaseLLM
from lagent.utils import create_object


class DualStageLLM(BaseLLM):
    """Coordinator that runs a planner model followed by an actor model.

    The prompt framework is expected to produce samples whose
    ``origin_prompt`` is a dictionary with the following structure::

        {
            "planner_messages": [...],
            "actor_messages": [...],
            "plan_directive": {
                "role": "system",
                "template": "Plan:\\n{plan}",
                "position": "prepend"  # or "append"
            }
        }

    The coordinator will call the planner on ``planner_messages`` and
    then inject the planner output into the actor messages according to
    the directive before querying the actor model.
    """

    DEFAULT_PLAN_ROLE = "system"
    DEFAULT_PLAN_TEMPLATE = "Planner outline:\n{plan}"
    DEFAULT_PLAN_POSITION = "prepend"

    def __init__(
        self,
        planner: Union[BaseLLM, Dict[str, Any]],
        actor: Union[BaseLLM, Dict[str, Any]],
        *,
        plan_role: str = DEFAULT_PLAN_ROLE,
        plan_template: str = DEFAULT_PLAN_TEMPLATE,
        plan_position: str = DEFAULT_PLAN_POSITION,
        store_traces: bool = True,
        **kwargs,
    ):
        super().__init__(path="dual-stage", tokenizer_only=True, meta_template=None, **kwargs)
        self.planner = self._ensure_llm(planner)
        self.actor = self._ensure_llm(actor)
        self.plan_defaults = dict(role=plan_role, template=plan_template, position=plan_position)
        self.store_traces = store_traces
        self.latest_traces: List[Dict[str, Any]] = []

    @staticmethod
    def _ensure_llm(llm: Union[BaseLLM, Dict[str, Any]]) -> BaseLLM:
        if isinstance(llm, BaseLLM):
            return llm
        if isinstance(llm, dict):
            return create_object(llm)
        raise TypeError(f"Unsupported planner/actor specification: {type(llm)}")

    # BaseLLM.chat is overridden to avoid template parsing.
    def chat(
        self,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        batched = isinstance(inputs, list)
        batch = inputs if batched else [inputs]
        responses: List[str] = []
        traces: List[Dict[str, Any]] = []
        for payload in batch:
            response, trace = self._run_single(payload, do_sample=do_sample, **kwargs)
            responses.append(response)
            if trace is not None:
                traces.append(trace)
        if self.store_traces:
            self.latest_traces = traces
        return responses if batched else responses[0]

    def _run_single(
        self,
        payload: Dict[str, Any],
        *,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            # Fallback to actor-only execution.
            actor_messages = copy.deepcopy(payload)
            actor_response = self.actor.chat(actor_messages, do_sample=do_sample, **kwargs)
            return actor_response, None

        planner_messages = copy.deepcopy(payload.get("planner_messages", []))
        actor_messages_spec = payload.get("actor_messages")
        if actor_messages_spec is None:
            actor_messages_spec = copy.deepcopy(payload.get("planner_messages", []))

        plan_directive = self._merge_plan_directive(
            payload.get("plan_directive"), payload.get("actor_plan_directive")
        )

        plan_text = None
        planner_trace = None
        if planner_messages:
            plan_text = self.planner.chat(planner_messages, do_sample=do_sample, **kwargs)
            planner_trace = dict(messages=planner_messages, plan=plan_text)

        actor_messages = self._prepare_actor_messages(actor_messages_spec, plan_text, plan_directive)
        actor_response = self.actor.chat(actor_messages, do_sample=do_sample, **kwargs)

        trace = dict(planner=planner_trace, actor_messages=actor_messages, response=actor_response)
        return actor_response, trace

    def _merge_plan_directive(self, *directives: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self.plan_defaults)
        for directive in directives:
            if directive:
                merged.update(directive)
        return merged

    def _prepare_actor_messages(
        self,
        actor_spec: Union[List[Dict[str, Any]], Dict[str, Any]],
        plan_text: Optional[str],
        directive: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if isinstance(actor_spec, dict):
            messages = copy.deepcopy(actor_spec.get("messages", []))
            directive = self._merge_plan_directive(actor_spec.get("plan_directive"), directive)
        else:
            messages = copy.deepcopy(actor_spec)

        if plan_text:
            plan_message = {
                "role": directive.get("role", self.DEFAULT_PLAN_ROLE),
                "content": directive.get("template", self.DEFAULT_PLAN_TEMPLATE).format(plan=str(plan_text)),
            }
            position = directive.get("position", self.DEFAULT_PLAN_POSITION)
            if position == "append":
                messages.append(plan_message)
            else:
                messages.insert(0, plan_message)
        return messages

    # The streaming interface can be implemented later if needed.
    def stream_chat(self, inputs: Iterable[Dict[str, Any]], **kwargs):
        raise NotImplementedError("DualStageLLM does not support streaming yet.")
