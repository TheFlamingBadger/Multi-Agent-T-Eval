"""Prompt framework registry and base classes for T-Eval.

The registry allows the evaluation runners to obtain prompt-transform
logic dynamically without mutating the source datasets. Framework
implementations live in :mod:`teval.prompts.registry`.
"""

from .registry import (
    PromptFramework,
    DatasetPassthroughFramework,
    build_prompt_framework,
    list_prompt_frameworks,
    register_prompt_framework,
)
from .dual_plan import DualPlanFramework
from .react import ReActFramework

__all__ = [
    "PromptFramework",
    "DatasetPassthroughFramework",
    "DualPlanFramework",
    "ReActFramework",
    "build_prompt_framework",
    "list_prompt_frameworks",
    "register_prompt_framework",
]
