from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Type, Union


class PromptFramework:
    """Base class for prompt-framework implementations.

    A prompt framework receives a dataset sample and can transform the
    message list before it is passed to the language model. Returning a
    meta-template key allows the runner to select an appropriate prompt
    format for the target model.
    """

    def __init__(self, **kwargs):
        self.config = kwargs or {}

    def build_sample(
        self, sample: Dict
    ) -> Tuple[Iterable[Dict], Optional[Union[str, Dict[str, Optional[str]]]]]:
        """Prepare a sample for inference.

        Args:
            sample (dict): The dataset entry to transform. Callers are
                expected to treat the returned value as authoritative.

        Returns:
            tuple(object, str | dict | None): A pair containing the prompt
            payload and optional meta-template information. The payload can
            be any structure understood by the downstream model wrapper.
        """
        raise NotImplementedError


_PROMPT_FRAMEWORK_REGISTRY: Dict[str, Type[PromptFramework]] = {}


def register_prompt_framework(name: Optional[str] = None):
    """Class decorator to register a prompt-framework implementation."""

    def decorator(cls: Type[PromptFramework]):
        if not issubclass(cls, PromptFramework):
            raise TypeError("Prompt framework must inherit from PromptFramework")
        key = (name or cls.__name__).lower()
        if key in _PROMPT_FRAMEWORK_REGISTRY:
            raise KeyError(f"Prompt framework '{key}' already registered")
        _PROMPT_FRAMEWORK_REGISTRY[key] = cls
        return cls

    return decorator


def build_prompt_framework(name: str, **kwargs) -> PromptFramework:
    """Instantiate a prompt framework by name."""
    key = name.lower()
    if key not in _PROMPT_FRAMEWORK_REGISTRY:
        available = ", ".join(sorted(_PROMPT_FRAMEWORK_REGISTRY))
        raise KeyError(f"Prompt framework '{name}' not found. Available: {available}")
    cls = _PROMPT_FRAMEWORK_REGISTRY[key]
    return cls(**kwargs)


def list_prompt_frameworks() -> Tuple[str, ...]:
    """Return the registered framework names."""
    return tuple(sorted(_PROMPT_FRAMEWORK_REGISTRY))


@register_prompt_framework("passthrough")
class DatasetPassthroughFramework(PromptFramework):
    """Default framework that keeps dataset prompts unchanged."""

    def build_sample(self, sample: Dict) -> Tuple[Iterable[Dict], Optional[str]]:
        origin_prompt = sample.get("origin_prompt")
        if origin_prompt is None:
            raise KeyError("Sample does not contain 'origin_prompt'")
        return origin_prompt, self.config.get("meta_template")
