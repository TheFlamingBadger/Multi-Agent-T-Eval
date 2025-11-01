from typing import List, Dict, Union
from time import perf_counter
from .base import BaseOrchestrator
import copy


class ThinkingTokensOrchestrator(BaseOrchestrator):
    """
    Orchestrator that implements a two-phase generation with explicit thinking tokens.
    
    This orchestrator first prompts the model to generate reasoning/thinking tokens,
    then uses that reasoning as context to generate the final response. This approach
    can improve performance on complex reasoning tasks by making the model's thought
    process explicit.
    
    Args:
        llm: The underlying language model
        thinking_prompt: The prompt to inject for the thinking phase.
                        Default: "First, let's think step by step about how to approach this."
        thinking_max_tokens: Maximum tokens for the thinking phase. Default: 512
        separator: Text to separate thinking from final response. Default: "\n\nFinal response:"
        include_thinking_in_context: Whether to include thinking in context for final generation.
                                     Default: True
    
    Example:
        >>> orchestrator = ThinkingTokensOrchestrator(
        ...     llm,
        ...     thinking_prompt="Let's analyze this carefully:",
        ...     thinking_max_tokens=256
        ... )
        >>> messages = [{"role": "user", "content": "Solve this problem..."}]
        >>> response = orchestrator.completion(messages)
    """
    
    def __init__(
        self, 
        llm, 
        thinking_prompt: str = "First, let's think step by step about how to approach this.",
        thinking_max_tokens: int = 512,
        separator: str = "\n\nFinal response:",
        include_thinking_in_context: bool = True,
        **kwargs
    ):
        super().__init__(llm, **kwargs)
        self.thinking_prompt = thinking_prompt
        self.thinking_max_tokens = thinking_max_tokens
        self.separator = separator
        self.include_thinking_in_context = include_thinking_in_context
    
    def completion(
        self, 
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]], 
        **kwargs
    ) -> List[str]:
        """
        Generate completions using a two-phase thinking + response approach.
        
        Phase 1: Generate thinking/reasoning tokens
        Phase 2: Generate final response using thinking as context
        
        Args:
            message_histories: Message history or batch of message histories
            **kwargs: Parameters to pass to llm.chat()
        
        Returns:
            List[str]: Final responses (without thinking tokens, unless configured otherwise)
        """
        # Normalize to batch format
        histories, was_single = self._normalize_input(message_histories)
        
        # Phase 1: Generate thinking tokens
        thinking_histories = self._prepare_thinking_prompts(histories)
        thinking_kwargs = kwargs.copy()
        thinking_kwargs['max_new_tokens'] = thinking_kwargs.get(
            'max_new_tokens', 
            self.thinking_max_tokens
        )
        
        thinking_start = perf_counter()
        thinking_responses = self.llm.chat(thinking_histories, **thinking_kwargs)
        thinking_elapsed = perf_counter() - thinking_start
        underlying_thinking_traces = getattr(self.llm, "last_trace", []) or []
        
        # Phase 2: Generate final responses using thinking as context
        final_histories = self._prepare_final_prompts(
            histories, 
            thinking_responses
        )
        
        final_start = perf_counter()
        final_responses = self.llm.chat(final_histories, **kwargs)
        final_elapsed = perf_counter() - final_start
        underlying_final_traces = getattr(self.llm, "last_trace", []) or []
        total_elapsed = thinking_elapsed + final_elapsed

        traces = []
        for index, (original_history, thinking_history, final_history, thinking, final) in enumerate(zip(
            histories,
            thinking_histories,
            final_histories,
            thinking_responses,
            final_responses
        )):
            thinking_call_trace = underlying_thinking_traces[index] if index < len(underlying_thinking_traces) else None
            final_call_trace = underlying_final_traces[index] if index < len(underlying_final_traces) else None
            traces.append({
                "strategy": "thinking",
                "config": {
                    "thinking_prompt": self.thinking_prompt,
                    "thinking_max_tokens": self.thinking_max_tokens,
                    "include_thinking_in_context": self.include_thinking_in_context,
                    "separator": self.separator,
                },
                "steps": [
                    {
                        "phase": "thinking",
                        "messages": thinking_history,
                        "response": thinking,
                        "elapsed_seconds": thinking_elapsed,
                    },
                    {
                        "phase": "final",
                        "messages": final_history,
                        "response": final,
                        "elapsed_seconds": final_elapsed,
                    },
                ],
                "original_messages": original_history,
                "underlying_calls": {
                    "thinking": thinking_call_trace,
                    "final": final_call_trace,
                },
                "total_elapsed_seconds": total_elapsed,
            })

        self._record_trace(traces)

        return self._denormalize_output(final_responses, was_single)
    
    def _prepare_thinking_prompts(
        self, 
        histories: List[List[Dict[str, str]]]
    ) -> List[List[Dict[str, str]]]:
        """
        Prepare message histories for the thinking phase.
        
        Appends the thinking prompt to each message history.
        
        Args:
            histories: Batch of message histories
            
        Returns:
            Modified histories with thinking prompts appended
        """
        thinking_histories = []
        
        for history in histories:
            # Deep copy to avoid modifying original
            thinking_history = copy.deepcopy(history)
            
            # Add thinking prompt as a user message
            thinking_history.append({
                "role": "user",
                "content": self.thinking_prompt
            })
            
            thinking_histories.append(thinking_history)
        
        return thinking_histories
    
    def _prepare_final_prompts(
        self, 
        original_histories: List[List[Dict[str, str]]],
        thinking_responses: List[str]
    ) -> List[List[Dict[str, str]]]:
        """
        Prepare message histories for the final response phase.
        
        Args:
            original_histories: Original message histories
            thinking_responses: Generated thinking tokens from phase 1
            
        Returns:
            Histories with thinking incorporated for final generation
        """
        final_histories = []
        
        for history, thinking in zip(original_histories, thinking_responses):
            final_history = copy.deepcopy(history)
            
            if self.include_thinking_in_context:
                # Add the thinking as assistant's reasoning
                final_history.append({
                    "role": "assistant",
                    "content": f"Reasoning: {thinking}"
                })
                
                # Add prompt for final response
                final_history.append({
                    "role": "user",
                    "content": "Now provide your final response based on the reasoning above."
                })
            else:
                # Just prompt for the response without including thinking
                # (thinking was still generated, but not passed as context)
                final_history.append({
                    "role": "user",
                    "content": "Now provide your final response."
                })
            
            final_histories.append(final_history)
        
        return final_histories
