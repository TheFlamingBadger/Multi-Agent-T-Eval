from typing import List, Dict, Union
from .base import BaseOrchestrator


class DirectOrchestrator(BaseOrchestrator):
    """
    Direct pass-through orchestrator with no meta-orchestration.
    
    This is the baseline orchestrator that directly calls the underlying LLM
    without any additional orchestration logic. It provides behavior identical
    to the original T-Eval implementation.
    
    Example:
        >>> from lagent.llms.openai import GPTAPI
        >>> llm = GPTAPI("gpt-4")
        >>> orchestrator = DirectOrchestrator(llm)
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = orchestrator.completion(messages, do_sample=False)
    """
    
    def completion(
        self, 
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]], 
        **kwargs
    ) -> List[str]:
        """
        Generate completions by directly calling the underlying LLM.
        
        Args:
            message_histories: Message history or batch of message histories
            **kwargs: Parameters passed directly to llm.chat()
                     (e.g., do_sample, temperature, max_new_tokens)
        
        Returns:
            List[str]: Generated responses from the LLM
        """
        histories, was_single = self._normalize_input(message_histories)
        responses = self.llm.chat(histories, **kwargs)

        model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "path", None) or self.llm.__class__.__name__
        traces = []

        for history, response in zip(histories, responses):
            traces.append({
                "strategy": "direct",
                "model": model_name,
                "steps": [
                    {
                        "type": "llm_call",
                        "messages": history,
                        "response": response,
                    }
                ]
            })

        self._record_trace(traces)

        return self._denormalize_output(responses, was_single)
