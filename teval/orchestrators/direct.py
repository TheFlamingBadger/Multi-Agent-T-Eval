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
        return self.llm.chat(message_histories, **kwargs)
