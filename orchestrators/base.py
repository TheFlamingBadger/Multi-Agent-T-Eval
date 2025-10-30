from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseOrchestrator(ABC):
    """
    Abstract base class for LLM orchestration strategies.
    
    This class provides a standard interface for different orchestration approaches,
    allowing meta-orchestration of LLM responses (e.g., thinking tokens, multi-model routing).
    
    Args:
        llm: The primary language model instance (from lagent.llms).
             Should have a .chat() method that accepts message histories.
        **kwargs: Additional configuration parameters specific to the orchestrator.
    """
    
    def __init__(self, llm, **kwargs):
        """
        Initialize the orchestrator with a language model.
        
        Args:
            llm: Language model instance (GPTAPI, HFTransformerCasualLM, etc.)
            **kwargs: Orchestrator-specific configuration
        """
        self.llm = llm
        self.config = kwargs
    
    @abstractmethod
    def completion(
        self, 
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]], 
        **kwargs
    ) -> List[str]:
        """
        Generate completions using the orchestration strategy.
        
        This method accepts message histories in the standard format used by T-Eval
        and returns the final model responses after applying the orchestration logic.
        
        Args:
            message_histories: Either a batch of message histories (List[List[Dict]]) or 
                             a single message history (List[Dict]). Each message dict should
                             have 'role' and 'content' keys.
                             Example single history:
                                 [
                                     {"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": "Hello!"}
                                 ]
            **kwargs: Additional parameters to pass to the underlying LLM.
                     Common parameters include:
                     - do_sample (bool): Whether to use sampling
                     - temperature (float): Sampling temperature
                     - max_new_tokens (int): Maximum tokens to generate
                     
        Returns:
            List[str]: List of generated responses, one per input message history.
                      For a single message history input, returns a list with one element.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
    
    def _normalize_input(
        self, 
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]]
    ) -> tuple[List[List[Dict[str, str]]], bool]:
        """
        Normalize input to always be a batch of message histories.
        
        Args:
            message_histories: Either a batch or single message history
            
        Returns:
            Tuple of (normalized batch, was_single_input)
        """
        # Check if this is a single message history (first element is a dict)
        if message_histories and isinstance(message_histories[0], dict):
            return [message_histories], True
        return message_histories, False
    
    def _denormalize_output(self, outputs: List[str], was_single_input: bool) -> List[str]:
        """
        Keep output as list even for single inputs (consistent with lagent behavior).
        
        Args:
            outputs: List of generated responses
            was_single_input: Whether the input was a single message history
            
        Returns:
            The outputs list unchanged (lagent always returns lists)
        """
        return outputs
