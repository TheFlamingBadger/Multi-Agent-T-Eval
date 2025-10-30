from typing import List, Dict, Union, Callable, Optional
from .base import BaseOrchestrator
import copy


class MultiModelOrchestrator(BaseOrchestrator):
    """
    Orchestrator that can route queries to different models or use multiple models.
    
    This orchestrator supports several multi-model strategies:
    1. Routing: Use a routing function to select which model handles each query
    2. Sequential: Use one model for reasoning, another for final response
    3. Ensemble: Generate from multiple models and select/combine responses
    
    Args:
        primary_llm: The primary/default language model
        secondary_llms: Dict of additional models, keyed by name
        routing_fn: Optional function that takes a message history and returns 
                   the name of the model to use. If None, uses primary_llm.
        strategy: One of 'routing', 'sequential', or 'ensemble'. Default: 'routing'
        ensemble_selection: For ensemble strategy, how to select final response.
                          One of 'first', 'longest', 'shortest'. Default: 'first'
    
    Example - Routing:
        >>> def router(history):
        ...     # Route complex queries to stronger model
        ...     if len(history[-1]['content']) > 500:
        ...         return 'strong_model'
        ...     return 'primary'
        >>> 
        >>> orchestrator = MultiModelOrchestrator(
        ...     primary_llm=base_model,
        ...     secondary_llms={'strong_model': gpt4},
        ...     routing_fn=router,
        ...     strategy='routing'
        ... )
    
    Example - Sequential:
        >>> # Use one model for reasoning, another for final answer
        >>> orchestrator = MultiModelOrchestrator(
        ...     primary_llm=reasoning_model,
        ...     secondary_llms={'final': fast_model},
        ...     strategy='sequential'
        ... )
    
    Example - Ensemble:
        >>> # Generate from multiple models and select longest response
        >>> orchestrator = MultiModelOrchestrator(
        ...     primary_llm=model1,
        ...     secondary_llms={'model2': model2, 'model3': model3},
        ...     strategy='ensemble',
        ...     ensemble_selection='longest'
        ... )
    """
    
    def __init__(
        self,
        primary_llm,
        secondary_llms: Optional[Dict[str, any]] = None,
        routing_fn: Optional[Callable] = None,
        strategy: str = 'routing',
        ensemble_selection: str = 'first',
        **kwargs
    ):
        super().__init__(primary_llm, **kwargs)
        self.secondary_llms = secondary_llms or {}
        self.routing_fn = routing_fn
        self.strategy = strategy
        self.ensemble_selection = ensemble_selection
        
        # Validate strategy
        valid_strategies = ['routing', 'sequential', 'ensemble']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}, got: {strategy}")
        
        # Validate ensemble selection method
        valid_selections = ['first', 'longest', 'shortest']
        if ensemble_selection not in valid_selections:
            raise ValueError(
                f"ensemble_selection must be one of {valid_selections}, got: {ensemble_selection}"
            )
    
    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generate completions using the configured multi-model strategy.
        
        Args:
            message_histories: Message history or batch of message histories
            **kwargs: Parameters to pass to llm.chat()
        
        Returns:
            List[str]: Generated responses
        """
        if self.strategy == 'routing':
            return self._routing_completion(message_histories, **kwargs)
        elif self.strategy == 'sequential':
            return self._sequential_completion(message_histories, **kwargs)
        elif self.strategy == 'ensemble':
            return self._ensemble_completion(message_histories, **kwargs)
    
    def _routing_completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Route each query to an appropriate model based on routing function.
        
        Args:
            message_histories: Message histories
            **kwargs: LLM parameters
            
        Returns:
            Responses from routed models
        """
        histories, was_single = self._normalize_input(message_histories)
        
        if self.routing_fn is None:
            # No routing function, use primary model for all
            return self._denormalize_output(
                self.llm.chat(histories, **kwargs),
                was_single
            )
        
        # Group histories by routed model
        model_groups = {}  # model_name -> list of (index, history)
        
        for idx, history in enumerate(histories):
            model_name = self.routing_fn(history)
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append((idx, history))
        
        # Generate responses from each model
        responses = [None] * len(histories)
        
        for model_name, group in model_groups.items():
            # Get the appropriate model
            if model_name == 'primary' or model_name not in self.secondary_llms:
                model = self.llm
            else:
                model = self.secondary_llms[model_name]
            
            # Extract just the histories for this batch
            indices, batch_histories = zip(*group)
            
            # Generate
            batch_responses = model.chat(list(batch_histories), **kwargs)
            
            # Place responses in correct positions
            for idx, response in zip(indices, batch_responses):
                responses[idx] = response
        
        return self._denormalize_output(responses, was_single)
    
    def _sequential_completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Use primary model for intermediate reasoning, secondary for final response.
        
        Args:
            message_histories: Message histories
            **kwargs: LLM parameters
            
        Returns:
            Final responses from secondary model
        """
        histories, was_single = self._normalize_input(message_histories)
        
        # Phase 1: Get reasoning from primary model
        reasoning_responses = self.llm.chat(histories, **kwargs)
        
        # Phase 2: Use reasoning with secondary model for final response
        # If no secondary model specified, use primary for both phases
        final_model = self.secondary_llms.get('final', self.llm)
        
        final_histories = []
        for history, reasoning in zip(histories, reasoning_responses):
            final_history = copy.deepcopy(history)
            final_history.append({
                "role": "assistant",
                "content": f"Intermediate reasoning: {reasoning}"
            })
            final_history.append({
                "role": "user",
                "content": "Based on the reasoning above, provide your final response."
            })
            final_histories.append(final_history)
        
        final_responses = final_model.chat(final_histories, **kwargs)
        
        return self._denormalize_output(final_responses, was_single)
    
    def _ensemble_completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generate from multiple models and select best response.
        
        Args:
            message_histories: Message histories
            **kwargs: LLM parameters
            
        Returns:
            Selected responses based on ensemble_selection strategy
        """
        histories, was_single = self._normalize_input(message_histories)
        
        # Generate from all models
        all_model_responses = []
        
        # Generate from primary model
        all_model_responses.append(self.llm.chat(histories, **kwargs))
        
        # Generate from secondary models
        for model_name, model in self.secondary_llms.items():
            all_model_responses.append(model.chat(histories, **kwargs))
        
        # Select best response for each input based on strategy
        final_responses = []
        
        for response_idx in range(len(histories)):
            # Get all responses for this input
            candidate_responses = [
                model_responses[response_idx] 
                for model_responses in all_model_responses
            ]
            
            # Select based on strategy
            if self.ensemble_selection == 'first':
                selected = candidate_responses[0]
            elif self.ensemble_selection == 'longest':
                selected = max(candidate_responses, key=len)
            elif self.ensemble_selection == 'shortest':
                selected = min(candidate_responses, key=len)
            
            final_responses.append(selected)
        
        return self._denormalize_output(final_responses, was_single)
