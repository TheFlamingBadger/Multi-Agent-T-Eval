from typing import List, Dict, Union, Callable, Optional
from time import perf_counter
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

    def _model_identifier(self, model) -> str:
        """
        Attempt to extract a human-readable identifier for the given model object.
        """
        return getattr(model, "model_name", None) or getattr(model, "path", None) or model.__class__.__name__
    
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
        traces = [
            {
                "strategy": "multi_model",
                "mode": "routing",
                "steps": []
            }
            for _ in histories
        ]
        total_elapsed = [0.0] * len(histories)

        if self.routing_fn is None:
            call_start = perf_counter()
            responses = self.llm.chat(histories, **kwargs)
            call_elapsed = perf_counter() - call_start
            model_name = self._model_identifier(self.llm)
            for idx, (history, response) in enumerate(zip(histories, responses)):
                traces[idx]["steps"].append({
                    "type": "llm_call",
                    "selected_model": model_name,
                    "messages": history,
                    "response": response,
                    "elapsed_seconds": call_elapsed,
                })
                total_elapsed[idx] += call_elapsed
                traces[idx]["total_elapsed_seconds"] = total_elapsed[idx]
            self._record_trace(traces)
            return self._denormalize_output(responses, was_single)
        
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
                resolved_name = self._model_identifier(self.llm)
            else:
                model = self.secondary_llms[model_name]
                resolved_name = self._model_identifier(model)
            
            # Extract just the histories for this batch
            indices, batch_histories = zip(*group)
            
            # Generate
            batch_histories_list = list(batch_histories)
            call_start = perf_counter()
            batch_responses = model.chat(batch_histories_list, **kwargs)
            call_elapsed = perf_counter() - call_start
            model_trace = getattr(model, "last_trace", []) or []
            
            # Place responses in correct positions and record trace
            for local_pos, (idx, history, response) in enumerate(zip(indices, batch_histories_list, batch_responses)):
                responses[idx] = response
                step_payload = {
                    "type": "llm_call",
                    "selected_model": model_name,
                    "resolved_model": resolved_name,
                    "messages": history,
                    "response": response,
                    "elapsed_seconds": call_elapsed,
                }
                if local_pos < len(model_trace) and model_trace[local_pos]:
                    step_payload["underlying_trace"] = model_trace[local_pos]
                traces[idx]["steps"].append(step_payload)
                total_elapsed[idx] += call_elapsed

        for idx in range(len(traces)):
            traces[idx]["total_elapsed_seconds"] = total_elapsed[idx]

        self._record_trace(traces)
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
        traces = [
            {
                "strategy": "multi_model",
                "mode": "sequential",
                "steps": []
            }
            for _ in histories
        ]
        total_elapsed = [0.0] * len(histories)

        # Phase 1: Get reasoning from primary model
        reasoning_start = perf_counter()
        reasoning_responses = self.llm.chat(histories, **kwargs)
        reasoning_elapsed = perf_counter() - reasoning_start
        reasoning_trace = getattr(self.llm, "last_trace", []) or []
        primary_model_name = self._model_identifier(self.llm)

        # Phase 2: Use reasoning with secondary model for final response
        # If no secondary model specified, use primary for both phases
        final_model = self.secondary_llms.get('final', self.llm)
        final_model_name = self._model_identifier(final_model)
        
        final_histories = []
        for idx, (history, reasoning) in enumerate(zip(histories, reasoning_responses)):
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
            reasoning_step = {
                "phase": "reasoning",
                "type": "llm_call",
                "model": primary_model_name,
                "messages": history,
                "response": reasoning,
                "elapsed_seconds": reasoning_elapsed,
            }
            if idx < len(reasoning_trace) and reasoning_trace[idx]:
                reasoning_step["underlying_trace"] = reasoning_trace[idx]
            traces[idx]["steps"].append(reasoning_step)
            total_elapsed[idx] += reasoning_elapsed

        final_start = perf_counter()
        final_responses = final_model.chat(final_histories, **kwargs)
        final_elapsed = perf_counter() - final_start
        final_trace = getattr(final_model, "last_trace", []) or []

        for idx, (final_history, final_response) in enumerate(zip(final_histories, final_responses)):
            final_step = {
                "phase": "final",
                "type": "llm_call",
                "model": final_model_name,
                "messages": final_history,
                "response": final_response,
                "elapsed_seconds": final_elapsed,
            }
            if idx < len(final_trace) and final_trace[idx]:
                final_step["underlying_trace"] = final_trace[idx]
            traces[idx]["steps"].append(final_step)
            total_elapsed[idx] += final_elapsed

        for idx in range(len(traces)):
            traces[idx]["total_elapsed_seconds"] = total_elapsed[idx]

        self._record_trace(traces)
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
        traces = [
            {
                "strategy": "multi_model",
                "mode": "ensemble",
                "selection": self.ensemble_selection,
                "steps": [],
                "candidates": []
            }
            for _ in histories
        ]
        total_elapsed = [0.0] * len(histories)
        
        # Generate from all models in a deterministic order
        model_sequence = [('primary', self.llm)] + list(self.secondary_llms.items())
        all_model_responses = []
        
        for alias, model in model_sequence:
            call_start = perf_counter()
            model_outputs = model.chat(histories, **kwargs)
            call_elapsed = perf_counter() - call_start
            model_trace = getattr(model, "last_trace", []) or []
            model_name = self._model_identifier(model)
            all_model_responses.append((alias, model_name, model_outputs))
            
            for idx, (history, output) in enumerate(zip(histories, model_outputs)):
                candidate_entry = {
                    "model_alias": alias,
                    "model": model_name,
                    "messages": history,
                    "response": output,
                    "elapsed_seconds": call_elapsed,
                }
                if idx < len(model_trace) and model_trace[idx]:
                    candidate_entry["underlying_trace"] = model_trace[idx]
                traces[idx]["candidates"].append(candidate_entry)
                total_elapsed[idx] += call_elapsed
        
        # Select best response for each input based on strategy
        final_responses = []
        
        for response_idx in range(len(histories)):
            # Get all responses for this input alongside provenance
            candidate_bundle = [
                {
                    "alias": alias,
                    "model": model_name,
                    "response": outputs[response_idx]
                }
                for alias, model_name, outputs in all_model_responses
            ]
            candidate_responses = [candidate["response"] for candidate in candidate_bundle]
            
            # Select based on strategy
            if self.ensemble_selection == 'first':
                selected_idx = 0
            elif self.ensemble_selection == 'longest':
                selected_idx = max(range(len(candidate_responses)), key=lambda i: len(candidate_responses[i]))
            elif self.ensemble_selection == 'shortest':
                selected_idx = min(range(len(candidate_responses)), key=lambda i: len(candidate_responses[i]))
            selected = candidate_responses[selected_idx]
            final_responses.append(selected)
            
            selected_candidate = candidate_bundle[selected_idx]
            traces[response_idx]["steps"].append({
                "type": "selection",
                "selected_alias": selected_candidate["alias"],
                "selected_model": selected_candidate["model"],
                "selection_strategy": self.ensemble_selection,
            })
            traces[response_idx]["final_response"] = selected

        for idx in range(len(traces)):
            traces[idx]["total_elapsed_seconds"] = total_elapsed[idx]

        self._record_trace(traces)
        return self._denormalize_output(final_responses, was_single)
