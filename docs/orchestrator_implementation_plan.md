# T-Eval Orchestrator Implementation Plan

## Overview

This document describes the orchestrator system added to T-Eval to enable meta-orchestration of LLM responses. The system allows models to generate "thinking tokens," make calls to other models, and implement various reasoning strategies before providing final outputs.

## Architecture

### Design Principles

1. **Backward Compatibility**: The system maintains full compatibility with existing T-Eval workflows
2. **Drop-in Replacement**: Orchestrators can replace direct LLM calls with minimal code changes
3. **Extensibility**: New orchestration strategies can be added by extending the base class
4. **Parameter Preservation**: All LLM parameters (`do_sample`, `temperature`, etc.) are preserved through the orchestration layer

### Component Structure

```
teval/orchestrators/
├── __init__.py              # Package exports
├── base.py                  # Abstract base class (BaseOrchestrator)
├── direct.py                # Baseline - no orchestration
├── thinking_tokens.py       # Chain-of-thought reasoning
└── multi_model.py           # Multi-model routing/ensemble
```

## Core Components

### 1. BaseOrchestrator (Abstract Base Class)

**Location**: `teval/orchestrators/base.py`

**Purpose**: Defines the standard interface that all orchestrators must implement.

**Key Method**:
```python
@abstractmethod
def completion(
    self, 
    message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]], 
    **kwargs
) -> List[str]:
    """
    Generate completions using the orchestration strategy.
    
    Args:
        message_histories: Either a batch of message histories or single history.
                          Each message has 'role' and 'content' keys.
        **kwargs: Additional LLM parameters (do_sample, temperature, etc.)
    
    Returns:
        List[str]: Generated responses
    """
```

**Helper Methods**:
- `_normalize_input()`: Converts single histories to batch format
- `_denormalize_output()`: Maintains list output format (consistent with lagent)

### 2. DirectOrchestrator

**Location**: `teval/orchestrators/direct.py`

**Purpose**: Baseline orchestrator with no meta-orchestration. Provides identical behavior to original T-Eval.

**Usage**:
```python
from teval.orchestrators import DirectOrchestrator

orchestrator = DirectOrchestrator(llm)
responses = orchestrator.completion(message_histories, do_sample=False)
```

**Behavior**: Direct pass-through to `llm.chat()`

### 3. ThinkingTokensOrchestrator

**Location**: `teval/orchestrators/thinking_tokens.py`

**Purpose**: Implements two-phase generation with explicit reasoning/thinking tokens.

**Parameters**:
- `thinking_prompt` (str): Prompt to trigger reasoning phase
  - Default: `"First, let's think step by step about how to approach this."`
- `thinking_max_tokens` (int): Maximum tokens for thinking phase
  - Default: `512`
- `separator` (str): Text separating thinking from response
  - Default: `"\n\nFinal response:"`
- `include_thinking_in_context` (bool): Whether to pass thinking to final generation
  - Default: `True`

**Phases**:

**Phase 1 - Thinking**: 
- Appends thinking prompt to message history
- Generates reasoning tokens
- Uses configured max tokens for thinking

**Phase 2 - Final Response**:
- Incorporates thinking as assistant's reasoning (if `include_thinking_in_context=True`)
- Prompts for final response based on reasoning
- Uses original kwargs for generation

**Usage Example**:
```python
orchestrator = ThinkingTokensOrchestrator(
    llm,
    thinking_prompt="Let's analyze this problem carefully:",
    thinking_max_tokens=256
)
responses = orchestrator.completion(messages, do_sample=False)
```

### 4. MultiModelOrchestrator

**Location**: `teval/orchestrators/multi_model.py`

**Purpose**: Enables routing queries to different models or using multiple models in sequence/ensemble.

**Parameters**:
- `primary_llm`: The primary/default language model
- `secondary_llms` (Dict[str, model]): Additional models keyed by name
- `routing_fn` (Callable): Function to select model for each query
- `strategy` (str): One of `'routing'`, `'sequential'`, `'ensemble'`
- `ensemble_selection` (str): For ensemble - `'first'`, `'longest'`, `'shortest'`

**Strategies**:

#### Routing Strategy
Route each query to appropriate model based on routing function.

```python
def router(history):
    if len(history[-1]['content']) > 500:
        return 'strong_model'
    return 'primary'

orchestrator = MultiModelOrchestrator(
    primary_llm=base_model,
    secondary_llms={'strong_model': gpt4},
    routing_fn=router,
    strategy='routing'
)
```

#### Sequential Strategy
Use primary model for reasoning, secondary for final response.

```python
orchestrator = MultiModelOrchestrator(
    primary_llm=reasoning_model,
    secondary_llms={'final': fast_model},
    strategy='sequential'
)
```

#### Ensemble Strategy
Generate from multiple models and select best response.

```python
orchestrator = MultiModelOrchestrator(
    primary_llm=model1,
    secondary_llms={'model2': model2, 'model3': model3},
    strategy='ensemble',
    ensemble_selection='longest'
)
```

## Integration with T-Eval

### Modified Components

#### test.py Changes

**1. Imports**:
```python
from teval.orchestrators import DirectOrchestrator, ThinkingTokensOrchestrator, MultiModelOrchestrator
```

**2. New Command-Line Arguments**:
```bash
--orchestrator {direct,thinking,multi_model}
    # Orchestration strategy (default: direct)

--thinking_prompt STR
    # Prompt for thinking phase (default: "First, let's think step by step...")

--thinking_max_tokens INT
    # Max tokens for thinking phase (default: 512)
```

**3. Orchestrator Initialization**:
```python
# After LLM initialization
if args.orchestrator == 'direct':
    orchestrator = DirectOrchestrator(llm)
elif args.orchestrator == 'thinking':
    orchestrator = ThinkingTokensOrchestrator(
        llm,
        thinking_prompt=args.thinking_prompt,
        thinking_max_tokens=args.thinking_max_tokens
    )
elif args.orchestrator == 'multi_model':
    orchestrator = MultiModelOrchestrator(llm, strategy='sequential')
```

**4. Modified infer() Function**:
```python
# Changed from: predictions = llm.chat(batch_infer_list, do_sample=False)
# To:
predictions = orchestrator.completion(batch_infer_list, do_sample=False)
```

### Evaluation Pipeline

The orchestrator system is **transparent to evaluators**. Evaluators receive predictions in the same format as before, so no changes are needed to:
- `teval/evaluators/instruct_evaluator.py`
- `teval/evaluators/planning_evaluator.py`
- `teval/evaluators/review_evaluator.py`
- `teval/evaluators/reason_retrieve_understand_evaluator.py`

## Usage Guide

### Running with Different Orchestrators

#### Baseline (Direct - No Orchestration)
```bash
# Explicit
python test.py --model_type api --model_path gpt-4 \
  --orchestrator direct \
  --dataset_path data/instruct_v2.json \
  --eval instruct

# Or omit --orchestrator (defaults to direct)
python test.py --model_type api --model_path gpt-4 \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

#### Thinking Tokens (Chain-of-Thought)
```bash
python test.py --model_type api --model_path gpt-4 \
  --orchestrator thinking \
  --thinking_prompt "Let's approach this step-by-step:" \
  --thinking_max_tokens 512 \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

#### Multi-Model (Sequential Reasoning)
```bash
python test.py --model_type api --model_path gpt-4 \
  --orchestrator multi_model \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

### Running Full Benchmark Suite

Update the test scripts (`test_all_en.sh`, `test_all_zh.sh`) to include orchestrator arguments:

```bash
#!/bin/bash
MODEL_TYPE=$1
MODEL_PATH=$2
MODEL_NAME=$3
ORCHESTRATOR=${4:-direct}  # Default to direct

python test.py --model_type $MODEL_TYPE --model_path $MODEL_PATH \
  --orchestrator $ORCHESTRATOR \
  --resume --out_name instruct_${MODEL_NAME}.json \
  --out_dir work_dirs/${MODEL_NAME}/ \
  --dataset_path data/instruct_v2.json \
  --eval instruct --prompt_type json \
  --model_display_name $MODEL_NAME

# ... repeat for other eval types
```

Usage:
```bash
# Direct orchestration
sh test_all_en.sh api gpt-4-1106-preview gpt4 direct

# Thinking tokens orchestration
sh test_all_en.sh api gpt-4-1106-preview gpt4_thinking thinking
```

## Extending the System

### Creating a Custom Orchestrator

To implement a new orchestration strategy:

**1. Create a new file in `teval/orchestrators/`**:

```python
# teval/orchestrators/my_custom_orchestrator.py

from typing import List, Dict, Union
from .base import BaseOrchestrator

class MyCustomOrchestrator(BaseOrchestrator):
    """
    Description of your orchestration strategy.
    """
    
    def __init__(self, llm, custom_param=None, **kwargs):
        super().__init__(llm, **kwargs)
        self.custom_param = custom_param
    
    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Implement your orchestration logic here.
        """
        # Normalize input
        histories, was_single = self._normalize_input(message_histories)
        
        # Your orchestration logic
        # ...
        
        # Call LLM(s)
        responses = self.llm.chat(histories, **kwargs)
        
        # Post-process if needed
        # ...
        
        return self._denormalize_output(responses, was_single)
```

**2. Add to `__init__.py`**:

```python
from .my_custom_orchestrator import MyCustomOrchestrator

__all__ = [
    # ... existing exports
    'MyCustomOrchestrator',
]
```

**3. Integrate into `test.py`**:

```python
# Add import
from teval.orchestrators import MyCustomOrchestrator

# Add argument choice
parser.add_argument('--orchestrator', type=str, default='direct',
                   choices=['direct', 'thinking', 'multi_model', 'my_custom'])

# Add initialization
elif args.orchestrator == 'my_custom':
    orchestrator = MyCustomOrchestrator(llm, custom_param=value)
```

### Example: Self-Consistency Orchestrator

```python
# teval/orchestrators/self_consistency.py

from typing import List, Dict, Union
from collections import Counter
from .base import BaseOrchestrator

class SelfConsistencyOrchestrator(BaseOrchestrator):
    """
    Generates multiple responses with sampling and selects most common.
    """
    
    def __init__(self, llm, num_samples=5, temperature=0.7, **kwargs):
        super().__init__(llm, **kwargs)
        self.num_samples = num_samples
        self.temperature = temperature
    
    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        histories, was_single = self._normalize_input(message_histories)
        
        final_responses = []
        
        for history in histories:
            # Generate multiple samples
            samples = []
            for _ in range(self.num_samples):
                response = self.llm.chat(
                    [history],
                    do_sample=True,
                    temperature=self.temperature
                )[0]
                samples.append(response)
            
            # Select most common response
            most_common = Counter(samples).most_common(1)[0][0]
            final_responses.append(most_common)
        
        return self._denormalize_output(final_responses, was_single)
```

## Performance Considerations

### Computational Cost

| Orchestrator | LLM Calls per Query | Relative Cost |
|--------------|---------------------|---------------|
| Direct | 1 | 1x (baseline) |
| ThinkingTokens | 2 | ~2x |
| MultiModel (Sequential) | 2 | ~2x |
| MultiModel (Ensemble, N models) | N | Nx |

### Token Usage

**ThinkingTokensOrchestrator**:
- Additional tokens = `thinking_max_tokens` + overhead for reasoning prompt
- Recommendation: Start with 256-512 tokens for thinking phase

**MultiModelOrchestrator**:
- Sequential: Similar to ThinkingTokens (~2x baseline)
- Ensemble: Multiplied by number of models

### Optimization Tips

1. **Batch Processing**: The orchestrators preserve batch processing from T-Eval
2. **Caching**: Consider implementing response caching for identical prompts
3. **Async Generation**: For ensemble strategies, implement async model calls
4. **Selective Orchestration**: Use routing to apply complex orchestration only when needed

## Testing

### Unit Testing

Create tests in `tests/test_orchestrators.py`:

```python
import unittest
from teval.orchestrators import DirectOrchestrator, ThinkingTokensOrchestrator

class MockLLM:
    def chat(self, histories, **kwargs):
        return ["response"] * len(histories)

class TestOrchestrators(unittest.TestCase):
    def test_direct_orchestrator(self):
        llm = MockLLM()
        orchestrator = DirectOrchestrator(llm)
        messages = [{"role": "user", "content": "test"}]
        result = orchestrator.completion(messages)
        self.assertEqual(len(result), 1)
    
    def test_thinking_orchestrator(self):
        llm = MockLLM()
        orchestrator = ThinkingTokensOrchestrator(llm)
        messages = [{"role": "user", "content": "test"}]
        result = orchestrator.completion(messages)
        self.assertEqual(len(result), 1)
```

### Integration Testing

Run small subset of T-Eval benchmark:

```bash
# Test with 10 samples
python test.py --model_type api --model_path gpt-3.5-turbo \
  --orchestrator thinking \
  --test_num 10 \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

## Experimental Comparison

### Recommended Experimental Setup

To compare orchestration strategies systematically:

```bash
# 1. Baseline
python test.py --orchestrator direct --model_path gpt-4 \
  --out_name results_direct.json --dataset_path data/instruct_v2.json --eval instruct

# 2. Thinking tokens
python test.py --orchestrator thinking --model_path gpt-4 \
  --out_name results_thinking.json --dataset_path data/instruct_v2.json --eval instruct

# 3. Multi-model (if you have multiple models)
python test.py --orchestrator multi_model --model_path gpt-4 \
  --out_name results_multimodel.json --dataset_path data/instruct_v2.json --eval instruct
```

### Analyzing Results

Use `teval/utils/convert_results.py` to aggregate results:

```bash
python teval/utils/convert_results.py --result_path work_dirs/model_name/model_name_-1.json
```

## Migration Guide

### From Original T-Eval

**No changes required** for basic usage! The default orchestrator is `direct`, which behaves identically to the original implementation.

**To experiment with orchestration**:
1. Add `--orchestrator thinking` (or other strategy) to your command
2. Optionally tune orchestration parameters
3. Results will be in the same format as before

### For Custom LLM Integrations

If you've added custom LLM classes:

```python
# Old way
class MyCustomLLM:
    def chat(self, messages, **kwargs):
        # Your implementation
        pass

# Still works! Just wrap with orchestrator
llm = MyCustomLLM()
orchestrator = DirectOrchestrator(llm)  # or ThinkingTokensOrchestrator, etc.
```

## Troubleshooting

### Issue: "orchestrator.completion() not found"

**Cause**: Old code still calling `llm.chat()` directly

**Solution**: Replace `llm.chat()` with `orchestrator.completion()`

### Issue: Thinking orchestrator doubles token costs

**Expected behavior**. Each query makes 2 LLM calls (thinking + final).

**Solutions**:
- Reduce `thinking_max_tokens`
- Use selective orchestration (only for complex queries)
- Cache thinking outputs for similar queries

### Issue: Message format errors

**Cause**: Orchestrators expect standard message format with `role` and `content` keys

**Solution**: Ensure `origin_prompt` in dataset follows format:
```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
```

## Future Enhancements

Potential additions to the orchestrator system:

1. **Reflection/Critique Orchestrator**: Generate, critique, and refine responses
2. **Tool-Use Orchestrator**: Integrate tool calls between reasoning steps
3. **Mixture-of-Agents**: Route different aspects to specialized models
4. **Adaptive Orchestration**: Dynamically select strategy based on query complexity
5. **Logging/Tracing**: Built-in logging of orchestration decisions and intermediate outputs

## References

- [T-Eval Paper](https://arxiv.org/abs/2312.14033)
- [Lagent Framework](https://github.com/InternLM/lagent)
- Chain-of-Thought Prompting: Wei et al., 2022
- Self-Consistency: Wang et al., 2022

## Support

For issues or questions:
1. Check this documentation
2. Review example usage in `test.py`
3. Open an issue on the T-Eval GitHub repository
