# Azure OpenAI Setup Guide

This guide explains how to use the Azure OpenAI orchestrator with T-Eval.

## Prerequisites

1. An Azure subscription with OpenAI service access
2. A deployed model in Azure OpenAI (e.g., GPT-4, GPT-3.5-Turbo)
3. Python packages: `openai` and `python-dotenv` (included in requirements.txt)

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- `openai>=1.0.0` - Azure OpenAI Python SDK
- `python-dotenv>=1.0.0` - For loading environment variables from .env

## Configuration

### 1. Create .env File

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

### 2. Fill in Azure OpenAI Credentials

Edit `.env` with your Azure OpenAI details:

```bash
# API version - use the version supported by your deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Your Azure OpenAI endpoint URL
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/

# Your deployment name
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Your API key
AZURE_OPENAI_API_KEY=your_api_key_here
```

### Finding Your Credentials

**In Azure Portal:**

1. Navigate to your Azure OpenAI resource
2. Go to "Keys and Endpoint" section
3. Copy:
   - **Endpoint**: Your endpoint URL
   - **Key**: Either KEY 1 or KEY 2

**Deployment Name:**

1. Go to "Model deployments" in your Azure OpenAI resource
2. Note the deployment name (not the model name)
   - Example: If you deployed GPT-4, your deployment might be named "gpt-4-deployment" or "my-gpt4"

**API Version:**

- Use the latest stable version: `2024-02-15-preview`
- Or check [Azure OpenAI API versions](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation)

## Usage

### Basic Usage (Direct Orchestrator)

Run T-Eval with Azure OpenAI using the direct orchestrator:

```bash
python test.py \
  --model_type azure \
  --dataset_path data/instruct_v2.json \
  --eval instruct \
  --out_name azure_results.json \
  --out_dir work_dirs/azure/
```

### With Custom .env Path

If your .env file is in a different location:

```bash
python test.py \
  --model_type azure \
  --azure_env_path /path/to/your/.env \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

### With Thinking Tokens Orchestrator

Use chain-of-thought reasoning with Azure OpenAI:

```bash
python test.py \
  --model_type azure \
  --orchestrator thinking \
  --thinking_prompt "Let's approach this step-by-step:" \
  --thinking_max_tokens 512 \
  --dataset_path data/instruct_v2.json \
  --eval instruct
```

### Complete Example

Full command with all evaluation types:

```bash
# Instruct evaluation
python test.py --model_type azure --orchestrator direct \
  --resume --out_name instruct_azure.json \
  --out_dir work_dirs/azure/ \
  --dataset_path data/instruct_v2.json \
  --eval instruct --prompt_type json \
  --model_display_name azure_gpt4

# Plan evaluation
python test.py --model_type azure --orchestrator direct \
  --resume --out_name plan_json_azure.json \
  --out_dir work_dirs/azure/ \
  --dataset_path data/plan_json_v2.json \
  --eval plan --prompt_type json \
  --model_display_name azure_gpt4

# Continue for other evaluation types...
```

## Architecture

The `AzureOpenAIOrchestrator` is a direct orchestrator that:

1. Loads credentials from `.env` file using `python-dotenv`
2. Creates an Azure OpenAI client using the official `openai` SDK
3. Implements the standard `completion()` interface
4. Handles batch processing by iterating over message histories
5. Maps T-Eval parameters (like `do_sample`, `max_new_tokens`) to Azure OpenAI API parameters

### Parameter Mapping

| T-Eval Parameter | Azure OpenAI Parameter | Notes |
|-----------------|------------------------|-------|
| `do_sample=False` | `temperature=0` | Forces deterministic output |
| `max_new_tokens` | `max_tokens` | Maximum tokens to generate |
| `temperature` | `temperature` | Sampling temperature (0-2) |
| `top_p` | `top_p` | Nucleus sampling |
| `frequency_penalty` | `frequency_penalty` | Penalize frequent tokens |
| `presence_penalty` | `presence_penalty` | Penalize repeated tokens |

## Advanced Usage

### Composing with Other Orchestrators

The Azure OpenAI orchestrator can be composed with other orchestration strategies:

```python
from teval.orchestrators import AzureOpenAIOrchestrator, ThinkingTokensOrchestrator

# Create base Azure orchestrator
base = AzureOpenAIOrchestrator(env_path=".env")

# Wrap with thinking tokens
orchestrator = ThinkingTokensOrchestrator(
    base,
    thinking_prompt="Let's reason through this:",
    thinking_max_tokens=256
)

# Use in evaluation
messages = [{"role": "user", "content": "Your question"}]
responses = orchestrator.completion(messages, temperature=0)
```

### Programmatic Usage

Use the orchestrator directly in Python code:

```python
from teval.orchestrators import AzureOpenAIOrchestrator

# Initialize
orchestrator = AzureOpenAIOrchestrator(env_path=".env")

# Single message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
response = orchestrator.completion(messages, temperature=0.7)
print(response[0])  # Returns list with one response

# Batch messages
batch = [
    [{"role": "user", "content": "Question 1"}],
    [{"role": "user", "content": "Question 2"}],
    [{"role": "user", "content": "Question 3"}],
]
responses = orchestrator.completion(batch, temperature=0)
for i, resp in enumerate(responses):
    print(f"Response {i+1}: {resp}")
```

### Override Credentials Programmatically

You can override .env credentials programmatically:

```python
orchestrator = AzureOpenAIOrchestrator(
    api_version="2024-02-15-preview",
    endpoint="https://my-resource.openai.azure.com/",
    deployment="gpt-4",
    api_key="my-secret-key"
)
```

**Note:** This is less secure than using .env files. Only use for testing.

## Security Best Practices

1. **Never commit .env files**: Add `.env` to your `.gitignore`
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use .env for local development only**: For production, use:
   - Azure Key Vault
   - Environment variables set by your deployment system
   - Managed identities (when running on Azure)

3. **Rotate keys regularly**: Azure Portal -> Your OpenAI resource -> Keys and Endpoint -> Regenerate

4. **Use separate credentials for dev/prod**: Create separate Azure OpenAI resources

## Troubleshooting

### Error: "Missing required Azure OpenAI credentials"

**Cause**: .env file not found or incomplete

**Solution:**
1. Ensure .env file exists in the project root (or specify path with `--azure_env_path`)
2. Verify all four required variables are set:
   - `AZURE_OPENAI_API_VERSION`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT`
   - `AZURE_OPENAI_API_KEY`

### Error: "The 'openai' package is required"

**Cause**: OpenAI SDK not installed

**Solution:**
```bash
pip install openai>=1.0.0
```

### Error: "Error calling Azure OpenAI API"

**Common causes:**
1. **Invalid API key**: Double-check key in Azure Portal
2. **Wrong endpoint**: Ensure endpoint matches your resource
3. **Wrong deployment name**: Verify deployment name in Azure Portal (not model name)
4. **API version mismatch**: Try a different API version (e.g., `2023-12-01-preview`)
5. **Rate limiting**: You may be hitting rate limits. Add retry logic or reduce concurrency.

### Rate Limiting

Azure OpenAI has rate limits based on your deployment tier:

- Tokens per minute (TPM)
- Requests per minute (RPM)

**Solution:**
```bash
# Reduce batch size to avoid rate limits
python test.py --model_type azure --batch_size 1 \
  --dataset_path data/instruct_v2.json --eval instruct
```

Or add retry logic to the orchestrator (future enhancement).

### Debugging

Enable verbose output to see API calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your test
```

## Cost Considerations

Azure OpenAI charges based on:
- Input tokens (prompt)
- Output tokens (completion)

**Thinking Tokens Orchestrator**: Approximately 2x the cost (two API calls per query)

**Multi-Model Orchestrator**: Varies based on strategy
- Sequential: ~2x cost
- Ensemble: N × cost (where N = number of models)

**Monitoring costs:**
1. Azure Portal -> Your OpenAI resource -> Cost Management
2. Set up budget alerts

## Performance Tips

1. **Batch processing**: T-Eval processes batches, but Azure orchestrator handles them sequentially. For true parallelism, consider async implementation.

2. **Caching**: For repeated queries, consider caching responses:
   ```python
   # Future enhancement - add caching layer
   ```

3. **Model selection**: Use appropriate model for task:
   - GPT-3.5-Turbo: Faster, cheaper, good for simple tasks
   - GPT-4: Slower, more expensive, better for complex reasoning

4. **Token limits**: Set `max_tokens` appropriately to avoid unnecessary costs:
   ```bash
   python test.py --model_type azure --max_tokens 256 ...
   ```

## Comparison with OpenAI API

| Feature | Azure OpenAI | OpenAI API (Lagent) |
|---------|-------------|---------------------|
| Authentication | API key via .env | API key via environment |
| Endpoint | Custom Azure endpoint | api.openai.com |
| Models | Your deployments | OpenAI model IDs |
| Billing | Azure subscription | OpenAI account |
| Rate limits | Based on deployment | Based on tier |
| Features | May lag behind OpenAI | Latest features first |

## Next Steps

1. ✅ Set up .env file with credentials
2. ✅ Run a small test (--test_num 10)
3. ✅ Verify results
4. Run full benchmark
5. Compare with other orchestrators (thinking, multi_model)
6. Analyze results using `teval/utils/convert_results.py`

## Support

For Azure OpenAI specific issues:
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure OpenAI API Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference)

For T-Eval orchestrator issues:
- Check `docs/orchestrator_implementation_plan.md`
- Review code in `teval/orchestrators/azure_openai.py`
