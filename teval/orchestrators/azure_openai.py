from typing import List, Dict, Union, Optional
from time import perf_counter
from .base import BaseOrchestrator
import os
from dotenv import load_dotenv


class AzureOpenAIOrchestrator(BaseOrchestrator):
    """
    Direct orchestrator using Azure OpenAI API.
    
    This orchestrator loads Azure OpenAI credentials from a .env file and makes
    direct API calls without any meta-orchestration. It's designed to work with
    the Azure OpenAI service endpoint.
    
    Environment Variables (loaded from .env):
        AZURE_OPENAI_API_VERSION: API version (e.g., "2024-02-15-preview")
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
        AZURE_OPENAI_DEPLOYMENT: Deployment name
        AZURE_OPENAI_API_KEY: API key for authentication
    
    Args:
        env_path: Path to .env file. If None, searches in current directory.
        api_version: Override API version from .env
        endpoint: Override endpoint from .env
        deployment: Override deployment from .env
        api_key: Override API key from .env
        **kwargs: Additional configuration parameters
    
    Example:
        >>> orchestrator = AzureOpenAIOrchestrator(env_path=".env")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = orchestrator.completion(messages, temperature=0.7)
    
    Example .env file:
        AZURE_OPENAI_API_VERSION=2024-02-15-preview
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT=gpt-4
        AZURE_OPENAI_API_KEY=your_api_key_here
    """
    
    def __init__(
        self,
        env_path: Optional[str] = None,
        api_version: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Azure OpenAI orchestrator.
        
        Args:
            env_path: Path to .env file (default: searches current directory)
            api_version: Override for AZURE_OPENAI_API_VERSION
            endpoint: Override for AZURE_OPENAI_ENDPOINT
            deployment: Override for AZURE_OPENAI_DEPLOYMENT
            api_key: Override for AZURE_OPENAI_API_KEY
        """
        # Load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()  # Searches for .env in current directory and parents
        
        # Get credentials from environment or overrides
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        
        # Validate required credentials
        missing_creds = []
        if not self.api_version:
            missing_creds.append('AZURE_OPENAI_API_VERSION')
        if not self.endpoint:
            missing_creds.append('AZURE_OPENAI_ENDPOINT')
        if not self.deployment:
            missing_creds.append('AZURE_OPENAI_DEPLOYMENT')
        if not self.api_key:
            missing_creds.append('AZURE_OPENAI_API_KEY')
        
        if missing_creds:
            raise ValueError(
                f"Missing required Azure OpenAI credentials: {', '.join(missing_creds)}. "
                f"Please set them in your .env file or pass as arguments."
            )
        
        # Initialize Azure OpenAI client
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for AzureOpenAIOrchestrator. "
                "Install it with: pip install openai"
            )
        
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )
        
        # Store for BaseOrchestrator (llm=None since we use self.client)
        super().__init__(llm=None, **kwargs)
    
    def completion(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generate completions using Azure OpenAI API.
        
        Args:
            message_histories: Message history or batch of message histories.
                             Each message should have 'role' and 'content' keys.
            **kwargs: Additional parameters for the API call:
                     - temperature (float): Sampling temperature (0-2)
                     - max_tokens (int): Maximum tokens to generate
                     - top_p (float): Nucleus sampling parameter
                     - frequency_penalty (float): Frequency penalty (0-2)
                     - presence_penalty (float): Presence penalty (0-2)
                     - do_sample (bool): Ignored for Azure OpenAI (use temperature instead)
        
        Returns:
            List[str]: Generated responses, one per input message history
        """
        # Normalize input to batch format
        histories, was_single = self._normalize_input(message_histories)
        
        # Prepare API parameters
        api_params = self._prepare_api_params(kwargs)
        
        # Generate responses for each history
        responses = []
        traces = []
        for history in histories:
            try:
                call_start = perf_counter()
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=history,
                    **api_params
                )
                call_elapsed = perf_counter() - call_start
                # Extract the message content
                content = response.choices[0].message.content
                responses.append(content)
                usage = getattr(response, "usage", None)
                usage_dict = None
                if usage is not None:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                traces.append({
                    "strategy": "azure_direct",
                    "deployment": self.deployment,
                    "total_elapsed_seconds": call_elapsed,
                    "steps": [
                        {
                            "type": "api_call",
                            "messages": history,
                            "response": content,
                            "usage": usage_dict,
                            "elapsed_seconds": call_elapsed,
                        }
                    ],
                    "usage": usage_dict,
                })
            except Exception as e:
                call_elapsed = perf_counter() - call_start if 'call_start' in locals() else 0.0
                print(f"Error calling Azure OpenAI API: {e}")
                # Return empty string on error to maintain batch consistency
                responses.append("")
                traces.append({
                    "strategy": "azure_direct",
                    "deployment": self.deployment,
                    "total_elapsed_seconds": call_elapsed,
                    "steps": [
                        {
                            "type": "api_call",
                            "messages": history,
                            "error": str(e),
                            "elapsed_seconds": call_elapsed,
                        }
                    ],
                    "usage": None,
                })
        self._record_trace(traces)
        return self._denormalize_output(responses, was_single)
    
    def _prepare_api_params(self, kwargs: dict) -> dict:
        """
        Prepare parameters for Azure OpenAI API call.
        
        Converts T-Eval style parameters to Azure OpenAI API parameters.
        
        Args:
            kwargs: Original parameters from T-Eval
        
        Returns:
            Dict with Azure OpenAI compatible parameters
        """
        api_params = {}
        
        # Map common parameters
        param_mapping = {
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'max_new_tokens': 'max_tokens',  # T-Eval uses max_new_tokens
            'top_p': 'top_p',
            'frequency_penalty': 'frequency_penalty',
            'presence_penalty': 'presence_penalty',
            'n': 'n',  # Number of completions
        }
        
        for src_key, dst_key in param_mapping.items():
            if src_key in kwargs:
                api_params[dst_key] = kwargs[src_key]
        
        # Handle do_sample parameter (T-Eval style)
        # If do_sample=False, set temperature=0 for deterministic output
        if 'do_sample' in kwargs and not kwargs['do_sample']:
            api_params['temperature'] = 0
        
        # Set default temperature if not specified
        if 'temperature' not in api_params:
            api_params['temperature'] = 0  # Deterministic by default
        
        return api_params
    
    def __repr__(self) -> str:
        """String representation of the orchestrator."""
        return (
            f"AzureOpenAIOrchestrator("
            f"deployment={self.deployment}, "
            f"endpoint={self.endpoint}, "
            f"api_version={self.api_version})"
        )

    # Compatibility shim so this orchestrator can be treated like an LLM in higher-level orchestrators.
    def chat(
        self,
        message_histories: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        return self.completion(message_histories, **kwargs)
