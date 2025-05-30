from dataclasses import dataclass
from typing import Optional
from convorator.client.anthropic_client import AnthropicLLM
from convorator.client.gemini_client import GeminiLLM
from convorator.client.llm_client import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, LLMInterface
from convorator.client.openai_client import OpenAILLM
from convorator.conversations.types import LoggerProtocol
from convorator.exceptions import LLMClientError, LLMConfigurationError


_PROVIDER_MAP = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "gemini": GeminiLLM,
    # Add aliases if needed
    "gpt": OpenAILLM,
    "claude": AnthropicLLM,
    "google": GeminiLLM,
}


@dataclass
class LLMClientConfig:
    client_type: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    system_message: Optional[str] = None
    role_name: Optional[str] = None


def create_llm_client(
    logger: LoggerProtocol,
    client_type: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,  # If None, provider class will use its default
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_message: Optional[str] = None,
    role_name: Optional[str] = None,
    use_responses_api: bool = False,  # New parameter for OpenAI Responses API
) -> LLMInterface:
    """
    Factory function to create LLM client instances based on type.

    Args:
        client_type: The type of client ('openai', 'anthropic', 'gemini', or aliases).
        api_key: The API key (optional, will check environment variables).
        model: The specific model name (optional, uses provider default if None).
        temperature: The generation temperature.
        max_tokens: The maximum number of tokens to generate.
        system_message: The initial system message/instruction.
        role_name: An optional cosmetic name for the assistant role.
        use_responses_api: For OpenAI only - use Responses API instead of Chat Completions.
                          Enables built-in tools (web search, file search, computer use).

    Returns:
        An instance of a class implementing LLMInterface.

    Raises:
        LLMConfigurationError: If client_type is unsupported or initialization fails due to config.
        LLMClientError: For other unexpected errors during creation.
    """

    client_type_lower = client_type.lower()
    logger.info(f"Attempting to create LLM client of type: '{client_type_lower}'")

    provider_class = _PROVIDER_MAP.get(client_type_lower)

    if not provider_class:
        supported = list(_PROVIDER_MAP.keys())
        raise LLMConfigurationError(
            f"Unsupported LLM client type: '{client_type}'. Supported types: {supported}."
        )

    try:
        # Prepare arguments, letting None be passed if model is not specified
        # The provider class __init__ will handle default model logic
        client_args = {
            "logger": logger,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_message": system_message,
            "role_name": role_name,
        }
        if model:
            client_args["model"] = model

        # Add use_responses_api parameter only for OpenAI
        if client_type_lower in ["openai", "gpt"]:
            client_args["use_responses_api"] = use_responses_api
            if use_responses_api:
                logger.info(f"Creating OpenAI client with Responses API enabled")

        # Instantiate the chosen provider class
        instance = provider_class(**client_args)

        # Get provider name safely - handles mocks in tests
        provider_name = (
            provider_class.__name__ if hasattr(provider_class, "__name__") else str(provider_class)
        )
        logger.info(f"Successfully created LLM client instance: {provider_name}")
        return instance

    except LLMConfigurationError as e:
        # Catch config errors during specific client init and re-raise
        logger.error(f"Configuration error creating LLM client '{client_type}': {e}")
        raise
    except LLMClientError as e:
        # Catch other client errors during init
        logger.error(f"Client error creating LLM client '{client_type}': {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors during client creation
        logger.exception(f"Unexpected error creating LLM client '{client_type}': {e}")
        # Wrap in LLMClientError for consistency
        raise LLMClientError(
            f"An unexpected error occurred while creating LLM client '{client_type}': {e}"
        ) from e
