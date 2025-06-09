import os
from typing import Any, Dict, List, Optional

import tiktoken
from convorator.client.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Conversation,
    LLMInterface,
)
from convorator.conversations.types import LoggerProtocol
from convorator.exceptions import LLMClientError, LLMConfigurationError, LLMResponseError


# Model family to latest stable version mapping
# This allows users to specify model families and get the latest version automatically
MODEL_FAMILY_MAPPING: Dict[str, str] = {
    # GPT-4.1 series (latest flagship models)
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    # GPT-4o series (omni models)
    "gpt-4o": "gpt-4o-2024-11-20",  # Latest stable GPT-4o
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",  # Latest GPT-4o mini
    # GPT-4.5 Preview
    "gpt-4.5-preview": "gpt-4.5-preview-2025-02-27",
    # o-series reasoning models
    "o3": "o3-2025-04-16",
    "o4-mini": "o4-mini-2025-04-16",
    "o3-mini": "o3-mini-2025-01-31",
    "o1": "o1-2024-12-17",
    "o1-preview": "o1-preview-2024-09-12",
    "o1-mini": "o1-mini-2024-09-12",
    # GPT-4 Turbo series
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    # GPT-4 classic
    "gpt-4": "gpt-4-0613",  # Latest stable GPT-4
    # GPT-3.5 series
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",  # Latest GPT-3.5
    "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct-0914",
}

# Updated context limits with latest models
OPENAI_CONTEXT_LIMITS: Dict[str, Optional[int]] = {
    # GPT-4.1 series (latest flagship models with 1M+ context)
    "gpt-4.1-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "gpt-4.1-mini-2025-04-14": 1047576,
    # GPT-4.5 Preview
    "gpt-4.5-preview-2025-02-27": 128000,
    # o-series reasoning models
    "o3-2025-04-16": 200000,
    "o4-mini-2025-04-16": 200000,
    "o3-mini-2025-01-31": 200000,
    "o1-2024-12-17": 200000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini-2024-09-12": 128000,
    # GPT-4o series (updated versions)
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    # Legacy model names (for backward compatibility)
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-05-13": 128000,  # Specific version of gpt-4o
    # GPT-4 Turbo series
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,  # Vision model, context includes image tokens
    # GPT-4 series
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    # GPT-3.5 Turbo series
    "gpt-3.5-turbo": 16384,  # Often refers to gpt-3.5-turbo-0125
    "gpt-3.5-turbo-16k": 16384,  # Explicitly 16k
    "gpt-3.5-turbo-0125": 16384,
    "gpt-3.5-turbo-1106": 16384,
    # Older gpt-3.5-turbo variants might have been 4096, ensure this is handled if used.
    "gpt-3.5-turbo-0613": 4096,  # Older 4k variant
    "gpt-3.5-turbo-16k-0613": 16384,
    # Instruct models (typically smaller context)
    "gpt-3.5-turbo-instruct": 4096,
    "gpt-3.5-turbo-instruct-0914": 4096,
}
# A conservative default, typically the smallest of the widely available models.
DEFAULT_OPENAI_CONTEXT_LIMIT = 4096


class OpenAILLM(LLMInterface):
    """
    Concrete implementation for OpenAI's API (GPT models).

    This client supports automatic model family resolution, allowing users to specify
    model families (e.g., "gpt-4o") which automatically resolve to the latest stable
    version (e.g., "gpt-4o-2024-11-20"). Users can still specify exact model versions
    for full control.

    Model Family Examples:
    - "gpt-4o" → "gpt-4o-2024-11-20" (latest GPT-4o)
    - "gpt-4.1" → "gpt-4.1-2025-04-14" (latest GPT-4.1)
    - "o4-mini" → "o4-mini-2025-04-16" (latest o4-mini reasoning model)
    - "gpt-4o-2024-05-13" → "gpt-4o-2024-05-13" (specific version, no resolution)

    This ensures backward compatibility while providing access to the latest models
    without requiring code changes when new model versions are released.
    """

    # Updated supported models list with latest OpenAI models
    SUPPORTED_MODELS = [
        # GPT-4.1 series (latest flagship models)
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        # GPT-4.5 Preview
        "gpt-4.5-preview",
        "gpt-4.5-preview-2025-02-27",
        # o-series reasoning models
        "o3",
        "o3-2025-04-16",
        "o4-mini",
        "o4-mini-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o1",
        "o1-2024-12-17",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        # GPT-4o series (updated)
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        # GPT-4 Turbo series
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        # GPT-4 classic series
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        # GPT-3.5 series
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        # GPT-3.5 Instruct
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo-instruct-0914",
    ]

    # Updated tiktoken encoding mapping with new models
    # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    MODEL_TO_ENCODING = {
        # GPT-4.1 series - use o200k_base for latest models
        "gpt-4.1": "o200k_base",
        "gpt-4.1-2025-04-14": "o200k_base",
        "gpt-4.1-nano": "o200k_base",
        "gpt-4.1-nano-2025-04-14": "o200k_base",
        "gpt-4.1-mini": "o200k_base",
        "gpt-4.1-mini-2025-04-14": "o200k_base",
        # GPT-4.5 Preview
        "gpt-4.5-preview": "o200k_base",
        "gpt-4.5-preview-2025-02-27": "o200k_base",
        # o-series reasoning models
        "o3": "o200k_base",
        "o3-2025-04-16": "o200k_base",
        "o4-mini": "o200k_base",
        "o4-mini-2025-04-16": "o200k_base",
        "o3-mini": "o200k_base",
        "o3-mini-2025-01-31": "o200k_base",
        "o1": "o200k_base",
        "o1-2024-12-17": "o200k_base",
        "o1-preview": "o200k_base",
        "o1-preview-2024-09-12": "o200k_base",
        "o1-mini": "o200k_base",
        "o1-mini-2024-09-12": "o200k_base",
        # GPT-4o series - uses o200k_base for -o models
        "gpt-4o": "o200k_base",
        "gpt-4o-2024-11-20": "o200k_base",
        "gpt-4o-2024-08-06": "o200k_base",
        "gpt-4o-2024-05-13": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4o-mini-2024-07-18": "o200k_base",
        # GPT-4 Turbo series - uses cl100k_base
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-2024-04-09": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4-0125-preview": "cl100k_base",
        "gpt-4-1106-preview": "cl100k_base",
        "gpt-4-vision-preview": "cl100k_base",
        # GPT-4 classic series - uses cl100k_base
        "gpt-4": "cl100k_base",
        "gpt-4-0613": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-32k-0613": "cl100k_base",
        # GPT-3.5 series - uses cl100k_base
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-0125": "cl100k_base",
        "gpt-3.5-turbo-1106": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-0613": "cl100k_base",
        "gpt-3.5-turbo-16k-0613": "cl100k_base",
        # GPT-3.5 Instruct - uses p50k_base
        "gpt-3.5-turbo-instruct": "p50k_base",
        "gpt-3.5-turbo-instruct-0914": "p50k_base",
    }
    DEFAULT_ENCODING = "o200k_base"  # Updated default to latest encoding

    def __init__(
        self,
        logger: LoggerProtocol,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",  # Default to the latest powerful model family
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
        use_responses_api: bool = False,  # New parameter to enable Responses API
    ):
        """Initializes the OpenAI client.

        Requires the 'openai' package to be installed (`pip install openai`).
        Requires the 'tiktoken' package for accurate token counting (`pip install tiktoken`).

        Args:
            api_key: The OpenAI API key.
            model: The model name to use. Can be a specific version (e.g., "gpt-4o-2024-11-20")
                   or a model family (e.g., "gpt-4o") which will resolve to the latest stable version.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.
            system_message: The system message.
            role_name: The name for the assistant's role.
            use_responses_api: If True, use the new Responses API instead of Chat Completions.
                              This enables access to built-in tools (web search, file search, computer use)
                              and server-side conversation state management, affecting how conversation
                              history is handled with the API.

        Raises:
            LLMConfigurationError: If API key is missing or client initialization fails.
        """
        # Set logger first so it's available for model resolution
        self.logger = logger

        # Resolve model family to specific version if needed
        resolved_model = self._resolve_model_name(model)

        super().__init__(model=resolved_model, max_tokens=max_tokens)
        try:
            import openai

            self.openai = openai  # Store the module for access to error types
        except ImportError as e:
            raise LLMConfigurationError(
                "OpenAI Python package not found. Please install it using 'pip install openai'."
            ) from e

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or pass it during initialization."
            )

        # Store both original and resolved model names
        self.original_model = model
        self.resolved_model = resolved_model

        # Use the class attribute list for checking
        # self._model is now set by super().__init__ to resolved_model
        if self._model not in self.SUPPORTED_MODELS and resolved_model not in self.SUPPORTED_MODELS:
            self.logger.warning(
                f"Model '{self._model}' is not in the explicitly supported list for OpenAILLM context/token estimation. Proceeding, but compatibility is not guaranteed."
            )
        # self.model = model # This is now self._model, set by superclass
        self.temperature = temperature
        self._encoding: Optional[Any] = None  # Store tiktoken encoding object lazily

        # Responses API support
        self.use_responses_api = use_responses_api
        self._last_response_id: Optional[str] = (
            None  # Track last response ID for conversation chaining
        )

        try:
            # Initialize the OpenAI client instance
            self.client = self.openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            # Catch potential errors during client init (e.g., invalid key format - though API call usually catches this)
            raise LLMConfigurationError(f"Failed to initialize OpenAI client: {e}") from e

        self._system_message = system_message
        self._role_name = role_name or "Assistant"
        self.conversation = Conversation(system_message=self._system_message)

        api_type = "Responses API" if self.use_responses_api else "Chat Completions API"
        self.logger.info(
            f"OpenAILLM initialized. Original Model: {self.original_model}, Resolved Model: {self._model}, API: {api_type}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}"
        )

        # Fetch context limit during initialization
        self._context_limit = (
            OPENAI_CONTEXT_LIMITS.get(self._model, DEFAULT_OPENAI_CONTEXT_LIMIT)
            or DEFAULT_OPENAI_CONTEXT_LIMIT
        )
        if self._model not in OPENAI_CONTEXT_LIMITS:
            self.logger.warning(
                f"Context limit not defined for OpenAI model '{self._model}'. Using default: {self._context_limit}"
            )

        # Check max_tokens against the known limit
        if self.max_tokens > self._context_limit:
            self.logger.warning(
                f"Requested max_tokens ({self.max_tokens}) exceeds the known context limit ({self._context_limit}) for model '{self._model}'. API calls might fail."
            )
        elif self.max_tokens > self._context_limit * 0.8:
            self.logger.info(
                f"Requested max_tokens ({self.max_tokens}) is close to the context limit ({self._context_limit}). Ensure input + output fits."
            )

    def _resolve_model_name(self, model: str) -> str:
        """
        Resolves a model name to its specific version.

        If the model is in MODEL_FAMILY_MAPPING, returns the latest stable version.
        Otherwise, returns the model name as-is for backward compatibility.

        Args:
            model: The model name or family (e.g., "gpt-4o" or "gpt-4o-2024-11-20")

        Returns:
            The resolved model name (specific version)
        """
        if model in MODEL_FAMILY_MAPPING:
            resolved = MODEL_FAMILY_MAPPING[model]
            self.logger.debug(f"Resolved model family '{model}' to specific version '{resolved}'")
            return resolved
        else:
            # Model is already a specific version or not in family mapping
            self.logger.debug(f"Using model name as-is: '{model}'")
            return model

    def get_model_info(self) -> Dict[str, str]:
        """
        Returns information about the model being used.

        Returns:
            Dictionary containing original model name, resolved model name, and family info
        """
        return {
            "original_model": getattr(self, "original_model", self._model),
            "resolved_model": self._model,
            "is_family_resolved": hasattr(self, "original_model")
            and self.original_model != self._model,
            "family_mapping_available": (
                self.original_model in MODEL_FAMILY_MAPPING
                if hasattr(self, "original_model")
                else False
            ),
        }

    @classmethod
    def get_available_model_families(cls) -> Dict[str, str]:
        """
        Returns the available model families and their current latest versions.

        Returns:
            Dictionary mapping model families to their latest versions
        """
        return MODEL_FAMILY_MAPPING.copy()

    @classmethod
    def get_latest_model_for_family(cls, family: str) -> Optional[str]:
        """
        Gets the latest model for a given family.

        Args:
            family: The model family name (e.g., "gpt-4o")

        Returns:
            The latest model version for that family, or None if family not found
        """
        return MODEL_FAMILY_MAPPING.get(family)

    def _get_tiktoken_encoding(self):
        """Lazily loads and returns the tiktoken encoding for the current model."""
        if self._encoding is None:
            # First, check if the tiktoken library was imported successfully.
            # No warning here; the public `count_tokens` method will issue the warning.
            if not tiktoken:
                self._encoding = "FAILED_TO_LOAD"
                return None

            encoding_name = self.MODEL_TO_ENCODING.get(self._model, self.DEFAULT_ENCODING)
            if encoding_name != self.DEFAULT_ENCODING and self._model not in self.MODEL_TO_ENCODING:
                self.logger.warning(
                    f"Using default encoding '{self.DEFAULT_ENCODING}' for unknown OpenAI model '{self._model}'. Token count may be inaccurate."
                )

            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
                self.logger.debug(
                    f"Loaded tiktoken encoding: {encoding_name} for model {self._model}"
                )
            except ValueError as e:
                self.logger.error(
                    f"Failed to get tiktoken encoding '{encoding_name}': {e}. Falling back to default."
                )
                try:
                    self._encoding = tiktoken.get_encoding(self.DEFAULT_ENCODING)
                    self.logger.debug(f"Using fallback tiktoken encoding: {self.DEFAULT_ENCODING}")
                except ValueError as e_fallback:
                    self.logger.error(
                        f"Failed to get fallback tiktoken encoding '{self.DEFAULT_ENCODING}': {e_fallback}. Token counting disabled."
                    )
                    # Set encoding to a value indicating failure, but not None to avoid retrying
                    self._encoding = "FAILED_TO_LOAD"

        # Return None if loading failed
        return None if self._encoding == "FAILED_TO_LOAD" else self._encoding

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the appropriate OpenAI API based on configuration."""
        if self.use_responses_api:
            return self._call_responses_api(messages)
        else:
            return self._call_chat_completions_api(messages)

    def _call_chat_completions_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the OpenAI Chat Completions API."""
        if not messages:
            raise LLMClientError("Cannot call OpenAI API with an empty message list.")

        self.logger.debug(
            f"Calling OpenAI Chat Completions API ({self._model}) with {len(messages)} messages. First message role: {messages[0].get('role')}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore  # API expects list of {'role': str, 'content': str}
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # --- Response Validation ---
            if not response.choices:
                self.logger.error(f"OpenAI response missing 'choices'. Response: {response}")
                raise LLMResponseError(f"Invalid response structure from OpenAI: No 'choices'.")

            first_choice = response.choices[0]
            if not first_choice.message:
                self.logger.error(f"OpenAI response choice missing 'message'. Response: {response}")
                raise LLMResponseError(
                    f"Invalid response structure from OpenAI: Choice missing 'message'."
                )

            content = first_choice.message.content

            if content is None:
                # Check finish reason if content is None
                finish_reason = first_choice.finish_reason
                self.logger.error(
                    f"OpenAI response content is null. Finish Reason: {finish_reason}. Response: {response}"
                )
                # Map common reasons to errors
                if finish_reason == "content_filter":
                    raise LLMResponseError("OpenAI response blocked due to content filter.")
                elif finish_reason == "length":
                    raise LLMResponseError(
                        f"OpenAI response truncated due to max_tokens ({self.max_tokens}) or other length limits."
                    )
                else:
                    raise LLMResponseError(
                        f"OpenAI returned null content. Finish Reason: {finish_reason}."
                    )

            content = content.strip()
            self.logger.debug(
                f"Received content from OpenAI Chat Completions API. Length: {len(content)}. Finish Reason: {first_choice.finish_reason}"
            )
            if not content and first_choice.finish_reason == "stop":
                self.logger.warning(
                    f"Received empty content string from OpenAI API, but finish reason was 'stop'. Response: {response}"
                )
                # Decide if empty content with 'stop' is an error or valid (e.g., model chose not to respond)
                # For now, let's return it, but a stricter check might raise LLMResponseError here.

            return content

        # --- Error Handling ---
        except self.openai.APIConnectionError as e:
            self.logger.error(f"OpenAI API connection error: {e}")
            raise LLMClientError(f"Network error connecting to OpenAI API: {e}") from e
        except self.openai.RateLimitError as e:
            self.logger.error(f"OpenAI API rate limit exceeded: {e}")
            raise LLMResponseError(
                f"OpenAI API rate limit exceeded. Check your plan and usage limits. Error: {e}"
            ) from e
        except self.openai.AuthenticationError as e:
            self.logger.error(f"OpenAI API authentication error: {e}")
            raise LLMConfigurationError(
                f"OpenAI API authentication failed. Check your API key. Error: {e}"
            ) from e
        except self.openai.PermissionDeniedError as e:
            self.logger.error(f"OpenAI API permission denied: {e}")
            raise LLMConfigurationError(
                f"OpenAI API permission denied. Check API key permissions or organization access. Error: {e}"
            ) from e
        except self.openai.NotFoundError as e:
            self.logger.error(f"OpenAI API resource not found (e.g., model): {e}")
            raise LLMConfigurationError(
                f"OpenAI API resource not found (check model name '{self._model}'?). Error: {e}"
            ) from e
        except self.openai.BadRequestError as e:  # Often indicates input schema issues
            self.logger.error(f"OpenAI API bad request error: {e}")
            raise LLMResponseError(
                f"OpenAI API reported a bad request (check message format or parameters?). Error: {e}"
            ) from e
        except self.openai.APIStatusError as e:  # Catch other 4xx/5xx errors
            self.logger.error(f"OpenAI API status error: {e.status_code} - {e.response}")
            # Attempt to extract a message from the response body if possible
            error_detail = str(e)
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", str(e))
            except:
                pass
            raise LLMResponseError(
                f"OpenAI API returned an error (Status {e.status_code}): {error_detail}"
            ) from e
        # Catch other potential OpenAI client errors
        except self.openai.OpenAIError as e:
            self.logger.error(f"An unexpected OpenAI client error occurred: {e}")
            raise LLMClientError(f"An unexpected OpenAI client error occurred: {e}") from e
        # --- Allow specific LLMResponseErrors from validation to pass through ---
        except LLMResponseError as e:
            # This catches errors raised explicitly in the validation logic above
            # (e.g., missing choices, null content)
            self.logger.error(f"LLM Response Error: {e}")  # Log it, but let it propagate
            raise
        # General exception catcher (less likely with specific catches above)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during OpenAI API call: {e}")
            raise LLMClientError(f"Unexpected error during OpenAI API call: {e}") from e

    def _call_responses_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the OpenAI Responses API with server-side conversation state management."""
        if not messages:
            raise LLMClientError("Cannot call OpenAI Responses API with an empty message list.")

        self.logger.debug(
            f"Calling OpenAI Responses API ({self._model}) with {len(messages)} messages. Last response ID: {self._last_response_id}"
        )

        try:
            request_params = {
                "model": self._model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,  # Responses API uses max_output_tokens
                "store": True,  # Enable server-side conversation storage
            }

            # Handle conversation continuation vs new conversation
            # If _last_response_id is set, we are continuing a conversation.
            # The 'messages' list in this case should ideally be just the new user message.
            if self._last_response_id:
                if not messages or messages[-1].get("role") != "user":
                    raise LLMClientError(
                        "For continued Responses API conversation, messages should end with the new user prompt."
                    )
                request_params["input"] = [messages[-1]]  # Send only the last user message
                request_params["previous_response_id"] = self._last_response_id
                self.logger.debug(
                    f"Continuing Responses API conversation with previous_response_id: {self._last_response_id}. Input: {messages[-1]}"
                )
            else:
                # New conversation: extract system message for 'instructions', rest for 'input'
                system_message_content = None
                input_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        system_message_content = msg.get("content")
                    else:
                        input_messages.append(msg)

                if system_message_content:
                    request_params["instructions"] = system_message_content
                    self.logger.debug(
                        f"Using system message as instructions: {system_message_content[:100]}..."
                    )

                if not input_messages:
                    raise LLMClientError(
                        "Cannot start Responses API conversation with no user/assistant messages."
                    )
                request_params["input"] = input_messages
                self.logger.debug(
                    f"Starting new Responses API conversation with {len(input_messages)} input messages."
                )

            response = self.client.responses.create(**request_params)
            content = self._extract_responses_content(response)

            if hasattr(response, "id"):
                self._last_response_id = response.id
                self.logger.debug(
                    f"Stored response ID for conversation continuation: {self._last_response_id}"
                )
            return content

        except self.openai.APIConnectionError as e:
            self.logger.error(f"OpenAI Responses API connection error: {e}")
            self._last_response_id = None  # Clear potentially stale ID on connection issues
            raise LLMClientError(f"Network error connecting to OpenAI Responses API: {e}") from e
        except self.openai.RateLimitError as e:
            self.logger.error(f"OpenAI Responses API rate limit exceeded: {e}")
            raise LLMResponseError(
                f"OpenAI Responses API rate limit exceeded. Check your plan and usage limits. Error: {e}"
            ) from e
        except self.openai.AuthenticationError as e:
            self.logger.error(f"OpenAI Responses API authentication error: {e}")
            self._last_response_id = None  # Clear ID on auth failure
            raise LLMConfigurationError(
                f"OpenAI Responses API authentication failed. Check your API key. Error: {e}"
            ) from e
        except self.openai.PermissionDeniedError as e:
            self.logger.error(f"OpenAI Responses API permission denied: {e}")
            self._last_response_id = None  # Clear ID on permission failure
            raise LLMConfigurationError(
                f"OpenAI Responses API permission denied. Check API key permissions or organization access. Error: {e}"
            ) from e
        except self.openai.NotFoundError as e:
            self.logger.error(
                f"OpenAI Responses API resource not found (e.g., model, response ID): {e}"
            )
            # If it's a continued conversation, the previous_response_id might be invalid
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self.logger.warning(
                    f"Clearing potentially invalid previous_response_id: {self._last_response_id}"
                )
                self._last_response_id = None
            raise LLMConfigurationError(
                f"OpenAI Responses API resource not found (check model '{self._model}' or conversation state?). Error: {e}"
            ) from e
        except self.openai.BadRequestError as e:  # Often indicates input schema issues
            self.logger.error(f"OpenAI Responses API bad request error: {e}")
            # Potentially reset _last_response_id if it seems to be causing the issue
            if self._last_response_id and (
                "previous_response_id" in str(e).lower() or "input" in str(e).lower()
            ):
                self.logger.warning(
                    f"Bad request with Responses API, clearing potentially problematic previous_response_id: {self._last_response_id}"
                )
                self._last_response_id = None
            raise LLMResponseError(
                f"OpenAI Responses API reported a bad request (check message format, parameters, or conversation state?). Error: {e}"
            ) from e
        except self.openai.APIStatusError as e:  # Catch other 4xx/5xx errors
            self.logger.error(f"OpenAI Responses API status error: {e.status_code} - {e.response}")
            error_detail = str(e)
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", str(e))
            except:
                pass  # Keep original error_detail if parsing fails
            if e.status_code == 500:
                self._last_response_id = (
                    None  # Clear ID on server errors as state might be corrupted
                )
            raise LLMResponseError(
                f"OpenAI Responses API returned an error (Status {e.status_code}): {error_detail}"
            ) from e
        except self.openai.OpenAIError as e:
            self.logger.error(f"An unexpected OpenAI Responses API client error occurred: {e}")
            self._last_response_id = None  # Clear ID on unknown OpenAI errors
            raise LLMClientError(
                f"An unexpected OpenAI Responses API client error occurred: {e}"
            ) from e
        except (
            LLMResponseError
        ):  # Re-raise specific LLMResponseErrors from _extract_responses_content
            raise
        except Exception as e:
            self.logger.exception(
                f"An unexpected error occurred during OpenAI Responses API call: {e}"
            )
            self._last_response_id = None  # Clear ID on any other unexpected error
            raise LLMClientError(f"Unexpected error during OpenAI Responses API call: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using tiktoken for the configured model."""
        encoding = self._get_tiktoken_encoding()

        if encoding:
            try:
                num_tokens = len(encoding.encode(text))
                self.logger.debug(
                    f"Counted {num_tokens} tokens for text (length {len(text)}) using {encoding.name}."
                )
                return num_tokens
            except Exception as e:
                fallback_reason = f"tiktoken encoding error ({encoding.name}): {e}"
                self.logger.error(
                    f"Error using tiktoken encoding to count tokens: {e}", exc_info=True
                )
        else:
            fallback_reason = "tiktoken library not available or encoding failed to load"

        # Fallback approximation
        estimated_tokens = len(text) // 4
        self.logger.warning(
            f"Using approximate token count ({estimated_tokens}) due to: {fallback_reason}. "
            f"For accurate counts, ensure 'tiktoken' is installed and the model name is correct."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) fetched during initialization."""
        self.logger.debug(f"Returning context limit for {self._model}: {self._context_limit}")
        return self._context_limit

    # --- OpenAI-Specific Provider Methods ---

    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Returns OpenAI-specific capabilities."""
        base_capabilities = super().get_provider_capabilities()
        base_capabilities.update(
            {
                "supports_tiktoken": True,
                "supports_streaming": True,  # OpenAI supports streaming
                "supported_models": self.SUPPORTED_MODELS,
                "encoding_info": {
                    "default_encoding": self.DEFAULT_ENCODING,
                    "model_encodings": self.MODEL_TO_ENCODING,
                },
                "context_limits": OPENAI_CONTEXT_LIMITS,
                # Model family resolution capabilities
                "supports_model_families": True,
                "model_family_mapping": MODEL_FAMILY_MAPPING,
                "current_model_info": self.get_model_info(),
                # Responses API capabilities
                "supports_responses_api": True,
                "responses_api_enabled": self.use_responses_api,
                "supports_web_search": self.use_responses_api,
                "supports_file_search": self.use_responses_api,
                "supports_computer_use": self.use_responses_api,
                "supports_server_side_conversation": self.use_responses_api,
            }
        )
        return base_capabilities

    def get_provider_settings(self) -> Dict[str, Any]:
        """Returns OpenAI-specific settings."""
        base_settings = super().get_provider_settings()
        base_settings.update(
            {
                "api_key": self.api_key[:10] + "..." if self.api_key else None,
                "temperature": self.temperature,
                "encoding": (
                    self._get_tiktoken_encoding().name
                    if self._get_tiktoken_encoding()
                    else "unavailable"
                ),
                "use_responses_api": self.use_responses_api,
                "last_response_id": getattr(self, "_last_response_id", None),
                # Model family resolution settings
                "original_model": getattr(self, "original_model", self._model),
                "resolved_model": self._model,
                "model_family_resolved": hasattr(self, "original_model")
                and self.original_model != self._model,
            }
        )
        return base_settings

    def set_provider_setting(self, setting_name: str, value: Any) -> bool:
        """
        Sets OpenAI-specific settings and overrides base settings where applicable.
        """
        # Handle OpenAI-specific settings first
        if setting_name == "temperature":
            if isinstance(value, (int, float)) and 0 <= value <= 2:
                self.temperature = float(value)
                self.logger.info(f"Updated OpenAI temperature to {self.temperature}")
                return True
            else:
                self.logger.warning(f"Invalid temperature value: {value}. Must be between 0 and 2.")
                return False

        if setting_name == "use_responses_api":
            if isinstance(value, bool):
                old_value = self.use_responses_api
                self.use_responses_api = value
                if old_value != value:
                    # Reset conversation state when switching APIs
                    self._last_response_id = None
                    self.logger.info(
                        f"Switched OpenAI API mode: Responses API {'enabled' if value else 'disabled'}"
                    )
                return True
            else:
                self.logger.warning(f"Invalid use_responses_api value: {value}. Must be boolean.")
                return False

        # If not an OpenAI-specific setting, delegate to the base class
        return super().set_provider_setting(setting_name, value)

    def supports_feature(self, feature_name: str) -> bool:
        """Checks OpenAI-specific feature support."""
        if super().supports_feature(feature_name):
            return True

        openai_features = {
            "tiktoken_encoding",
            "streaming",
            "function_calling",  # Most OpenAI models support this
            "vision",  # Some models like gpt-4-vision-preview
            "model_family_resolution",  # New feature for automatic model family to version mapping
            "model_family_mapping",  # New feature for model family support
        }

        # Responses API specific features
        responses_api_features = {
            "responses_api",
            "web_search",
            "file_search",
            "computer_use",
            "server_side_conversation",
        }

        if feature_name in responses_api_features:
            return self.use_responses_api

        return feature_name in openai_features

    def get_tiktoken_encoding_info(self) -> Dict[str, Any]:
        """
        Returns information about the tiktoken encoding for this model.
        OpenAI-specific method.

        Returns:
            Dictionary with encoding information.
        """
        encoding = self._get_tiktoken_encoding()
        if encoding:
            return {
                "encoding_name": encoding.name,
                "model": self._model,
                "available": True,
            }
        else:
            return {
                "encoding_name": None,
                "model": self._model,
                "available": False,
                "fallback_used": True,
            }

    # --- Responses API Methods ---

    def clear_conversation(self, keep_system: bool = True):
        """Clears the internal conversation history and resets response ID for Responses API."""
        super().clear_conversation(keep_system=keep_system)
        # Reset Responses API conversation state
        if self.use_responses_api:
            self._last_response_id = None
            self.logger.debug("Reset Responses API conversation state (response ID cleared)")

    def query_with_web_search(
        self, prompt: str, user_location: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Query using the Responses API with built-in web search tool.

        Args:
            prompt: The user's query
            user_location: Optional location context for search (e.g., {"country": "US", "city": "New York"})

        Returns:
            The response with web search results incorporated

        Raises:
            LLMConfigurationError: If not using Responses API
        """
        if not self.use_responses_api:
            raise LLMConfigurationError(
                "Web search tool requires Responses API. Set use_responses_api=True when initializing OpenAILLM."
            )
        self.logger.debug(f"Querying with web search: {prompt[:100]}...")
        try:
            request_params = {
                "model": self._model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "store": True,
                "tools": [{"type": "web_search_preview"}],
            }
            if user_location:
                request_params["tools"][0]["user_location"] = user_location

            if self._last_response_id:
                request_params["input"] = [{"role": "user", "content": prompt}]
                request_params["previous_response_id"] = self._last_response_id
            else:
                if self._system_message:
                    request_params["instructions"] = self._system_message
                request_params["input"] = [{"role": "user", "content": prompt}]

            response = self.client.responses.create(**request_params)
            content = self._extract_responses_content(response)
            if hasattr(response, "id"):
                self._last_response_id = response.id
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(content)
            return content
        except self.openai.APIConnectionError as e:
            self.logger.error(f"OpenAI Web Search query connection error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Network error during OpenAI Web Search: {e}") from e
        except self.openai.RateLimitError as e:
            self.logger.error(f"OpenAI Web Search query rate limit exceeded: {e}")
            raise LLMResponseError(f"OpenAI Web Search rate limit exceeded: {e}") from e
        except self.openai.AuthenticationError as e:
            self.logger.error(f"OpenAI Web Search query authentication error: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Web Search authentication failed: {e}") from e
        except self.openai.PermissionDeniedError as e:
            self.logger.error(f"OpenAI Web Search query permission denied: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Web Search permission denied: {e}") from e
        except self.openai.NotFoundError as e:
            self.logger.error(f"OpenAI Web Search query resource not found: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Web Search resource not found: {e}") from e
        except self.openai.BadRequestError as e:
            self.logger.error(f"OpenAI Web Search query bad request: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMResponseError(f"OpenAI Web Search bad request: {e}") from e
        except self.openai.APIStatusError as e:
            self.logger.error(f"OpenAI Web Search query API status error: {e.status_code}")
            if e.status_code == 500:
                self._last_response_id = None
            raise LLMResponseError(
                f"OpenAI Web Search API error (Status {e.status_code}): {e}"
            ) from e
        except self.openai.OpenAIError as e:
            self.logger.error(f"OpenAI Web Search query client error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"OpenAI Web Search client error: {e}") from e
        except LLMResponseError:
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error in web search query: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Unexpected error in web search query: {e}") from e

    def query_with_file_search(
        self, prompt: str, vector_store_ids: List[str], filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Query using the Responses API with built-in file search tool.

        Args:
            prompt: The user's query
            vector_store_ids: List of vector store IDs to search
            filters: Optional filters for the search

        Returns:
            The response with file search results incorporated

        Raises:
            LLMConfigurationError: If not using Responses API
        """
        if not self.use_responses_api:
            raise LLMConfigurationError(
                "File search tool requires Responses API. Set use_responses_api=True when initializing OpenAILLM."
            )
        self.logger.debug(f"Querying with file search: {prompt[:100]}...")
        try:
            file_search_tool = {"type": "file_search", "vector_store_ids": vector_store_ids}
            if filters:
                file_search_tool["filters"] = filters
            request_params = {
                "model": self._model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "store": True,
                "tools": [file_search_tool],
            }
            if self._last_response_id:
                request_params["input"] = [{"role": "user", "content": prompt}]
                request_params["previous_response_id"] = self._last_response_id
            else:
                if self._system_message:
                    request_params["instructions"] = self._system_message
                request_params["input"] = [{"role": "user", "content": prompt}]

            response = self.client.responses.create(**request_params)
            content = self._extract_responses_content(response)
            if hasattr(response, "id"):
                self._last_response_id = response.id
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(content)
            return content
        except self.openai.APIConnectionError as e:
            self.logger.error(f"OpenAI File Search query connection error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Network error during OpenAI File Search: {e}") from e
        except self.openai.RateLimitError as e:
            self.logger.error(f"OpenAI File Search query rate limit exceeded: {e}")
            raise LLMResponseError(f"OpenAI File Search rate limit exceeded: {e}") from e
        except self.openai.AuthenticationError as e:
            self.logger.error(f"OpenAI File Search query authentication error: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI File Search authentication failed: {e}") from e
        except self.openai.PermissionDeniedError as e:
            self.logger.error(f"OpenAI File Search query permission denied: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI File Search permission denied: {e}") from e
        except self.openai.NotFoundError as e:
            self.logger.error(f"OpenAI File Search query resource not found: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI File Search resource not found: {e}") from e
        except self.openai.BadRequestError as e:
            self.logger.error(f"OpenAI File Search query bad request: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMResponseError(f"OpenAI File Search bad request: {e}") from e
        except self.openai.APIStatusError as e:
            self.logger.error(f"OpenAI File Search query API status error: {e.status_code}")
            if e.status_code == 500:
                self._last_response_id = None
            raise LLMResponseError(
                f"OpenAI File Search API error (Status {e.status_code}): {e}"
            ) from e
        except self.openai.OpenAIError as e:
            self.logger.error(f"OpenAI File Search query client error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"OpenAI File Search client error: {e}") from e
        except LLMResponseError:
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error in file search query: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Unexpected error in file search query: {e}") from e

    def query_with_computer_use(
        self,
        prompt: str,
        display_width: int = 1024,
        display_height: int = 768,
        environment: str = "browser",
    ) -> str:
        """
        Query using the Responses API with built-in computer use tool.

        Args:
            prompt: The user's query/instruction for computer interaction
            display_width: Display width for computer use
            display_height: Display height for computer use
            environment: Environment type ("browser", "mac", "windows", "ubuntu")

        Returns:
            The response from computer use interaction

        Raises:
            LLMConfigurationError: If not using Responses API
        """
        if not self.use_responses_api:
            raise LLMConfigurationError(
                "Computer use tool requires Responses API. Set use_responses_api=True when initializing OpenAILLM."
            )
        self.logger.debug(f"Querying with computer use: {prompt[:100]}...")
        try:
            request_params = {
                "model": self._model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "store": True,
                "tools": [
                    {
                        "type": "computer_use_preview",
                        "display_width": display_width,
                        "display_height": display_height,
                        "environment": environment,
                    }
                ],
            }
            if self._last_response_id:
                request_params["input"] = [{"role": "user", "content": prompt}]
                request_params["previous_response_id"] = self._last_response_id
            else:
                if self._system_message:
                    request_params["instructions"] = self._system_message
                request_params["input"] = [{"role": "user", "content": prompt}]

            response = self.client.responses.create(**request_params)
            content = self._extract_responses_content(response)
            if hasattr(response, "id"):
                self._last_response_id = response.id
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(content)
            return content
        except self.openai.APIConnectionError as e:
            self.logger.error(f"OpenAI Computer Use query connection error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Network error during OpenAI Computer Use: {e}") from e
        except self.openai.RateLimitError as e:
            self.logger.error(f"OpenAI Computer Use query rate limit exceeded: {e}")
            raise LLMResponseError(f"OpenAI Computer Use rate limit exceeded: {e}") from e
        except self.openai.AuthenticationError as e:
            self.logger.error(f"OpenAI Computer Use query authentication error: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Computer Use authentication failed: {e}") from e
        except self.openai.PermissionDeniedError as e:
            self.logger.error(f"OpenAI Computer Use query permission denied: {e}")
            self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Computer Use permission denied: {e}") from e
        except self.openai.NotFoundError as e:
            self.logger.error(f"OpenAI Computer Use query resource not found: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMConfigurationError(f"OpenAI Computer Use resource not found: {e}") from e
        except self.openai.BadRequestError as e:
            self.logger.error(f"OpenAI Computer Use query bad request: {e}")
            if self._last_response_id and "previous_response_id" in str(e).lower():
                self._last_response_id = None
            raise LLMResponseError(f"OpenAI Computer Use bad request: {e}") from e
        except self.openai.APIStatusError as e:
            self.logger.error(f"OpenAI Computer Use query API status error: {e.status_code}")
            if e.status_code == 500:
                self._last_response_id = None
            raise LLMResponseError(
                f"OpenAI Computer Use API error (Status {e.status_code}): {e}"
            ) from e
        except self.openai.OpenAIError as e:
            self.logger.error(f"OpenAI Computer Use query client error: {e}")
            self._last_response_id = None
            raise LLMClientError(f"OpenAI Computer Use client error: {e}") from e
        except LLMResponseError:
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error in computer use query: {e}")
            self._last_response_id = None
            raise LLMClientError(f"Unexpected error in computer use query: {e}") from e

    def _extract_responses_content(self, response) -> str:
        """Helper method to extract text content from Responses API response."""
        content = ""

        if hasattr(response, "output_text"):
            # Simple case - direct text output
            return response.output_text.strip()

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                if hasattr(output_item, "type") and output_item.type == "message":
                    if hasattr(output_item, "content"):
                        for content_item in output_item.content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                content += content_item.text

        if not content:
            self.logger.error(
                f"No text content found in Responses API output. Response: {response}"
            )
            raise LLMResponseError("No text content found in OpenAI Responses API response.")

        return content.strip()
