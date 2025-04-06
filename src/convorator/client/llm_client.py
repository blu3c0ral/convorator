# llm_client.py
"""
llm_client: Unified Interface for Large Language Model APIs

This module provides a consistent, provider-agnostic interface for interacting with various
Large Language Model APIs including OpenAI's GPT models, Google's Gemini, and Anthropic's Claude.

Key Features:
- Multi-provider support with unified interface
- Conversation management with history tracking
- System message configuration
- Provider switching capabilities
- Temperature and model customization
- Simulated responses for testing without API keys

Architecture:
- LLMInterface: Main entry point providing a unified API for all supported providers
- LLMProvider: Abstract base class defining the common interface for all providers
- Provider implementations: Concrete implementations for OpenAI, Gemini, and Claude
- Conversation: Helper class for maintaining conversation state
- Message: Represents individual messages in a conversation

Dependencies:
- requests: For making HTTP requests to LLM APIs
- logging: For operational logging and debugging

Basic Usage:
    from convorator.llm_client import LLMInterface

    # Initialize with default provider (OpenAI)
    llm = LLMInterface(api_key="your-api-key")

    # Simple query
    response = llm.query("Explain quantum computing in simple terms")

    # Using conversation context
    llm = LLMInterface(system_message="You are a helpful AI assistant")
    response1 = llm.query("What is Python?", use_conversation=True)
    response2 = llm.query("What are its main features?", use_conversation=True)

    # Switching providers
    llm.switch_provider("claude", api_key="your-claude-api-key")

Authors:
    Development Team (convorator)

Version:
    1.0.0

TODO: Provider API versioning and compatibility checks
"""

from datetime import datetime, timezone
import functools
import json
import os
import time
from typing import Dict, List, Any, Optional
import uuid
import requests
from abc import ABC, abstractmethod


import convorator.utils.logger as setup_logger

# Configure logging
logger = setup_logger.setup_logger("llm_client")


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails (missing or invalid API key)."""

    pass


class APIError(LLMError):
    """Raised when the LLM API returns an error response."""

    def __init__(self, message, provider, status_code=None, response_text=None, request_info=None):
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text
        self.request_info = request_info  # redacted payload
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""

    pass


class ConnectionError(LLMError):
    """Raised for network connectivity issues."""

    pass


class ProviderError(LLMError):
    """Raised for provider-specific errors (e.g., parsing issues)."""

    def __init__(self, provider, original_error):
        self.provider = provider
        self.original_error = original_error
        message = f"Error from {provider} provider: {str(original_error)}"
        super().__init__(message)


# ---------------------------
# Centralized Exception Handler
# ---------------------------
def handle_provider_exceptions(provider_name: str):
    """
    Decorator that standardizes exception handling across LLM providers.

    This decorator wraps provider API calls to ensure consistent error handling
    by transforming provider-specific exceptions into the application's
    exception hierarchy. It also adds logging with request tracking.

    Args:
        provider_name (str): Name of the LLM provider being used (e.g., "OpenAI", "Claude")

    Returns:
        callable: Decorated function with standardized exception handling

    Raises:
        ConnectionError: For network timeouts and connectivity issues
        AuthenticationError: For authentication and authorization failures (401, 403)
        RateLimitError: When API rate limits are exceeded (429)
        APIError: For other API errors with context about the failure
        ProviderError: For unexpected errors from the provider (should be rare if specific errors are caught)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_id = uuid.uuid4().hex
            try:
                logger.info(
                    f"Sending request to {provider_name}",
                    extra={
                        "provider": provider_name,
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                logger.error(f"{provider_name} API request timed out.", exc_info=True)
                raise ConnectionError(f"{provider_name} API request timed out.") from e
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Network connection issue with {provider_name} API.", exc_info=True)
                raise ConnectionError(f"Network connection issue with {provider_name} API.") from e
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                response_text = e.response.text
                logger.error(
                    f"{provider_name} API returned HTTP {status_code}: {response_text}",
                    exc_info=True,
                )
                if status_code in (401, 403):
                    raise AuthenticationError(f"{provider_name} authentication failed.") from e
                elif status_code == 429:
                    raise RateLimitError(
                        "Rate limit exceeded",
                        provider=provider_name,
                        status_code=429,
                        response_text=response_text,
                    ) from e
                else:
                    raise APIError(
                        f"{provider_name} API error",
                        provider=provider_name,
                        status_code=status_code,
                        response_text=response_text,
                    ) from e
            # Catch specific ProviderErrors raised internally first
            except ProviderError as e:
                logger.error(
                    f"Provider-specific error during {provider_name} call: {e}", exc_info=True
                )
                raise  # Re-raise ProviderError as is
            # FIX: Catch other unexpected exceptions last
            except Exception as e:
                logger.exception(
                    f"Unexpected error during {provider_name} API call.", exc_info=True
                )
                # Wrap unexpected errors in ProviderError for consistency
                raise ProviderError(provider_name, e) from e

        return wrapper

    return decorator


# ---------------------------
# Retry Mechanism
# ---------------------------
def retry(max_attempts=3, backoff_factor=1, retry_statuses=(429, 500, 502, 503, 504)):
    """
    Decorator that implements exponential backoff retry logic for API calls.

    This decorator automatically retries failed API calls that result in
    transient errors such as rate limits or server errors. Each retry
    uses exponential backoff to increase the delay between attempts.

    Args:
        max_attempts (int): Maximum number of retry attempts (default: 3)
        backoff_factor (int): Base factor for exponential backoff calculation (default: 1)
        retry_statuses (tuple): HTTP status codes that should trigger a retry (default: 429, 500, 502, 503, 504)

    Returns:
        callable: Decorated function with retry capability

    Raises:
        ConnectionError: When maximum retry attempts are exceeded for network errors or retryable API errors.
        APIError: For non-retriable API errors.
        AuthenticationError: For authentication errors (not retried).
        ProviderError: For provider-specific errors (not retried by default).
        Other exceptions: Original exceptions for non-retriable errors.

    Note:
        The backoff delay is calculated as: backoff_factor * (2 ** attempt_number)
    """

    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            attempts = 0
            last_exception = None
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                # Catch specific retryable errors first
                except (RateLimitError, APIError) as e:
                    last_exception = e
                    # Check if it's an APIError with a retryable status code
                    if isinstance(e, APIError) and e.status_code in retry_statuses:
                        attempts += 1
                        if attempts >= max_attempts:
                            logger.error(
                                f"Max retry attempts ({max_attempts}) reached for API error."
                            )
                            break  # Exit loop to raise ConnectionError below
                        sleep_time = backoff_factor * (2 ** (attempts - 1))
                        logger.warning(
                            f"Retrying attempt {attempts}/{max_attempts} after {sleep_time:.2f}s due to API error: {e}"
                        )
                        time.sleep(sleep_time)
                    else:
                        # Non-retryable APIError or RateLimitError (if 429 not in retry_statuses, though unlikely)
                        raise
                except (
                    ConnectionError
                ) as e:  # Catch ConnectionError raised by handle_provider_exceptions
                    last_exception = e
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) reached for connection error."
                        )
                        break  # Exit loop to raise ConnectionError below
                    sleep_time = backoff_factor * (2 ** (attempts - 1))
                    logger.warning(
                        f"Retrying attempt {attempts}/{max_attempts} after {sleep_time:.2f}s due to connection error: {e}"
                    )
                    time.sleep(sleep_time)
                # Let AuthenticationError and non-retryable ProviderError propagate immediately
                except (AuthenticationError, ProviderError) as e:
                    raise
                # Catch other unexpected exceptions (should be rare if handle_provider_exceptions is robust)
                except Exception as e:
                    logger.exception("Caught unexpected exception in retry loop.")
                    raise  # Re-raise unexpected exceptions immediately

            # If loop finished due to max attempts, raise ConnectionError
            raise ConnectionError(
                f"Maximum retry attempts ({max_attempts}) exceeded."
            ) from last_exception

        return wrapper_retry

    return decorator_retry


class Message:
    """
    Represent a message in a conversation.
    """

    def __init__(self, role: str, content: str):
        """
        Initialize a message.

        Args:
            role: The role of the message sender
            content: The content of the message

        Raises:
            TypeError: If role or content are not strings.
        """
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("Role and content must be strings")
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the message to a dictionary format.

        Returns:
            Dict[str, str]: Dictionary with 'role' and 'content' keys.
        """
        return {"role": self.role, "content": self.content}


class ProviderMessage(Message):
    """
    Represents a single message in an LLM conversation.
    """

    def __init__(self, role: str, content: str):
        """
        Initialize a message.

        Args:
            role: The role of the message sender ('system', 'user', or 'assistant')
            content: The content of the message

        Raises:
            TypeError: If role or content are not strings.
            ValueError: If role is not one of 'system', 'user', or 'assistant'.
        """
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("Role and content must be strings")
        if role not in ["system", "user", "assistant"]:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        super().__init__(role, content)


class Conversation:
    """
    Maintains the state of a conversation with an LLM.
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize a conversation.

        Args:
            system_message (Optional[str]): Initial system message. Defaults to None.
        """
        self.messages: List[ProviderMessage] = []
        if system_message:
            # Use the proper method to ensure correct placement/replacement
            self.set_system_message(system_message)

    def set_system_message(self, content: str) -> None:
        """
        Set or update the system message for the conversation.

        If a system message exists, it's updated. Otherwise, it's inserted at the beginning.

        Args:
            content (str): The system message content.
        """
        # Check if a system message already exists and update it
        for i, message in enumerate(self.messages):
            if message.role == "system":
                self.messages[i] = ProviderMessage("system", content)
                logger.debug("Updated existing system message.")
                return

        # Otherwise insert a new system message at the beginning
        self.messages.insert(0, ProviderMessage("system", content))
        logger.debug("Inserted new system message.")

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            content (str): The user message content.
        """
        self.messages.append(ProviderMessage("user", content))
        logger.debug("Added user message.")

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.

        Args:
            content (str): The assistant message content.
        """
        self.messages.append(ProviderMessage("assistant", content))
        logger.debug("Added assistant message.")

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in dictionary format for API requests.

        Returns:
            List[Dict[str, str]]: List of message dictionaries.
        """
        return [message.to_dict() for message in self.messages]

    def clear(self) -> None:
        """
        Clear all messages except the system message (if one exists).
        """
        system_message_content = None
        for message in self.messages:
            if message.role == "system":
                system_message_content = message.content
                break

        self.messages = []
        if system_message_content is not None:
            # Re-add the system message using the proper method
            self.set_system_message(system_message_content)
        logger.info("Conversation cleared (system message preserved if existed).")

    def is_system_message_set(self) -> bool:
        """
        Check if a system message has been set in the conversation.

        Returns:
            bool: True if a system message exists, False otherwise.
        """
        return any(message.role == "system" for message in self.messages)

    def switch_conversation_roles(self) -> None:
        """
        Switch the roles of 'user' and 'assistant' messages. System messages remain unchanged.
        """
        for message in self.messages:
            if message.role == "user":
                message.role = "assistant"
            elif message.role == "assistant":
                message.role = "user"
        logger.debug("Switched user/assistant roles in conversation.")


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize the abstract LLM provider.

        Args:
            temperature (float): Controls randomness (0.0 to 1.0). Defaults to 0.5.
        """
        # Clamp temperature during initialization
        self._temperature = max(0.0, min(1.0, temperature))
        if self._temperature != temperature:
            logger.warning(f"Temperature {temperature} clamped to {self._temperature}.")

    @abstractmethod
    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to the LLM and return the response. Must be implemented by subclasses.

        Args:
            prompt (str): The prompt to send.
            conversation (Optional[Conversation]): Conversation context. Defaults to None.

        Returns:
            str: Text response from the LLM.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            AuthenticationError: If authentication fails.
            RateLimitError: If rate limits are exceeded.
            APIError: If the provider API returns an error.
            ConnectionError: If there's a network issue.
            ProviderError: For provider-specific parsing or unexpected errors.
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's API (GPT models).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.5,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key (Optional[str]): API key. Uses OPENAI_API_KEY env var if None.
            model (str): Model identifier. Defaults to "gpt-4".
            temperature (float): Controls randomness. Defaults to 0.5.

        Raises:
            AuthenticationError: If no API key is found.
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            logger.error(
                "No OpenAI API key provided or found in environment variable OPENAI_API_KEY."
            )
            raise AuthenticationError("No OpenAI API key provided.")
        logger.info(f"OpenAIProvider initialized with model: {self.model}")

    @retry()  # Use default retry settings
    @handle_provider_exceptions("OpenAI")
    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to OpenAI's API. See base class for Args and Raises.
        """
        # API key check is implicitly handled by handle_provider_exceptions via HTTPError 401/403
        # but we ensure it exists during init.

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepare messages
        current_messages: List[Dict[str, str]]
        if conversation:
            # Add the user's prompt *before* getting messages for the API call
            conversation.add_user_message(prompt)
            current_messages = conversation.get_messages()
        else:
            current_messages = [{"role": "user", "content": prompt}]

        data = {
            "model": self.model,
            "messages": current_messages,
            "temperature": self._temperature,
        }
        logger.debug(f"Sending request to OpenAI: {data}")

        # Make the API call
        response = requests.post(
            self.api_url, headers=headers, json=data, timeout=30
        )  # Standard timeout

        # Let handle_provider_exceptions deal with HTTP errors (4xx, 5xx)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        # Process successful response (200 OK)
        try:
            response_data = response.json()
            logger.debug(f"Received response from OpenAI: {response_data}")
            # Safely access the content
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content")

            if content is None:  # Check for None or missing content explicitly
                logger.error(
                    f"Received unexpected response structure or empty content from OpenAI: {response_data}"
                )
                raise ValueError("Received empty or malformed content from API")

            # Add the assistant's response to the conversation *after* successful retrieval
            if conversation:
                conversation.add_assistant_message(content)

            return content

        # FIX: Catch only specific parsing/validation errors here
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f"Error parsing OpenAI response: {response.text} | Error: {e}", exc_info=True
            )
            raise ProviderError("OpenAI", e) from e
        # Other exceptions (like requests exceptions) are handled by the decorators


class GeminiProvider(LLMProvider):
    """
    Provider for Google's Gemini API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.5,
    ):
        """
        Initialize the Google Gemini provider.

        Args:
            api_key (Optional[str]): API key. Uses GEMINI_API_KEY env var if None.
            model (str): Model identifier. Defaults to "gemini-pro".
            temperature (float): Controls randomness. Defaults to 0.5.

        Raises:
            AuthenticationError: If no API key is found.
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        # Correct API endpoint structure
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

        if not self.api_key:
            logger.error(
                "No Gemini API key provided or found in environment variable GEMINI_API_KEY."
            )
            raise AuthenticationError("No Gemini API key provided.")
        logger.info(f"GeminiProvider initialized with model: {self.model}")

    @retry()  # Use default retry settings
    @handle_provider_exceptions("Gemini")
    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to Google's Gemini API. See base class for Args and Raises.
        """
        # API key check handled by init and handle_provider_exceptions

        # Format the conversation for Gemini API
        contents = []
        system_instruction = None  # Gemini v1beta uses systemInstruction field

        if conversation:
            # Add user prompt first
            conversation.add_user_message(prompt)
            # Process messages, extracting system message if present
            for message in conversation.messages:
                if message.role == "system":
                    # Use the *last* system message found as the system instruction
                    system_instruction = {"parts": [{"text": message.content}]}
                else:
                    # Map roles: user -> user, assistant -> model
                    role = "user" if message.role == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": message.content}]})
        else:
            # Simple query, no history
            contents = [{"role": "user", "parts": [{"text": prompt}]}]

        data: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": self._temperature},
        }
        # Add system instruction if it was found
        if system_instruction:
            data["systemInstruction"] = system_instruction

        logger.debug(f"Sending request to Gemini: {data}")

        # Make the API call (key is passed as query param)
        url = f"{self.api_url}?key={self.api_key}"
        response = requests.post(url, json=data, timeout=60)  # Gemini can be slower

        # Let handle_provider_exceptions deal with HTTP errors
        response.raise_for_status()

        # Process successful response (200 OK)
        try:
            response_data = response.json()
            logger.debug(f"Received response from Gemini: {response_data}")

            # Safely extract content using .get()
            candidates = response_data.get("candidates")
            if not candidates:
                logger.error(f"No 'candidates' field found in Gemini response: {response_data}")
                raise ValueError("No candidates found in Gemini API response")

            first_candidate = candidates[0]
            content_part = first_candidate.get("content", {}).get("parts", [{}])[0]
            content = content_part.get("text")

            if content is None:
                logger.error(f"Missing 'text' field in Gemini response part: {response_data}")
                raise ValueError("Missing 'text' field in response part")

            # Add the assistant's response to the conversation *after* successful retrieval
            if conversation:
                conversation.add_assistant_message(content)

            return content

        # FIX: Catch only specific parsing/validation errors here
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f"Error parsing Gemini response: {response.text} | Error: {e}", exc_info=True
            )
            raise ProviderError("Gemini", e) from e
        # Other exceptions handled by decorators


class ClaudeProvider(LLMProvider):
    """
    Provider for Anthropic's Claude API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",  # Default to a known recent model
        temperature: float = 0.5,
    ):
        """
        Initialize the Anthropic Claude provider.

        Args:
            api_key (Optional[str]): API key. Uses ANTHROPIC_API_KEY env var if None.
            model (str): Model identifier. Defaults to "claude-3-opus-20240229".
            temperature (float): Controls randomness. Defaults to 0.5.

        Raises:
            AuthenticationError: If no API key is found.
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.api_version = "2023-06-01"  # Required header

        if not self.api_key:
            logger.error(
                "No Anthropic API key provided or found in environment variable ANTHROPIC_API_KEY."
            )
            raise AuthenticationError("No Anthropic API key provided.")
        logger.info(f"ClaudeProvider initialized with model: {self.model}")

    @retry()  # Use default retry settings
    @handle_provider_exceptions("Claude")
    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to Anthropic's Claude API. See base class for Args and Raises.
        """
        # API key check handled by init and handle_provider_exceptions

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

        # Format conversation for Claude API
        system_prompt_content = None
        current_messages = []
        if conversation:
            # Add user prompt first
            conversation.add_user_message(prompt)
            # Separate system message from user/assistant messages
            for message in conversation.messages:
                if message.role == "system":
                    # Use the *last* system message found
                    system_prompt_content = message.content
                else:
                    # Roles match Claude's API (user, assistant)
                    current_messages.append({"role": message.role, "content": message.content})
        else:
            current_messages = [{"role": "user", "content": prompt}]

        # Construct payload
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": current_messages,
            "temperature": self._temperature,
            "max_tokens": 4096,  # Recommended practice for Claude
        }
        if system_prompt_content:
            data["system"] = system_prompt_content  # Add system prompt if present

        logger.debug(f"Sending request to Claude: {data}")

        # Make the API call
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=60,  # Slightly longer timeout for Claude
        )

        # Let handle_provider_exceptions deal with HTTP errors
        response.raise_for_status()

        # Process successful response (200 OK)
        try:
            response_data = response.json()
            logger.debug(f"Received response from Claude: {response_data}")

            # Ensure 'content' exists and is a list
            content_blocks = response_data.get("content")
            if not isinstance(content_blocks, list):
                logger.error(f"Response 'content' is not a list or is missing: {response_data}")
                raise ValueError("Response missing or invalid 'content' field")

            # Find the first text block and extract its text
            text_content = ""
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")  # Use get with default
                    break  # Assuming we only want the first text block

            # Check if any text content was actually found
            # Allow empty string as a valid response, but log if needed.
            if not text_content and text_content != "":
                logger.warning(f"No 'text' content block found in Claude response: {response_data}")
                # Decide if this is an error or just an empty response
                # For now, let's allow empty string through but raise if no text block found at all
                raise ValueError("No 'text' content block found in response")

            # Add the assistant's response to the conversation *after* successful retrieval
            if conversation:
                conversation.add_assistant_message(text_content)

            return text_content

        # FIX: Catch only specific parsing/validation errors here
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(
                f"Error parsing Claude response: {response.text} | Error: {e}", exc_info=True
            )
            raise ProviderError("Claude", e) from e
        # Other exceptions handled by decorators


class LLMInterfaceConfig:
    """
    Configuration data class for LLMInterface.
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
    ):
        """
        Initialize the configuration.

        Args:
            provider (str): 'openai', 'gemini', or 'claude'.
            api_key (Optional[str]): API key. Defaults to None.
            model (Optional[str]): Model identifier. Defaults to None.
            system_message (Optional[str]): System message. Defaults to None.
            temperature (float): Generation temperature. Defaults to 0.5.
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.system_message = system_message
        # Clamp temperature on config creation as well
        self.temperature = max(0.0, min(1.0, temperature))


class LLMInterface:
    """
    Unified interface for working with multiple LLM providers.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
        role_name: Optional[str] = None,
        role_description: Optional[str] = None,
    ):
        """
        Initialize the unified LLM interface.

        Args:
            provider (str): 'openai', 'gemini', or 'claude'. Defaults to "openai".
            api_key (Optional[str]): API key. Defaults to None (uses env vars).
            model (Optional[str]): Model identifier. Defaults to provider default.
            system_message (Optional[str]): System message. Defaults to None.
            temperature (float): Generation temperature (0.0-1.0). Defaults to 0.5.

        Raises:
            ValueError: If an unsupported provider is specified.
            AuthenticationError: If the selected provider requires an API key and none is found.
        """
        # Store raw init args for get_current_config
        self._init_api_key = api_key
        self._init_model = model

        # Clamp temperature immediately
        self.temperature = max(0.0, min(1.0, temperature))
        if self.temperature != temperature:
            logger.warning(f"Initial temperature {temperature} clamped to {self.temperature}.")

        self.provider_name = provider.lower()
        # Provider initialization might raise AuthenticationError
        self.provider = self._initialize_provider(self.provider_name, api_key, model)

        # Initialize conversation only if a system message is provided
        self.conversation: Optional[Conversation] = None
        self.system_message: Optional[str] = None  # Track the intended system message separately
        if system_message:
            self.update_system_message(
                system_message
            )  # Use method to ensure conversation is created

        # Initialize role name and description if provided
        self.role_name = role_name
        self.role_description = role_description

        logger.info(f"LLMInterface initialized for provider: {self.provider_name}")

    def _initialize_provider(
        self,
        provider_name: str,  # Use consistent naming
        api_key: Optional[str],
        model: Optional[str],
    ) -> LLMProvider:
        """
        Internal factory method to create and return the appropriate provider instance.

        Args:
            provider_name (str): The LLM provider name.
            api_key (Optional[str]): API key.
            model (Optional[str]): Model identifier.

        Returns:
            LLMProvider: An initialized provider instance.

        Raises:
            ValueError: If provider_name is unsupported.
            AuthenticationError: If API key is required and missing for the provider.
        """
        provider_name_lower = provider_name.lower()
        if provider_name_lower == "openai":
            return OpenAIProvider(
                api_key=api_key, model=model or "gpt-4", temperature=self.temperature
            )
        elif provider_name_lower == "gemini":
            return GeminiProvider(
                api_key=api_key,
                model=model or "gemini-pro",
                temperature=self.temperature,
            )
        elif provider_name_lower == "claude":
            return ClaudeProvider(
                api_key=api_key,
                model=model or "claude-3-opus-20240229",
                temperature=self.temperature,
            )
        else:
            logger.error(f"Attempted to initialize unsupported provider: {provider_name}")
            raise ValueError(
                f"Unsupported provider: {provider_name}. Choose from 'openai', 'gemini', or 'claude'."
            )

    def get_role_name(self) -> Optional[str]:
        """
        Get the role name for the assistant.

        Returns:
            Optional[str]: The role name, or None if not set.
        """
        return self.role_name

    def get_role_description(self) -> Optional[str]:
        """
        Get the role description for the assistant.

        Returns:
            Optional[str]: The role description, or None if not set.
        """
        return self.role_description

    def _start_conversation(self, system_message_content: Optional[str] = None) -> None:
        """
        Starts a new conversation, replacing the existing one.

        Args:
            system_message_content (Optional[str]): Content for the system message. Defaults to None.
        """
        self.conversation = Conversation(system_message_content)
        # Update the interface's tracked system message
        self.system_message = system_message_content
        logger.info(f"Started new conversation for {self.provider_name} provider.")
        if system_message_content:
            logger.debug(f"New conversation started with system message.")

    def update_system_message(self, system_message_content: str) -> None:
        """
        Update the system message. Creates a conversation if one doesn't exist.

        Args:
            system_message_content (str): The new system message content.
        """
        self.system_message = system_message_content  # Update tracked system message
        if not self.conversation:
            # If no conversation exists, start a new one with this system message
            self._start_conversation(system_message_content)
        else:
            # If a conversation exists, update its system message
            self.conversation.set_system_message(system_message_content)
            logger.info("Updated system message in the current conversation.")

    def is_system_message_set(self) -> bool:
        """
        Check if a system message is currently set for the interface.

        Returns:
            bool: True if a system message is set, False otherwise.
        """
        return self.system_message is not None

    def get_system_message(self) -> Optional[str]:
        """
        Get the current system message set for the interface.

        Returns:
            Optional[str]: The system message content, or None if not set.
        """
        return self.system_message

    def query(
        self,
        prompt: str,
        use_conversation: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a query to the current LLM provider.

        The interface's configured `system_message` is ALWAYS used.
        If `conversation_history` is provided, any system message within it is IGNORED,
        and the interface's system message is applied instead.

        Args:
            prompt (str): The prompt to send.
            use_conversation (bool): Use and update the interface's internal conversation state. Defaults to False.
            conversation_history (Optional[List[Dict[str, Any]]]): Initialize a *temporary*
                conversation from this history for a single query (ignoring its system message).
                Ignored if use_conversation is True. Defaults to None.

        Returns:
            str: Text response from the LLM.

        Raises:
            LLMError subclasses: Propagates errors from the provider's query method.
        """
        target_conversation: Optional[Conversation] = None

        if use_conversation:
            # Use the internal conversation state
            if not self.conversation:
                self._start_conversation(
                    self.system_message
                )  # Ensure internal conversation exists with correct system msg
            target_conversation = self.conversation
            logger.debug("Using internal conversation state for query.")

        elif conversation_history:
            # Create a temporary conversation using the INTERFACE'S system message,
            # then populate with user/assistant messages from the provided history.
            logger.debug(
                "Using provided history; applying INTERFACE system message and ignoring history's system message."
            )
            # Start temporary conversation with the interface's system message
            target_conversation = Conversation(self.system_message)

            # Add only user/assistant messages from the provided history
            for msg in conversation_history:
                role = msg.get("role")
                content = msg.get("content")
                if role == "user" and content is not None:
                    target_conversation.add_user_message(content)
                elif role == "assistant" and content is not None:
                    target_conversation.add_assistant_message(content)
                # System messages from history are deliberately ignored
            logger.debug(
                f"Temporary context created with interface system message and {len(target_conversation.messages) - (1 if self.system_message else 0)} user/assistant messages from history."
            )

        else:
            # Simple query: Create a temporary conversation context that ONLY includes
            # the interface's system message, if one is set.
            logger.debug("Performing simple query. Including interface system message if set.")
            if self.system_message:
                target_conversation = Conversation(self.system_message)
                logger.debug("Temporary context created with interface system message.")
            else:
                target_conversation = None  # No system message, no history -> pass None
                logger.debug("No interface system message set. Querying without system context.")

        # --- Query Execution ---
        try:
            # Pass the prompt and the prepared conversation object (or None)
            # The provider's query method will add the prompt to the conversation object.
            response = self.provider.query(prompt, target_conversation)
            logger.info(f"Query successful for provider {self.provider_name}.")

            # If use_conversation=True, target_conversation is self.conversation,
            # and the provider added the assistant response to it.
            # If temporary, we don't care about the assistant response being added to it.
            return response
        except LLMError as e:
            logger.error(f"LLM query failed: {type(e).__name__} - {e}", exc_info=False)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during provider query execution: {e}")
            raise LLMError(f"Unexpected error during query: {e}") from e

    def export_conversation(self) -> Dict[str, Any]:
        """
        Export the current internal conversation state.

        Returns:
            Dict[str, Any]: Dictionary with provider, messages, timestamp. Empty if no conversation.
        """
        if not self.conversation:
            logger.warning("Attempted to export, but no active conversation exists.")
            return {}

        export_data = {
            "provider": self.provider_name,  # Reflects current provider
            "messages": self.conversation.get_messages(),
            "timestamp": time.time(),
        }
        logger.info(f"Exported conversation with {len(export_data['messages'])} messages.")
        return export_data

    def import_conversation(
        self, conversation_data: Dict[str, Any], include_system_message: bool = True
    ) -> None:
        """
        Import conversation data, replacing the current internal conversation.

        Args:
            conversation_data (Dict[str, Any]): Data from export_conversation. Must contain 'messages'.
            include_system_message (bool): If True, use system message from data.
                                           If False, keep the interface's current system message. Defaults to True.
        """
        if "messages" not in conversation_data:
            logger.error("Import failed: 'messages' key missing in conversation_data.")
            raise ValueError("'messages' key is required in conversation_data for import.")

        imported_messages = conversation_data["messages"]
        new_system_message_content: Optional[str] = None

        # Determine the system message for the new conversation
        if include_system_message:
            # Try to find system message in imported data
            for message in imported_messages:
                if message.get("role") == "system":
                    new_system_message_content = message.get("content")
                    break  # Use the first one found
            logger.debug(
                f"Importing with system message from data: {bool(new_system_message_content)}"
            )
        else:
            # Keep the existing system message of the interface
            new_system_message_content = self.system_message
            logger.debug(
                f"Importing, keeping existing system message: {bool(new_system_message_content)}"
            )

        # Start a new conversation with the determined system message
        # This replaces self.conversation and sets self.system_message correctly
        self._start_conversation(new_system_message_content)

        # Add all non-system messages from the imported data
        messages_added = 0
        if self.conversation:  # Should always be true after _start_conversation
            for message in imported_messages:
                role = message.get("role")
                content = message.get("content")
                if role == "system":
                    continue  # System message already handled
                elif role == "user" and content is not None:
                    self.conversation.add_user_message(content)
                    messages_added += 1
                elif role == "assistant" and content is not None:
                    self.conversation.add_assistant_message(content)
                    messages_added += 1
                else:
                    logger.warning(f"Skipping invalid message during import: {message}")

        logger.info(f"Imported conversation. Added {messages_added} user/assistant messages.")

    def switch_provider(
        self,
        new_provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,  # Option to set a new system message on switch
        switch_conversation_roles: bool = True,
    ) -> None:
        """
        Switch to a different LLM provider, preserving conversation context.

        Args:
            new_provider (str): The provider to switch to ('openai', 'gemini', 'claude').
            api_key (Optional[str]): API key for the new provider. Defaults to None (uses env vars).
            model (Optional[str]): Model for the new provider. Defaults to None (provider default).
            temperature (Optional[float]): New temperature. Defaults to None (keeps current).
            system_message (Optional[str]): Optionally set a new system message. Defaults to None (keeps current).
            switch_conversation_roles (bool): Switch user/assistant roles in history. Defaults to True.

        Raises:
            ValueError: If new_provider is unsupported.
            AuthenticationError: If API key is required and missing for the new provider.
        """
        logger.info(f"Switching provider from {self.provider_name} to {new_provider}")

        # Update temperature if provided
        if temperature is not None:
            new_temp_clamped = max(0.0, min(1.0, temperature))
            if new_temp_clamped != self.temperature:
                self.temperature = new_temp_clamped
                logger.info(f"Temperature updated to {self.temperature}")
            if new_temp_clamped != temperature:
                logger.warning(f"Provided temperature {temperature} clamped to {self.temperature}.")

        # Store new init keys/models for get_current_config consistency
        self._init_api_key = api_key
        self._init_model = model

        # Initialize the new provider (might raise errors)
        self.provider = self._initialize_provider(new_provider, api_key, model)
        self.provider_name = new_provider.lower()

        # Update system message if provided
        if system_message is not None:
            self.update_system_message(system_message)  # This handles conversation creation/update

        # Switch roles in the existing conversation if requested and conversation exists
        if switch_conversation_roles and self.conversation:
            self.conversation.switch_conversation_roles()
            logger.info("Switched conversation roles due to provider switch.")
        elif switch_conversation_roles and not self.conversation:
            logger.warning("Requested role switch, but no active conversation exists.")

        logger.info(f"Successfully switched to {self.provider_name} provider.")

    def switch_conversation_roles(self) -> None:
        """
        Switch the roles ('user' <-> 'assistant') in the current internal conversation.
        """
        if not self.conversation:
            logger.warning("Attempted to switch roles, but no active conversation exists.")
            return

        self.conversation.switch_conversation_roles()
        logger.info("Switched conversation roles.")

    def get_current_config(self) -> LLMInterfaceConfig:
        """
        Get the current configuration of the LLM interface.

        Returns:
            LLMInterfaceConfig: Object containing current configuration parameters.
        """
        return LLMInterfaceConfig(
            provider=self.provider_name,
            # Return the keys/models used at the *last* init/switch for consistency
            api_key=self._init_api_key,
            model=self._init_model,
            system_message=self.system_message,  # Use the tracked system message
            temperature=self.temperature,
        )

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the current conversation without querying the LLM.
        Creates a conversation if one doesn't exist.

        Args:
            content (str): The user message content.
        """
        if not self.conversation:
            # Start conversation using the interface's tracked system message
            self._start_conversation(self.system_message)

        # We know self.conversation exists now
        self.conversation.add_user_message(content)
        logger.info("Manually added user message to conversation.")

    def clear_conversation(self) -> None:
        """
        Clear the current conversation history, preserving the system message setting.
        """
        if self.conversation:
            # Start a new conversation, preserving the current system message setting
            self._start_conversation(self.system_message)
            # Note: _start_conversation already logs "Started new conversation..."
            # logger.info("Conversation cleared (system message preserved).") # Redundant log
        else:
            logger.info("Attempted to clear conversation, but none exists.")
