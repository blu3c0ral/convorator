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

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Assuming standard project structure where 'convorator' is a top-level package
try:
    from convorator.utils.logger import setup_logger
    from convorator.exceptions import (
        LLMClientError,
        LLMConfigurationError,
        LLMResponseError,
    )
    import tiktoken  # Add tiktoken import
except ImportError:
    # Fallback for running the script directly or if structure differs
    import logging

    setup_logger = lambda name: logging.getLogger(name)  # Basic logger fallback

    # Define basic exceptions if import fails (not ideal for production)
    class LLMClientError(Exception):
        """Base class for LLM client errors."""

        pass

    class LLMConfigurationError(LLMClientError):
        """Raised when there are configuration issues (API keys, model names, etc.)."""

        pass

    # Define tiktoken as None if it couldn't be imported
    tiktoken = None

    logging.warning(
        "Could not import from convorator package or tiktoken. Using basic logging and exception definitions. Token counting may be approximate."
    )


logger = setup_logger("llm_client")  # Use a specific logger name

# Constants
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# --- Helper Classes (Message, Conversation) ---


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # Typically 'system', 'user', or 'assistant'/'model'
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Converts the message to a dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """Manages the conversation history for an LLM interaction."""

    messages: List[Message] = field(default_factory=list)
    system_message: Optional[str] = None  # Stores the intended system message content

    def __post_init__(self):
        """Ensures the system message is correctly placed if provided."""
        if self.system_message and (not self.messages or self.messages[0].role != "system"):
            # Insert system message at the beginning if it's not already there
            self.messages.insert(0, Message(role="system", content=self.system_message))
        elif not self.system_message and self.messages and self.messages[0].role == "system":
            # If system_message is None but history starts with one, update internal state
            self.system_message = self.messages[0].content

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation, handling system message updates."""
        role = role.lower()  # Normalize role

        if role == "system":
            self.system_message = content  # Update the stored system message content
            # Check if a system message already exists in the list
            for i, msg in enumerate(self.messages):
                if msg.role == "system":
                    if msg.content != content:
                        logger.debug("Updating existing system message in history.")
                        self.messages[i] = Message(role="system", content=content)
                    return  # Found and potentially updated
            # If no system message exists, add it to the beginning
            logger.debug("Inserting new system message at the beginning of history.")
            self.messages.insert(0, Message(role="system", content=content))
        else:
            # Check for consecutive non-system messages with the same role
            last_non_system_role = None
            if self.messages:
                # Find the role of the last non-system message
                for msg in reversed(self.messages):
                    if msg.role != "system":
                        last_non_system_role = msg.role
                        break

            if last_non_system_role == role:
                logger.warning(
                    f"Adding consecutive messages with the same role '{role}'. This might cause issues with some LLM APIs (e.g., Anthropic, Gemini)."
                )
            elif role not in ["user", "assistant", "model"]:  # 'model' is used by Gemini
                logger.warning(
                    f"Adding message with non-standard role '{role}'. Ensure the target API supports this."
                )

            self.messages.append(Message(role=role, content=content))

    def add_user_message(self, content: str):
        """Convenience method to add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str):
        """Convenience method to add an assistant/model message."""
        # Use 'assistant' as the generic internal role
        self.add_message("assistant", content)

    def get_messages(self) -> List[Dict[str, str]]:
        """Returns the conversation history as a list of dictionaries."""
        return [msg.to_dict() for msg in self.messages]

    def clear(self, keep_system: bool = True):
        """Clears the conversation history."""
        if keep_system and self.system_message:
            # Keep only the system message if it exists
            self.messages = [Message(role="system", content=self.system_message)]
        else:
            # Clear everything
            self.messages = []
            self.system_message = None  # Also clear the stored system message content
        logger.debug(
            f"Conversation cleared (System message {'kept' if keep_system and self.system_message else 'removed'})."
        )


# --- LLM Interface Definition ---


class LLMInterface(ABC):
    """
    Abstract base class defining the interface for LLM clients.
    Manages conversation state and provides a unified query method.
    """

    conversation: Conversation
    _system_message: Optional[str]  # Stored desired system message
    _role_name: Optional[str]  # Optional name for the assistant role

    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Abstract method to handle the actual API call for the specific provider.
        Must be implemented by subclasses. Should raise appropriate convorator exceptions
        (LLMClientError, LLMConfigurationError, LLMResponseError) on failure.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimates the number of tokens for the given text for this specific LLM."""
        pass

    @abstractmethod
    def get_context_limit(self) -> int:
        """Returns the maximum context token limit for the configured model."""
        pass

    def query(
        self,
        prompt: str,
        use_conversation: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Sends a prompt to the LLM and returns the response.

        Manages conversation history internally if use_conversation is True.
        If use_conversation is False, it uses the provided conversation_history (if any)
        or just the system message (if set) and the current prompt for a stateless call.

        Args:
            prompt: The user's input prompt.
            use_conversation: If True, use and update the internal conversation history.
                              If False, perform a stateless query.
            conversation_history: Optional list of messages for stateless queries.
                                  Ignored if use_conversation is True.

        Returns:
            The LLM's response content as a string.

        Raises:
            LLMConfigurationError: For configuration issues (API key, model).
            LLMClientError: For client-side issues (network, unexpected errors).
            LLMResponseError: For API errors (rate limits, bad request, blocked content).
        """
        messages_to_send: List[Dict[str, str]]
        last_user_message_added_to_internal_convo = False  # Track changes to internal state

        try:
            if use_conversation:
                # --- Stateful Query ---
                # Ensure the internal conversation object has the correct system message
                if self._system_message != self.conversation.system_message:
                    self.conversation.add_message(role="system", content=self._system_message)

                # Add the current user prompt to the internal conversation
                self.conversation.add_user_message(prompt)
                last_user_message_added_to_internal_convo = True
                messages_to_send = self.conversation.get_messages()
                logger.debug(f"Using internal conversation with {len(messages_to_send)} messages.")

            else:
                # --- Stateless Query ---
                if conversation_history is not None:
                    # Use provided history
                    messages_to_send = list(conversation_history)  # Create a copy

                    # Remove any existing system message from the provided history
                    if self._system_message is not None:
                        messages_to_send = [
                            msg for msg in messages_to_send if msg.get("role") != "system"
                        ]
                        # Insert the interface's system message at the beginning
                        messages_to_send.insert(
                            0, {"role": "system", "content": self._system_message}
                        )
                    # If no system message is set on interface but there's one in history, keep it (handled by default)
                else:
                    # Start fresh, only system message (if any) and prompt
                    messages_to_send = []
                    if self._system_message:
                        messages_to_send.append({"role": "system", "content": self._system_message})

                # Add the current user prompt
                messages_to_send.append({"role": "user", "content": prompt})
                logger.debug(f"Performing stateless query with {len(messages_to_send)} messages.")

            # --- API Call (Common for both stateful/stateless) ---
            response_content = self._call_api(messages_to_send)
            # --- Success ---

            if use_conversation:
                # Add successful assistant response to internal conversation
                # This implicitly assumes the last message added was the user prompt
                self.conversation.add_assistant_message(response_content)
                logger.debug("Added assistant response to internal conversation.")

            return response_content

        except (LLMClientError, LLMConfigurationError, LLMResponseError) as e:
            logger.error(f"Error during LLM query for {self.__class__.__name__}: {e}")
            # If using internal conversation and we added the user message that failed, remove it
            if use_conversation and last_user_message_added_to_internal_convo:
                if self.conversation.messages and self.conversation.messages[-1].role == "user":
                    removed_msg = self.conversation.messages.pop()
                    logger.debug(
                        f"Removed last user message ('{removed_msg.content[:50]}...') from internal history due to API error."
                    )
                else:
                    logger.warning(
                        "Attempted to remove last user message due to error, but history state was unexpected."
                    )
            raise  # Re-raise the caught convorator exception

        except Exception as e:
            logger.exception(
                f"Unexpected error during LLM query for {self.__class__.__name__}: {e}"
            )
            # Handle removal for unexpected errors too
            if use_conversation and last_user_message_added_to_internal_convo:
                if self.conversation.messages and self.conversation.messages[-1].role == "user":
                    removed_msg = self.conversation.messages.pop()
                    logger.debug(
                        f"Removed last user message ('{removed_msg.content[:50]}...') from internal history due to unexpected error."
                    )
                else:
                    logger.warning(
                        "Attempted to remove last user message due to unexpected error, but history state was unexpected."
                    )
            # Wrap unexpected errors in LLMClientError
            raise LLMClientError(f"An unexpected error occurred during LLM query: {e}") from e

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Returns the current internal conversation history."""
        return self.conversation.get_messages()

    def clear_conversation(self, keep_system: bool = True):
        """Clears the internal conversation history.

        Args:
            keep_system: If True, retains the system message (if one is set).
                         If False, clears all messages including the system message.
        """
        self.conversation.clear(keep_system=keep_system)
        # If clearing completely, also clear the interface's system message attribute
        if not keep_system:
            self._system_message = None

    def get_system_message(self) -> Optional[str]:
        """Returns the configured system message."""
        return self._system_message

    def set_system_message(self, message: Optional[str]):
        """Sets or updates the system message for subsequent queries."""
        if message != self._system_message:
            logger.debug(f"Setting system message to: '{message}'")
            self._system_message = message
            # Update the conversation object immediately if it exists
            if hasattr(self, "conversation"):
                self.conversation.add_message(role="system", content=message)
        else:
            logger.debug("System message is already set to the desired value.")

    def get_role_name(self) -> Optional[str]:
        """Returns the configured assistant role name (cosmetic)."""
        return self._role_name

    def set_role_name(self, name: str):
        """Sets the assistant role name (cosmetic)."""
        logger.debug(f"Setting role name to: '{name}'")
        self._role_name = name


# --- Concrete LLM Client Implementations ---


# Default context limit fallback if API fetch fails
DEFAULT_CONTEXT_LIMIT_FALLBACK = 8192


class OpenAILLM(LLMInterface):
    """Concrete implementation for OpenAI's API (GPT models)."""

    SUPPORTED_MODELS = [
        # GPT-4 / 4o / Turbo
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
        # GPT-3.5
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        # GPT-3.5 Instruct
        "gpt-3.5-turbo-instruct",
    ]

    # Tiktoken encoding mapping (add more as needed)
    # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    MODEL_TO_ENCODING = {
        # --- GPT-4 / 4o / Turbo --- uses o200k_base for -o models, cl100k_base otherwise
        "gpt-4o": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4-0125-preview": "cl100k_base",
        "gpt-4-1106-preview": "cl100k_base",
        "gpt-4-vision-preview": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4-0613": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-32k-0613": "cl100k_base",
        # --- GPT-3.5 --- uses cl100k_base
        "gpt-3.5-turbo-0125": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-1106": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-0613": "cl100k_base",
        "gpt-3.5-turbo-16k-0613": "cl100k_base",
        # --- GPT-3.5 Instruct --- uses p50k_base
        "gpt-3.5-turbo-instruct": "p50k_base",
        # Add other models (like text-embedding-ada-002, older models) if needed
    }
    DEFAULT_ENCODING = "cl100k_base"  # Fallback encoding

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",  # Default to the latest powerful model
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
    ):
        """Initializes the OpenAI client.

        Requires the 'openai' package to be installed (`pip install openai`).
        Requires the 'tiktoken' package for accurate token counting (`pip install tiktoken`).

        Raises:
            LLMConfigurationError: If API key is missing or client initialization fails.
        """
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

        # Use the class attribute list for checking
        if model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model}' is not in the explicitly supported list for OpenAILLM context/token estimation. Proceeding, but compatibility is not guaranteed."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._encoding: Optional[Any] = None  # Store tiktoken encoding object lazily

        try:
            # Initialize the OpenAI client instance
            self.client = self.openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            # Catch potential errors during client init (e.g., invalid key format - though API call usually catches this)
            raise LLMConfigurationError(f"Failed to initialize OpenAI client: {e}") from e

        self._system_message = system_message
        self._role_name = role_name or "Assistant"
        self.conversation = Conversation(system_message=self._system_message)
        logger.info(
            f"OpenAILLM initialized. Model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}"
        )

        # Fetch context limit during initialization
        self._context_limit = self._fetch_context_limit()

    def _fetch_context_limit(self) -> int:
        """Fetches the context window size (in tokens) for the current model from the OpenAI API.

        Returns:
            int: The context window size in tokens, or a fallback value if the fetch fails.
        """
        try:
            model_info = self.client.models.retrieve(self.model)
            context_window = getattr(model_info, "context_window", None)
            if context_window:
                logger.debug(
                    f"Successfully fetched context limit for {self.model}: {context_window}"
                )
                return context_window
            else:
                logger.warning(
                    f"Could not find context_window attribute for {self.model}. Using fallback."
                )
                return DEFAULT_CONTEXT_LIMIT_FALLBACK
        except self.openai.NotFoundError as e:
            logger.warning(
                f"Model {self.model} not found during context limit fetch: {e}. Using fallback."
            )
            return DEFAULT_CONTEXT_LIMIT_FALLBACK
        except (
            self.openai.APIConnectionError,
            self.openai.RateLimitError,
            self.openai.APIStatusError,
        ) as e:
            logger.warning(f"Error fetching context limit for {self.model}: {e}. Using fallback.")
            return DEFAULT_CONTEXT_LIMIT_FALLBACK

    def _get_tiktoken_encoding(self):
        """Lazily loads and returns the tiktoken encoding for the current model."""
        if self._encoding is None:
            if tiktoken is None:
                logger.warning(
                    "'tiktoken' library not found. Cannot accurately count tokens for OpenAI. Install using 'pip install tiktoken'"
                )
                return None  # Indicate failure to load

            encoding_name = self.MODEL_TO_ENCODING.get(self.model, self.DEFAULT_ENCODING)
            if encoding_name != self.DEFAULT_ENCODING and self.model not in self.MODEL_TO_ENCODING:
                logger.warning(
                    f"Using default encoding '{self.DEFAULT_ENCODING}' for unknown OpenAI model '{self.model}'. Token count may be inaccurate."
                )

            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Loaded tiktoken encoding: {encoding_name} for model {self.model}")
            except ValueError as e:
                logger.error(
                    f"Failed to get tiktoken encoding '{encoding_name}': {e}. Falling back to default."
                )
                try:
                    self._encoding = tiktoken.get_encoding(self.DEFAULT_ENCODING)
                    logger.debug(f"Using fallback tiktoken encoding: {self.DEFAULT_ENCODING}")
                except ValueError as e_fallback:
                    logger.error(
                        f"Failed to get fallback tiktoken encoding '{self.DEFAULT_ENCODING}': {e_fallback}. Token counting disabled."
                    )
                    # Set encoding to a value indicating failure, but not None to avoid retrying
                    self._encoding = "FAILED_TO_LOAD"

        # Return None if loading failed
        return None if self._encoding == "FAILED_TO_LOAD" else self._encoding

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the OpenAI Chat Completions API."""
        if not messages:
            raise LLMClientError("Cannot call OpenAI API with an empty message list.")

        logger.debug(
            f"Calling OpenAI API ({self.model}) with {len(messages)} messages. First message role: {messages[0].get('role')}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # API expects list of {'role': str, 'content': str}
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # --- Response Validation ---
            if not response.choices:
                logger.error(f"OpenAI response missing 'choices'. Response: {response}")
                raise LLMResponseError(f"Invalid response structure from OpenAI: No 'choices'.")

            first_choice = response.choices[0]
            if not first_choice.message:
                logger.error(f"OpenAI response choice missing 'message'. Response: {response}")
                raise LLMResponseError(
                    f"Invalid response structure from OpenAI: Choice missing 'message'."
                )

            content = first_choice.message.content

            if content is None:
                # Check finish reason if content is None
                finish_reason = first_choice.finish_reason
                logger.error(
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
            logger.debug(
                f"Received content from OpenAI API. Length: {len(content)}. Finish Reason: {first_choice.finish_reason}"
            )
            if not content and first_choice.finish_reason == "stop":
                logger.warning(
                    f"Received empty content string from OpenAI API, but finish reason was 'stop'. Response: {response}"
                )
                # Decide if empty content with 'stop' is an error or valid (e.g., model chose not to respond)
                # For now, let's return it, but a stricter check might raise LLMResponseError here.

            return content

        # --- Error Handling ---
        except self.openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise LLMClientError(f"Network error connecting to OpenAI API: {e}") from e
        except self.openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}")
            raise LLMResponseError(
                f"OpenAI API rate limit exceeded. Check your plan and usage limits. Error: {e}"
            ) from e
        except self.openai.AuthenticationError as e:
            logger.error(f"OpenAI API authentication error: {e}")
            raise LLMConfigurationError(
                f"OpenAI API authentication failed. Check your API key. Error: {e}"
            ) from e
        except self.openai.PermissionDeniedError as e:
            logger.error(f"OpenAI API permission denied: {e}")
            raise LLMConfigurationError(
                f"OpenAI API permission denied. Check API key permissions or organization access. Error: {e}"
            ) from e
        except self.openai.NotFoundError as e:
            logger.error(f"OpenAI API resource not found (e.g., model): {e}")
            raise LLMConfigurationError(
                f"OpenAI API resource not found (check model name '{self.model}'?). Error: {e}"
            ) from e
        except self.openai.BadRequestError as e:  # Often indicates input schema issues
            logger.error(f"OpenAI API bad request error: {e}")
            raise LLMResponseError(
                f"OpenAI API reported a bad request (check message format or parameters?). Error: {e}"
            ) from e
        except self.openai.APIStatusError as e:  # Catch other 4xx/5xx errors
            logger.error(f"OpenAI API status error: {e.status_code} - {e.response}")
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
            logger.error(f"An unexpected OpenAI client error occurred: {e}")
            raise LLMClientError(f"An unexpected OpenAI client error occurred: {e}") from e
        # --- Allow specific LLMResponseErrors from validation to pass through ---
        except LLMResponseError as e:
            # This catches errors raised explicitly in the validation logic above
            # (e.g., missing choices, null content)
            logger.error(f"LLM Response Error: {e}")  # Log it, but let it propagate
            raise
        # General exception catcher (less likely with specific catches above)
        except Exception as e:
            logger.exception(f"An unexpected error occurred during OpenAI API call: {e}")
            raise LLMClientError(f"Unexpected error during OpenAI API call: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using tiktoken for the configured model."""
        encoding = self._get_tiktoken_encoding()
        if encoding:
            try:
                num_tokens = len(encoding.encode(text))
                logger.debug(
                    f"Counted {num_tokens} tokens for text (length {len(text)}) using {encoding.name}."
                )
                return num_tokens
            except Exception as e:
                logger.error(
                    f"Error using tiktoken encoding '{encoding.name}' to count tokens: {e}"
                )
                # Fall through to approximation
        else:
            logger.warning("Tiktoken encoding not available.")
            # Fall through to approximation

        # Fallback approximation (e.g., 4 chars per token)
        estimated_tokens = len(text) // 4
        logger.warning(
            f"Using approximate token count: {estimated_tokens} (based on character count). Install 'tiktoken' for accuracy."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) fetched during initialization."""
        logger.debug(f"Returning context limit for {self.model}: {self._context_limit}")
        return self._context_limit


ANTHROPIC_CONTEXT_LIMITS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}
DEFAULT_ANTHROPIC_CONTEXT_LIMIT = 100000  # Fallback


class AnthropicLLM(LLMInterface):
    """Concrete implementation for Anthropic's API (Claude models)."""

    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",  # Default to fast/cheap Haiku
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
    ):
        """Initializes the Anthropic client.

        Requires the 'anthropic' package to be installed (`pip install anthropic`).

        Note: Anthropic context limits are currently hardcoded as the API/SDK
        does not provide a standard way to fetch them dynamically.

        Raises:
            LLMConfigurationError: If API key is missing or client initialization fails.
        """
        try:
            import anthropic

            self.anthropic = anthropic
        except ImportError as e:
            raise LLMConfigurationError(
                "Anthropic Python package not found. Please install it using 'pip install anthropic'."
            ) from e

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "Anthropic API key not provided. Set the ANTHROPIC_API_KEY environment variable or pass it during initialization."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Get context limit from hardcoded dict
        self._context_limit = ANTHROPIC_CONTEXT_LIMITS.get(
            self.model, DEFAULT_ANTHROPIC_CONTEXT_LIMIT
        )
        if self.model not in ANTHROPIC_CONTEXT_LIMITS:
            logger.warning(
                f"Context limit not defined for Anthropic model '{self.model}'. Using default: {self._context_limit}"
            )

        try:
            self.client = self.anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise LLMConfigurationError(f"Failed to initialize Anthropic client: {e}") from e

        # Check max_tokens against the known limit
        if self.max_tokens > self._context_limit:
            logger.warning(
                f"Requested max_tokens ({self.max_tokens}) exceeds the known context limit ({self._context_limit}) for model '{self.model}'. API calls might fail."
            )
        elif self.max_tokens > self._context_limit * 0.8:
            logger.info(
                f"Requested max_tokens ({self.max_tokens}) is close to the context limit ({self._context_limit}). Ensure input + output fits."
            )

        self._system_message = system_message
        self._role_name = role_name or "Assistant"
        # Conversation object doesn't need the system message initially for Anthropic
        self.conversation = Conversation()
        logger.info(
            f"AnthropicLLM initialized. Model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Context Limit: {self._context_limit} (Hardcoded)"
        )

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the Anthropic Messages API."""
        system_prompt = self._system_message  # System prompt is passed separately
        # Filter out system message from history, ensure roles are 'user'/'assistant'
        filtered_messages = []
        approx_input_tokens = 0  # Estimate input tokens
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                # Use the _system_message attribute, ignore system messages in list
                continue
            elif role in ["user", "assistant"]:
                if content is None:
                    logger.warning(f"Skipping message with role '{role}' because content is None.")
                    continue
                filtered_messages.append({"role": role, "content": content})
                approx_input_tokens += self.count_tokens(content)  # Add to token estimate
            else:
                logger.warning(
                    f"Skipping message with unsupported role '{role}' for Anthropic API."
                )

        if not filtered_messages:
            raise LLMClientError(
                "Cannot call Anthropic API with an empty message list (after filtering)."
            )

        # Check estimated tokens against limit
        if approx_input_tokens + self.max_tokens > self._context_limit:
            logger.warning(
                f"Estimated input tokens (~{approx_input_tokens}) + max_tokens ({self.max_tokens}) exceeds context limit ({self._context_limit}). API call might fail."
            )
        elif approx_input_tokens > self._context_limit * 0.95:
            logger.info(
                f"Estimated input tokens (~{approx_input_tokens}) is very close to context limit ({self._context_limit})."
            )

        # --- Anthropic API Constraints Validation ---
        if filtered_messages[0]["role"] != "user":
            logger.error(
                f"Anthropic message list must start with 'user' role. First role found: '{filtered_messages[0]['role']}'."
            )
            # Attempt to fix by removing leading non-user messages? Or just error out?
            # Let's error out for now to be explicit.
            raise LLMClientError("Anthropic requires messages to start with the 'user' role.")

        # Check for alternating roles (simple check)
        for i in range(len(filtered_messages) - 1):
            if filtered_messages[i]["role"] == filtered_messages[i + 1]["role"]:
                logger.warning(
                    f"Consecutive messages with role '{filtered_messages[i]['role']}' found at index {i}. Anthropic API requires alternating roles."
                )
                # Consider raising LLMClientError here if strict adherence is needed.

        logger.debug(
            f"Calling Anthropic API ({self.model}) with {len(filtered_messages)} messages. System prompt: {'Yes' if system_prompt else 'No'}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt if system_prompt else None,  # Pass system prompt here
                messages=filtered_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # --- Response Validation ---
            if not response.content:
                stop_reason = getattr(response, "stop_reason", "Unknown")
                logger.error(
                    f"Anthropic response missing 'content'. Stop Reason: {stop_reason}. Response: {response}"
                )
                # Check if stop reason indicates max tokens
                if stop_reason == "max_tokens":
                    raise LLMResponseError(
                        f"Anthropic response likely stopped due to reaching max_tokens ({self.max_tokens}) or context limit ({self._context_limit})."
                    )
                else:
                    raise LLMResponseError(
                        f"Invalid response structure from Anthropic: No 'content'. Stop Reason: {stop_reason}."
                    )

            # Content is a list of blocks, usually one text block for simple chat
            if not isinstance(response.content, list) or len(response.content) == 0:
                logger.error(
                    f"Anthropic response 'content' is not a non-empty list. Type: {type(response.content)}. Response: {response}"
                )
                raise LLMResponseError(
                    f"Invalid response structure from Anthropic: 'content' is not a non-empty list."
                )

            # Extract text from the first text block if available
            content_text = None
            if hasattr(response.content[0], "text"):
                content_text = response.content[0].text

            if content_text is None:
                stop_reason = getattr(response, "stop_reason", "Unknown")
                logger.error(
                    f"Anthropic response content block missing 'text'. Stop Reason: {stop_reason}. Response: {response}"
                )
                # Check stop reason for clues
                if stop_reason == "max_tokens":
                    raise LLMResponseError(
                        f"Anthropic response truncated due to max_tokens ({self.max_tokens}) or context limit ({self._context_limit})."
                    )
                # Add other specific stop reason checks if needed (e.g., 'tool_use' if tools were involved)
                else:
                    raise LLMResponseError(
                        f"Anthropic response content block missing 'text'. Stop Reason: {stop_reason}."
                    )

            content = content_text.strip()
            stop_reason = getattr(response, "stop_reason", "Unknown")
            logger.debug(
                f"Received content from Anthropic API. Length: {len(content)}. Stop Reason: {stop_reason}"
            )

            if not content and stop_reason == "stop_sequence":
                logger.warning(
                    f"Received empty content string from Anthropic API, but stop reason was 'stop_sequence'. Response: {response}"
                )
                # Similar to OpenAI, decide if this is valid or an error. Return empty for now.

            # Check for error stop reasons even if content exists (e.g., partial output before error)
            if stop_reason == "error":
                logger.error(
                    f"Anthropic response indicates an error stop reason. Response: {response}"
                )
                # Potentially check if error type indicates overload_error (related to context)
                error_type = getattr(response, "error", {}).get("type")
                if error_type == "overload_error":
                    raise LLMResponseError(
                        "Anthropic API Error: Overloaded, potentially due to exceeding context limits."
                    )
                else:
                    raise LLMResponseError(
                        f"Anthropic API reported an error during generation (Type: {error_type})."
                    )

            return content

        # --- Error Handling ---
        except self.anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {e}")
            raise LLMClientError(f"Network error connecting to Anthropic API: {e}") from e
        except self.anthropic.RateLimitError as e:
            logger.error(f"Anthropic API rate limit exceeded: {e}")
            raise LLMResponseError(
                f"Anthropic API rate limit exceeded. Check your plan and usage limits. Error: {e}"
            ) from e
        except self.anthropic.AuthenticationError as e:
            logger.error(f"Anthropic API authentication error: {e}")
            raise LLMConfigurationError(
                f"Anthropic API authentication failed. Check your API key. Error: {e}"
            ) from e
        except self.anthropic.PermissionDeniedError as e:
            logger.error(f"Anthropic API permission denied: {e}")
            raise LLMConfigurationError(
                f"Anthropic API permission denied. Check API key permissions. Error: {e}"
            ) from e
        except self.anthropic.NotFoundError as e:
            logger.error(f"Anthropic API resource not found (e.g., model): {e}")
            raise LLMConfigurationError(
                f"Anthropic API resource not found (check model name '{self.model}'?). Error: {e}"
            ) from e
        except (
            self.anthropic.BadRequestError
        ) as e:  # Covers invalid request structure, role issues etc.
            logger.error(f"Anthropic API bad request error: {e}")
            # Try to get more details from the error if possible
            error_detail = str(e)
            error_type = None
            if hasattr(e, "body") and e.body and "error" in e.body:
                error_type = e.body["error"].get("type")
                if "message" in e.body["error"]:
                    error_detail = e.body["error"]["message"]

            # Check if the error indicates context length issues
            if error_type == "invalid_request_error" and (
                "prompt is too long" in error_detail or "exceeds the context window" in error_detail
            ):
                raise LLMResponseError(
                    f"Anthropic API Error: Request likely exceeded context limit ({self._context_limit}). Details: {error_detail}"
                ) from e
            elif error_type == "overload_error":  # Sometimes context issues manifest as overload
                raise LLMResponseError(
                    f"Anthropic API Error: Overloaded, potentially due to context limit ({self._context_limit}). Details: {error_detail}"
                ) from e
            else:
                raise LLMResponseError(
                    f"Anthropic API reported a bad request (check message format/roles?). Type: {error_type}. Error: {error_detail}"
                ) from e
        except self.anthropic.APIStatusError as e:  # Catch other 4xx/5xx
            logger.error(f"Anthropic API status error: {e.status_code} - {e.response}")
            error_detail = str(e)
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", str(e))
            except:
                pass
            raise LLMResponseError(
                f"Anthropic API returned an error (Status {e.status_code}): {error_detail}"
            ) from e
        # Catch other potential Anthropic client errors
        except self.anthropic.AnthropicError as e:
            logger.error(f"An unexpected Anthropic client error occurred: {e}")
            raise LLMClientError(f"An unexpected Anthropic client error occurred: {e}") from e
        # --- Allow specific LLMResponseErrors from validation to pass through ---
        except LLMResponseError as e:
            # This catches errors raised explicitly in the validation logic above
            # (e.g., missing content)
            logger.error(f"LLM Response Error: {e}")  # Log it, but let it propagate
            raise
        # General exception catcher
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Anthropic API call: {e}")
            raise LLMClientError(f"Unexpected error during Anthropic API call: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using the Anthropic SDK."""
        try:
            # Use the client instance's count_tokens method
            token_count = self.client.count_tokens(text)
            # Removed debug log
            return token_count
        except AttributeError:
            logger.warning(
                "Anthropic client does not have 'count_tokens' method (older version?). Falling back to approximation."
            )
        except Exception as e:
            logger.error(
                f"Error counting tokens with Anthropic SDK: {e}. Falling back to approximation."
            )

        # Fallback approximation (e.g., 4 chars per token)
        estimated_tokens = len(text) // 4
        logger.warning(
            f"Using approximate token count: {estimated_tokens} (based on character count). Update 'anthropic' package for accuracy."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) for the configured Anthropic model.

        Note: These limits are currently hardcoded as the Anthropic API/SDK
        does not provide a standard way to fetch them dynamically.
        """
        limit = self._context_limit  # Use the value set in __init__
        logger.debug(f"Returning context limit for {self.model}: {limit} (Hardcoded)")
        return limit


GEMINI_CONTEXT_LIMITS = {
    "gemini-1.5-flash-latest": 1048576,  # 1M context
    "gemini-1.5-pro-latest": 1048576,  # 1M context (up to 2M available)
    "gemini-1.0-pro": 32768,  # 32k context (includes input+output)
}
DEFAULT_GEMINI_CONTEXT_LIMIT = 32768


class GeminiLLM(LLMInterface):
    """Concrete implementation for Google's Gemini API."""

    SUPPORTED_MODELS = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash-latest",  # Default to flash for speed/cost
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,  # Gemini uses max_output_tokens
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
    ):
        """Initializes the Google Gemini client.

        Requires the 'google-generativeai' package (`pip install google-generativeai`).
        Attempts to fetch the model's input/output token limits during initialization.

        Raises:
            LLMConfigurationError: If API key is missing, configuration fails, or client initialization fails.
        """
        try:
            import google.generativeai as genai

            # Import google API core exceptions for specific error handling
            import google.api_core.exceptions as google_exceptions

            self.genai = genai
            self.google_exceptions = google_exceptions
        except ImportError as e:
            raise LLMConfigurationError(
                "Google Generative AI package not found. Please install it using 'pip install google-generativeai'."
            ) from e

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "Google API key not provided. Set the GOOGLE_API_KEY environment variable or pass it during initialization."
            )

        try:
            # Configure the API key globally for the genai module
            self.genai.configure(api_key=self.api_key)
            logger.info("Google Generative AI SDK configured successfully.")
        except Exception as e:
            # Catch potential issues during configure()
            raise LLMConfigurationError(f"Failed to configure Google API key: {e}") from e

        # Normalize model name (remove 'models/' prefix if present)
        if model.startswith("models/"):
            model = model.split("/", 1)[1]

        if model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model}' is not in the explicitly supported list for GeminiLLM ({self.SUPPORTED_MODELS}). Proceeding, but compatibility is not guaranteed."
            )
        self.model_name = model  # Store the potentially normalized name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            # Define generation configuration
            self.generation_config = self.genai.types.GenerationConfig(
                # candidate_count=1 # Default is 1, usually no need to change
                # stop_sequences=... # Optional stop sequences
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                # top_p=... # Optional nucleus sampling
                # top_k=... # Optional top-k sampling
            )
        except Exception as e:
            # Catch potential errors creating GenerationConfig (e.g., invalid values)
            raise LLMConfigurationError(f"Failed to create Gemini GenerationConfig: {e}") from e

        # Define safety settings (adjust as needed)
        # Defaults are generally BLOCK_MEDIUM_AND_ABOVE for most categories.
        # Setting to BLOCK_NONE disables safety filtering for that category (USE WITH CAUTION).
        self.safety_settings = {
            # Example: Relax harassment slightly (BLOCK_ONLY_HIGH)
            # self.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: self.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            # Example: Disable hate speech filter (BLOCK_NONE) - Use responsibly!
            # self.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
        if self.safety_settings:
            logger.warning(
                f"Using custom safety settings for Gemini: {self.safety_settings}. Be aware of the implications."
            )

        self._system_message = system_message
        self._role_name = role_name or "Model"  # Gemini uses 'model' role

        try:
            # Initialize the generative model instance
            # Pass system_instruction directly here
            self.model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self._system_message if self._system_message else None,
            )
            logger.info(f"Gemini GenerativeModel initialized for '{self.model_name}'.")
        except self.google_exceptions.NotFound as e:
            logger.error(f"Gemini model '{self.model_name}' not found or access denied: {e}")
            raise LLMConfigurationError(
                f"Gemini model '{self.model_name}' not found or access denied. Check model name and API key permissions. Error: {e}"
            ) from e
        except Exception as e:
            # Catch other potential errors during model initialization
            logger.exception(
                f"Failed to initialize Gemini GenerativeModel '{self.model_name}': {e}"
            )
            raise LLMConfigurationError(
                f"Failed to initialize Gemini GenerativeModel '{self.model_name}': {e}"
            ) from e

        # Conversation state for Gemini
        # Internal conversation uses standard roles ('user', 'assistant')
        self.conversation = Conversation()
        # Actual chat session with Gemini API (uses 'user', 'model' roles)
        self.chat: Optional[genai.ChatSession] = None  # Initialize chat session lazily
        logger.info(
            f"GeminiLLM initialized. Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}"
        )

    def _translate_roles_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Translates roles from internal standard ('system', 'user', 'assistant')
        to Gemini's required format ('user', 'model') within a 'contents' list.
        System message is handled by model initialization or system_instruction,
        so 'system' roles in the message list are typically skipped.
        """
        gemini_history = []
        valid_roles = {"user": "user", "assistant": "model"}
        expected_role = (
            "user"  # Gemini API expects alternating user/model roles, starting with user
        )

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if content is None:  # Skip messages without content
                logger.warning(
                    f"Skipping message with role '{role}' at index {i} due to None content."
                )
                continue

            if role == "system":
                if i == 0:
                    # Skip initial system message as it's handled by system_instruction
                    logger.debug(
                        "Skipping initial system message in history (handled by system_instruction)."
                    )
                    continue
                else:
                    # System messages mid-conversation are not directly supported.
                    # Log a warning and skip. Consider merging with next user message if needed.
                    logger.warning(
                        f"System message found mid-conversation at index {i}; skipping for Gemini API call."
                    )
                    continue

            translated_role = valid_roles.get(role)
            if not translated_role:
                logger.warning(
                    f"Unsupported role '{role}' encountered at index {i} for Gemini. Skipping message."
                )
                continue

            # --- Role Alternation Check ---
            # If the history is not empty, check if the current role matches the last one
            if gemini_history and gemini_history[-1]["role"] == translated_role:
                logger.warning(
                    f"Role alternation mismatch for Gemini: Consecutive '{translated_role}' roles found. API call might fail or behave unexpectedly."
                )
                # Simple fix attempt: skip this message to maintain alternation? Or let API handle?
                # For now, log warning and include it. API might handle it.
                # If issues arise, consider raising an error or attempting merging logic here.

            # --- Append to Gemini History ---
            # Gemini expects content in the format: {'role': ..., 'parts': [{'text': ...}]}
            gemini_history.append({"role": translated_role, "parts": [{"text": content}]})
            # Update expected role for the next iteration (not strictly needed as we check against the last appended role)
            # expected_role = 'model' if translated_role == 'user' else 'user'

        # --- Final Validation ---
        if gemini_history and gemini_history[0]["role"] != "user":
            logger.warning(
                "Gemini history (after translation) does not start with 'user' role. Attempting to proceed, but API may reject."
            )
            # Could potentially insert a dummy user message or raise error here.

        return gemini_history

    def _handle_gemini_response(self, response, context: str = "generate_content") -> str:
        """
        Helper to process Gemini response object (from generate_content or chat.send_message),
        extract text content, and handle potential errors like blocking.

        Args:
            response: The response object from the Gemini API call.
            context: String indicating call context ('generate_content' or 'chat') for logging.

        Returns:
            The extracted text content.

        Raises:
            LLMResponseError: If the response is blocked, invalid, empty, or indicates an error.
        """
        logger.debug(f"Handling Gemini response from '{context}'.")

        # 1. Check for Prompt Feedback (blocking before generation)
        # Sometimes present even if candidates exist but generation was stopped early.
        if hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason.name  # Get the enum name
            safety_ratings_str = str(getattr(response.prompt_feedback, "safety_ratings", "N/A"))
            logger.error(
                f"Gemini prompt blocked in '{context}'. Reason: {block_reason}. Safety Ratings: {safety_ratings_str}. Response: {response}"
            )
            raise LLMResponseError(
                f"Gemini prompt blocked due to {block_reason}. Safety: {safety_ratings_str}"
            )

        # 2. Check for Candidates
        if not response.candidates:
            # This might happen if the prompt itself was blocked, or other issues.
            prompt_feedback_str = str(getattr(response, "prompt_feedback", "N/A"))
            logger.error(
                f"Gemini response in '{context}' has no candidates. Prompt Feedback: {prompt_feedback_str}. Response: {response}"
            )
            # Check if prompt feedback gives a reason
            if hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                raise LLMResponseError(
                    f"Gemini response has no candidates, likely due to prompt blocking. Reason: {block_reason}."
                )
            else:
                raise LLMResponseError(
                    f"Gemini response has no candidates. Prompt Feedback: {prompt_feedback_str}"
                )

        # 3. Access Content and Check Finish Reason/Safety (within the first candidate)
        candidate = response.candidates[0]
        content_text = None
        finish_reason = "UNKNOWN"  # Default
        safety_ratings_str = "N/A"  # Default

        try:
            # Finish reason and safety ratings are usually on the candidate
            finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            safety_ratings_str = str(getattr(candidate, "safety_ratings", "N/A"))

            # Content can be inside candidate.content.parts
            if candidate.content and candidate.content.parts:
                # Aggregate text from all parts (usually just one)
                content_text = "".join(
                    part.text for part in candidate.content.parts if hasattr(part, "text")
                )

            # Sometimes, the response object has a direct .text attribute (convenience)
            # Let's prefer the explicit parts extraction but use .text as fallback
            if content_text is None and hasattr(response, "text"):
                content_text = response.text
                logger.debug("Extracted content using response.text fallback.")

            if content_text is None:
                # If still no text, something is wrong
                logger.error(
                    f"Could not extract text content from Gemini candidate in '{context}'. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}. Candidate: {candidate}"
                )
                raise LLMResponseError(
                    f"Could not extract text content from Gemini response. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
                )

        except AttributeError as e:
            logger.error(
                f"AttributeError accessing Gemini response candidate details in '{context}': {e}. Response: {response}",
                exc_info=True,
            )
            raise LLMResponseError(f"Error accessing Gemini response structure: {e}") from e
        except ValueError as e:
            # This might occur if accessing .text fails due to blocking (though prompt_feedback is primary check)
            logger.error(
                f"ValueError accessing Gemini response text in '{context}', potentially due to blocking. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}. Error: {e}. Response: {response}",
                exc_info=True,
            )
            raise LLMResponseError(
                f"Gemini response blocked or invalid. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
            ) from e

        # 4. Post-extraction Checks (Finish Reason, Safety, Emptiness)
        content = content_text.strip()
        logger.debug(
            f"Received content from Gemini API ({context}). Length: {len(content)}. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
        )

        # Check finish reason for potential issues even if some text exists
        if finish_reason == "SAFETY":
            logger.error(
                f"Gemini response flagged for SAFETY in '{context}'. Safety Ratings: {safety_ratings_str}. Content (may be partial): '{content[:100]}...'. Response: {response}"
            )
            raise LLMResponseError(
                f"Gemini response blocked or cut short due to SAFETY. Ratings: {safety_ratings_str}"
            )
        elif finish_reason == "RECITATION":
            logger.warning(
                f"Gemini response flagged for RECITATION in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )
            # Recitation might be acceptable depending on use case, but log it.
        elif finish_reason == "OTHER":
            logger.warning(
                f"Gemini response finished with OTHER reason in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )
            # May indicate unexpected issues.
        elif finish_reason not in [
            "STOP",
            "MAX_TOKENS",
            "UNKNOWN",
        ]:  # Check for unexpected valid reasons
            logger.warning(
                f"Gemini response finished with unexpected reason '{finish_reason}' in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )

        # Check for empty content after stripping
        if not content and finish_reason == "STOP":
            logger.warning(
                f"Received empty content string from Gemini API ({context}), but finish reason was 'STOP'. Response: {response}"
            )
            # Return empty string as it might be intentional.
        elif not content and finish_reason != "STOP":
            logger.error(
                f"Received empty content string from Gemini API ({context}) with finish reason '{finish_reason}'. Response: {response}"
            )
            raise LLMResponseError(
                f"Received empty content from Gemini API. Finish Reason: {finish_reason}"
            )

        return content

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Internal method for stateless Gemini API calls using generate_content.
        The public 'query' method handles choosing between this and stateful chat.
        """
        logger.debug(
            f"Calling Gemini API (stateless generate_content) for model '{self.model_name}'. Translating {len(messages)} messages."
        )
        gemini_history = self._translate_roles_for_gemini(messages)

        if not gemini_history:
            logger.error(
                "Cannot call Gemini API (generate_content) with empty history after role translation."
            )
            # Check if original messages only contained system messages
            if all(m.get("role") == "system" for m in messages):
                raise LLMClientError(
                    "Cannot call Gemini API (generate_content): Input contained only system messages."
                )
            else:
                raise LLMClientError(
                    "No valid messages found for Gemini API call (generate_content) after role translation."
                )

        logger.debug(f"Calling generate_content with {len(gemini_history)} translated messages.")
        try:
            # Use the initialized self.model instance
            response = self.model.generate_content(
                contents=gemini_history,
                # generation_config and safety_settings are part of self.model
                stream=False,  # Use non-streaming for simple query interface
            )
            # Process response using the common handler
            return self._handle_gemini_response(response, context="generate_content")

        # --- Specific Google API Error Handling ---
        except self.google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini API permission denied (generate_content): {e}")
            raise LLMConfigurationError(
                f"Gemini API permission denied. Check API key/permissions. Error: {e}"
            ) from e
        except self.google_exceptions.InvalidArgument as e:
            # Often indicates issues with the request structure (e.g., roles, content format)
            logger.error(f"Gemini API invalid argument (generate_content): {e}")
            raise LLMResponseError(
                f"Gemini API invalid argument (check message roles/format?). Error: {e}"
            ) from e
        except self.google_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini API resource exhausted (generate_content - rate limit?): {e}")
            raise LLMResponseError(
                f"Gemini API resource exhausted (likely rate limit). Error: {e}"
            ) from e
        except self.google_exceptions.NotFound as e:
            # Should be caught at init, but maybe model becomes unavailable later?
            logger.error(
                f"Gemini API resource not found (generate_content - model '{self.model_name}'?): {e}"
            )
            raise LLMConfigurationError(
                f"Gemini API resource not found (model '{self.model_name}'?). Error: {e}"
            ) from e
        except self.google_exceptions.InternalServerError as e:
            logger.error(f"Gemini API internal server error (generate_content): {e}")
            raise LLMResponseError(
                f"Gemini API reported an internal server error. Try again later. Error: {e}"
            ) from e
        except self.google_exceptions.GoogleAPIError as e:  # Catch-all for other Google API errors
            logger.error(f"Gemini API error (generate_content): {e}")
            raise LLMResponseError(f"Gemini API returned an error: {e}") from e
        # --- General Exceptions ---
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during Gemini API call (generate_content): {e}"
            )
            # Attempt to get a more specific message if available
            error_message = getattr(e, "message", str(e))
            raise LLMClientError(
                f"Unexpected error during Gemini API call (generate_content): {error_message}"
            ) from e

    def query(
        self,
        prompt: str,
        use_conversation: bool = True,
        conversation_history: Optional[
            List[Dict[str, str]]
        ] = None,  # Used only if use_conversation=False
    ) -> str:
        """
        Sends a prompt to the Gemini LLM.

        If use_conversation is True, it utilizes a stateful ChatSession, managing
        history internally (translating roles as needed).
        If use_conversation is False, it performs a stateless generate_content call
        using the logic from the base LLMInterface.query method.

        Args:
            prompt: The user's input prompt.
            use_conversation: If True, use stateful chat session. If False, use stateless call.
            conversation_history: Optional history for stateless calls (ignored if use_conversation=True).

        Returns:
            The LLM's response content as a string.

        Raises:
            LLMConfigurationError: For configuration issues.
            LLMClientError: For client-side issues.
            LLMResponseError: For API errors or problematic responses.
        """
        if use_conversation:
            # --- Stateful Chat Session ---
            logger.debug("Using stateful Gemini chat session.")

            # Initialize chat session if it doesn't exist
            if not self.chat:
                logger.info("Starting new Gemini chat session.")
                # Translate existing internal history (user/assistant) to Gemini format (user/model)
                # Use the current state of self.conversation
                initial_gemini_history = self._translate_roles_for_gemini(
                    self.conversation.get_messages()
                )
                logger.debug(
                    f"Initializing chat with {len(initial_gemini_history)} translated messages."
                )
                try:
                    # Start chat using the initialized self.model (which includes system instruction etc.)
                    self.chat = self.model.start_chat(history=initial_gemini_history)
                except self.google_exceptions.InvalidArgument as e:
                    logger.error(
                        f"Failed to start Gemini chat session due to invalid argument (check history format/roles?): {e}"
                    )
                    raise LLMResponseError(
                        f"Failed to start Gemini chat session (invalid history/roles?): {e}"
                    ) from e
                except Exception as e:
                    logger.exception(f"Failed to start Gemini chat session: {e}")
                    raise LLMClientError(f"Failed to start Gemini chat session: {e}") from e

            # --- Send Message via Chat ---
            user_message_added_to_internal = False
            try:
                logger.debug(f"Sending message to Gemini chat: '{prompt[:100]}...'")
                # Send the prompt using the chat session
                response = self.chat.send_message(prompt, stream=False)

                # --- Process response ---
                content = self._handle_gemini_response(response, context="chat")

                # --- Update Internal State (on success) ---
                # Add the user prompt that was successfully sent and processed.
                self.conversation.add_user_message(prompt)
                user_message_added_to_internal = True  # Mark success
                # Add the successful assistant/model response.
                self.conversation.add_assistant_message(content)
                logger.debug("Updated internal conversation history after successful chat message.")

                # Update the chat object's history (optional, but good practice if reusing chat object elsewhere)
                # Note: The google library might update chat.history automatically, but explicit sync can be safer
                # self.chat.history = self._translate_roles_for_gemini(self.conversation.get_messages())

                return content

            # --- Error Handling for Chat Session ---
            except (LLMClientError, LLMConfigurationError, LLMResponseError) as e:
                # These errors are already logged by _handle_gemini_response or raised directly
                # If the error happened *after* the user message was added internally (shouldn't happen often), log it.
                if user_message_added_to_internal:
                    logger.warning(
                        "Error occurred after user message was added to internal state but before assistant response - state might be inconsistent."
                    )
                    # Consider removing the user message here if necessary, though it indicates partial success followed by failure.
                # No need to pop user message here as it wasn't added on the failure path of send_message or _handle_gemini_response
                raise  # Re-raise the specific error
            except self.google_exceptions.PermissionDenied as e:
                logger.error(f"Gemini API chat permission denied: {e}")
                raise LLMConfigurationError(f"Gemini chat permission denied: {e}") from e
            except self.google_exceptions.InvalidArgument as e:
                logger.error(f"Gemini API chat invalid argument: {e}")
                raise LLMResponseError(f"Gemini chat invalid argument: {e}") from e
            except self.google_exceptions.ResourceExhausted as e:
                logger.error(f"Gemini API chat resource exhausted (rate limit?): {e}")
                raise LLMResponseError(f"Gemini chat resource exhausted: {e}") from e
            except self.google_exceptions.InternalServerError as e:
                logger.error(f"Gemini API internal server error (chat): {e}")
                raise LLMResponseError(
                    f"Gemini API reported an internal server error during chat. Try again later. Error: {e}"
                ) from e
            except (
                self.google_exceptions.GoogleAPIError
            ) as e:  # Catch-all Google API errors for chat
                logger.error(f"Gemini API chat error: {e}")
                raise LLMResponseError(f"Gemini chat API returned an error: {e}") from e
            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred during Gemini chat session send_message: {e}"
                )
                error_message = getattr(e, "message", str(e))
                # Don't modify internal conversation state here, as failure happened during API call
                raise LLMClientError(
                    f"Unexpected error during Gemini chat session: {error_message}"
                ) from e

        else:
            # --- Stateless Call ---
            logger.debug("Using stateless Gemini API call (generate_content via base class query).")
            # Delegate to the base class query method, which will call our _call_api (stateless version)
            return super().query(
                prompt, use_conversation=False, conversation_history=conversation_history
            )

    def clear_conversation(self, keep_system: bool = True):
        """Clears the internal conversation history and resets the chat session."""
        super().clear_conversation(keep_system=keep_system)
        # Also reset the stateful chat session object
        if self.chat:
            logger.debug("Resetting Gemini chat session object.")
            self.chat = None

    def set_system_message(self, message: Optional[str]):
        """Sets the system message and re-initializes the underlying Gemini model if needed."""
        if message != self._system_message:
            logger.info(f"System message changed for Gemini. Re-initializing GenerativeModel.")
            super().set_system_message(message)  # Update internal state and conversation object
            try:
                # Re-initialize the model with the new system instruction
                self.model = self.genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    system_instruction=self._system_message if self._system_message else None,
                )
                # Reset any existing chat session as its context is now based on the old system message
                self.chat = None
                logger.info(f"Gemini GenerativeModel re-initialized with new system instruction.")
            except Exception as e:
                # Log error but try to continue, state might be inconsistent
                logger.exception(
                    f"Failed to re-initialize Gemini GenerativeModel after system message change: {e}"
                )
                # Raise config error as the client might be unusable
                raise LLMConfigurationError(
                    f"Failed to re-initialize Gemini model after system message change: {e}"
                ) from e
        else:
            logger.debug("System message unchanged for Gemini.")

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using the Gemini SDK (model.count_tokens)."""
        if not hasattr(self, "model") or self.model is None:
            logger.error("Gemini model not initialized. Cannot count tokens.")
            # Fall back to approximation if model isn't ready
            estimated_tokens = len(text) // 4
            logger.warning(
                f"Using approximate token count: {estimated_tokens} (model not initialized)."
            )
            return estimated_tokens

        try:
            # Use the model instance's count_tokens method
            # It requires the content to be tokenized (can be string, list of strings, or dict)
            count_response = self.model.count_tokens(text)
            token_count = count_response.total_tokens
            logger.debug(
                f"Counted {token_count} tokens for text (length {len(text)}) using Gemini SDK."
            )
            return token_count
        except AttributeError:
            logger.warning(
                "Gemini model object does not have 'count_tokens' method (older version or initialization issue?). Falling back to approximation."
            )
        except self.google_exceptions.GoogleAPIError as e:
            logger.error(
                f"Gemini API error during token count: {e}. Falling back to approximation."
            )
        except Exception as e:
            logger.error(
                f"Unexpected error counting tokens with Gemini SDK: {e}. Falling back to approximation."
            )

        # Fallback approximation (e.g., 4 chars per token)
        estimated_tokens = len(text) // 4
        logger.warning(
            f"Using approximate token count: {estimated_tokens} (based on character count). Check SDK version or API errors."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) for the configured Gemini model."""
        # Note: Gemini 1.5 models have large context, but actual usable input/output limits might vary.
        # Gemini 1.0 limits often include both input and output tokens.
        limit = GEMINI_CONTEXT_LIMITS.get(self.model_name)  # Use model_name which is normalized
        if limit:
            logger.debug(f"Context limit for {self.model_name}: {limit}")
            return limit
        else:
            logger.warning(
                f"Context limit not defined for Gemini model '{self.model_name}'. Returning default: {DEFAULT_GEMINI_CONTEXT_LIMIT}"
            )
            return DEFAULT_GEMINI_CONTEXT_LIMIT


# --- Factory Function ---

_PROVIDER_MAP = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "gemini": GeminiLLM,
    # Add aliases if needed
    "gpt": OpenAILLM,
    "claude": AnthropicLLM,
    "google": GeminiLLM,
}


def create_llm_client(
    client_type: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,  # If None, provider class will use its default
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_message: Optional[str] = None,
    role_name: Optional[str] = None,
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
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_message": system_message,
            "role_name": role_name,
        }
        if model:
            client_args["model"] = model

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


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly
    # Requires environment variables (e.g., OPENAI_API_KEY) to be set

    print("Running LLM Client Example...")
    logger.setLevel(logging.DEBUG)  # Show debug messages for example
    logging.basicConfig(level=logging.DEBUG)  # Configure root logger for dependencies

    try:
        # --- OpenAI Example ---
        print("\n--- OpenAI Example ---")
        try:
            openai_client = create_llm_client(
                "openai", system_message="You are a concise assistant."
            )
            # Stateless query
            response_stateless = openai_client.query(
                "What is the capital of France?", use_conversation=False
            )
            print(f"OpenAI Stateless Response: {response_stateless}")
            # Stateful query
            response1 = openai_client.query("What is Python?")
            print(f"OpenAI Response 1: {response1}")
            response2 = openai_client.query("What are its main uses?")
            print(f"OpenAI Response 2: {response2}")
            print("OpenAI Conversation History:")
            for msg in openai_client.get_conversation_history():
                print(f"  {msg['role']}: {msg['content'][:80]}...")
            openai_client.clear_conversation()
            print("OpenAI Conversation Cleared.")
        except LLMConfigurationError as e:
            print(f"OpenAI Configuration Error: {e}")
        except (LLMClientError, LLMResponseError) as e:
            print(f"OpenAI Client/Response Error: {e}")

        # --- Anthropic Example ---
        print("\n--- Anthropic Example ---")
        try:
            # Ensure ANTHROPIC_API_KEY is set in env
            anthropic_client = create_llm_client(
                "anthropic",
                model="claude-3-haiku-20240307",
                system_message="Respond like a pirate.",
            )
            response1 = anthropic_client.query("Why is the sky blue?")
            print(f"Anthropic Response 1: {response1}")
            response2 = anthropic_client.query("Tell me a short joke about the sea.")
            print(f"Anthropic Response 2: {response2}")
            print("Anthropic Conversation History:")
            for msg in anthropic_client.get_conversation_history():
                print(f"  {msg['role']}: {msg['content'][:80]}...")
            anthropic_client.clear_conversation(keep_system=False)  # Clear system too
            print("Anthropic Conversation Cleared (including system).")
            # Query again without system message
            response3 = anthropic_client.query("What is water?")
            print(f"Anthropic Response 3 (no system): {response3}")
        except LLMConfigurationError as e:
            print(f"Anthropic Configuration Error: {e}")
        except (LLMClientError, LLMResponseError) as e:
            print(f"Anthropic Client/Response Error: {e}")

        # --- Gemini Example ---
        print("\n--- Gemini Example ---")
        try:
            # Ensure GOOGLE_API_KEY is set in env
            gemini_client = create_llm_client(
                "gemini", system_message="Explain things simply, like I'm five."
            )
            response1 = gemini_client.query("How does a car engine work?")
            print(f"Gemini Response 1: {response1}")
            response2 = gemini_client.query("Why do we need fuel?")
            print(f"Gemini Response 2: {response2}")
            print("Gemini Conversation History (Internal):")
            for msg in gemini_client.get_conversation_history():
                print(f"  {msg['role']}: {msg['content'][:80]}...")
            # Test changing system message
            gemini_client.set_system_message("Explain like a rocket scientist.")
            response3 = gemini_client.query(
                "How does gravity work?"
            )  # Should start new chat context
            print(f"Gemini Response 3 (new system msg): {response3}")
            gemini_client.clear_conversation()
            print("Gemini Conversation Cleared.")
        except LLMConfigurationError as e:
            print(f"Gemini Configuration Error: {e}")
        except (LLMClientError, LLMResponseError) as e:
            print(f"Gemini Client/Response Error: {e}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the example: {e}")
