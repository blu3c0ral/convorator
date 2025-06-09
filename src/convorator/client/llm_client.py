# llm_client.py
from __future__ import annotations

"""
llm_client: Unified and Robust Interface for Large Language Model APIs

This module provides a consistent, provider-agnostic interface for interacting with
various Large Language Model (LLM) APIs, including OpenAI's GPT models, 
Anthropic's Claude models, and Google's Gemini models.

CRITICAL UPDATE NEEDED - OpenAI Responses API Migration:
===========================================================
The current OpenAI implementation uses the Chat Completions API, which is now
considered the "previous standard" by OpenAI. The new Responses API is the 
"primary API for interacting with OpenAI models" and offers significant advantages:

1. Built-in tools: web_search_preview, file_search, computer_use_preview
2. Server-side conversation state management (no need to send full history)
3. Better performance and tool orchestration
4. Priority for new features and models

Migration Priority: HIGH - The Responses API represents OpenAI's strategic direction
and provides capabilities not available in Chat Completions.

Current Status: The implementation below maintains Chat Completions for backward
compatibility but should be extended to support Responses API for new features.
===========================================================

It is designed for reliability, ease of use, and to abstract away the complexities
of individual provider SDKs, offering a standardized approach to LLM interactions.

Key Features:
- Multi-Provider Support: Seamlessly work with OpenAI, Anthropic, and Gemini APIs
  through a unified `LLMInterface`.
- Factory Function: Easy client creation using `create_llm_client('provider_name', ...)`.
- Conversation Management: Robust internal conversation history tracking (`Conversation` class)
  with helper methods for adding user and assistant messages.
- System Message Handling: Consistent `set_system_message` and `get_system_message` across
  providers, with provider-specific optimizations (e.g., `system_instruction` for Gemini,
  `system` parameter for Anthropic).
- Role Normalization & Translation: Internally uses standard roles ('user', 'assistant', 'system')
  and automatically translates them to provider-specific roles (e.g., 'model' for Gemini).
  Handles role alternation requirements for APIs like Gemini and Anthropic, including
  automatic merging of consecutive messages with the same role for Gemini.
- Token Counting: Provider-aware `count_tokens` method using official SDKs (tiktoken for OpenAI,
  `client.messages.count_tokens` for Anthropic, `model.count_tokens` for Gemini) with graceful
  fallback to approximation if SDK methods fail.
- Context Limit Awareness: `get_context_limit` method provides model-specific context window sizes,
  utilizing dynamically fetched limits (for Gemini 1.5+) or up-to-date hardcoded values.
- Standardized Model Naming: Consistent access to model name via `get_model_name()` and internal
  `_model` attribute.
- Max Tokens Configuration: `max_tokens` parameter consistently handled across providers.
- Temperature Control: Standardized `temperature` setting.
- Comprehensive Error Handling: 
    - Custom exceptions (`LLMClientError`, `LLMConfigurationError`, `LLMResponseError`)
      for clear distinction of error sources.
    - Detailed logging of API calls, errors, and fallbacks.
    - Robust handling of API-specific errors (rate limits, authentication, content filtering, etc.).
    - Atomic updates for critical settings (e.g., Gemini system message and safety settings)
      with rollback mechanisms on failure.
    - Improved conversation state consistency on API errors during `query()`.
- Provider-Specific Capabilities: `get_provider_capabilities()` and `supports_feature()` allow
  querying specific features (e.g., streaming, safety settings, tiktoken support) for each provider.
- Stateless and Stateful Queries: `query()` method supports both modes (`use_conversation` flag).
- Up-to-Date API Usage: Ensured usage of General Availability (GA) endpoints (e.g., Anthropic token counting).

Architecture:
- `LLMInterface`: Abstract base class defining the common interface for all LLM clients.
- Concrete Implementations: `OpenAILLM`, `AnthropicLLM`, `GeminiLLM` inherit from `LLMInterface`.
- `Message` & `Conversation`: Dataclasses for managing individual messages and conversation history.
- `LLMClientConfig`: Dataclass for configuring LLM clients (used by the factory).
- `create_llm_client`: Factory function for easy instantiation of specific LLM clients.

Dependencies:
- `openai`: For OpenAI API interaction.
- `anthropic`: For Anthropic API interaction.
- `google-generativeai`: For Google Gemini API interaction.
- `tiktoken`: For OpenAI token counting.
- `logging`: For detailed operational logging.

Basic Usage:
    from convorator.client.llm_client import create_llm_client

    # Create an OpenAI client
    openai_llm = create_llm_client(
        client_type='openai',
        api_key="your-openai-api-key",
        model="gpt-4o",
        system_message="You are a helpful and concise AI assistant."
    )
    response = openai_llm.query("What is the capital of France?")
    print(f"OpenAI: {response}")

    # Create an Anthropic client
    anthropic_llm = create_llm_client(
        client_type='anthropic',
        api_key="your-anthropic-api-key",
        model="claude-3-haiku-20240307",
        max_tokens=150
    )
    anthropic_llm.set_system_message("Respond in a witty tone.")
    response = anthropic_llm.query("Why is the sky blue?", use_conversation=True)
    response = anthropic_llm.query("Any other fun facts?", use_conversation=True)
    print(f"Anthropic: {response}")

    # Create a Gemini client
    gemini_llm = create_llm_client(
        client_type='gemini',
        api_key="your-google-api-key",
        model="gemini-1.5-flash-latest"
    )
    response = gemini_llm.query("Explain black holes to a child.")
    print(f"Gemini: {response}")

Authors:
    Development Team (convorator)

Version:
    1.1.0 (Reflects significant enhancements and bug fixes)

Considerations:
- API Keys: Ensure appropriate API keys are set as environment variables 
  (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY) or passed during client creation.
- Package Installation: Required provider-specific packages (`openai`, `anthropic`, `google-generativeai`)
  and `tiktoken` must be installed.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# from openai.resources.chat.completions import


# Assuming standard project structure where 'convorator' is a top-level package

from convorator.utils.logger import setup_logger
from convorator.exceptions import (
    LLMClientError,
    LLMConfigurationError,
    LLMResponseError,
)
import tiktoken


logger = setup_logger("llm_client")  # Use a specific logger name

# Constants
DEFAULT_MAX_TOKENS = 100000
DEFAULT_TEMPERATURE = 0.6

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

    messages: List[Message] = field(default_factory=list)  # type: ignore
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
    _model: str  # Standardized attribute for the model name
    max_tokens: int

    def __init__(self, model: str, max_tokens: int):
        """
        Initializes the LLMInterface with essential parameters.

        Args:
            model: The specific model name (e.g., "gpt-4o", "claude-3-haiku-20240307").
            max_tokens: The maximum number of tokens to generate in a response.
                        This is a fundamental parameter for controlling LLM output length.
        """
        if not model:
            raise LLMConfigurationError("Model name cannot be empty.")
        self._model = model
        self.max_tokens = max_tokens

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

    # --- Standardized Provider-Specific Methods ---

    def get_provider_name(self) -> str:
        """
        Returns the name of the LLM provider.

        Returns:
            The provider name (e.g., "openai", "anthropic", "gemini").
        """
        class_name = self.__class__.__name__.lower()
        if "openai" in class_name:
            return "openai"
        elif "anthropic" in class_name:
            return "anthropic"
        elif "gemini" in class_name:
            return "gemini"
        else:
            return "unknown"

    def get_provider_capabilities(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the capabilities and features of this provider.

        Returns:
            A dictionary with provider-specific capabilities information.
        """
        return {
            "provider": self.get_provider_name(),
            "model": self._model,
            "max_tokens": self.max_tokens,
            "supports_conversation": True,
            "supports_system_message": True,
            "supports_temperature": hasattr(self, "temperature"),
            "context_limit": self.get_context_limit(),
        }

    def get_provider_settings(self) -> Dict[str, Any]:
        """
        Returns current provider-specific settings.
        Base implementation returns common settings; providers can override for specific settings.

        Returns:
            A dictionary of current provider settings.
        """
        settings = {
            "model": self._model,
            "max_tokens": self.max_tokens,
            "system_message": self._system_message,
            "role_name": self._role_name,
        }

        # Add temperature if available
        if hasattr(self, "temperature"):
            settings["temperature"] = self.temperature

        return settings

    def set_provider_setting(self, setting_name: str, value: Any) -> bool:
        """
        Sets a provider-specific setting if supported.
        Base implementation handles common settings; providers can override for specific settings.

        Args:
            setting_name: The name of the setting to change.
            value: The new value for the setting.

        Returns:
            True if the setting was successfully changed, False if not supported.
        """
        # NOTE: Temperature validation is intentionally NOT handled here.
        # Each provider should implement its own temperature validation
        # in their override of this method to ensure proper bounds checking.

        if setting_name == "max_tokens":
            self.max_tokens = value
            return True
        elif setting_name == "system_message":
            self.set_system_message(value)
            return True
        elif setting_name == "role_name":
            self.set_role_name(value)
            return True
        else:
            # Provider-specific settings should be handled by subclass overrides
            return False

    def supports_feature(self, feature_name: str) -> bool:
        """
        Checks if the provider supports a specific feature.

        Args:
            feature_name: The name of the feature to check.

        Returns:
            True if the feature is supported, False otherwise.
        """
        base_features = {
            "conversation_history",
            "system_message",
            "role_name",
            "token_counting",
            "context_limit",
            "stateless_query",
            "stateful_query",
        }

        return feature_name in base_features

    # --- Existing methods remain unchanged ---

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
        user_message_added_internally_this_turn = False

        try:
            if use_conversation:
                # --- Stateful Query ---
                # Ensure the internal conversation object has the correct system message
                if (self._system_message is not None) and (
                    self._system_message != self.conversation.system_message
                ):
                    self.conversation.add_message(role="system", content=self._system_message)

                # Add the current user prompt to the internal conversation
                self.conversation.add_user_message(prompt)
                user_message_added_internally_this_turn = True
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

                # Add the current user prompt - but only if it's not empty/whitespace
                # This handles the orchestrator pattern where prompt="" and the actual prompt
                # is already included in the provided conversation_history
                if prompt and prompt.strip():
                    messages_to_send.append({"role": "user", "content": prompt})
                    logger.debug(
                        f"Performing stateless query with {len(messages_to_send)} messages (prompt appended)."
                    )
                else:
                    logger.debug(
                        f"Performing stateless query with {len(messages_to_send)} messages (empty prompt not appended)."
                    )

            # --- API Call (Common for both stateful/stateless) ---
            response_content = self._call_api(messages_to_send)
            # --- If _call_api succeeded, the user prompt (if stateful) should remain ---

            # --- Add assistant response (if stateful and API call succeeded) ---
            if use_conversation:
                try:
                    self.conversation.add_assistant_message(response_content)
                    logger.debug("Added assistant response to internal conversation.")
                except Exception as e_assist_add:
                    # API call was successful, but adding assistant's response to OUR internal history failed.
                    # Log this failure but do NOT pop the user message. The prompt was successfully processed by the LLM.
                    logger.error(
                        f"LLM API call successful for {self.__class__.__name__}, but failed to add assistant's response to internal conversation history. "
                        f"User Prompt: '{prompt[:100]}...', LLM Response: '{response_content[:100]}...'. Internal Error: {e_assist_add}",
                        exc_info=True,
                    )
                    # The 'response_content' will still be returned.
                    # The internal conversation state will have the user message but not this assistant response. This is an acknowledged inconsistency.

            return response_content

        except (LLMClientError, LLMConfigurationError, LLMResponseError) as e:
            # These errors typically originate from _call_api or configuration issues before it.
            logger.error(
                f"LLM API Error or Configuration Error during query for {self.__class__.__name__}: {e}"
            )
            if use_conversation and user_message_added_internally_this_turn:
                # If add_user_message succeeded but _call_api (or something before it that didn't prevent add_user_message) failed.
                if self.conversation.messages and self.conversation.messages[-1].role == "user":
                    # Basic check: is the last message the one we added?
                    # Content check could be self.conversation.messages[-1].content == prompt
                    removed_msg = self.conversation.messages.pop()
                    logger.debug(
                        f"Removed last user message ('{removed_msg.content[:50]}...') from internal history due to API/Config error before assistant response."
                    )
            raise  # Re-raise the original, specific error

        except Exception as e_unexpected:
            # Catch-all for truly unexpected things not caught by the specific handlers above.
            logger.exception(
                f"Unexpected error during LLM query for {self.__class__.__name__}: {e_unexpected}"
            )
            if use_conversation and user_message_added_internally_this_turn:
                # If user message was added, but then an unexpected crash occurred before/during _call_api or during add_assistant_message
                # (though add_assistant_message failure is caught separately now).
                # This path means an unexpected error happened likely during _call_api or setup.
                if self.conversation.messages and self.conversation.messages[-1].role == "user":
                    removed_msg = self.conversation.messages.pop()
                    logger.debug(
                        f"Removed last user message ('{removed_msg.content[:50]}...') from internal history due to unexpected error."
                    )
            # Wrap unexpected errors in LLMClientError for consistent error type from query()
            raise LLMClientError(
                f"An unexpected error occurred during LLM query for {self.__class__.__name__}: {e_unexpected}"
            ) from e_unexpected

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
        if (message is not None) and (message != self._system_message):
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

    def get_model_name(self) -> str:
        """Returns the configured model name."""
        return self._model
