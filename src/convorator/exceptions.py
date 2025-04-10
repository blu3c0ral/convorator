# convorator.exceptions.py
"""Custom exception classes for the Convorator package."""

from typing import Any, Dict, Optional

# --- Base Exception ---


class ConvoratorError(Exception):
    """Base class for all custom exceptions in the Convorator package."""

    pass


# --- Orchestration Errors ---


class LLMOrchestrationError(ConvoratorError):
    """Indicates an error during the orchestration logic."""

    pass


class MaxIterationsExceededError(LLMOrchestrationError):
    """Raised when max iterations are reached without a satisfactory result."""

    pass


# --- LLM Interaction Errors ---


class LLMResponseError(ConvoratorError):
    """Indicates an error related to the LLM's response (e.g., API error, empty response)."""

    def __init__(
        self, message: str, original_exception: Optional[Exception] = None, *args: Any
    ) -> None:
        super().__init__(message, *args)
        self.original_exception = original_exception


# Added from llm_client.py
class LLMClientError(ConvoratorError):
    """Base class for errors specific to the LLM client operations."""

    pass


# Renamed from LLMError in llm_client.py and made more specific
class LLMConfigurationError(LLMClientError):
    """Indicates an error in the LLM client configuration (e.g., missing API key)."""

    pass


# --- Validation Errors ---


class SchemaValidationError(ValueError, ConvoratorError):
    """
    Raised when JSON schema validation fails, inheriting from ValueError
    for semantic compatibility but also from ConvoratorError for package hierarchy.
    """

    def __init__(
        self,
        message: str,
        schema: Optional[Dict[str, Any]] = None,
        instance: Optional[Any] = None,
        *args: Any,
    ) -> None:
        super().__init__(message, *args)
        self.schema = schema
        self.instance = instance  # The instance that failed validation


# --- Prompt/Template Errors ---


# Added from conversations/utils.py
class MissingVariableError(ConvoratorError, KeyError):
    """Raised when a required variable is missing for prompt formatting.

    Inherits from KeyError for potential compatibility if caught broadly,
    but primarily from ConvoratorError for package hierarchy.
    """

    pass
