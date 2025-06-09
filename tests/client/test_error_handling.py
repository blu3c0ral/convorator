import pytest
import os
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from convorator.client.openai_client import OpenAILLM
from convorator.exceptions import (
    LLMClientError,
    LLMConfigurationError,
    LLMResponseError,
)


# --- Fixtures ---


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_openai():
    """Mock the OpenAI module to prevent actual API calls during initialization."""
    mock_openai_module = Mock()
    mock_openai_module.OpenAI.return_value = Mock()

    # Create mock exception classes that behave like the real ones
    mock_openai_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mock_openai_module.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_openai_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
    mock_openai_module.PermissionDeniedError = type("PermissionDeniedError", (Exception,), {})
    mock_openai_module.NotFoundError = type("NotFoundError", (Exception,), {})
    mock_openai_module.BadRequestError = type("BadRequestError", (Exception,), {})

    # APIStatusError needs special handling as it has status_code and response attributes
    class MockAPIStatusError(Exception):
        def __init__(self, message, status_code=500, response=None):
            super().__init__(message)
            self.status_code = status_code
            self.response = response if response else Mock()
            self.response.json.return_value = {"error": {"message": message}}

    mock_openai_module.APIStatusError = MockAPIStatusError
    mock_openai_module.OpenAIError = type("OpenAIError", (Exception,), {})

    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        yield mock_openai_module


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken to avoid encoding issues in error tests."""
    return Mock()


@pytest.fixture
def openai_client(mock_logger, mock_openai, mock_tiktoken):
    """Create a test OpenAI client with mocked dependencies."""
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)
            return client


# --- Test Category 1: Custom Exception Mapping ---


def test_api_connection_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI APIConnectionError is properly mapped to LLMClientError.
    """
    # Mock the API call to raise APIConnectionError
    mock_response = Mock()
    openai_client.client.chat.completions.create.side_effect = mock_openai.APIConnectionError(
        "Network timeout"
    )

    with pytest.raises(
        LLMClientError, match="Network error connecting to OpenAI API: Network timeout"
    ):
        openai_client.query("Test prompt", use_conversation=False)


def test_rate_limit_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI RateLimitError is properly mapped to LLMResponseError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.RateLimitError(
        "Rate limit exceeded"
    )

    with pytest.raises(LLMResponseError, match="OpenAI API rate limit exceeded"):
        openai_client.query("Test prompt", use_conversation=False)


def test_authentication_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI AuthenticationError is properly mapped to LLMConfigurationError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.AuthenticationError(
        "Invalid API key"
    )

    with pytest.raises(LLMConfigurationError, match="OpenAI API authentication failed"):
        openai_client.query("Test prompt", use_conversation=False)


def test_permission_denied_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI PermissionDeniedError is properly mapped to LLMConfigurationError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.PermissionDeniedError(
        "Access denied"
    )

    with pytest.raises(LLMConfigurationError, match="OpenAI API permission denied"):
        openai_client.query("Test prompt", use_conversation=False)


def test_not_found_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI NotFoundError is properly mapped to LLMConfigurationError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.NotFoundError(
        "Model not found"
    )

    with pytest.raises(LLMConfigurationError, match="OpenAI API resource not found"):
        openai_client.query("Test prompt", use_conversation=False)


def test_bad_request_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI BadRequestError is properly mapped to LLMResponseError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.BadRequestError(
        "Invalid request format"
    )

    with pytest.raises(LLMResponseError, match="OpenAI API reported a bad request"):
        openai_client.query("Test prompt", use_conversation=False)


def test_api_status_error_mapping(openai_client, mock_openai):
    """
    Tests that OpenAI APIStatusError is properly mapped to LLMResponseError.
    """
    status_error = mock_openai.APIStatusError("Server error", status_code=500)
    openai_client.client.chat.completions.create.side_effect = status_error

    with pytest.raises(LLMResponseError, match="OpenAI API returned an error \\(Status 500\\)"):
        openai_client.query("Test prompt", use_conversation=False)


def test_generic_openai_error_mapping(openai_client, mock_openai):
    """
    Tests that generic OpenAI errors are mapped to LLMClientError.
    """
    openai_client.client.chat.completions.create.side_effect = mock_openai.OpenAIError(
        "Unknown OpenAI error"
    )

    with pytest.raises(LLMClientError, match="An unexpected OpenAI client error occurred"):
        openai_client.query("Test prompt", use_conversation=False)


def test_unexpected_error_mapping(openai_client):
    """
    Tests that unexpected errors are wrapped in LLMClientError.
    """
    openai_client.client.chat.completions.create.side_effect = ValueError("Unexpected Python error")

    with pytest.raises(LLMClientError, match="Unexpected error during OpenAI API call"):
        openai_client.query("Test prompt", use_conversation=False)


# --- Test Category 2: Conversation State Consistency During Errors ---


def test_conversation_rollback_on_api_error(openai_client, mock_openai):
    """
    Tests that conversation state is properly rolled back when API calls fail.
    """
    # Set up initial conversation state
    openai_client.set_system_message("You are a helpful assistant")

    # Mock successful response for first query
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "First response"
    mock_response.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.return_value = mock_response
    openai_client.query("First successful message", use_conversation=True)

    # Now set up the error for the second call
    openai_client.client.chat.completions.create.side_effect = mock_openai.RateLimitError(
        "Rate limit exceeded"
    )

    initial_history = openai_client.get_conversation_history()
    initial_length = len(initial_history)

    # Attempt a query that will fail
    with pytest.raises(LLMResponseError):
        openai_client.query("This will fail", use_conversation=True)

    # Conversation should be rolled back to initial state
    final_history = openai_client.get_conversation_history()
    assert len(final_history) == initial_length
    assert final_history == initial_history


def test_conversation_rollback_preserves_system_message(openai_client, mock_openai):
    """
    Tests that system messages are preserved during rollback.
    """
    openai_client.set_system_message("System prompt")
    openai_client.client.chat.completions.create.side_effect = mock_openai.APIConnectionError(
        "Network error"
    )

    with pytest.raises(LLMClientError):
        openai_client.query("Failed message", use_conversation=True)

    # System message should be preserved
    history = openai_client.get_conversation_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "System prompt"


def test_stateless_query_no_rollback_needed(openai_client, mock_openai):
    """
    Tests that stateless queries don't affect conversation state even on errors.
    """
    # Set up initial conversation
    openai_client.conversation.add_user_message("Existing message")
    initial_history = openai_client.get_conversation_history()

    # Mock API error
    openai_client.client.chat.completions.create.side_effect = mock_openai.RateLimitError(
        "Rate limit"
    )

    with pytest.raises(LLMResponseError):
        openai_client.query("Stateless failed query", use_conversation=False)

    # Conversation should remain unchanged
    final_history = openai_client.get_conversation_history()
    assert final_history == initial_history


def test_multiple_error_recovery(openai_client, mock_openai):
    """
    Tests conversation consistency through multiple error scenarios.
    """
    openai_client.set_system_message("System")

    # Successful query
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Success 1"
    mock_response.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.return_value = mock_response
    openai_client.query("Success 1", use_conversation=True)

    # First error
    openai_client.client.chat.completions.create.side_effect = mock_openai.RateLimitError("Error 1")
    with pytest.raises(LLMResponseError):
        openai_client.query("Fail 1", use_conversation=True)

    # Second error
    openai_client.client.chat.completions.create.side_effect = mock_openai.APIConnectionError(
        "Error 2"
    )
    with pytest.raises(LLMClientError):
        openai_client.query("Fail 2", use_conversation=True)

    # Another successful query
    mock_response2 = Mock()
    mock_response2.choices = [Mock()]
    mock_response2.choices[0].message = Mock()
    mock_response2.choices[0].message.content = "Success 2"
    mock_response2.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.side_effect = None
    openai_client.client.chat.completions.create.return_value = mock_response2
    openai_client.query("Success 2", use_conversation=True)

    # Final state should only have successful messages
    history = openai_client.get_conversation_history()
    user_messages = [msg for msg in history if msg["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages[0]["content"] == "Success 1"
    assert user_messages[1]["content"] == "Success 2"


# --- Test Category 3: Response Validation and Edge Cases ---


def test_empty_response_choices(openai_client):
    """
    Tests handling of API responses with empty choices array.
    """
    mock_response = Mock()
    mock_response.choices = []

    openai_client.client.chat.completions.create.return_value = mock_response

    with pytest.raises(
        LLMResponseError, match="Invalid response structure from OpenAI: No 'choices'"
    ):
        openai_client.query("Test prompt", use_conversation=False)


def test_missing_message_in_choice(openai_client):
    """
    Tests handling of API responses where choice is missing message.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = None

    openai_client.client.chat.completions.create.return_value = mock_response

    with pytest.raises(
        LLMResponseError, match="Invalid response structure from OpenAI: Choice missing 'message'"
    ):
        openai_client.query("Test prompt", use_conversation=False)


def test_null_content_with_content_filter(openai_client):
    """
    Tests handling of null content due to content filtering.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].finish_reason = "content_filter"

    openai_client.client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMResponseError, match="OpenAI response blocked due to content filter"):
        openai_client.query("Filtered content", use_conversation=False)


def test_null_content_with_length_limit(openai_client):
    """
    Tests handling of null content due to length limits.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].finish_reason = "length"

    openai_client.client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMResponseError, match="OpenAI response truncated due to max_tokens"):
        openai_client.query("Long content", use_conversation=False)


def test_null_content_unknown_reason(openai_client):
    """
    Tests handling of null content with unknown finish reason.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].finish_reason = "unknown_reason"

    openai_client.client.chat.completions.create.return_value = mock_response

    with pytest.raises(
        LLMResponseError, match="OpenAI returned null content. Finish Reason: unknown_reason"
    ):
        openai_client.query("Test prompt", use_conversation=False)


def test_empty_content_with_stop_reason(openai_client, mock_logger):
    """
    Tests handling of empty content with 'stop' finish reason (edge case).
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = ""  # Empty string, not None
    mock_response.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.return_value = mock_response

    # Should succeed but log a warning
    result = openai_client.query("Test prompt", use_conversation=False)
    assert result == ""

    # Should log a warning about empty content
    mock_logger.warning.assert_any_call(
        "Received empty content string from OpenAI API, but finish reason was 'stop'. Response: "
        + str(mock_response)
    )


# --- Test Category 4: Input Validation Edge Cases ---


def test_empty_message_list_error(openai_client):
    """
    Tests that empty message lists are rejected.
    """
    # This should not happen in normal operation, but we test the guard
    with pytest.raises(LLMClientError, match="Cannot call OpenAI API with an empty message list"):
        openai_client._call_api([])


def test_query_with_none_prompt(openai_client):
    """
    Tests behavior with None as prompt (should be handled gracefully).
    """
    # The query method should handle this, likely by converting to string
    with pytest.raises((TypeError, AttributeError)):
        # This will fail at the conversation level when trying to add None as content
        openai_client.query(None, use_conversation=True)


def test_query_with_very_long_prompt(openai_client):
    """
    Tests behavior with extremely long prompts.
    """
    # Create a very long prompt (100KB)
    long_prompt = "x" * 100000

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Response to long prompt"
    mock_response.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.return_value = mock_response

    # Should succeed (actual token limits would be enforced by OpenAI API)
    result = openai_client.query(long_prompt, use_conversation=False)
    assert result == "Response to long prompt"


# --- Test Category 5: Responses API Error Handling ---


def test_responses_api_connection_error_handling(openai_client, mock_openai):
    """
    Tests error handling in Responses API mode.
    """
    openai_client.use_responses_api = True
    openai_client.client.responses.create.side_effect = mock_openai.APIConnectionError(
        "Network error"
    )

    with pytest.raises(LLMClientError, match="Network error connecting to OpenAI Responses API"):
        openai_client.query("Test prompt", use_conversation=False)

    # Should clear response ID on connection issues
    assert openai_client._last_response_id is None


def test_responses_api_invalid_previous_response_id(openai_client, mock_openai):
    """
    Tests handling of invalid previous_response_id in continued conversations.
    """
    openai_client.use_responses_api = True
    openai_client._last_response_id = "invalid-id"

    # Mock NotFoundError that mentions previous_response_id
    error = mock_openai.NotFoundError("Invalid previous_response_id: invalid-id")
    openai_client.client.responses.create.side_effect = error

    with pytest.raises(LLMConfigurationError):
        openai_client.query("Test prompt", use_conversation=True)

    # Should clear the invalid response ID
    assert openai_client._last_response_id is None


def test_responses_api_bad_request_clears_state(openai_client, mock_openai):
    """
    Tests that bad requests in Responses API clear potentially problematic state.
    """
    openai_client.use_responses_api = True
    openai_client._last_response_id = "some-id"

    # Mock BadRequestError that mentions input issues
    error = mock_openai.BadRequestError("Invalid input format")
    openai_client.client.responses.create.side_effect = error

    with pytest.raises(LLMResponseError):
        openai_client.query("Test prompt", use_conversation=True)

    # Should clear response ID on input-related bad requests
    assert openai_client._last_response_id is None


def test_responses_api_server_error_clears_state(openai_client, mock_openai):
    """
    Tests that server errors clear conversation state for safety.
    """
    openai_client.use_responses_api = True
    openai_client._last_response_id = "some-id"

    # Mock 500 server error
    error = mock_openai.APIStatusError("Internal server error", status_code=500)
    openai_client.client.responses.create.side_effect = error

    with pytest.raises(LLMResponseError):
        openai_client.query("Test prompt", use_conversation=True)

    # Should clear response ID on server errors
    assert openai_client._last_response_id is None


# --- Test Category 6: Initialization and Configuration Errors ---


def test_missing_api_key_error():
    """
    Tests proper error when API key is missing.
    """
    with patch.dict("sys.modules", {"openai": Mock(), "tiktoken": Mock()}):
        with patch.dict(os.environ, {}, clear=True):  # Clear environment
            with pytest.raises(LLMConfigurationError, match="OpenAI API key not provided"):
                OpenAILLM(logger=Mock(), model="gpt-4o")


def test_openai_import_error():
    """
    Tests proper error when OpenAI package is not installed.
    """
    with patch.dict("sys.modules", {"tiktoken": Mock()}):
        # Remove openai from sys.modules to simulate ImportError
        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'openai'")):
                with pytest.raises(LLMConfigurationError, match="OpenAI Python package not found"):
                    OpenAILLM(logger=Mock(), model="gpt-4o", api_key="test-key")


def test_client_initialization_failure(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests proper error when OpenAI client initialization fails.
    """
    # Mock the OpenAI constructor to raise an exception
    mock_openai.OpenAI.side_effect = Exception("Client init failed")

    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(LLMConfigurationError, match="Failed to initialize OpenAI client"):
                OpenAILLM(logger=mock_logger, model="gpt-4o")


# --- Test Category 7: Complex Error Scenarios ---


def test_error_during_assistant_response_addition(openai_client, mock_openai, mock_logger, caplog):
    """
    Tests error handling when adding assistant response to conversation fails.
    """
    # Mock successful API call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "API Success"
    mock_response.choices[0].finish_reason = "stop"

    openai_client.client.chat.completions.create.return_value = mock_response

    # Mock conversation.add_assistant_message to fail
    original_add_assistant = openai_client.conversation.add_assistant_message
    openai_client.conversation.add_assistant_message = Mock(
        side_effect=Exception("Conversation error")
    )

    # Capture logs at ERROR level for the llm_client logger
    with caplog.at_level(logging.ERROR, logger="llm_client"):
        # API call succeeds, but conversation update fails
        result = openai_client.query("Test prompt", use_conversation=True)

    # Should still return the API response
    assert result == "API Success"

    # Should log the error about failing to add assistant's response
    error_messages = [record.message for record in caplog.records if record.levelname == "ERROR"]
    assert any(
        "failed to add assistant's response to internal conversation history" in msg
        for msg in error_messages
    )


def test_conversation_state_corruption_recovery(openai_client):
    """
    Tests recovery from conversation state corruption.
    """
    # Corrupt the conversation object
    openai_client.conversation = None

    # Should handle gracefully or raise appropriate error
    with pytest.raises((AttributeError, LLMClientError)):
        openai_client.query("Test prompt", use_conversation=True)


def test_api_status_error_with_malformed_response(openai_client, mock_openai):
    """
    Tests handling of APIStatusError with malformed error response.
    """
    # Create APIStatusError with response that fails JSON parsing
    error_response = Mock()
    error_response.json.side_effect = Exception("JSON parse error")

    error = mock_openai.APIStatusError("Server error", status_code=503, response=error_response)
    openai_client.client.chat.completions.create.side_effect = error

    # Should handle the JSON parsing failure gracefully
    with pytest.raises(LLMResponseError, match="OpenAI API returned an error \\(Status 503\\)"):
        openai_client.query("Test prompt", use_conversation=False)


# --- Test Category 8: Responses API Edge Cases ---


def test_responses_api_malformed_output(openai_client):
    """
    Tests handling of malformed Responses API output.
    """
    openai_client.use_responses_api = True

    # Mock response with missing expected structure
    mock_response = Mock()
    # Make sure output_text exists but has issues in the extraction method
    del mock_response.output_text  # Remove the attribute entirely
    mock_response.output = []  # Empty output array

    openai_client.client.responses.create.return_value = mock_response

    with pytest.raises(
        LLMResponseError, match="No text content found in OpenAI Responses API response"
    ):
        openai_client.query("Test prompt", use_conversation=False)


def test_responses_api_empty_message_list_error(openai_client):
    """
    Tests Responses API with empty message list.
    """
    openai_client.use_responses_api = True

    with pytest.raises(
        LLMClientError, match="Cannot call OpenAI Responses API with an empty message list"
    ):
        openai_client._call_responses_api([])


def test_responses_api_conversation_continuation_without_user_message(openai_client):
    """
    Tests error when trying to continue Responses API conversation without user message.
    """
    openai_client.use_responses_api = True
    openai_client._last_response_id = "existing-id"

    # Try to continue conversation but last message is not from user
    assistant_only_messages = [{"role": "assistant", "content": "I am assistant"}]

    with pytest.raises(LLMClientError, match="messages should end with the new user prompt"):
        openai_client._call_responses_api(assistant_only_messages)


# --- Test Category 9: Feature-Specific Error Testing ---


def test_web_search_without_responses_api_error(openai_client):
    """
    Tests that web search features require Responses API.
    """
    openai_client.use_responses_api = False

    with pytest.raises(LLMConfigurationError, match="Web search tool requires Responses API"):
        openai_client.query_with_web_search("Search query")


def test_file_search_without_responses_api_error(openai_client):
    """
    Tests that file search features require Responses API.
    """
    openai_client.use_responses_api = False

    with pytest.raises(LLMConfigurationError, match="File search tool requires Responses API"):
        openai_client.query_with_file_search("Search query", ["vector-store-1"])


def test_computer_use_without_responses_api_error(openai_client):
    """
    Tests that computer use features require Responses API.
    """
    openai_client.use_responses_api = False

    with pytest.raises(LLMConfigurationError, match="Computer use tool requires Responses API"):
        openai_client.query_with_computer_use("Use computer")
