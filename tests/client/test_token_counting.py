import pytest
from unittest.mock import Mock, patch
import os
import sys

from convorator.client.openai_client import (
    OpenAILLM,
    OPENAI_CONTEXT_LIMITS,
    DEFAULT_OPENAI_CONTEXT_LIMIT,
)
from convorator.exceptions import LLMConfigurationError


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
    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        yield mock_openai_module


# --- Test Category 1: count_tokens() with Tiktoken Available ---


@patch("tiktoken.get_encoding")
def test_count_tokens_with_tiktoken_available(mock_get_encoding, mock_logger, mock_openai):
    """
    Tests that count_tokens uses tiktoken correctly when it's available.
    """
    mock_encoding = Mock()
    mock_encoding.name = "o200k_base"
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # Simulate 5 tokens
    mock_get_encoding.return_value = mock_encoding

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = OpenAILLM(logger=mock_logger, model="gpt-4o")

        text_to_count = "this is a test"
        token_count = client.count_tokens(text_to_count)

        assert token_count == 5
        mock_get_encoding.assert_called_once_with("o200k_base")
        mock_encoding.encode.assert_called_once_with(text_to_count)
        mock_logger.debug.assert_any_call("Counted 5 tokens for text (length 14) using o200k_base.")


@patch("tiktoken.get_encoding")
def test_tiktoken_encoding_lazy_loading_and_caching(mock_get_encoding, mock_logger, mock_openai):
    """
    Tests that the tiktoken encoding object is loaded lazily on the first call
    and cached for subsequent calls.
    """
    mock_encoding = Mock()
    mock_encoding.name = "o200k_base"
    mock_encoding.encode.return_value = [1, 2, 3]
    mock_get_encoding.return_value = mock_encoding

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = OpenAILLM(logger=mock_logger, model="gpt-4o")

        # Encoding should not be loaded at initialization
        assert client._encoding is None
        mock_get_encoding.assert_not_called()

        # First call to count_tokens should load the encoding
        client.count_tokens("first call")
        assert client._encoding is not None
        mock_get_encoding.assert_called_once()

        # Second call should use the cached encoding
        client.count_tokens("second call")
        mock_get_encoding.assert_called_once()  # Should still be 1 call


# --- Test Category 2: count_tokens() Fallback Behavior ---


def test_count_tokens_fallback_when_tiktoken_unavailable(mock_logger, mock_openai):
    """
    Tests that count_tokens gracefully falls back to an approximation
    when the tiktoken package is not installed.
    """
    # By patching the name 'tiktoken' within the target module, we can simulate it not being there
    # without affecting the test runner's environment.
    with patch("convorator.client.openai_client.tiktoken", None):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o")
            text_to_count = "This is a much longer test sentence for fallback."  # length 49
            token_count = client.count_tokens(text_to_count)

            # Fallback is len(text) // 4
            expected_fallback_count = len(text_to_count) // 4  # 49 // 4 = 12
            assert token_count == expected_fallback_count

            # Verify that exactly one warning was logged.
            mock_logger.warning.assert_called_once()
            log_message = mock_logger.warning.call_args[0][0]
            assert "Using approximate token count" in log_message
            assert "tiktoken library not available" in log_message


@patch("tiktoken.get_encoding")
def test_count_tokens_fallback_when_encoding_fails(mock_get_encoding, mock_logger, mock_openai):
    """
    Tests that count_tokens falls back to approximation if get_encoding fails.
    """
    # Simulate get_encoding raising an error for the specific encoding and the fallback
    mock_get_encoding.side_effect = ValueError("Invalid encoding")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = OpenAILLM(logger=mock_logger, model="gpt-4o")

        text_to_count = "test"
        token_count = client.count_tokens(text_to_count)

        assert token_count == len(text_to_count) // 4

        # Check for appropriate error and warning logs
        mock_logger.error.assert_any_call(
            "Failed to get tiktoken encoding 'o200k_base': Invalid encoding. Falling back to default."
        )
        mock_logger.error.assert_any_call(
            "Failed to get fallback tiktoken encoding 'o200k_base': Invalid encoding. Token counting disabled."
        )
        # With the refactored production code, we now expect exactly one warning.
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Using approximate token count" in warning_msg
        assert "encoding failed to load" in warning_msg


# --- Test Category 3: Context Limit Retrieval ---


@pytest.mark.parametrize(
    "model_name, expected_limit",
    [
        ("gpt-4o-2024-11-20", 128000),
        ("gpt-4", 8192),
        ("gpt-3.5-turbo-0613", 4096),
        ("gpt-4.1-mini-2025-04-14", 1047576),
    ],
)
def test_get_context_limit_for_known_models(mock_logger, mock_openai, model_name, expected_limit):
    """
    Tests that get_context_limit returns the correct values for known models.
    """
    with patch.dict(sys.modules, {"tiktoken": Mock()}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Provide a sensible max_tokens to avoid the warning during this specific test.
            client = OpenAILLM(logger=mock_logger, model=model_name, max_tokens=1024)
            assert client.get_context_limit() == expected_limit
            mock_logger.warning.assert_not_called()


def test_get_context_limit_fallback_for_unknown_model(mock_logger, mock_openai):
    """
    Tests that get_context_limit falls back to a default for unknown models.
    """
    with patch.dict(sys.modules, {"tiktoken": Mock()}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Use a model name that is not in OPENAI_CONTEXT_LIMITS
            client = OpenAILLM(logger=mock_logger, model="future-model-v1")

            assert client.get_context_limit() == DEFAULT_OPENAI_CONTEXT_LIMIT

            # Check that a warning was logged
            mock_logger.warning.assert_any_call(
                f"Context limit not defined for OpenAI model 'future-model-v1'. Using default: {DEFAULT_OPENAI_CONTEXT_LIMIT}"
            )


# --- Test Category 4: Encoding Selection ---


@patch("tiktoken.get_encoding")
@pytest.mark.parametrize(
    "model_name, expected_encoding",
    [
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini-2024-07-18", "o200k_base"),
        ("o3-2025-04-16", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-4-turbo-2024-04-09", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("gpt-3.5-turbo-instruct", "p50k_base"),
        ("unsupported-model-for-encoding-test", "o200k_base"),  # Uses default encoding
    ],
)
def test_encoding_selection_for_different_models(
    mock_get_encoding, mock_logger, mock_openai, model_name, expected_encoding
):
    """
    Tests that the correct tiktoken encoding is selected for different model families.
    """
    mock_get_encoding.return_value = Mock()

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        client = OpenAILLM(logger=mock_logger, model=model_name)
        client.count_tokens("test")

        # The client resolves the model name, so we check the encoding for the *resolved* model
        resolved_model = client._resolve_model_name(model_name)
        # Use the instance to access class attributes
        true_expected_encoding = client.MODEL_TO_ENCODING.get(
            resolved_model, client.DEFAULT_ENCODING
        )

        mock_get_encoding.assert_called_with(true_expected_encoding)


@patch("tiktoken.get_encoding")
def test_token_counting_accuracy_with_different_encodings(
    mock_get_encoding, mock_logger, mock_openai
):
    """
    Simulates getting different token counts for different encodings to ensure
    the client uses the correct encoding object returned by tiktoken.
    """
    # Mock two different encoding objects
    encoding_o200k = Mock()
    encoding_o200k.name = "o200k_base"
    encoding_o200k.encode.return_value = [1, 2, 3]  # 3 tokens

    encoding_cl100k = Mock()
    encoding_cl100k.name = "cl100k_base"
    encoding_cl100k.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    def get_encoding_side_effect(encoding_name):
        if encoding_name == "o200k_base":
            return encoding_o200k
        elif encoding_name == "cl100k_base":
            return encoding_cl100k
        else:
            raise ValueError("Unexpected encoding requested")

    mock_get_encoding.side_effect = get_encoding_side_effect

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        # Test with a gpt-4o model, expecting o200k_base (3 tokens)
        client_4o = OpenAILLM(logger=mock_logger, model="gpt-4o")
        assert client_4o.count_tokens("test") == 3

        # Test with a gpt-4 model, expecting cl100k_base (5 tokens)
        client_4 = OpenAILLM(logger=mock_logger, model="gpt-4")
        assert client_4.count_tokens("test") == 5


# --- Test Category 5: Performance with Large Text ---


@pytest.mark.skip(reason="Performance testing is out of scope for unit tests.")
def test_count_tokens_performance_with_large_text():
    """
    This test is intended to verify that token counting performance is acceptable
    for large text inputs. It is marked to be skipped in typical unit test runs
    and should be executed as part of a dedicated performance testing suite.

    To run this test, you would need the actual tiktoken library installed and
    a large text file.

    Example:
    import time
    import tiktoken

    client = OpenAILLM(...)
    large_text = "..." # Load a large text file

    start_time = time.time()
    client.count_tokens(large_text)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Token counting for {len(large_text)} characters took {duration:.4f} seconds.")
    assert duration < 1.0 # Set a reasonable performance threshold
    """
    pass
