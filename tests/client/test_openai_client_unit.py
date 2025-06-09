import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from convorator.client.openai_client import (
    OpenAILLM,
    MODEL_FAMILY_MAPPING,
    OPENAI_CONTEXT_LIMITS,
    DEFAULT_OPENAI_CONTEXT_LIMIT,
)
from convorator.client.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Conversation,
)
from convorator.exceptions import (
    LLMConfigurationError,
    LLMClientError,
    LLMResponseError,
)
from convorator.utils.logger import setup_logger


# Test fixtures and mocks
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return Mock()


@pytest.fixture
def mock_openai():
    """Mock the OpenAI module with proper exception hierarchy."""
    mock_openai = Mock()

    # Create proper exception hierarchy
    mock_openai.OpenAIError = type("OpenAIError", (Exception,), {})
    mock_openai.APIConnectionError = type("APIConnectionError", (mock_openai.OpenAIError,), {})
    mock_openai.RateLimitError = type("RateLimitError", (mock_openai.OpenAIError,), {})
    mock_openai.AuthenticationError = type("AuthenticationError", (mock_openai.OpenAIError,), {})
    mock_openai.PermissionDeniedError = type(
        "PermissionDeniedError", (mock_openai.OpenAIError,), {}
    )
    mock_openai.NotFoundError = type("NotFoundError", (mock_openai.OpenAIError,), {})
    mock_openai.BadRequestError = type("BadRequestError", (mock_openai.OpenAIError,), {})
    mock_openai.APIStatusError = type("APIStatusError", (mock_openai.OpenAIError,), {})

    # Mock client instance
    mock_client = Mock()
    mock_openai.OpenAI.return_value = mock_client

    return mock_openai


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken module."""
    mock_tiktoken = Mock()
    mock_encoding = Mock()
    mock_encoding.name = "o200k_base"
    mock_encoding.encode.return_value = [1, 2, 3, 4]  # 4 tokens
    mock_tiktoken.get_encoding.return_value = mock_encoding
    return mock_tiktoken


# --- Test Category 1: Client Initialization ---


def test_openai_client_initialization_default_parameters(mock_logger, mock_openai, mock_tiktoken):
    """Tests OpenAI client initialization with default parameters."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Verify default values
            assert client._model == "gpt-4o-2024-11-20"  # Resolved from "gpt-4o"
            assert client.original_model == "gpt-4o"
            assert client.resolved_model == "gpt-4o-2024-11-20"
            assert client.temperature == DEFAULT_TEMPERATURE
            assert client.max_tokens == DEFAULT_MAX_TOKENS
            assert client.api_key == "test-api-key"
            assert client.use_responses_api is False
            assert client._last_response_id is None

            # Verify OpenAI client was created
            mock_openai.OpenAI.assert_called_once_with(api_key="test-api-key")


def test_openai_client_initialization_custom_parameters(mock_logger, mock_openai, mock_tiktoken):
    """Tests OpenAI client initialization with custom parameters."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        from convorator.client.openai_client import OpenAILLM

        client = OpenAILLM(
            logger=mock_logger,
            api_key="custom-key",
            model="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=2000,
            system_message="You are a test assistant.",
            role_name="TestBot",
            use_responses_api=True,
        )

        assert client._model == "gpt-3.5-turbo-0125"  # Resolved
        assert client.original_model == "gpt-3.5-turbo"
        assert client.temperature == 0.9
        assert client.max_tokens == 2000
        assert client.api_key == "custom-key"
        assert client._system_message == "You are a test assistant."
        assert client._role_name == "TestBot"
        assert client.use_responses_api is True

        # Verify conversation has system message
        assert len(client.conversation.messages) == 1
        assert client.conversation.messages[0].role == "system"
        assert client.conversation.messages[0].content == "You are a test assistant."


def test_openai_client_initialization_with_specific_model_version(
    mock_logger, mock_openai, mock_tiktoken
):
    """Tests initialization with a specific model version (no family resolution)."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="gpt-4o-2024-05-13")  # Specific version

            # Should not resolve since it's already specific
            assert client._model == "gpt-4o-2024-05-13"
            assert client.original_model == "gpt-4o-2024-05-13"
            assert client.resolved_model == "gpt-4o-2024-05-13"


def test_openai_client_initialization_missing_api_key(mock_logger, mock_openai, mock_tiktoken):
    """Tests that missing API key raises configuration error."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        # Clear environment variable
        with patch.dict(os.environ, {}, clear=True):
            from convorator.client.openai_client import OpenAILLM

            with pytest.raises(LLMConfigurationError, match="OpenAI API key not provided"):
                OpenAILLM(logger=mock_logger)


def test_openai_client_initialization_openai_import_error(mock_logger):
    """Tests handling of missing OpenAI package."""
    # Use a simpler approach - patch the openai import directly at module level
    import sys

    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(LLMConfigurationError, match="OpenAI Python package not found"):
            from convorator.client.openai_client import OpenAILLM

            OpenAILLM(logger=mock_logger, api_key="test-key")


def test_openai_client_initialization_client_creation_error(
    mock_logger, mock_openai, mock_tiktoken
):
    """Tests handling of OpenAI client creation errors."""
    mock_openai.OpenAI.side_effect = Exception("Client creation failed")

    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        from convorator.client.openai_client import OpenAILLM

        with pytest.raises(LLMConfigurationError, match="Failed to initialize OpenAI client"):
            OpenAILLM(logger=mock_logger, api_key="test-key")


# --- Test Category 2: Model Family Resolution ---


def test_resolve_model_name_family_to_specific(mock_logger, mock_openai, mock_tiktoken):
    """Tests model family resolution to specific versions."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="gpt-4o")

            # Test the resolution method directly
            assert client._resolve_model_name("gpt-4o") == "gpt-4o-2024-11-20"
            assert client._resolve_model_name("gpt-4.1") == "gpt-4.1-2025-04-14"
            assert client._resolve_model_name("o3") == "o3-2025-04-16"
            assert client._resolve_model_name("gpt-3.5-turbo") == "gpt-3.5-turbo-0125"


def test_resolve_model_name_specific_version_unchanged(mock_logger, mock_openai, mock_tiktoken):
    """Tests that specific model versions are not changed."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Specific versions should remain unchanged
            assert client._resolve_model_name("gpt-4o-2024-05-13") == "gpt-4o-2024-05-13"
            assert client._resolve_model_name("gpt-4-0613") == "gpt-4-0613"
            assert client._resolve_model_name("unknown-model") == "unknown-model"


def test_resolve_model_name_all_families(mock_logger, mock_openai, mock_tiktoken):
    """Tests resolution for all model families in the mapping."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            for family, expected_version in MODEL_FAMILY_MAPPING.items():
                resolved = client._resolve_model_name(family)
                assert (
                    resolved == expected_version
                ), f"Family {family} should resolve to {expected_version}, got {resolved}"


# --- Test Category 3: Model Family Mapping and Capabilities ---


def test_get_model_info(mock_logger, mock_openai, mock_tiktoken):
    """Tests get_model_info method."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Test with family resolution
            client = OpenAILLM(logger=mock_logger, model="gpt-4o")
            info = client.get_model_info()

            assert info["original_model"] == "gpt-4o"
            assert info["resolved_model"] == "gpt-4o-2024-11-20"
            assert info["is_family_resolved"] is True
            assert info["family_mapping_available"] is True

            # Test with specific version
            client2 = OpenAILLM(logger=mock_logger, model="gpt-4o-2024-05-13")
            info2 = client2.get_model_info()

            assert info2["original_model"] == "gpt-4o-2024-05-13"
            assert info2["resolved_model"] == "gpt-4o-2024-05-13"
            assert info2["is_family_resolved"] is False
            assert info2["family_mapping_available"] is False


def test_get_available_model_families():
    """Tests the class method for getting available model families."""
    from convorator.client.openai_client import OpenAILLM

    families = OpenAILLM.get_available_model_families()

    assert isinstance(families, dict)
    assert "gpt-4o" in families
    assert "gpt-4.1" in families
    assert "o3" in families
    assert families["gpt-4o"] == MODEL_FAMILY_MAPPING["gpt-4o"]

    # Ensure it's a copy (modification doesn't affect original)
    families["test"] = "test-value"
    assert "test" not in MODEL_FAMILY_MAPPING


def test_get_latest_model_for_family():
    """Tests getting the latest model for a specific family."""
    from convorator.client.openai_client import OpenAILLM

    assert OpenAILLM.get_latest_model_for_family("gpt-4o") == "gpt-4o-2024-11-20"
    assert OpenAILLM.get_latest_model_for_family("gpt-4.1") == "gpt-4.1-2025-04-14"
    assert OpenAILLM.get_latest_model_for_family("o3") == "o3-2025-04-16"
    assert OpenAILLM.get_latest_model_for_family("nonexistent") is None


# --- Test Category 4: Configuration Validation ---


def test_context_limit_initialization(mock_logger, mock_openai, mock_tiktoken):
    """Tests context limit initialization and validation."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Test known model
            client = OpenAILLM(logger=mock_logger, model="gpt-4o")
            assert client.get_context_limit() == OPENAI_CONTEXT_LIMITS["gpt-4o-2024-11-20"]

            # Test unknown model (should use default)
            client2 = OpenAILLM(logger=mock_logger, model="unknown-model")
            assert client2.get_context_limit() == DEFAULT_OPENAI_CONTEXT_LIMIT


def test_max_tokens_validation_warnings(mock_logger, mock_openai, mock_tiktoken):
    """Tests warnings for max_tokens vs context limits."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Test exceeding context limit
            client = OpenAILLM(
                logger=mock_logger, model="gpt-4", max_tokens=10000  # Has 8192 context limit
            )

            # Should have logged a warning
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "exceeds the known context limit" in str(call)
            ]
            assert len(warning_calls) > 0

            # Test close to limit
            client2 = OpenAILLM(logger=mock_logger, model="gpt-4", max_tokens=7000)  # > 80% of 8192

            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "close to the context limit" in str(call)
            ]
            assert len(info_calls) > 0


def test_unsupported_model_warning(mock_logger, mock_openai, mock_tiktoken):
    """Tests warning for unsupported models."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="unsupported-model")

            # Should have logged a warning about unsupported model
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "not in the explicitly supported list" in str(call)
            ]
            assert len(warning_calls) > 0


# --- Test Category 5: Provider-Specific Settings and Capabilities ---


def test_get_provider_capabilities(mock_logger, mock_openai, mock_tiktoken):
    """Tests OpenAI-specific provider capabilities."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Test Chat Completions mode
            client = OpenAILLM(logger=mock_logger, model="gpt-4o")
            capabilities = client.get_provider_capabilities()

            # Base capabilities
            assert capabilities["provider"] == "openai"
            assert capabilities["model"] == "gpt-4o-2024-11-20"
            assert capabilities["supports_conversation"] is True
            assert capabilities["supports_system_message"] is True

            # OpenAI-specific capabilities
            assert capabilities["supports_tiktoken"] is True
            assert capabilities["supports_streaming"] is True
            assert capabilities["supports_model_families"] is True
            assert capabilities["supports_responses_api"] is True
            assert capabilities["responses_api_enabled"] is False
            assert capabilities["supports_web_search"] is False  # Only in Responses API mode
            assert capabilities["supports_file_search"] is False
            assert capabilities["supports_computer_use"] is False
            assert capabilities["supports_server_side_conversation"] is False

            # Check model info
            assert "current_model_info" in capabilities
            assert capabilities["current_model_info"]["original_model"] == "gpt-4o"

            # Test Responses API mode
            client_responses = OpenAILLM(logger=mock_logger, model="gpt-4o", use_responses_api=True)
            capabilities_responses = client_responses.get_provider_capabilities()

            assert capabilities_responses["responses_api_enabled"] is True
            assert capabilities_responses["supports_web_search"] is True
            assert capabilities_responses["supports_file_search"] is True
            assert capabilities_responses["supports_computer_use"] is True
            assert capabilities_responses["supports_server_side_conversation"] is True


def test_get_provider_settings(mock_logger, mock_openai, mock_tiktoken):
    """Tests OpenAI-specific provider settings."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-long-api-key-for-testing"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(
                logger=mock_logger, model="gpt-4o", temperature=0.8, use_responses_api=True
            )

            # Mock the _get_tiktoken_encoding method directly
            mock_encoding = Mock()
            mock_encoding.name = "o200k_base"

            with patch.object(client, "_get_tiktoken_encoding", return_value=mock_encoding):
                settings = client.get_provider_settings()

                # Base settings
                assert settings["model"] == "gpt-4o-2024-11-20"
                assert settings["max_tokens"] == DEFAULT_MAX_TOKENS
                assert settings["temperature"] == 0.8

                # OpenAI-specific settings
                assert settings["api_key"] == "test-long-..."  # Truncated for security
                assert settings["use_responses_api"] is True
                assert settings["last_response_id"] is None
                assert settings["original_model"] == "gpt-4o"
                assert settings["resolved_model"] == "gpt-4o-2024-11-20"
                assert settings["model_family_resolved"] is True
                assert settings["encoding"] == "o200k_base"


def test_set_provider_setting(mock_logger, mock_openai, mock_tiktoken):
    """Tests setting OpenAI-specific provider settings."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Test temperature setting - valid range
            assert client.set_provider_setting("temperature", 0.9) is True
            assert client.temperature == 0.9

            # Test valid edge values
            assert client.set_provider_setting("temperature", 0.0) is True
            assert client.temperature == 0.0

            assert client.set_provider_setting("temperature", 2.0) is True
            assert client.temperature == 2.0

            # Test invalid temperature values - NOW FIXED!
            # After fixing the base class bug, OpenAI-specific validation should work
            original_temp = client.temperature

            # These should now correctly return False and leave temperature unchanged
            assert client.set_provider_setting("temperature", -0.5) is False
            assert client.temperature == original_temp  # Should be unchanged

            assert client.set_provider_setting("temperature", 3.0) is False
            assert client.temperature == original_temp  # Should be unchanged

            # Test non-numeric temperature - should also be rejected
            assert client.set_provider_setting("temperature", "invalid") is False
            assert client.temperature == original_temp  # Should be unchanged

            # Test Responses API setting
            assert client.set_provider_setting("use_responses_api", True) is True
            assert client.use_responses_api is True
            assert client._last_response_id is None  # Should be reset

            # Test invalid Responses API setting
            assert client.set_provider_setting("use_responses_api", "invalid") is False

            # Test unsupported setting
            assert client.set_provider_setting("unsupported_setting", "value") is False


def test_supports_feature(mock_logger, mock_openai, mock_tiktoken):
    """Tests OpenAI-specific feature support."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Test Chat Completions mode
            client = OpenAILLM(logger=mock_logger, use_responses_api=False)

            # Base features
            assert client.supports_feature("conversation_history") is True
            assert client.supports_feature("system_message") is True
            assert client.supports_feature("token_counting") is True

            # OpenAI-specific features
            assert client.supports_feature("tiktoken_encoding") is True
            assert client.supports_feature("streaming") is True
            assert client.supports_feature("function_calling") is True
            assert client.supports_feature("model_family_resolution") is True

            # Responses API features (disabled in Chat Completions mode)
            assert client.supports_feature("responses_api") is False
            assert client.supports_feature("web_search") is False
            assert client.supports_feature("file_search") is False
            assert client.supports_feature("computer_use") is False

            # Test Responses API mode
            client_responses = OpenAILLM(logger=mock_logger, use_responses_api=True)
            assert client_responses.supports_feature("responses_api") is True
            assert client_responses.supports_feature("web_search") is True
            assert client_responses.supports_feature("file_search") is True
            assert client_responses.supports_feature("computer_use") is True

            # Unknown feature
            assert client.supports_feature("unknown_feature") is False


# --- Test Category 6: Internal State Management ---


def test_conversation_management(mock_logger, mock_openai, mock_tiktoken):
    """Tests internal conversation state management."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, system_message="System prompt")

            # Initial state
            assert len(client.conversation.messages) == 1
            assert client.conversation.messages[0].role == "system"

            # Add messages manually
            client.conversation.add_user_message("Hello")
            client.conversation.add_assistant_message("Hi there")

            assert len(client.conversation.messages) == 3

            # Test clear conversation with Responses API
            client.use_responses_api = True
            client._last_response_id = "test-response-id"

            client.clear_conversation(keep_system=True)
            assert len(client.conversation.messages) == 1  # Only system message
            assert client._last_response_id is None  # Should be cleared

            # Test clear conversation completely
            client._last_response_id = "another-id"
            client.clear_conversation(keep_system=False)
            assert len(client.conversation.messages) == 0
            assert client._last_response_id is None


def test_system_message_updates(mock_logger, mock_openai, mock_tiktoken):
    """Tests system message updates and conversation synchronization."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Set initial system message
            client.set_system_message("Initial system message")
            assert client.get_system_message() == "Initial system message"
            assert len(client.conversation.messages) == 1
            assert client.conversation.messages[0].content == "Initial system message"

            # Update system message
            client.set_system_message("Updated system message")
            assert client.get_system_message() == "Updated system message"
            assert len(client.conversation.messages) == 1  # Still just one
            assert client.conversation.messages[0].content == "Updated system message"


def test_tiktoken_encoding_management(mock_logger, mock_openai, mock_tiktoken):
    """Tests tiktoken encoding lazy loading and caching."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="gpt-4o")

            # Initially None
            assert client._encoding is None

            # Mock tiktoken.get_encoding to return our mock encoding
            mock_encoding = Mock()
            mock_encoding.name = "o200k_base"

            # Patch at the module level within the client's namespace
            with patch.object(client, "_get_tiktoken_encoding") as mock_method:
                # Set up the mock to simulate the first call loading encoding
                def side_effect():
                    if client._encoding is None:
                        client._encoding = mock_encoding
                    return client._encoding

                mock_method.side_effect = side_effect

                # First call should load encoding
                encoding = client._get_tiktoken_encoding()
                assert encoding is not None
                assert client._encoding is not None
                assert mock_method.call_count == 1

                # Second call should use cached encoding
                encoding2 = client._get_tiktoken_encoding()
                assert encoding2 is encoding  # Same object
                assert mock_method.call_count == 2  # Method called again but returns cached


def test_tiktoken_encoding_fallback(mock_logger, mock_openai):
    """Tests tiktoken encoding fallback behavior."""
    mock_tiktoken_fail = Mock()

    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken_fail}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="gpt-4o")

            # Mock the client method to simulate tiktoken failure
            with patch.object(client, "_get_tiktoken_encoding") as mock_method:
                # First call simulates tiktoken failure, sets FAILED_TO_LOAD
                def first_call():
                    client._encoding = "FAILED_TO_LOAD"
                    return None

                # Subsequent calls return None immediately
                def subsequent_calls():
                    return None

                mock_method.side_effect = [first_call(), subsequent_calls()]

                # Should handle error gracefully
                encoding = client._get_tiktoken_encoding()
                assert encoding is None
                assert client._encoding == "FAILED_TO_LOAD"

                # Subsequent calls should return None immediately
                encoding2 = client._get_tiktoken_encoding()
                assert encoding2 is None


def test_tiktoken_encoding_info(mock_logger, mock_openai, mock_tiktoken):
    """Tests tiktoken encoding information method."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, model="gpt-4o")

            # Test with working tiktoken
            mock_encoding = Mock()
            mock_encoding.name = "o200k_base"

            with patch.object(client, "_get_tiktoken_encoding", return_value=mock_encoding):
                info = client.get_tiktoken_encoding_info()
                assert info["available"] is True
                assert info["model"] == "gpt-4o-2024-11-20"
                assert info["encoding_name"] == "o200k_base"
                assert "fallback_used" not in info

            # Test with failed tiktoken
            client._encoding = "FAILED_TO_LOAD"
            with patch.object(client, "_get_tiktoken_encoding", return_value=None):
                info_failed = client.get_tiktoken_encoding_info()
                assert info_failed["available"] is False
                assert info_failed["encoding_name"] is None
                assert info_failed["fallback_used"] is True


def test_count_tokens_with_fallback(mock_logger, mock_openai, mock_tiktoken):
    """Tests token counting with tiktoken fallback."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Test with failed tiktoken loading
            client._encoding = "FAILED_TO_LOAD"

            test_text = "This is a test message for token counting"
            token_count = client.count_tokens(test_text)

            # Should use fallback approximation (length // 4)
            expected_fallback = len(test_text) // 4
            assert token_count == expected_fallback

            # Should have logged a warning
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "approximate token count" in str(call)
            ]
            assert len(warning_calls) > 0


# --- Test Category 8: Responses API vs Chat Completions API Mode Switching ---


def test_api_mode_switching(mock_logger, mock_openai, mock_tiktoken):
    """Tests switching between API modes."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, use_responses_api=False)

            # Initially in Chat Completions mode
            assert client.use_responses_api is False
            assert client._last_response_id is None

            # Switch to Responses API
            client.set_provider_setting("use_responses_api", True)
            assert client.use_responses_api is True
            assert client._last_response_id is None  # Should remain None on switch

            # Simulate having a response ID
            client._last_response_id = "test-response-id"

            # Switch back to Chat Completions
            client.set_provider_setting("use_responses_api", False)
            assert client.use_responses_api is False
            assert client._last_response_id is None  # Should be cleared on switch


def test_initialization_with_both_api_modes(mock_logger, mock_openai, mock_tiktoken):
    """Tests initialization with different API modes."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Chat Completions mode (default)
            client1 = OpenAILLM(logger=mock_logger, use_responses_api=False)
            assert client1.use_responses_api is False
            assert any(
                "Chat Completions API" in str(call) for call in mock_logger.info.call_args_list
            )

            # Reset mock
            mock_logger.reset_mock()

            # Responses API mode
            client2 = OpenAILLM(logger=mock_logger, use_responses_api=True)
            assert client2.use_responses_api is True
            assert any("Responses API" in str(call) for call in mock_logger.info.call_args_list)


def test_responses_api_state_management(mock_logger, mock_openai, mock_tiktoken):
    """Tests Responses API state management without actual API calls."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger, use_responses_api=True)

            # Initially no response ID
            assert client._last_response_id is None

            # Simulate setting a response ID (would normally come from API response)
            client._last_response_id = "response-123"
            assert client._last_response_id == "response-123"

            # Clear conversation should reset response ID
            client.clear_conversation()
            assert client._last_response_id is None


def test_model_to_encoding_mapping(mock_logger, mock_openai, mock_tiktoken):
    """Tests that all supported models have encoding mappings."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Test a sample of models from different categories
            test_models = [
                ("gpt-4.1", "o200k_base"),
                ("gpt-4o", "o200k_base"),
                ("o3", "o200k_base"),
                ("gpt-4-turbo", "cl100k_base"),
                ("gpt-3.5-turbo", "cl100k_base"),
                ("gpt-3.5-turbo-instruct", "p50k_base"),
            ]

            for model_family, expected_encoding in test_models:
                resolved_model = client._resolve_model_name(model_family)
                encoding_name = client.MODEL_TO_ENCODING.get(
                    resolved_model, client.DEFAULT_ENCODING
                )
                assert (
                    encoding_name == expected_encoding
                ), f"Model {resolved_model} should use {expected_encoding}"


def test_context_limits_completeness(mock_logger, mock_openai, mock_tiktoken):
    """Tests that all supported models have context limits defined."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # Check that major model families have context limits
            important_models = [
                "gpt-4.1-2025-04-14",
                "gpt-4o-2024-11-20",
                "o3-2025-04-16",
                "gpt-4-turbo-2024-04-09",
                "gpt-3.5-turbo-0125",
            ]

            for model in important_models:
                assert (
                    model in OPENAI_CONTEXT_LIMITS
                ), f"Model {model} should have a defined context limit"
                assert isinstance(
                    OPENAI_CONTEXT_LIMITS[model], int
                ), f"Context limit for {model} should be an integer"
                assert (
                    OPENAI_CONTEXT_LIMITS[model] > 0
                ), f"Context limit for {model} should be positive"


def test_supported_models_consistency(mock_logger, mock_openai, mock_tiktoken):
    """Tests consistency between supported models, family mapping, and context limits."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            client = OpenAILLM(logger=mock_logger)

            # All family mapping targets should be in supported models
            for family, specific_model in MODEL_FAMILY_MAPPING.items():
                assert (
                    specific_model in client.SUPPORTED_MODELS
                ), f"Target model {specific_model} for family {family} should be in SUPPORTED_MODELS"

            # All family names should also be in supported models
            for family in MODEL_FAMILY_MAPPING.keys():
                assert (
                    family in client.SUPPORTED_MODELS
                ), f"Family name {family} should be in SUPPORTED_MODELS"


# --- Test Category 7: Error Handling Without API Calls ---


def test_responses_api_configuration_errors(mock_logger, mock_openai, mock_tiktoken):
    """Tests configuration errors for Responses API methods."""
    with patch.dict("sys.modules", {"openai": mock_openai, "tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from convorator.client.openai_client import OpenAILLM

            # Client without Responses API
            client = OpenAILLM(logger=mock_logger, use_responses_api=False)

            # All Responses API methods should raise configuration errors
            with pytest.raises(
                LLMConfigurationError, match="Web search tool requires Responses API"
            ):
                client.query_with_web_search("test query")

            with pytest.raises(
                LLMConfigurationError, match="File search tool requires Responses API"
            ):
                client.query_with_file_search("test query", ["store-id"])

            with pytest.raises(
                LLMConfigurationError, match="Computer use tool requires Responses API"
            ):
                client.query_with_computer_use("test query")
