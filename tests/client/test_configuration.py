import pytest
import os
from unittest.mock import Mock, patch

from convorator.client.openai_client import (
    OpenAILLM,
    MODEL_FAMILY_MAPPING,
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


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken to avoid encoding issues in configuration tests."""
    return Mock()


# --- Test Category 1: Model Family Resolution ---


@pytest.mark.parametrize(
    "family, expected_resolved",
    [
        ("gpt-4o", "gpt-4o-2024-11-20"),
        ("gpt-4.1", "gpt-4.1-2025-04-14"),
        ("o3", "o3-2025-04-16"),
        ("gpt-4", "gpt-4-0613"),
        ("gpt-3.5-turbo", "gpt-3.5-turbo-0125"),
    ],
)
def test_model_family_resolution(
    mock_logger, mock_openai, mock_tiktoken, family, expected_resolved
):
    """
    Tests that model families are correctly resolved to their latest specific versions.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model=family, max_tokens=1024)

            # Test the internal resolution method
            resolved = client._resolve_model_name(family)
            assert resolved == expected_resolved

            # Test that the client uses the resolved model
            assert client._model == expected_resolved
            assert client.original_model == family


def test_direct_model_names_no_resolution(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests that specific model versions are used as-is without resolution.
    """
    specific_model = "gpt-4o-2024-08-06"  # Not in family mapping

    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model=specific_model, max_tokens=1024)

            resolved = client._resolve_model_name(specific_model)
            assert resolved == specific_model  # Should be unchanged
            assert client._model == specific_model


def test_unknown_model_family_handling(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests that unknown model names are handled gracefully without resolution.
    """
    unknown_model = "future-gpt-model-v10"

    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model=unknown_model, max_tokens=1024)

            resolved = client._resolve_model_name(unknown_model)
            assert resolved == unknown_model  # Should be unchanged
            assert client._model == unknown_model


# --- Test Category 2: Model Family Availability Queries ---


def test_get_available_model_families(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests retrieval of all available model families and their mappings.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            families = client.get_available_model_families()

            # Should return a copy of the family mapping
            assert isinstance(families, dict)
            assert "gpt-4o" in families
            assert "gpt-4.1" in families
            assert "o3" in families
            assert families["gpt-4o"] == "gpt-4o-2024-11-20"

            # Should be a copy, not the original
            assert families is not MODEL_FAMILY_MAPPING


@pytest.mark.parametrize(
    "family, expected_latest",
    [
        ("gpt-4o", "gpt-4o-2024-11-20"),
        ("gpt-4.1", "gpt-4.1-2025-04-14"),
        ("o3", "o3-2025-04-16"),
        ("nonexistent-family", None),
    ],
)
def test_get_latest_model_for_family(
    mock_logger, mock_openai, mock_tiktoken, family, expected_latest
):
    """
    Tests retrieval of the latest model version for specific families.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            latest = client.get_latest_model_for_family(family)
            assert latest == expected_latest


def test_get_model_info_with_family_resolution(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests model info reporting when a family is resolved.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            info = client.get_model_info()

            assert info["original_model"] == "gpt-4o"
            assert info["resolved_model"] == "gpt-4o-2024-11-20"
            assert info["is_family_resolved"] is True
            assert info["family_mapping_available"] is True


def test_get_model_info_without_family_resolution(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests model info reporting when no family resolution occurs.
    """
    specific_model = "gpt-4o-2024-08-06"

    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model=specific_model, max_tokens=1024)

            info = client.get_model_info()

            assert info["original_model"] == specific_model
            assert info["resolved_model"] == specific_model
            assert info["is_family_resolved"] is False
            assert info["family_mapping_available"] is False


# --- Test Category 3: Provider Capabilities ---


def test_get_provider_capabilities_basic(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests basic provider capabilities reporting.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            capabilities = client.get_provider_capabilities()

            # Base capabilities
            assert capabilities["provider"] == "openai"
            assert capabilities["model"] == "gpt-4o-2024-11-20"  # Resolved
            assert capabilities["max_tokens"] == 1024
            assert capabilities["supports_conversation"] is True
            assert capabilities["supports_system_message"] is True
            assert capabilities["supports_temperature"] is True

            # OpenAI-specific capabilities
            assert capabilities["supports_tiktoken"] is True
            assert capabilities["supports_streaming"] is True
            assert capabilities["supports_model_families"] is True
            assert "model_family_mapping" in capabilities
            assert "current_model_info" in capabilities

            # Responses API capabilities
            assert capabilities["supports_responses_api"] is True
            assert capabilities["responses_api_enabled"] is False  # Default
            assert capabilities["supports_web_search"] is False  # Depends on responses_api_enabled
            assert capabilities["supports_file_search"] is False
            assert capabilities["supports_computer_use"] is False


def test_get_provider_capabilities_with_responses_api(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests provider capabilities when Responses API is enabled.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(
                logger=mock_logger, model="gpt-4o", max_tokens=1024, use_responses_api=True
            )

            capabilities = client.get_provider_capabilities()

            assert capabilities["responses_api_enabled"] is True
            assert capabilities["supports_web_search"] is True
            assert capabilities["supports_file_search"] is True
            assert capabilities["supports_computer_use"] is True
            assert capabilities["supports_server_side_conversation"] is True


@pytest.mark.parametrize(
    "feature, expected",
    [
        ("tiktoken_encoding", True),
        ("streaming", True),
        ("function_calling", True),
        ("vision", True),
        ("model_family_resolution", True),
        ("model_family_mapping", True),
        ("responses_api", False),  # Default is False
        ("web_search", False),  # Depends on responses_api
        ("file_search", False),
        ("computer_use", False),
        ("nonexistent_feature", False),
    ],
)
def test_supports_feature(mock_logger, mock_openai, mock_tiktoken, feature, expected):
    """
    Tests feature support queries.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            assert client.supports_feature(feature) == expected


def test_supports_feature_with_responses_api(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests feature support when Responses API features are enabled.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(
                logger=mock_logger, model="gpt-4o", max_tokens=1024, use_responses_api=True
            )

            assert client.supports_feature("responses_api") is True
            assert client.supports_feature("web_search") is True
            assert client.supports_feature("file_search") is True
            assert client.supports_feature("computer_use") is True


# --- Test Category 4: Settings Validation and Updates ---


def test_get_provider_settings(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests retrieval of current provider settings.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(
                logger=mock_logger,
                model="gpt-4o",
                max_tokens=2048,
                temperature=0.8,
                use_responses_api=True,
            )

            # Mock the encoding retrieval to avoid tiktoken issues
            mock_encoding = Mock()
            mock_encoding.name = "o200k_base"
            with patch.object(client, "_get_tiktoken_encoding", return_value=mock_encoding):
                settings = client.get_provider_settings()

            # Base settings
            assert settings["model"] == "gpt-4o-2024-11-20"  # Resolved
            assert settings["max_tokens"] == 2048
            assert settings["temperature"] == 0.8

            # OpenAI-specific settings
            assert settings["api_key"].startswith("test-key")  # Truncated for security
            assert settings["use_responses_api"] is True
            assert settings["last_response_id"] is None
            assert settings["original_model"] == "gpt-4o"
            assert settings["resolved_model"] == "gpt-4o-2024-11-20"
            assert settings["model_family_resolved"] is True


def test_set_provider_setting_temperature_valid(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests setting valid temperature values.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Test valid temperature values
            assert client.set_provider_setting("temperature", 0.0) is True
            assert client.temperature == 0.0

            assert client.set_provider_setting("temperature", 1.5) is True
            assert client.temperature == 1.5

            assert client.set_provider_setting("temperature", 2.0) is True
            assert client.temperature == 2.0


def test_set_provider_setting_temperature_invalid(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests setting invalid temperature values.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024, temperature=1.0)

            original_temp = client.temperature

            # Test invalid temperature values
            assert client.set_provider_setting("temperature", -0.5) is False
            assert client.temperature == original_temp  # Should remain unchanged

            assert client.set_provider_setting("temperature", 2.5) is False
            assert client.temperature == original_temp

            assert client.set_provider_setting("temperature", "invalid") is False
            assert client.temperature == original_temp


def test_set_provider_setting_use_responses_api(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests setting the use_responses_api flag.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Initially False
            assert client.use_responses_api is False

            # Enable Responses API
            assert client.set_provider_setting("use_responses_api", True) is True
            assert client.use_responses_api is True

            # Disable Responses API
            assert client.set_provider_setting("use_responses_api", False) is True
            assert client.use_responses_api is False

            # Test invalid value
            assert client.set_provider_setting("use_responses_api", "invalid") is False


def test_set_provider_setting_base_class_delegation(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests that non-OpenAI-specific settings are delegated to the base class.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Test base class settings
            assert client.set_provider_setting("max_tokens", 2048) is True
            assert client.max_tokens == 2048

            assert client.set_provider_setting("system_message", "You are helpful") is True
            assert client.get_system_message() == "You are helpful"

            assert client.set_provider_setting("role_name", "Assistant") is True
            assert client.get_role_name() == "Assistant"

            # Test unsupported setting
            assert client.set_provider_setting("unsupported_setting", "value") is False


# --- Test Category 5: Context Limit vs max_tokens Validation ---


def test_context_limit_vs_max_tokens_warnings(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests warnings when max_tokens approaches or exceeds context limits.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Test exceeding context limit
            mock_logger.reset_mock()
            client = OpenAILLM(
                logger=mock_logger, model="gpt-3.5-turbo-0613", max_tokens=5000
            )  # Limit is 4096

            mock_logger.warning.assert_called_with(
                "Requested max_tokens (5000) exceeds the known context limit (4096) for model 'gpt-3.5-turbo-0613'. API calls might fail."
            )


def test_context_limit_vs_max_tokens_close_warning(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests info warning when max_tokens is close to context limits.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Test close to context limit (>80%)
            mock_logger.reset_mock()
            client = OpenAILLM(
                logger=mock_logger, model="gpt-3.5-turbo-0613", max_tokens=3500
            )  # 85% of 4096

            mock_logger.info.assert_called_with(
                "Requested max_tokens (3500) is close to the context limit (4096). Ensure input + output fits."
            )


def test_context_limit_vs_max_tokens_no_warning(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests no warnings when max_tokens is reasonable.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Test reasonable max_tokens
            mock_logger.reset_mock()
            client = OpenAILLM(
                logger=mock_logger, model="gpt-4o", max_tokens=1024
            )  # Well within 128k limit

            # Should not have any warnings about context limits
            warning_calls = [
                call for call in mock_logger.warning.call_args_list if "context limit" in str(call)
            ]
            assert len(warning_calls) == 0

            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "close to the context limit" in str(call)
            ]
            assert len(info_calls) == 0


# --- Test Category 6: Advanced Configuration Edge Cases ---


def test_tiktoken_encoding_info(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests tiktoken encoding information retrieval.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Mock the encoding retrieval
            mock_encoding = Mock()
            mock_encoding.name = "o200k_base"

            with patch.object(client, "_get_tiktoken_encoding", return_value=mock_encoding):
                info = client.get_tiktoken_encoding_info()

                assert info["encoding_name"] == "o200k_base"
                assert info["model"] == "gpt-4o-2024-11-20"
                assert info["available"] is True


def test_tiktoken_encoding_info_failed(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests tiktoken encoding information when encoding fails.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Mock failed encoding retrieval
            with patch.object(client, "_get_tiktoken_encoding", return_value=None):
                info = client.get_tiktoken_encoding_info()

                assert info["encoding_name"] is None
                assert info["model"] == "gpt-4o-2024-11-20"
                assert info["available"] is False
                assert info["fallback_used"] is True


def test_model_resolution_logging(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests that model resolution is properly logged.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = OpenAILLM(logger=mock_logger, model="gpt-4o", max_tokens=1024)

            # Check that initialization logged the resolution
            mock_logger.info.assert_any_call(
                "OpenAILLM initialized. Original Model: gpt-4o, Resolved Model: gpt-4o-2024-11-20, API: Chat Completions API, Temperature: 0.6, Max Tokens: 1024"
            )


def test_unsupported_model_warning(mock_logger, mock_openai, mock_tiktoken):
    """
    Tests warning for models not in the supported list.
    """
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            unknown_model = "future-model-x"
            client = OpenAILLM(logger=mock_logger, model=unknown_model, max_tokens=1024)

            # Should warn about unsupported model
            mock_logger.warning.assert_any_call(
                f"Model '{unknown_model}' is not in the explicitly supported list for OpenAILLM context/token estimation. Proceeding, but compatibility is not guaranteed."
            )
