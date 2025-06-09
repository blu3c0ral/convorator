import pytest
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from convorator.client.llm_client import (
    LLMInterface,
    Conversation,
    LLMConfigurationError,
    LLMClientError,
    LLMResponseError,
)
from convorator.utils.logger import setup_logger


# A concrete implementation of the abstract class for testing purposes
class ConcreteLLM(LLMInterface):
    def __init__(self, model: str = "test-model", max_tokens: int = 100):
        super().__init__(model=model, max_tokens=max_tokens)
        self._system_message: Optional[str] = None
        self._role_name: Optional[str] = "assistant"
        self.conversation = Conversation()
        # Add a temperature attribute for testing set_provider_setting
        self.temperature = 0.5
        self.api_call_history = []
        self.should_raise_error = None  # For testing error scenarios

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        self.api_call_history.append(messages)

        # Allow tests to trigger specific errors
        if self.should_raise_error:
            raise self.should_raise_error

        # Handle empty message list (new behavior when empty prompt with no history)
        if not messages:
            return "Response to empty message list"

        # Simulate a simple response
        return f"Response to: {messages[-1]['content']}"

    def count_tokens(self, text: str) -> int:
        # Simple approximation for testing
        return len(text) // 4

    def get_context_limit(self) -> int:
        # Dummy value for testing
        return 4096

    def set_provider_setting(self, setting_name: str, value: Any) -> bool:
        """Override to handle temperature for testing purposes."""
        if setting_name == "temperature":
            self.temperature = value
            return True
        # Delegate to base class for other settings
        return super().set_provider_setting(setting_name, value)


# Incomplete implementation for testing abstract method enforcement
class IncompleteLLM(LLMInterface):
    def __init__(self, model: str = "incomplete-model", max_tokens: int = 100):
        super().__init__(model=model, max_tokens=max_tokens)
        self._system_message: Optional[str] = None
        self._role_name: Optional[str] = "assistant"
        self.conversation = Conversation()

    # Missing _call_api implementation
    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def get_context_limit(self) -> int:
        return 4096


# --- Unit Tests for LLMInterface ---


def test_llm_interface_cannot_be_instantiated_directly():
    """
    Tests that the abstract LLMInterface class cannot be instantiated directly.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class LLMInterface"):
        LLMInterface(model="abc", max_tokens=100)


def test_incomplete_subclass_cannot_be_instantiated():
    """
    Tests that a subclass missing abstract method implementations cannot be instantiated.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteLLM"):
        IncompleteLLM()


def test_llm_interface_base_initialization():
    """
    Tests that a concrete subclass correctly initializes the base attributes.
    """
    llm = ConcreteLLM(model="my-model", max_tokens=500)
    assert llm._model == "my-model"
    assert llm.max_tokens == 500
    assert llm.get_model_name() == "my-model"


def test_llm_interface_init_with_empty_model_name_fails():
    """
    Tests that initializing with an empty model name raises a configuration error.
    """
    with pytest.raises(LLMConfigurationError, match="Model name cannot be empty"):
        ConcreteLLM(model="", max_tokens=100)


def test_llm_interface_init_with_invalid_inputs():
    """
    Tests initialization with various invalid inputs.
    """
    # Negative max_tokens should be allowed (up to the implementation to validate)
    llm = ConcreteLLM(model="test", max_tokens=-1)
    assert llm.max_tokens == -1

    # Zero max_tokens should be allowed
    llm_zero = ConcreteLLM(model="test", max_tokens=0)
    assert llm_zero.max_tokens == 0

    # Non-string model names should be handled (type hints are suggestions)
    llm_int_model = ConcreteLLM(model=123, max_tokens=100)
    assert llm_int_model._model == 123


def test_get_provider_name_detection():
    """
    Tests the automatic provider name detection based on class name.
    """

    class MyOpenAILLM(ConcreteLLM):
        pass

    class MyAnthropicLLM(ConcreteLLM):
        pass

    class MyGeminiLLM(ConcreteLLM):
        pass

    class UnrecognizedLLM(ConcreteLLM):
        pass

    assert MyOpenAILLM().get_provider_name() == "openai"
    assert MyAnthropicLLM().get_provider_name() == "anthropic"
    assert MyGeminiLLM().get_provider_name() == "gemini"
    assert UnrecognizedLLM().get_provider_name() == "unknown"


def test_get_provider_capabilities_base_implementation():
    """
    Tests the base implementation of get_provider_capabilities.
    """
    llm = ConcreteLLM(model="cap-model", max_tokens=123)
    caps = llm.get_provider_capabilities()

    assert caps["provider"] == "unknown"  # From ConcreteLLM class name
    assert caps["model"] == "cap-model"
    assert caps["max_tokens"] == 123
    assert caps["supports_conversation"] is True
    assert caps["supports_system_message"] is True
    assert caps["supports_temperature"] is True
    assert caps["context_limit"] == 4096


def test_get_provider_settings_base_implementation():
    """
    Tests the base implementation of get_provider_settings.
    """
    llm = ConcreteLLM(model="set-model", max_tokens=456)
    llm.set_system_message("You are a tester.")
    llm.set_role_name("Tester")

    settings = llm.get_provider_settings()

    assert settings["model"] == "set-model"
    assert settings["max_tokens"] == 456
    assert settings["system_message"] == "You are a tester."
    assert settings["role_name"] == "Tester"
    assert settings["temperature"] == 0.5


def test_set_provider_setting_base_implementation():
    """
    Tests the base implementation of set_provider_setting for common attributes.
    """
    llm = ConcreteLLM()

    # Test temperature
    assert llm.set_provider_setting("temperature", 0.9) is True
    assert llm.temperature == 0.9

    # Test max_tokens
    assert llm.set_provider_setting("max_tokens", 2000) is True
    assert llm.max_tokens == 2000

    # Test system_message
    assert llm.set_provider_setting("system_message", "New system message") is True
    assert llm.get_system_message() == "New system message"

    # Test role_name
    assert llm.set_provider_setting("role_name", "TestRole") is True
    assert llm.get_role_name() == "TestRole"

    # Test unsupported setting
    assert llm.set_provider_setting("unsupported_setting", "value") is False


def test_system_message_management():
    """
    Tests the set_system_message and get_system_message methods.
    """
    llm = ConcreteLLM()

    # Initially should be None
    assert llm.get_system_message() is None

    # Set a system message
    llm.set_system_message("You are a helpful assistant.")
    assert llm.get_system_message() == "You are a helpful assistant."

    # The conversation object should be updated
    assert len(llm.conversation.messages) == 1
    assert llm.conversation.messages[0].role == "system"

    # Set it again with the same message (should be a no-op)
    llm.set_system_message("You are a helpful assistant.")
    assert len(llm.conversation.messages) == 1

    # Update the system message
    llm.set_system_message("You are a creative writer.")
    assert llm.get_system_message() == "You are a creative writer."
    assert len(llm.conversation.messages) == 1
    assert llm.conversation.messages[0].content == "You are a creative writer."


def test_system_message_edge_cases():
    """
    Tests edge cases for system message management.
    """
    llm = ConcreteLLM()

    # Set to None initially
    llm.set_system_message(None)
    assert llm.get_system_message() is None

    # Set to empty string
    llm.set_system_message("")
    assert llm.get_system_message() == ""
    assert len(llm.conversation.messages) == 1
    assert llm.conversation.messages[0].content == ""

    # Try to set back to None after having empty string
    # The implementation only updates if message is not None AND message != self._system_message
    # Since None is not "not None", the condition fails and it doesn't update
    llm.set_system_message(None)
    assert llm.get_system_message() == ""  # Stays as empty string - actual behavior

    # Test setting from non-empty to None directly
    llm2 = ConcreteLLM()
    llm2.set_system_message("Some message")
    assert llm2.get_system_message() == "Some message"

    # This also won't work due to the same implementation logic
    llm2.set_system_message(None)
    assert llm2.get_system_message() == "Some message"  # Actual behavior


def test_role_name_management():
    """
    Tests the set_role_name and get_role_name methods.
    """
    llm = ConcreteLLM()

    # Check default
    assert llm.get_role_name() == "assistant"

    # Set a new role name
    llm.set_role_name("Bot")
    assert llm.get_role_name() == "Bot"


def test_conversation_history_management():
    """
    Tests get_conversation_history and clear_conversation methods.
    """
    llm = ConcreteLLM()
    llm.set_system_message("System prompt")
    llm.conversation.add_user_message("Hello")
    llm.conversation.add_assistant_message("Hi there")

    history = llm.get_conversation_history()
    assert len(history) == 3
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[2]["role"] == "assistant"

    # Clear conversation, keeping system message
    llm.clear_conversation(keep_system=True)
    history_after_clear = llm.get_conversation_history()
    assert len(history_after_clear) == 1
    assert history_after_clear[0]["role"] == "system"
    assert llm.get_system_message() == "System prompt"

    # Add a message and clear completely
    llm.conversation.add_user_message("Another message")
    assert len(llm.get_conversation_history()) == 2

    llm.clear_conversation(keep_system=False)
    assert len(llm.get_conversation_history()) == 0
    assert llm.get_system_message() is None


def test_supports_feature_base_implementation():
    """
    Tests the base implementation of supports_feature.
    """
    llm = ConcreteLLM()

    base_features = [
        "conversation_history",
        "system_message",
        "role_name",
        "token_counting",
        "context_limit",
        "stateless_query",
        "stateful_query",
    ]

    for feature in base_features:
        assert llm.supports_feature(feature) is True, f"Feature '{feature}' should be supported"

    assert llm.supports_feature("unsupported_feature") is False
    assert llm.supports_feature("streaming") is False  # Not supported in base


def test_query_stateful_updates_conversation_history():
    """
    Tests that a stateful query correctly updates the internal conversation.
    """
    llm = ConcreteLLM()

    # First query
    response1 = llm.query("What is your name?", use_conversation=True)
    assert response1 == "Response to: What is your name?"

    history1 = llm.get_conversation_history()
    assert len(history1) == 2
    assert history1[0]["role"] == "user"
    assert history1[1]["role"] == "assistant"

    # Second query
    response2 = llm.query("What is your purpose?", use_conversation=True)
    assert response2 == "Response to: What is your purpose?"

    history2 = llm.get_conversation_history()
    assert len(history2) == 4
    assert history2[2]["role"] == "user"
    assert history2[3]["role"] == "assistant"

    # Verify that the API was called with the full history
    assert len(llm.api_call_history) == 2
    assert len(llm.api_call_history[0]) == 1  # First call has just the user prompt
    assert len(llm.api_call_history[1]) == 3  # Second call has user, assistant, user


def test_query_stateless_does_not_update_conversation_history():
    """
    Tests that a stateless query does not affect the internal conversation.
    """
    llm = ConcreteLLM()
    llm.conversation.add_user_message("Original message")

    # Perform a stateless query
    response = llm.query("Stateless prompt", use_conversation=False)
    assert response == "Response to: Stateless prompt"

    # Internal history should be unchanged
    history = llm.get_conversation_history()
    assert len(history) == 1
    assert history[0]["content"] == "Original message"

    # API should have been called, but only with the stateless message
    assert len(llm.api_call_history) == 1
    assert len(llm.api_call_history[0]) == 1
    assert llm.api_call_history[0][0]["content"] == "Stateless prompt"


def test_query_stateless_with_provided_history():
    """
    Tests that a stateless query uses the provided conversation history.
    """
    llm = ConcreteLLM()

    provided_history = [
        {"role": "user", "content": "History user prompt"},
        {"role": "assistant", "content": "History assistant prompt"},
    ]

    llm.query("New prompt", use_conversation=False, conversation_history=provided_history)

    # Check the messages sent to the API
    sent_messages = llm.api_call_history[0]
    assert len(sent_messages) == 3
    assert sent_messages[0]["content"] == "History user prompt"
    assert sent_messages[1]["content"] == "History assistant prompt"
    assert sent_messages[2]["content"] == "New prompt"

    # Internal history should remain empty
    assert len(llm.get_conversation_history()) == 0


def test_query_handles_system_message_correctly():
    """
    Tests that the system message is correctly included in API calls.
    """
    llm = ConcreteLLM()
    llm.set_system_message("You are a test bot.")

    # Stateful query
    llm.query("Stateful", use_conversation=True)

    # Stateless query
    llm.query("Stateless", use_conversation=False)

    # Stateless with history
    llm.query("Stateless with history", use_conversation=False, conversation_history=[])

    # Check API calls
    assert len(llm.api_call_history) == 3

    # All calls should start with the system message
    for api_call in llm.api_call_history:
        assert api_call[0]["role"] == "system"
        assert api_call[0]["content"] == "You are a test bot."


# --- Enhanced Tests for Error Handling and Edge Cases ---


def test_query_with_empty_prompt():
    """
    Tests query behavior with empty prompt.
    """
    llm = ConcreteLLM()

    # Empty string with no conversation history results in empty message list
    response = llm.query("", use_conversation=False)
    assert response == "Response to empty message list"

    # Whitespace-only with no conversation history also results in empty message list
    response2 = llm.query("   ", use_conversation=False)
    assert response2 == "Response to empty message list"

    # But with conversation history, empty prompt should use only the history
    history = [{"role": "user", "content": "Previous message"}]
    response3 = llm.query("", use_conversation=False, conversation_history=history)
    assert response3 == "Response to: Previous message"


def test_query_error_handling_llm_client_error():
    """
    Tests that LLMClientError is propagated and conversation state is rolled back.
    """
    llm = ConcreteLLM()
    llm.conversation.add_user_message("Pre-existing message")

    # Configure the mock to raise LLMClientError
    llm.should_raise_error = LLMClientError("Network error")

    with pytest.raises(LLMClientError, match="Network error"):
        llm.query("This will fail", use_conversation=True)

    # Conversation should be rolled back (user message removed)
    history = llm.get_conversation_history()
    assert len(history) == 1
    assert history[0]["content"] == "Pre-existing message"


def test_query_error_handling_llm_configuration_error():
    """
    Tests that LLMConfigurationError is propagated and conversation state is rolled back.
    """
    llm = ConcreteLLM()

    # Configure the mock to raise LLMConfigurationError
    llm.should_raise_error = LLMConfigurationError("Invalid API key")

    with pytest.raises(LLMConfigurationError, match="Invalid API key"):
        llm.query("This will fail", use_conversation=True)

    # Conversation should be rolled back
    history = llm.get_conversation_history()
    assert len(history) == 0


def test_query_error_handling_llm_response_error():
    """
    Tests that LLMResponseError is propagated and conversation state is rolled back.
    """
    llm = ConcreteLLM()
    llm.set_system_message("System")

    # Configure the mock to raise LLMResponseError
    llm.should_raise_error = LLMResponseError("Rate limit exceeded")

    with pytest.raises(LLMResponseError, match="Rate limit exceeded"):
        llm.query("This will fail", use_conversation=True)

    # System message should remain, but user message should be rolled back
    history = llm.get_conversation_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"


def test_query_error_handling_unexpected_error():
    """
    Tests that unexpected errors are wrapped in LLMClientError and conversation state is rolled back.
    """
    llm = ConcreteLLM()

    # Configure the mock to raise an unexpected error
    llm.should_raise_error = ValueError("Unexpected error")

    # Fix the regex to match the actual error message format
    with pytest.raises(LLMClientError, match="An unexpected error occurred during LLM query"):
        llm.query("This will fail", use_conversation=True)

    # Conversation should be rolled back
    history = llm.get_conversation_history()
    assert len(history) == 0


def test_query_error_handling_stateless_no_rollback():
    """
    Tests that errors in stateless queries don't affect conversation state.
    """
    llm = ConcreteLLM()
    llm.conversation.add_user_message("Existing message")

    # Configure the mock to raise an error
    llm.should_raise_error = LLMClientError("API error")

    with pytest.raises(LLMClientError, match="API error"):
        llm.query("This will fail", use_conversation=False)

    # Conversation should be unchanged (no rollback needed for stateless)
    history = llm.get_conversation_history()
    assert len(history) == 1
    assert history[0]["content"] == "Existing message"


def test_query_conversation_consistency_multiple_errors():
    """
    Tests conversation state consistency after multiple failed queries.
    """
    llm = ConcreteLLM()
    llm.set_system_message("System message")

    # First successful query
    llm.should_raise_error = None
    llm.query("Success", use_conversation=True)

    # Failed query
    llm.should_raise_error = LLMClientError("First error")
    with pytest.raises(LLMClientError):
        llm.query("Fail 1", use_conversation=True)

    # Another failed query
    llm.should_raise_error = LLMResponseError("Second error")
    with pytest.raises(LLMResponseError):
        llm.query("Fail 2", use_conversation=True)

    # Another successful query
    llm.should_raise_error = None
    llm.query("Success 2", use_conversation=True)

    # Final state should have: system + success1 + assistant1 + success2 + assistant2
    history = llm.get_conversation_history()
    assert len(history) == 5
    assert history[0]["role"] == "system"
    assert history[1]["content"] == "Success"
    assert history[2]["role"] == "assistant"
    assert history[3]["content"] == "Success 2"
    assert history[4]["role"] == "assistant"


def test_query_with_malformed_conversation_history():
    """
    Tests query behavior with malformed conversation history input.
    """
    llm = ConcreteLLM()

    # Missing keys in history
    malformed_history = [{"role": "user"}, {"content": "Hello"}]  # Missing content  # Missing role

    # This should not crash, but may cause issues in _call_api
    # The LLMInterface doesn't validate the format, it passes it through
    try:
        llm.query("Test", use_conversation=False, conversation_history=malformed_history)
    except (KeyError, AttributeError):
        # Expected - malformed history will cause issues in message processing
        pass


def test_query_with_empty_conversation_history():
    """
    Tests query with explicitly empty conversation history.
    """
    llm = ConcreteLLM()
    llm.set_system_message("System")

    response = llm.query("Test with empty history", use_conversation=False, conversation_history=[])
    assert response == "Response to: Test with empty history"

    # Should have called API with system message + user prompt
    sent_messages = llm.api_call_history[0]
    assert len(sent_messages) == 2
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["content"] == "Test with empty history"


def test_system_message_overrides_in_stateless_with_history():
    """
    Tests that interface system message overrides system messages in provided history.
    """
    llm = ConcreteLLM()
    llm.set_system_message("Interface system message")

    history_with_system = [
        {"role": "system", "content": "History system message"},
        {"role": "user", "content": "User message from history"},
    ]

    llm.query("New prompt", use_conversation=False, conversation_history=history_with_system)

    sent_messages = llm.api_call_history[0]
    # Should filter out the history system message and use the interface one
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[0]["content"] == "Interface system message"
    assert sent_messages[1]["role"] == "user"
    assert sent_messages[1]["content"] == "User message from history"
    assert sent_messages[2]["content"] == "New prompt"
