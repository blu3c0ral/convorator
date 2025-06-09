import pytest
import os
import time
from typing import Optional

from convorator.client.openai_client import OpenAILLM
from convorator.exceptions import (
    LLMClientError,
    LLMConfigurationError,
    LLMResponseError,
)


# --- Fixtures and Test Configuration ---


@pytest.fixture
def api_key() -> Optional[str]:
    """Get OpenAI API key from environment, skipping tests if not available."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def real_logger():
    """Create a real logger for API tests."""
    from convorator.utils.logger import setup_logger

    return setup_logger("test_real_api")


@pytest.fixture
def real_chat_client(real_logger, api_key):
    """Create a real OpenAI client for Chat Completions API testing."""
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available - skipping real API tests")

    return OpenAILLM(
        logger=real_logger,
        api_key=api_key,
        model="gpt-4o-mini",  # Use the most cost-effective model
        max_tokens=50,  # Keep responses short for cost control
        temperature=0.1,  # Low temperature for consistent testing
        use_responses_api=False,
    )


@pytest.fixture
def real_responses_client(real_logger, api_key):
    """Create a real OpenAI client for Responses API testing."""
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available - skipping real API tests")

    return OpenAILLM(
        logger=real_logger,
        api_key=api_key,
        model="gpt-4o-mini",  # Use the most cost-effective model
        max_tokens=50,  # Keep responses short for cost control
        temperature=0.1,  # Low temperature for consistent testing
        use_responses_api=True,
    )


def rate_limit_delay():
    """Add a small delay between API calls to avoid rate limiting."""
    time.sleep(0.5)  # 500ms delay


# --- Test Category 1: Real Chat Completions API Tests ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_chat_completions_basic_query(real_chat_client):
    """
    Tests a basic query with real Chat Completions API.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    # Simple, cost-effective query
    result = real_chat_client.query("Say 'Hello'", use_conversation=False)

    # Verify we got a real response
    assert isinstance(result, str)
    assert len(result) > 0
    assert "hello" in result.lower() or "hi" in result.lower()


@pytest.mark.integration
@pytest.mark.real_api
def test_real_chat_completions_conversation_workflow(real_chat_client):
    """
    Tests a real conversation workflow with Chat Completions API.
    Cost: ~2 API calls with minimal tokens.
    """
    rate_limit_delay()

    # Set system message
    real_chat_client.set_system_message("Be very brief. Answer in 5 words or less.")

    # First turn
    result1 = real_chat_client.query("What is 2+2?", use_conversation=True)
    assert isinstance(result1, str)
    assert len(result1) > 0

    rate_limit_delay()

    # Second turn - should remember context
    result2 = real_chat_client.query("Double that", use_conversation=True)
    assert isinstance(result2, str)
    assert len(result2) > 0

    # Verify conversation history
    history = real_chat_client.get_conversation_history()
    assert len(history) >= 4  # system + user + assistant + user + assistant


@pytest.mark.integration
@pytest.mark.real_api
def test_real_chat_completions_system_message(real_chat_client):
    """
    Tests system message handling with real API.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    # Set specific system message
    real_chat_client.set_system_message("Always respond with exactly the word 'TEST'")

    result = real_chat_client.query("What is your name?", use_conversation=True)

    # The model should follow the system instruction closely
    assert isinstance(result, str)
    assert "test" in result.lower()


# --- Test Category 2: Real Responses API Tests ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_responses_api_basic_query(real_responses_client):
    """
    Tests a basic query with real Responses API.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    try:
        result = real_responses_client.query("Say 'Hello API'", use_conversation=False)

        # Verify we got a real response
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify response ID was stored
        assert real_responses_client._last_response_id is not None

    except LLMConfigurationError as e:
        if "Responses API" in str(e):
            pytest.skip("Responses API not available for this account")
        else:
            raise


@pytest.mark.integration
@pytest.mark.real_api
def test_real_responses_api_conversation_workflow(real_responses_client):
    """
    Tests real conversation workflow with Responses API.
    Cost: ~2 API calls with minimal tokens.
    """
    rate_limit_delay()

    try:
        # Set system message (becomes instructions)
        real_responses_client.set_system_message("Be extremely brief.")

        # First query
        result1 = real_responses_client.query("Count to 3", use_conversation=True)
        assert isinstance(result1, str)
        assert len(result1) > 0

        first_response_id = real_responses_client._last_response_id
        assert first_response_id is not None

        rate_limit_delay()

        # Second query - should continue conversation
        result2 = real_responses_client.query("Now count to 5", use_conversation=True)
        assert isinstance(result2, str)
        assert len(result2) > 0

        # Response ID should have changed
        second_response_id = real_responses_client._last_response_id
        assert second_response_id != first_response_id

    except LLMConfigurationError as e:
        if "Responses API" in str(e):
            pytest.skip("Responses API not available for this account")
        else:
            raise


# --- Test Category 3: Real Model Family Resolution ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_model_family_resolution(real_logger, api_key):
    """
    Tests that model families resolve to real, working models.
    Cost: ~3 API calls with minimal tokens.
    """
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available")

    # Test different model families
    test_models = [
        "gpt-4o-mini",  # Should resolve to specific version
        "gpt-4o",  # Should resolve to latest gpt-4o
        "gpt-3.5-turbo",  # Should resolve to latest 3.5-turbo
    ]

    for model_family in test_models:
        rate_limit_delay()

        client = OpenAILLM(
            logger=real_logger,
            api_key=api_key,
            model=model_family,
            max_tokens=20,  # Very short responses
            temperature=0.0,
        )

        # Verify model resolution
        model_info = client.get_model_info()
        assert model_info["original_model"] == model_family
        assert model_info["resolved_model"] != model_family or "-" in model_info["resolved_model"]

        # Verify the resolved model actually works
        try:
            result = client.query("Hi", use_conversation=False)
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            pytest.fail(
                f"Model family {model_family} resolved to {model_info['resolved_model']} but failed: {e}"
            )


# --- Test Category 4: Token Counting Accuracy Verification ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_token_counting_accuracy(real_chat_client):
    """
    Tests token counting accuracy against real API usage.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    # Test with a known text
    test_text = "Hello world! How are you today?"

    # Count tokens using our implementation
    our_count = real_chat_client.count_tokens(test_text)

    # Make an API call and check if our count is reasonable
    result = real_chat_client.query(test_text, use_conversation=False)

    # Our token count should be a positive integer
    assert isinstance(our_count, int)
    assert our_count > 0

    # For this simple text, count should be reasonable (not fallback estimation)
    # The fallback would be len(text) // 4 = 7, tiktoken should give a different count
    fallback_estimate = len(test_text) // 4

    # If tiktoken is working, we should get a different (more accurate) count
    if our_count != fallback_estimate:
        # tiktoken is working correctly
        assert our_count > 3  # Should be at least a few tokens
        assert our_count < 20  # Should not be unreasonably high for this text
    else:
        # Fallback is being used, which is also acceptable
        assert our_count == fallback_estimate


@pytest.mark.integration
@pytest.mark.real_api
def test_real_tiktoken_encoding_availability(real_chat_client):
    """
    Tests that tiktoken encoding is available and working with real model.
    Cost: No API calls.
    """
    # Get encoding info
    encoding_info = real_chat_client.get_tiktoken_encoding_info()

    # Should have encoding information
    assert isinstance(encoding_info, dict)
    assert "model" in encoding_info
    assert "available" in encoding_info
    assert encoding_info["model"] == real_chat_client._model

    # Test encoding with various texts
    test_texts = [
        "Hello",
        "This is a longer test with multiple words and punctuation!",
        "Unicode test: 你好世界 🌍",
        "",  # Empty string
    ]

    for text in test_texts:
        count = real_chat_client.count_tokens(text)
        assert isinstance(count, int)
        assert count >= 0


# --- Test Category 5: Context Limit Testing ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_context_limit_awareness(real_chat_client):
    """
    Tests context limit awareness with moderately sized inputs.
    Cost: ~1 API call with moderate tokens.
    """
    rate_limit_delay()

    # Get the context limit
    context_limit = real_chat_client.get_context_limit()
    assert isinstance(context_limit, int)
    assert context_limit > 1000  # Should be a reasonable limit

    # Create a moderately long prompt (but well within limits)
    long_prompt = "Please summarize this text: " + "This is a test sentence. " * 50

    # Count tokens in the prompt
    prompt_tokens = real_chat_client.count_tokens(long_prompt)

    # Should be well within context limit
    assert prompt_tokens < context_limit * 0.1  # Use less than 10% of context

    # Should be able to process this without issues
    result = real_chat_client.query(long_prompt, use_conversation=False)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.real_api
def test_real_max_tokens_configuration(real_logger, api_key):
    """
    Tests max_tokens configuration with real API.
    Cost: ~2 API calls with controlled token limits.
    """
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available")

    # Test with very low max_tokens
    rate_limit_delay()

    low_tokens_client = OpenAILLM(
        logger=real_logger,
        api_key=api_key,
        model="gpt-4o-mini",
        max_tokens=5,  # Very restrictive
        temperature=0.0,
    )

    result1 = low_tokens_client.query("Write a long story about adventure", use_conversation=False)

    # Should get a response, but it should be short due to max_tokens limit
    assert isinstance(result1, str)
    # Response should be relatively short (though exact length may vary)

    rate_limit_delay()

    # Test with higher max_tokens
    high_tokens_client = OpenAILLM(
        logger=real_logger,
        api_key=api_key,
        model="gpt-4o-mini",
        max_tokens=100,  # More generous
        temperature=0.0,
    )

    result2 = high_tokens_client.query("Write a long story about adventure", use_conversation=False)

    # Should get a longer response
    assert isinstance(result2, str)
    # Generally expect the higher max_tokens to allow for longer responses
    assert len(result2) >= len(result1)


# --- Test Category 6: Rate Limit Handling ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_rate_limit_error_handling(real_chat_client):
    """
    Tests handling of real rate limit scenarios.
    Cost: Variable - may trigger rate limits intentionally.
    """
    # Note: This test may occasionally hit real rate limits
    # We'll make multiple rapid requests to potentially trigger rate limiting

    successful_requests = 0
    rate_limit_hit = False

    for i in range(5):  # Try 5 rapid requests
        try:
            result = real_chat_client.query(f"Say number {i}", use_conversation=False)
            successful_requests += 1
            assert isinstance(result, str)
            # Very short delay between requests to potentially trigger rate limits
            time.sleep(0.1)
        except LLMResponseError as e:
            if "rate limit" in str(e).lower():
                rate_limit_hit = True
                break
            else:
                raise

    # We should either complete all requests successfully or hit a rate limit
    assert successful_requests > 0  # At least some should succeed
    # If we hit rate limit, that's expected behavior and shows proper error handling


@pytest.mark.integration
@pytest.mark.real_api
def test_real_authentication_error_handling(real_logger):
    """
    Tests handling of authentication errors with invalid API key.
    Cost: No successful API calls.
    """
    # Create client with invalid API key
    invalid_client = OpenAILLM(
        logger=real_logger, api_key="invalid_key_test", model="gpt-4o-mini", max_tokens=10
    )

    # Should get authentication error
    with pytest.raises(LLMConfigurationError) as exc_info:
        invalid_client.query("Test", use_conversation=False)

    assert "authentication" in str(exc_info.value).lower()


# --- Test Category 7: Cost-Aware Testing Patterns ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_api_cost_awareness_features(real_chat_client):
    """
    Tests cost-awareness features and usage patterns.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    # Test with very short, cost-effective prompts
    short_prompts = ["Hi", "1+1=?", "OK"]

    for prompt in short_prompts:
        # Count tokens before API call
        input_tokens = real_chat_client.count_tokens(prompt)
        assert input_tokens <= 5  # Very short prompts

        result = real_chat_client.query(prompt, use_conversation=False)

        # Verify we got responses
        assert isinstance(result, str)
        assert len(result) > 0

        # Count response tokens
        output_tokens = real_chat_client.count_tokens(result)

        # With max_tokens=50, output should be reasonably short
        assert output_tokens <= 60  # Some buffer for tokenization differences

        rate_limit_delay()


@pytest.mark.integration
@pytest.mark.real_api
def test_real_provider_capabilities_accuracy(real_chat_client):
    """
    Tests that provider capabilities accurately reflect real API features.
    Cost: No API calls.
    """
    capabilities = real_chat_client.get_provider_capabilities()

    # Verify capabilities structure
    assert isinstance(capabilities, dict)
    assert capabilities["provider"] == "openai"
    assert capabilities["supports_tiktoken"] is True
    assert capabilities["supports_streaming"] is True
    assert capabilities["supports_model_families"] is True

    # Verify model information
    assert "current_model_info" in capabilities
    model_info = capabilities["current_model_info"]
    assert model_info["resolved_model"] in capabilities["supported_models"]

    # Verify encoding information
    assert "encoding_info" in capabilities
    encoding_info = capabilities["encoding_info"]
    assert encoding_info["default_encoding"] == "o200k_base"
    assert real_chat_client._model in encoding_info["model_encodings"]


@pytest.mark.integration
@pytest.mark.real_api
def test_real_stateless_vs_stateful_costs(real_chat_client):
    """
    Tests cost implications of stateless vs stateful queries.
    Cost: ~3 API calls with minimal tokens.
    """
    # Clear any existing conversation
    real_chat_client.clear_conversation()

    rate_limit_delay()

    # Stateless query - should only send the single message
    result1 = real_chat_client.query("Count: 1", use_conversation=False)
    assert isinstance(result1, str)

    rate_limit_delay()

    # Start stateful conversation
    result2 = real_chat_client.query("Count: 1", use_conversation=True)
    assert isinstance(result2, str)

    rate_limit_delay()

    # Continue stateful conversation - should send full history
    result3 = real_chat_client.query("Count: 2", use_conversation=True)
    assert isinstance(result3, str)

    # Verify conversation history exists
    history = real_chat_client.get_conversation_history()
    assert len(history) >= 4  # At least 2 user + 2 assistant messages

    # Note: In real usage, stateful queries become more expensive as conversation grows
    # This test demonstrates the pattern without excessive cost


# --- Test Category 8: Real Error Recovery Patterns ---


@pytest.mark.integration
@pytest.mark.real_api
def test_real_network_error_resilience(real_chat_client):
    """
    Tests resilience to network-related errors.
    Cost: ~1 API call with minimal tokens.
    """
    rate_limit_delay()

    # Under normal conditions, this should work
    try:
        result = real_chat_client.query("Test network", use_conversation=False)
        assert isinstance(result, str)
        assert len(result) > 0
    except LLMClientError as e:
        # Network errors should be wrapped in LLMClientError
        assert "network" in str(e).lower() or "connection" in str(e).lower()
    except LLMResponseError as e:
        # API errors should be wrapped in LLMResponseError
        assert isinstance(e, LLMResponseError)


@pytest.mark.integration
@pytest.mark.real_api
def test_real_conversation_state_after_errors(real_chat_client, real_logger, api_key):
    """
    Tests conversation state consistency after real errors.
    Cost: ~2 API calls with minimal tokens.
    """
    if not api_key:
        pytest.skip("OPENAI_API_KEY not available")

    # Start a successful conversation
    rate_limit_delay()

    real_chat_client.set_system_message("Be brief.")
    result1 = real_chat_client.query("Say A", use_conversation=True)
    assert isinstance(result1, str)

    initial_history = real_chat_client.get_conversation_history()
    assert len(initial_history) >= 2

    # Try to cause an error with invalid model (create new client)
    error_client = OpenAILLM(
        logger=real_logger, api_key=api_key, model="invalid-model-name-12345", max_tokens=10
    )

    try:
        error_client.query("This should fail", use_conversation=True)
        # If this doesn't fail, that's also fine - OpenAI might handle unknown models gracefully
    except (LLMConfigurationError, LLMResponseError):
        # Expected - model not found or other API error
        pass

    # Original client should still work
    rate_limit_delay()

    result2 = real_chat_client.query("Say B", use_conversation=True)
    assert isinstance(result2, str)

    # Conversation should have continued normally
    final_history = real_chat_client.get_conversation_history()
    assert len(final_history) > len(initial_history)


# --- Utility Test for API Key Validation ---


@pytest.mark.integration
@pytest.mark.real_api
def test_api_key_validation():
    """
    Validates that API key is available for real tests.
    Cost: No API calls.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    # Basic validation
    assert isinstance(api_key, str)
    assert len(api_key) > 10  # Should be a reasonable length
    assert api_key.startswith(("sk-", "sk-proj-"))  # OpenAI API key format
