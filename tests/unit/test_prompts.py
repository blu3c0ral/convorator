import pytest
from unittest.mock import Mock, MagicMock

from convorator.conversations.configurations import OrchestratorConfig
from convorator.conversations.types import (
    SolutionLLMGroup,
    PromptBuilderInputs,
    LoggerProtocol,
)
from convorator.conversations.prompts import (
    default_build_initial_prompt,
    default_build_debate_user_prompt,
    default_build_moderator_context,
    default_build_moderator_instructions,
    default_build_primary_prompt,
    default_build_debater_prompt,
    default_build_summary_prompt,
    default_build_improvement_prompt,
    default_build_fix_prompt,
    _get_last_message_content_by_role,
    _get_nth_last_message_content_by_role,
)
from convorator.conversations.state import MultiAgentConversation
from convorator.client.llm_client import LLMInterface

# --- Fixtures --- #


@pytest.fixture
def mock_llm_interface():
    """Creates a mock LLMInterface."""
    mock_llm = Mock(spec=LLMInterface)
    mock_llm.get_role_name.return_value = "MockRole"
    mock_llm.get_system_message.return_value = "Mock system message."
    # Add other necessary mock methods/attributes as needed
    return mock_llm


@pytest.fixture
def mock_llm_group(mock_llm_interface):
    """Creates a mock SolutionLLMGroup (name updated)."""
    return SolutionLLMGroup(
        primary_llm=mock_llm_interface,
        debater_llm=mock_llm_interface,
        moderator_llm=mock_llm_interface,
        solution_generation_llm=mock_llm_interface,
    )


@pytest.fixture
def mock_logger():
    """Creates a mock logger."""
    return Mock()


@pytest.fixture
def basic_prompt_inputs(mock_logger, mock_llm_group):
    """Creates a basic PromptBuilderInputs instance."""
    return PromptBuilderInputs(
        topic="Test Topic",
        logger=mock_logger,
        llm_group=mock_llm_group,
        solution_schema={"type": "object", "properties": {"key": {"type": "string"}}},
        initial_solution='{"key": "initial"}',
        requirements="Test requirements",
        assessment_criteria="Test assessment criteria",
        moderator_instructions="Test moderator instructions",
        debate_context="Test debate context",
        primary_role_name="PrimaryTest",
        debater_role_name="DebaterTest",
        moderator_role_name="ModeratorTest",
        expect_json_output=True,
        conversation_history=MultiAgentConversation(),
        # Optional fields can be added here if needed for specific tests
        # e.g., initial_prompt_content, moderator_summary, last_response, errors
    )


@pytest.fixture
def complex_conversation_history():
    """Creates a MultiAgentConversation with multiple messages from different roles."""
    history = MultiAgentConversation()
    # Start with user prompt
    history.add_message(role="user", content="Initial user prompt")

    # Add multiple messages from each role
    history.add_message(role="DebaterTest", content="Debater message 1")
    history.add_message(role="PrimaryTest", content="Primary message 1")
    history.add_message(role="ModeratorTest", content="Moderator message 1")

    history.add_message(role="DebaterTest", content="Debater message 2")
    history.add_message(role="PrimaryTest", content="Primary message 2")
    history.add_message(role="ModeratorTest", content="Moderator message 2")

    history.add_message(role="DebaterTest", content="Debater message 3")
    history.add_message(role="PrimaryTest", content="Primary message 3")
    history.add_message(role="ModeratorTest", content="Moderator message 3")

    return history


# --- Test Cases --- #


def test_default_build_initial_prompt(basic_prompt_inputs):
    """Tests the default_build_initial_prompt function."""
    prompt = default_build_initial_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "Test Topic" in prompt
    assert "Test requirements" in prompt
    assert "Test assessment criteria" in prompt
    assert '{"key": "initial"}' in prompt
    assert "JSON format" not in prompt


def test_default_build_debate_user_prompt(basic_prompt_inputs):
    """Tests the default_build_debate_user_prompt function."""
    # Set the initial prompt content which this builder uses
    basic_prompt_inputs.initial_prompt_content = "Core debate instructions."

    prompt = default_build_debate_user_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "Core debate instructions." in prompt
    assert "Test debate context" in prompt


def test_default_build_moderator_context(basic_prompt_inputs):
    """Tests the default_build_moderator_context function."""
    # Add some history for context
    basic_prompt_inputs.conversation_history.add_message("user", "Initial prompt")
    basic_prompt_inputs.conversation_history.add_message("DebaterTest", "Debater response 1")
    basic_prompt_inputs.conversation_history.add_message("PrimaryTest", "Primary response 1")

    prompt = default_build_moderator_context(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "Test moderator instructions" in prompt
    assert "Debater response 1" in prompt
    assert "Primary response 1" in prompt
    assert "RECENT EXCHANGE TO MODERATE" in prompt


def test_default_build_moderator_instructions(basic_prompt_inputs):
    """Tests the default_build_moderator_instructions function."""
    # Note: The default builder uses Assessment Criteria
    prompt = default_build_moderator_instructions(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "You are the Moderator" in prompt
    assert "assessment criteria" in prompt
    assert "PrimaryTest" in prompt
    assert "DebaterTest" in prompt


def test_default_build_primary_prompt(basic_prompt_inputs):
    """Tests the default_build_primary_prompt function."""
    # Add history relevant to the primary
    basic_prompt_inputs.conversation_history.add_message("user", "Initial prompt")
    basic_prompt_inputs.conversation_history.add_message("DebaterTest", "Debater response 1")
    basic_prompt_inputs.conversation_history.add_message(
        "ModeratorTest", "Moderator feedback 1"
    )  # Feedback from previous round

    prompt = default_build_primary_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "DebaterTest" in prompt
    assert "Debater response 1" in prompt
    assert "MODERATORTEST" in prompt
    assert "Moderator feedback 1" in prompt
    assert "YOUR TASK AS PRIMARYTEST" in prompt


def test_default_build_debater_prompt(basic_prompt_inputs):
    """Tests the default_build_debater_prompt function."""
    # Add history relevant to the debater
    basic_prompt_inputs.conversation_history.add_message("user", "Initial prompt")
    basic_prompt_inputs.conversation_history.add_message(
        "DebaterTest", "Debater response 1"
    )  # Own previous
    basic_prompt_inputs.conversation_history.add_message("PrimaryTest", "Primary response 1")
    basic_prompt_inputs.conversation_history.add_message("ModeratorTest", "Moderator feedback 1")

    prompt = default_build_debater_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "PRIMARYTEST" in prompt
    assert "Primary response 1" in prompt
    assert "MODERATORTEST" in prompt
    assert "Moderator feedback 1" in prompt
    assert "YOUR TASK AS DEBATERTEST" in prompt


def test_default_build_summary_prompt(basic_prompt_inputs):
    """Tests the default_build_summary_prompt function."""
    # Add some conversation history
    basic_prompt_inputs.conversation_history.add_message("user", "Initial prompt")
    basic_prompt_inputs.conversation_history.add_message("DebaterTest", "Debate message 1")
    basic_prompt_inputs.conversation_history.add_message("PrimaryTest", "Debate message 2")
    basic_prompt_inputs.conversation_history.add_message("ModeratorTest", "Feedback 1")

    prompt = default_build_summary_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "concise summary of strengths" in prompt


def test_default_build_improvement_prompt(basic_prompt_inputs):
    """Tests the default_build_improvement_prompt function."""
    basic_prompt_inputs.moderator_summary = "Key insights from the debate."

    prompt = default_build_improvement_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "Key insights from the debate." in prompt
    assert basic_prompt_inputs.initial_solution in prompt
    assert "generate an improved solution" in prompt
    assert "Format your response as a valid JSON object." in prompt
    assert '"type": "object"' in prompt
    assert '"properties": {' in prompt
    assert '"key": {' in prompt
    assert '"type": "string"' in prompt


def test_default_build_fix_prompt(basic_prompt_inputs):
    """Tests the default_build_fix_prompt function."""
    basic_prompt_inputs.response_to_fix = '{"wrong_key": "value"}'  # Example incorrect response
    basic_prompt_inputs.errors_to_fix = (
        "JSON does not match schema. Missing required property: 'key'"
    )

    prompt = default_build_fix_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert '{"wrong_key": "value"}' in prompt
    assert "JSON does not match schema" in prompt
    assert "Missing required property: 'key'" in prompt
    assert "regenerate the response, correcting these errors" in prompt
    assert "Format your corrected response as a valid JSON object." in prompt
    assert "matching the schema provided previously." in prompt


# --- Test Edge Cases / Variations --- #


def test_initial_prompt_no_json(basic_prompt_inputs):
    """Tests initial prompt generation when JSON output is not expected."""
    basic_prompt_inputs.expect_json_output = False
    basic_prompt_inputs.solution_schema = None

    prompt = default_build_initial_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "JSON format" not in prompt
    assert "schema" not in prompt.lower()


def test_improvement_prompt_no_json(basic_prompt_inputs):
    """Tests improvement prompt generation when JSON output is not expected."""
    basic_prompt_inputs.moderator_summary = "Key insights from the debate."
    basic_prompt_inputs.expect_json_output = False
    basic_prompt_inputs.solution_schema = None

    prompt = default_build_improvement_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "JSON format" not in prompt
    assert "schema" not in prompt.lower()


def test_fix_prompt_no_json(basic_prompt_inputs):
    """Tests fix prompt generation when JSON output is not expected."""
    basic_prompt_inputs.response_to_fix = "Some text response that had an error."
    basic_prompt_inputs.errors_to_fix = "The response was too short."
    basic_prompt_inputs.expect_json_output = False
    basic_prompt_inputs.solution_schema = None

    prompt = default_build_fix_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    assert "Some text response" in prompt
    assert "too short" in prompt
    assert "regenerate the response, correcting these errors" in prompt
    assert "JSON format" not in prompt
    assert "schema" not in prompt.lower()


# --- Helper Function Tests --- #


def test_get_last_message_content_by_role():
    """Tests the _get_last_message_content_by_role helper function."""
    history = MultiAgentConversation()
    logger = Mock()

    # Empty history
    assert _get_last_message_content_by_role(history, "SomeRole", logger) is None

    # History with one message
    history.add_message("TestRole", "Test message")
    assert _get_last_message_content_by_role(history, "TestRole", logger) == "Test message"
    assert _get_last_message_content_by_role(history, "NonexistentRole", logger) is None

    # History with multiple messages from same role
    history.add_message("TestRole", "Second message")
    assert _get_last_message_content_by_role(history, "TestRole", logger) == "Second message"

    # Check None history handling
    assert _get_last_message_content_by_role(None, "TestRole", logger) is None
    logger.warning.assert_called()


def test_get_nth_last_message_content_by_role():
    """Tests the _get_nth_last_message_content_by_role helper function."""
    history = MultiAgentConversation()
    logger = Mock()

    # Empty history
    assert _get_nth_last_message_content_by_role(history, "SomeRole", 1, logger) is None

    # Add multiple messages from same role
    history.add_message("TestRole", "First message")
    history.add_message("TestRole", "Second message")
    history.add_message("TestRole", "Third message")

    # Test retrieving different positions
    assert _get_nth_last_message_content_by_role(history, "TestRole", 1, logger) == "Third message"
    assert _get_nth_last_message_content_by_role(history, "TestRole", 2, logger) == "Second message"
    assert _get_nth_last_message_content_by_role(history, "TestRole", 3, logger) == "First message"

    # Out of range
    assert _get_nth_last_message_content_by_role(history, "TestRole", 4, logger) is None

    # Nonexistent role
    assert _get_nth_last_message_content_by_role(history, "NonexistentRole", 1, logger) is None

    # Check None history handling
    assert _get_nth_last_message_content_by_role(None, "TestRole", 1, logger) is None
    logger.warning.assert_called()


# --- Missing Optional Field Tests --- #


def test_initial_prompt_missing_fields():
    """Tests initial prompt generation when optional fields are missing."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        conversation_history=MultiAgentConversation(),
        # Minimal required fields, omitting many optional ones
    )

    prompt = default_build_initial_prompt(inputs)
    assert isinstance(prompt, str)
    assert "[Not Provided]" in prompt
    logger.warning.assert_called()


def test_debate_user_prompt_missing_initial_content():
    """Tests debate user prompt when initial_prompt_content is missing."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        conversation_history=MultiAgentConversation(),
        primary_role_name="Primary",
        debater_role_name="Debater",
        moderator_role_name="Moderator",
        # Missing initial_prompt_content
    )

    prompt = default_build_debate_user_prompt(inputs)
    assert isinstance(prompt, str)
    assert "Error" in prompt
    logger.error.assert_called()


def test_moderator_context_empty_history():
    """Tests moderator context builder with empty conversation history."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        conversation_history=MultiAgentConversation(),  # Empty history
        primary_role_name="Primary",
        debater_role_name="Debater",
        moderator_role_name="Moderator",
        moderator_instructions="Test instructions",
        initial_prompt_content="Initial context",
        debate_context="Some debate context",
    )

    prompt = default_build_moderator_context(inputs)
    assert isinstance(prompt, str)
    assert "No preceding messages" in prompt
    assert "Test instructions" in prompt
    assert "Initial context" in prompt


def test_fix_prompt_missing_required_fields():
    """Tests fix prompt when required fields are missing."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        conversation_history=MultiAgentConversation(),
        # Missing response_to_fix and errors_to_fix
    )

    prompt = default_build_fix_prompt(inputs)
    assert isinstance(prompt, str)
    assert "Error" in prompt
    logger.error.assert_called()


# --- Custom Role Name Tests --- #


def test_default_role_names():
    """Tests that default role names are used when not provided."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        conversation_history=MultiAgentConversation(),
        # No custom role names provided, should use defaults
    )

    assert inputs.primary_role_name == "Primary"
    assert inputs.debater_role_name == "Debater"
    assert inputs.moderator_role_name == "Moderator"


def test_primary_prompt_with_default_roles(mock_logger, mock_llm_group):
    """Tests primary prompt with default role names."""
    # Create inputs with default role names
    inputs = PromptBuilderInputs(
        logger=mock_logger,
        llm_group=mock_llm_group,
        conversation_history=MultiAgentConversation(),
        # Using default role names
    )

    # Add relevant history
    inputs.conversation_history.add_message("user", "Initial prompt")
    inputs.conversation_history.add_message("Debater", "Debater response")
    inputs.conversation_history.add_message("Moderator", "Moderator feedback")

    prompt = default_build_primary_prompt(inputs)
    assert isinstance(prompt, str)
    assert "Debater" in prompt
    assert "MODERATOR" in prompt
    assert "PRIMARY" in prompt


# --- Complex History Tests --- #


def test_primary_prompt_with_complex_history(basic_prompt_inputs, complex_conversation_history):
    """Tests primary prompt with a complex conversation history."""
    # Use the complex history fixture
    basic_prompt_inputs.conversation_history = complex_conversation_history

    prompt = default_build_primary_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    # Should reference the most recent debater message (Debater message 3)
    assert "Debater message 3" in prompt
    # Should use the second-to-last primary message if retrieving own previous message
    assert "Primary message 2" in prompt or "PREVIOUSLY, YOU (PrimaryTest) SAID" in prompt
    # Should reference most recent moderator feedback
    assert "Moderator message 3" in prompt


def test_debater_prompt_with_complex_history(basic_prompt_inputs, complex_conversation_history):
    """Tests debater prompt with a complex conversation history."""
    # Use the complex history fixture
    basic_prompt_inputs.conversation_history = complex_conversation_history

    prompt = default_build_debater_prompt(basic_prompt_inputs)
    assert isinstance(prompt, str)
    # Should reference most recent primary message (Primary message 3)
    assert "Primary message 3" in prompt
    # Should reference most recent moderator feedback
    assert "Moderator message 3" in prompt
    # Should use the second-to-last debater message if retrieving own previous message
    assert "Debater message 2" in prompt or "PREVIOUSLY, YOU (DebaterTest) SAID" in prompt


def test_moderator_context_with_complex_history(basic_prompt_inputs, complex_conversation_history):
    """Tests moderator context builder with a complex conversation history."""
    # Use the complex history fixture
    basic_prompt_inputs.conversation_history = complex_conversation_history
    basic_prompt_inputs.initial_prompt_content = "Initial context for debate"

    prompt = default_build_moderator_context(basic_prompt_inputs)
    assert isinstance(prompt, str)
    # Should include most recent messages from each role
    assert "Debater message 3" in prompt
    assert "Primary message 3" in prompt
    # Should not include older messages directly in the prompt
    assert "Debater message 1" not in prompt
    assert "Primary message 1" not in prompt


# --- Edge Cases --- #


def test_improvement_prompt_missing_moderator_summary():
    """Tests improvement prompt when moderator summary is missing."""
    logger = Mock()
    inputs = PromptBuilderInputs(
        logger=logger,
        initial_solution='{"key": "value"}',
        requirements="Test requirements",
        expect_json_output=True,
        # Missing moderator_summary
    )

    prompt = default_build_improvement_prompt(inputs)
    assert isinstance(prompt, str)
    assert "[Not Provided]" in prompt
    logger.warning.assert_called()


def test_complex_json_schema():
    """Tests handling of a complex JSON schema."""
    logger = Mock()
    complex_schema = {
        "type": "object",
        "required": ["name", "items"],
        "properties": {
            "name": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "value": {"type": "string"}},
                },
            },
        },
    }

    inputs = PromptBuilderInputs(
        logger=logger,
        solution_schema=complex_schema,
        initial_solution='{"name": "test", "items": [{"id": 1, "value": "first"}]}',
        requirements="Must include nested items array",
        expect_json_output=True,
        moderator_summary="Needs more items in the array",
    )

    prompt = default_build_improvement_prompt(inputs)
    assert isinstance(prompt, str)
    # The schema should be included in the prompt
    assert '"required": [' in prompt
    assert '"items": {' in prompt
    assert '"type": "array"' in prompt
