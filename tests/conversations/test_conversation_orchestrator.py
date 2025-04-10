import pytest
import logging
from unittest.mock import MagicMock

# Imports from the module under test and its dependencies
from convorator.client.llm_client import LLMInterface, Conversation
from convorator.conversations.types import SolutionLLMGroup
from convorator.conversations.configurations import OrchestratorConfig
from convorator.conversations.prompts import PromptBuilderInputs
from convorator.conversations.state import MultiAgentConversation
from convorator.conversations.conversation_orchestrator import SolutionImprovementOrchestrator
from convorator.exceptions import (
    LLMResponseError,
    LLMOrchestrationError,
    MaxIterationsExceededError,
    SchemaValidationError,
    LLMClientError,
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
)


# Constants for test-specific inputs
TEST_INITIAL_SOLUTION = {"id": 123, "content": "Initial test content."}
TEST_REQUIREMENTS = "The solution must be a JSON object with 'id' and 'content'."
TEST_ASSESSMENT_CRITERIA = "Assess clarity and adherence to requirements."


# --- Foundational Fixtures ---


@pytest.fixture
def mock_llm_interface(mocker) -> MagicMock:
    """
    Provides a MagicMock object simulating an LLMInterface instance.

    Mocks methods essential for the orchestrator based on its usage:
    - query: Returns a default response.
    - count_tokens: Returns a simple token count (length of string).
    - get_context_limit: Returns a typical context limit.
    - get_role_name: Returns a default role name.
    - get_system_message: Returns a default system message.
    """
    mock = mocker.MagicMock(spec=LLMInterface)
    mock.query.return_value = "Default mock LLM response."
    # Simulate token counting based on simple length (adjust if specific logic needed)
    mock.count_tokens.side_effect = lambda text: len(text.split()) if text else 0
    mock.get_context_limit.return_value = 8192  # Example context limit
    mock.get_role_name.return_value = "MockLLM"
    mock.get_system_message.return_value = "You are a helpful mock assistant."
    # Ensure max_tokens attribute exists, as used in _prepare_history_for_llm
    mock.max_tokens = 1024  # Example max generation tokens
    return mock


@pytest.fixture
def mock_llm_group(mocker, mock_llm_interface) -> SolutionLLMGroup:
    """
    Provides a SolutionLLMGroup instance populated with distinct mock LLMInterfaces.

    Ensures each role (primary, debater, moderator, solution_generation)
    has its own mock LLMInterface instance with a specific role name assigned.
    """
    # Create distinct mocks for each role
    primary_mock = mocker.MagicMock(spec=LLMInterface)
    primary_mock.query.return_value = "Primary mock response."
    primary_mock.count_tokens.side_effect = lambda text: len(text.split()) if text else 0
    primary_mock.get_context_limit.return_value = 8192
    primary_mock.get_role_name.return_value = "Primary"
    primary_mock.get_system_message.return_value = "Primary system prompt."
    primary_mock.max_tokens = 1024

    debater_mock = mocker.MagicMock(spec=LLMInterface)
    debater_mock.query.return_value = "Debater mock response."
    debater_mock.count_tokens.side_effect = lambda text: len(text.split()) if text else 0
    debater_mock.get_context_limit.return_value = 8192
    debater_mock.get_role_name.return_value = "Debater"
    debater_mock.get_system_message.return_value = "Debater system prompt."
    debater_mock.max_tokens = 1024

    moderator_mock = mocker.MagicMock(spec=LLMInterface)
    moderator_mock.query.return_value = "Moderator mock response."
    moderator_mock.count_tokens.side_effect = lambda text: len(text.split()) if text else 0
    moderator_mock.get_context_limit.return_value = 8192
    moderator_mock.get_role_name.return_value = "Moderator"
    moderator_mock.get_system_message.return_value = "Moderator system prompt."
    moderator_mock.max_tokens = 1024

    solution_gen_mock = mocker.MagicMock(spec=LLMInterface)
    solution_gen_mock.query.return_value = (
        '{"result": "solution gen mock response"}'  # Default JSON
    )
    solution_gen_mock.count_tokens.side_effect = lambda text: len(text.split()) if text else 0
    solution_gen_mock.get_context_limit.return_value = 8192
    solution_gen_mock.get_role_name.return_value = "SolutionGenerator"
    solution_gen_mock.get_system_message.return_value = "Solution generator system prompt."
    solution_gen_mock.max_tokens = 1024

    return SolutionLLMGroup(
        primary_llm=primary_mock,
        debater_llm=debater_mock,
        moderator_llm=moderator_mock,
        solution_generation_llm=solution_gen_mock,
    )


@pytest.fixture
def valid_orchestrator_config(mock_llm_group) -> OrchestratorConfig:
    """
    Provides a valid OrchestratorConfig instance using the mock LLM group.

    Populates the config with minimal required fields and sensible defaults
    based on the OrchestratorConfig definition and typical usage.
    """
    return OrchestratorConfig(
        llm_group=mock_llm_group,
        topic="Test Topic: Optimizing a widget description",
        requirements="Placeholder requirements for testing.",
        assessment_criteria="Placeholder assessment criteria for testing.",
        debate_iterations=2,  # Default non-zero value
        improvement_iterations=1,  # Default non-zero value
        # Other fields like schema, prompt overrides, logger are optional
    )


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provides a mock logger instance."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def default_prompt_inputs(valid_orchestrator_config, mock_logger) -> PromptBuilderInputs:
    """
    Provides a standard PromptBuilderInputs instance populated with test data.

    Derives values from the valid_orchestrator_config and adds necessary
    runtime elements like initial solution, requirements, criteria, logger,
    and a conversation history object.
    """
    # Sample data required by PromptBuilderInputs based on its definition
    initial_solution_sample = {"widget_id": 1, "description": "Initial description."}
    requirements_sample = "Description must be under 50 chars and mention 'optimized'."
    assessment_criteria_sample = "Clarity and conciseness."
    conversation_history_sample = MultiAgentConversation()  # Fresh history

    return PromptBuilderInputs(
        topic=valid_orchestrator_config.topic,
        logger=mock_logger,  # Use the mock logger
        llm_group=valid_orchestrator_config.llm_group,
        solution_schema=valid_orchestrator_config.solution_schema,  # Will be None by default
        initial_solution=initial_solution_sample,
        requirements=requirements_sample,
        assessment_criteria=assessment_criteria_sample,
        moderator_instructions=valid_orchestrator_config.moderator_instructions_override,  # None
        debate_context=valid_orchestrator_config.debate_context_override,  # None
        primary_role_name=valid_orchestrator_config.llm_group.primary_llm.get_role_name(),
        debater_role_name=valid_orchestrator_config.llm_group.debater_llm.get_role_name(),
        moderator_role_name=valid_orchestrator_config.llm_group.moderator_llm.get_role_name(),
        expect_json_output=valid_orchestrator_config.expect_json_output,  # False by default
        conversation_history=conversation_history_sample,
        # Fields added later dynamically by orchestrator, init with None/defaults
        initial_prompt_content=None,
        moderator_summary=None,
        response_to_fix=None,
        errors_to_fix=None,
    )


# Placeholder for actual tests to be added later
def test_placeholder():
    """Placeholder test to ensure the file is valid."""
    assert True


# --- Phase 3: Orchestrator Initialization & Configuration Tests ---


class TestOrchestratorInitialization:
    def test_init_success_with_valid_config(self, valid_orchestrator_config):
        """Verify successful initialization with a valid config and arguments."""
        orchestrator = SolutionImprovementOrchestrator(
            config=valid_orchestrator_config,
            initial_solution=TEST_INITIAL_SOLUTION,
            requirements=TEST_REQUIREMENTS,
            assessment_criteria=TEST_ASSESSMENT_CRITERIA,
        )

        # Check direct config assignment
        assert orchestrator.config is valid_orchestrator_config

        # Check attributes derived from config
        assert orchestrator.llm_group is valid_orchestrator_config.llm_group
        assert orchestrator.primary_llm is valid_orchestrator_config.llm_group.primary_llm
        assert orchestrator.debater_llm is valid_orchestrator_config.llm_group.debater_llm
        assert orchestrator.moderator_llm is valid_orchestrator_config.llm_group.moderator_llm
        assert (
            orchestrator.solution_generation_llm
            is valid_orchestrator_config.llm_group.solution_generation_llm
        )
        assert orchestrator.logger is valid_orchestrator_config.logger
        assert orchestrator.debate_iterations == valid_orchestrator_config.debate_iterations
        assert (
            orchestrator.improvement_iterations == valid_orchestrator_config.improvement_iterations
        )
        assert (
            orchestrator.moderator_instructions
            is valid_orchestrator_config.moderator_instructions_override  # Should be None in default fixture
        )
        assert (
            orchestrator.debate_context
            is valid_orchestrator_config.debate_context_override  # Should be None in default fixture
        )

        # Check default prompt builder assignments (since not overridden in fixture)
        assert orchestrator.build_initial_prompt is default_build_initial_prompt
        assert orchestrator.build_debate_user_prompt is default_build_debate_user_prompt
        assert orchestrator.build_moderator_context is default_build_moderator_context
        assert (
            orchestrator.build_moderator_role_instructions is default_build_moderator_instructions
        )
        assert orchestrator.build_primary_prompt is default_build_primary_prompt
        assert orchestrator.build_debater_prompt is default_build_debater_prompt
        assert orchestrator.build_summary_prompt is default_build_summary_prompt
        assert orchestrator.build_improvement_prompt is default_build_improvement_prompt
        assert orchestrator.build_fix_prompt is default_build_fix_prompt

        # Check task-specific inputs
        assert orchestrator.initial_solution is TEST_INITIAL_SOLUTION
        assert orchestrator.requirements is TEST_REQUIREMENTS
        assert orchestrator.assessment_criteria is TEST_ASSESSMENT_CRITERIA

        # Check derived role names (using mock names from mock_llm_group)
        assert orchestrator.primary_name == "Primary"
        assert orchestrator.debater_name == "Debater"
        assert orchestrator.moderator_name == "Moderator"

        # Check initial state
        assert isinstance(orchestrator.main_conversation_log, MultiAgentConversation)
        assert len(orchestrator.main_conversation_log.get_messages()) == 0
        assert orchestrator.prompt_inputs is None

    def test_init_raises_type_error_for_invalid_config(self):
        """Verify TypeError is raised if config is not an OrchestratorConfig instance."""
        with pytest.raises(TypeError) as excinfo:
            SolutionImprovementOrchestrator(
                config="not a config object",  # Invalid type
                initial_solution=TEST_INITIAL_SOLUTION,
                requirements=TEST_REQUIREMENTS,
                assessment_criteria=TEST_ASSESSMENT_CRITERIA,
            )
        assert "config must be an instance of OrchestratorConfig" in str(excinfo.value)


# --- Phase 4: Core Internal Utility Tests (`_prepare_history_for_llm`) ---


@pytest.fixture
def orchestrator_instance(valid_orchestrator_config) -> SolutionImprovementOrchestrator:
    """Provides a standard orchestrator instance for testing internal methods."""
    return SolutionImprovementOrchestrator(
        config=valid_orchestrator_config,
        initial_solution=TEST_INITIAL_SOLUTION,
        requirements=TEST_REQUIREMENTS,
        assessment_criteria=TEST_ASSESSMENT_CRITERIA,
    )


# Helper to create messages easily
def create_message(role: str, words: int) -> dict:
    return {"role": role, "content": " ".join(["word"] * words)}


class TestHistoryPreparation:

    # Define a simple token counter for consistency in tests
    @staticmethod
    def simple_token_count(text: str) -> int:
        return len(text.split()) if text else 0

    @pytest.mark.parametrize(
        "test_id, system_msg_words, history_msgs_words, context_limit, max_gen_tokens, expected_len, expected_exception, expected_first_role",
        [
            (
                "no_truncation_needed",
                10,  # system words
                [20, 30],  # history words list
                100,  # context limit
                10,  # max generation tokens
                3,  # Expected len (sys + 2 hist)
                None,  # No exception expected
                "system",  # First message should be system
            ),
            (
                "no_truncation_needed_no_system",
                0,  # No system message
                [20, 30],  # history words list
                100,  # context limit
                10,  # max generation tokens
                2,  # Expected len (2 hist)
                None,
                "user",  # First message should be user
            ),
            (
                "truncation_needed",
                10,  # system words
                [20, 30, 40, 50],  # history (140 words total)
                100,  # context limit (sys=10 + hist=140 + buffer=10+10=20 => 170 > 100)
                10,  # max generation tokens
                2,  # Expected len (sys + last 2 hist [40, 50] = 90. 10 + 90 + 20 = 120 > 100. Error in calc -> needs last 1 -> 50. 10 + 50 + 20 = 80. Expect sys + last 1)
                # RETHINK: Limit 100. Buffer 10+10=20. Available = 80. Sys=10. Hist needs <= 70.
                # History: [20, 30, 40, 50]. Keep [40, 50] = 90 > 70. Keep [50] = 50 <= 70. OK.
                # Expected: sys + hist[3]. Length = 2.
                None,
                "system",
            ),
            (
                "truncation_needed_no_system",
                0,
                [20, 30, 40, 50],  # history (140 words)
                100,  # Limit 100. Buffer 20. Available = 80.
                10,  # <-- INSERTED max_gen_tokens value
                # Keep [40, 50] = 90 > 80. Keep [50] = 50 <= 80. OK.
                # Expected: hist[3]. Length = 1.
                1,
                None,
                "assistant",  # Assuming the last message role
            ),
            (
                "edge_system_plus_buffer_exceeds_limit",
                95,
                [10],  # history words
                100,  # context limit
                10,  # max generation tokens (buffer = 20)
                0,  # Expected len doesn't matter when exception is raised
                LLMOrchestrationError,  # WAS None -> Should expect an error
                None,
            ),
            (
                "edge_system_alone_exceeds_limit",
                110,
                [10],
                100,
                10,  # buffer 20
                0,
                LLMOrchestrationError,  # WAS None -> Should expect an error
                None,
            ),
            (
                "edge_cannot_truncate_enough",
                10,  # system words
                [95],  # Single large history message
                100,  # context limit
                10,  # buffer = 20. Available = 80. Sys=10. Hist needs <= 70.
                # Cannot remove the only history message if it's too big.
                # Truncation removes the [95] message, leaving only system message.
                1,  # WAS 0 -> Correct expected length is 1 (system message)
                None,
                "system",  # WAS None -> Role should be system
            ),
            (
                "edge_cannot_truncate_enough_no_system",
                0,  # no system
                [95],  # Single large history message
                100,  # context limit
                10,  # buffer = 20. Available = 80. Hist needs <= 80.
                # Cannot remove the only history message if it's too big.
                0,
                None,
                None,
            ),
        ],
    )
    def test_prepare_history_scenarios(
        self,
        orchestrator_instance,  # Fixture providing the orchestrator
        mocker,  # Fixture for mocking
        test_id,
        system_msg_words,
        history_msgs_words,
        context_limit,
        max_gen_tokens,
        expected_len,
        expected_exception,
        expected_first_role,
    ):
        """Tests _prepare_history_for_llm across various scenarios."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.llm_group.primary_llm  # Use any mock LLM from the group

        # Configure the mock LLM for this specific test case
        mock_llm.get_context_limit.return_value = context_limit
        mock_llm.max_tokens = max_gen_tokens
        mock_llm.count_tokens.side_effect = self.simple_token_count

        # Construct the initial message list
        messages = []
        if system_msg_words > 0:
            messages.append(create_message("system", system_msg_words))

        role_cycle = ["user", "assistant"]
        for i, words in enumerate(history_msgs_words):
            messages.append(create_message(role_cycle[i % 2], words))

        if expected_exception:
            with pytest.raises(expected_exception) as excinfo:
                orchestrator._prepare_history_for_llm(
                    llm_service=mock_llm, messages=messages, context=f"Test: {test_id}"
                )
            # Optional: Add more specific checks on the exception message if needed
            assert str(context_limit) in str(excinfo.value)  # Check limit mentioned in error
        else:
            prepared_history = orchestrator._prepare_history_for_llm(
                llm_service=mock_llm, messages=messages, context=f"Test: {test_id}"
            )

            assert len(prepared_history) == expected_len
            if expected_len > 0:
                assert prepared_history[0]["role"] == expected_first_role

            # Verify total tokens are within limit (including buffer)
            total_tokens = sum(self.simple_token_count(msg["content"]) for msg in prepared_history)
            response_buffer = mock_llm.max_tokens + 10
            assert total_tokens + response_buffer <= context_limit


# --- Phase 5: Agent Turn and Query Logic Tests ---


@pytest.fixture
def agent_conversation() -> Conversation:
    """Provides a fresh Conversation object for agent perspective history."""
    # Note: This is Conversation from LLMInterface module, not MultiAgentConversation
    conv = Conversation(system_message="Agent system message.")
    # Add some basic history
    conv.add_user_message("Initial user query for agent.")
    conv.add_assistant_message("Initial assistant response.")
    return conv


class TestAgentTurns:
    """Tests for agent turn execution and query logic."""

    def test_query_agent_basic_success(self, orchestrator_instance, agent_conversation, mocker):
        """Test the basic success path of _query_agent_and_update_logs."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.primary_llm
        test_response = "Test agent response"
        mock_llm.query.return_value = test_response

        # Set up history preparation spy
        prepare_spy = mocker.spy(orchestrator, "_prepare_history_for_llm")

        # Execute method
        response = orchestrator._query_agent_and_update_logs(
            llm_to_query=mock_llm, role_name="TestAgent", conversation_history=agent_conversation
        )

        # Verify response and method calls
        assert response == test_response
        assert prepare_spy.called
        assert mock_llm.query.called

        # Verify conversation was updated
        assert agent_conversation.get_messages()[-1]["content"] == test_response

        # Verify main log was updated
        assert orchestrator.main_conversation_log.get_messages()[-1]["content"] == test_response

    def test_query_agent_error_propagation(self, orchestrator_instance, agent_conversation, mocker):
        """Test that LLMResponseError is properly propagated."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.primary_llm
        error_msg = "Test LLM error"
        mock_llm.query.side_effect = LLMResponseError(error_msg)

        # Execute method and verify error
        with pytest.raises(LLMResponseError) as excinfo:
            orchestrator._query_agent_and_update_logs(
                llm_to_query=mock_llm,
                role_name="TestAgent",
                conversation_history=agent_conversation,
            )

        assert error_msg in str(excinfo.value)

    def test_execute_agent_turn_basic(
        self, orchestrator_instance, agent_conversation, default_prompt_inputs, mocker
    ):
        """Test the basic execution flow of _execute_agent_turn."""
        orchestrator = orchestrator_instance

        # Set prompt_inputs (required by method)
        orchestrator.prompt_inputs = default_prompt_inputs

        # Mock the prompt builder
        test_prompt = "Test built prompt"
        mock_builder = mocker.Mock(return_value=test_prompt)

        # Spy on _query_agent_and_update_logs
        query_spy = mocker.patch.object(
            orchestrator, "_query_agent_and_update_logs", return_value="Mock agent response"
        )

        # Execute the method
        orchestrator._execute_agent_turn(
            role_name="TestRole",
            llm=orchestrator.primary_llm,
            prompt_builder=mock_builder,
            agent_conv=agent_conversation,
            other_convs=[],
        )

        # Verify prompt builder was called correctly
        mock_builder.assert_called_once_with(orchestrator.prompt_inputs)

        # Verify user message was added to conversation
        assert agent_conversation.get_messages()[-1]["content"] == test_prompt

        # Verify query method was called
        assert query_spy.called

    def test_execute_agent_turn_prompt_error(
        self, orchestrator_instance, agent_conversation, default_prompt_inputs, mocker
    ):
        """Test error handling during prompt building in _execute_agent_turn."""
        orchestrator = orchestrator_instance
        orchestrator.prompt_inputs = default_prompt_inputs

        # Mock the prompt builder to raise an error
        error_msg = "Test prompt error"
        mock_builder = mocker.Mock(side_effect=ValueError(error_msg))

        # Execute and verify error wrapping
        with pytest.raises(LLMOrchestrationError) as excinfo:
            orchestrator._execute_agent_turn(
                role_name="TestRole",
                llm=orchestrator.primary_llm,
                prompt_builder=mock_builder,
                agent_conv=agent_conversation,
                other_convs=[],
            )

        assert "Failed to build prompt" in str(excinfo.value)
        assert error_msg in str(excinfo.value)


# --- Phase 6: Generation and Verification Logic Tests ---


@pytest.fixture
def simple_schema() -> dict:
    """Provides a simple JSON schema for validation testing."""
    return {
        "type": "object",
        "required": ["id", "name", "status"],
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
    }


class TestGenerationVerification:
    """Tests for _generate_and_verify_result method."""

    def test_json_success_path(
        self, orchestrator_instance, default_prompt_inputs, simple_schema, mocker
    ):
        """Test successful generation and validation of JSON result."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM to return valid JSON matching the schema
        valid_json_response = '{"id": 123, "name": "Test Item", "status": "active"}'
        mock_llm.query.return_value = valid_json_response

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a simple fix prompt builder that should NOT be called in success case
        mock_fix_builder = mocker.Mock(return_value="Fix prompt that shouldn't be used")

        # Execute the method
        result = orchestrator._generate_and_verify_result(
            llm_service=mock_llm,
            context="Test JSON Generation",
            main_prompt_or_template="Generate a valid item",
            fix_prompt_builder=mock_fix_builder,
            result_schema=simple_schema,
            use_conversation=False,
            max_improvement_iterations=3,
            json_result=True,
        )

        # Verify result is parsed JSON and matches expected structure
        assert isinstance(result, dict)
        assert result["id"] == 123
        assert result["name"] == "Test Item"
        assert result["status"] == "active"

        # Verify LLM was called once with correct params
        mock_llm.query.assert_called_once_with("Generate a valid item", use_conversation=False)

        # Verify fix builder was NOT called (since first response was valid)
        mock_fix_builder.assert_not_called()

    def test_non_json_success_path(self, orchestrator_instance, default_prompt_inputs, mocker):
        """Test successful generation of non-JSON result (no validation needed)."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM to return text
        text_response = "This is a plain text response."
        mock_llm.query.return_value = text_response

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a simple fix prompt builder that should NOT be called in success case
        mock_fix_builder = mocker.Mock(return_value="Fix prompt that shouldn't be used")

        # Execute the method
        result = orchestrator._generate_and_verify_result(
            llm_service=mock_llm,
            context="Test Text Generation",
            main_prompt_or_template="Generate a text response",
            fix_prompt_builder=mock_fix_builder,
            result_schema=None,  # No schema for text
            use_conversation=False,
            max_improvement_iterations=3,
            json_result=False,  # Expecting non-JSON
        )

        # Verify result is the raw text
        assert result == text_response

        # Verify LLM was called once
        mock_llm.query.assert_called_once()

        # Verify fix builder was NOT called
        mock_fix_builder.assert_not_called()

    def test_correction_loop_success(
        self, orchestrator_instance, default_prompt_inputs, simple_schema, mocker
    ):
        """Test the correction loop: Initial error + successful correction."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM to return invalid JSON first, then valid on retry
        # This will test the error-correction loop
        invalid_json = '{"id": "not_an_integer", "status": "unknown"}'  # Missing name, invalid id type, invalid status
        valid_json = '{"id": 456, "name": "Fixed Item", "status": "active"}'

        # Create a side effect that returns invalid JSON first, then valid JSON
        mock_llm.query.side_effect = [invalid_json, valid_json]

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a fix prompt builder that will be called after the first error
        fix_prompt = "Please fix the JSON errors."
        mock_fix_builder = mocker.Mock(return_value=fix_prompt)

        # Import the original function to use in side_effect
        from convorator.conversations import utils

        original_parse_json_response = utils.parse_json_response

        # Create a mock parser with the CORRECT patch target
        parse_patch = mocker.patch(
            "convorator.conversations.conversation_orchestrator.parse_json_response"
        )

        # First call raises SchemaValidationError, second calls original function
        schema_error = SchemaValidationError(
            "JSON validation failed: id must be integer, missing required property 'name', status must be one of [active, inactive]"
        )

        # Configure the mock to raise an error on first call, then return the parsed valid JSON on the second
        # Provide a context string and the correct logger for the direct call to the original function
        parsed_valid_json = original_parse_json_response(
            logger=orchestrator_instance.logger,  # Pass the logger first
            response=valid_json,
            schema=simple_schema,
            context="Test context",
        )
        parse_patch.side_effect = [schema_error, parsed_valid_json]  # Use the actual parsed result

        # Execute the method
        result = orchestrator._generate_and_verify_result(
            llm_service=mock_llm,
            context="Test Correction Loop",
            main_prompt_or_template="Generate an item with correction",
            fix_prompt_builder=mock_fix_builder,
            result_schema=simple_schema,
            use_conversation=False,
            max_improvement_iterations=3,
            json_result=True,
        )

        # Verify result is the corrected JSON
        assert isinstance(result, dict)
        assert result["id"] == 456
        assert result["name"] == "Fixed Item"
        assert result["status"] == "active"

        # Verify LLM was called twice with the right prompts
        assert mock_llm.query.call_count == 2
        mock_llm.query.assert_any_call("Generate an item with correction", use_conversation=False)
        mock_llm.query.assert_any_call(fix_prompt, use_conversation=False)

        # Verify fix builder was called once with correct info
        mock_fix_builder.assert_called_once()
        # Verify prompt_inputs was updated with error info
        assert orchestrator.prompt_inputs.response_to_fix == invalid_json
        assert "JSON validation failed" in orchestrator.prompt_inputs.errors_to_fix

    def test_max_iterations_exceeded(
        self, orchestrator_instance, default_prompt_inputs, simple_schema, mocker
    ):
        """Test hitting the max improvement iterations limit."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM to always return invalid JSON
        invalid_json = '{"id": "still_not_integer", "status": "still_bad"}'  # Persistently invalid
        mock_llm.query.return_value = invalid_json

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a fix prompt builder that will be called repeatedly
        mock_fix_builder = mocker.Mock(return_value="Please fix the JSON errors.")

        # Mock the parser with the CORRECT patch target
        parse_patch = mocker.patch(
            "convorator.conversations.conversation_orchestrator.parse_json_response"
        )

        # Configure parser to always raise the same error
        schema_error = SchemaValidationError("JSON validation fails repeatedly")
        parse_patch.side_effect = schema_error

        # Execute the method and expect MaxIterationsExceededError
        with pytest.raises(MaxIterationsExceededError) as excinfo:
            orchestrator._generate_and_verify_result(
                llm_service=mock_llm,
                context="Test Max Iterations",
                main_prompt_or_template="Generate with persistent errors",
                fix_prompt_builder=mock_fix_builder,
                result_schema=simple_schema,
                use_conversation=False,
                max_improvement_iterations=2,  # Set low for test efficiency
                json_result=True,
            )

        # Verify error message contains both the max iterations info and the original error
        assert "Failed to produce a valid result after 2 attempts" in str(excinfo.value)
        assert "JSON validation fails repeatedly" in str(excinfo.value)

        # Verify LLM was called exactly max_improvement_iterations times
        assert mock_llm.query.call_count == 2

        # Verify fix builder was called max_improvement_iterations-1 times
        assert mock_fix_builder.call_count == 1  # Called after first failure

    def test_llm_client_error_propagation(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test that LLMClientError is propagated immediately as a non-recoverable error."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM to raise LLMClientError
        error_msg = "API connection failure"
        mock_llm.query.side_effect = LLMClientError(error_msg)

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a fix prompt builder that should never be called
        mock_fix_builder = mocker.Mock()

        # Execute and verify the error is propagated directly
        with pytest.raises(LLMClientError) as excinfo:
            orchestrator._generate_and_verify_result(
                llm_service=mock_llm,
                context="Test Client Error",
                main_prompt_or_template="Prompt causing client error",
                fix_prompt_builder=mock_fix_builder,
                use_conversation=False,
                max_improvement_iterations=3,
                json_result=True,
            )

        assert error_msg in str(excinfo.value)

        # Verify fix builder was NOT called (as this is a non-recoverable error)
        mock_fix_builder.assert_not_called()

    def test_token_limit_check(self, orchestrator_instance, default_prompt_inputs, mocker):
        """Test the token limit check for stateless prompts."""
        orchestrator = orchestrator_instance
        mock_llm = orchestrator.solution_generation_llm

        # Configure mock LLM with specific limits
        context_limit = 100
        mock_llm.get_context_limit.return_value = context_limit
        mock_llm.max_tokens = 20  # This means buffer will be 20+10=30

        # Create a counting function that will make the prompt exceed the limit
        def token_count_exceeding(text):
            # Return a value that makes any prompt exceed available tokens
            # Available tokens = context_limit - buffer = 100 - 30 = 70
            return 80  # Any prompt will exceed the limit

        mock_llm.count_tokens.side_effect = token_count_exceeding

        # Ensure prompt_inputs is initialized
        orchestrator.prompt_inputs = default_prompt_inputs

        # Create a fix prompt builder that should never be called
        mock_fix_builder = mocker.Mock()

        # Execute and verify it raises MaxIterationsExceededError due to token limit
        with pytest.raises(MaxIterationsExceededError) as excinfo:
            orchestrator._generate_and_verify_result(
                llm_service=mock_llm,
                context="Test Token Limit",
                main_prompt_or_template="A prompt that will exceed token limit",
                fix_prompt_builder=mock_fix_builder,
                use_conversation=False,  # Ensuring stateless prompt checks happen
                max_improvement_iterations=2,
                json_result=False,
            )

        assert "Single prompt exceeded token limit" in str(excinfo.value)

        # Query should never be called as the token check prevents it
        mock_llm.query.assert_not_called()

        # Fix builder should never be called since we fail on the token check
        # before ever making a query or encountering a response that needs fixing
        mock_fix_builder.assert_not_called()


# --- Phase 7: Debate Logic Tests ---


class TestDebateFlow:
    """Tests for the _run_moderated_debate method."""

    @pytest.mark.parametrize("debate_iterations", [1, 2, 3])
    def test_debate_flow_standard_iterations(
        self, orchestrator_instance, default_prompt_inputs, mocker, debate_iterations
    ):
        """Test the debate flow for a positive number of iterations."""
        mocker.resetall()  # Explicitly reset mocks for isolation
        orchestrator = orchestrator_instance
        # Set iterations directly on the instance
        orchestrator.debate_iterations = debate_iterations
        orchestrator.prompt_inputs = default_prompt_inputs

        # Mock prompt builders
        test_initial_user_prompt = "User Prompt Debate Test"
        mock_user_prompt_builder = mocker.patch.object(
            orchestrator, "build_debate_user_prompt", return_value=test_initial_user_prompt
        )
        mock_primary_builder = mocker.patch.object(
            orchestrator, "build_primary_prompt", return_value="Primary Prompt Content"
        )
        mock_moderator_builder = mocker.patch.object(
            orchestrator, "build_moderator_context", return_value="Moderator Prompt Content"
        )
        mock_debater_builder = mocker.patch.object(
            orchestrator, "build_debater_prompt", return_value="Debater Prompt Content"
        )

        # Mock individual LLM query methods
        mock_primary_query = mocker.patch.object(orchestrator.primary_llm, "query")
        mock_debater_query = mocker.patch.object(orchestrator.debater_llm, "query")
        mock_moderator_query = mocker.patch.object(orchestrator.moderator_llm, "query")

        # Define side effects for LLM queries based on expected call sequence
        debater_responses = ["Initial Debater Response"]
        primary_responses = []
        moderator_responses = []

        for i in range(debate_iterations):
            primary_responses.append(f"Primary Response {i+1}")
            moderator_responses.append(f"Moderator Feedback {i+1}")
            if i < debate_iterations - 1:
                debater_responses.append(f"Debater Response {i+1}")

        mock_debater_query.side_effect = debater_responses
        mock_primary_query.side_effect = primary_responses
        mock_moderator_query.side_effect = moderator_responses

        # Execute the debate
        initial_prompt_content = "Ignored Initial Prompt Content for Context"
        debate_history = orchestrator._run_moderated_debate(initial_prompt_content)

        # --- Assertions ---

        # 1. Prompt Builder Calls
        mock_user_prompt_builder.assert_called_once_with(orchestrator.prompt_inputs)
        assert mock_primary_builder.call_count == debate_iterations
        assert mock_moderator_builder.call_count == debate_iterations
        assert mock_debater_builder.call_count == debate_iterations - 1

        # 2. LLM Query Calls (Check counts)
        assert mock_debater_query.call_count == len(debater_responses)
        assert mock_primary_query.call_count == len(primary_responses)
        assert mock_moderator_query.call_count == len(moderator_responses)

        # 3. Verify the main conversation log contains expected messages
        # Build expected messages based on the side effects
        expected_messages = []
        expected_messages.append({"role": "user", "content": test_initial_user_prompt})
        expected_messages.append(
            {"role": orchestrator.debater_name, "content": debater_responses[0]}
        )

        debater_response_idx = 1
        for i in range(debate_iterations):
            expected_messages.append(
                {"role": orchestrator.primary_name, "content": primary_responses[i]}
            )
            expected_messages.append(
                {"role": orchestrator.moderator_name, "content": moderator_responses[i]}
            )
            if i < debate_iterations - 1:
                expected_messages.append(
                    {
                        "role": orchestrator.debater_name,
                        "content": debater_responses[debater_response_idx],
                    }
                )
                debater_response_idx += 1

        actual_messages = orchestrator.main_conversation_log.get_messages()
        assert len(actual_messages) == len(expected_messages)
        assert actual_messages == expected_messages

        # Verify returned history matches the log
        assert debate_history == actual_messages

    def test_debate_flow_zero_iterations(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test the debate flow when debate_iterations is 0."""
        mocker.resetall()  # Explicitly reset mocks for isolation
        orchestrator = orchestrator_instance
        orchestrator.debate_iterations = 0  # Set directly on the instance
        orchestrator.prompt_inputs = default_prompt_inputs

        # Mock lower-level components
        test_initial_user_prompt = "User Prompt Zero Iter"
        mock_user_prompt_builder = mocker.patch.object(
            orchestrator, "build_debate_user_prompt", return_value=test_initial_user_prompt
        )
        # Mock the debater's query method directly
        mock_debater_query = mocker.patch.object(
            orchestrator.debater_llm, "query", return_value="Init Debater Only"
        )
        # Mock the prompt builders that should NOT be called
        mock_primary_builder = mocker.patch.object(orchestrator, "build_primary_prompt")
        mock_moderator_builder = mocker.patch.object(orchestrator, "build_moderator_context")
        mock_debater_builder = mocker.patch.object(orchestrator, "build_debater_prompt")
        # Mock other LLM queries that should NOT be called
        mock_primary_query = mocker.patch.object(orchestrator.primary_llm, "query")
        mock_moderator_query = mocker.patch.object(orchestrator.moderator_llm, "query")

        # Execute the debate
        debate_history = orchestrator._run_moderated_debate("Initial Context Zero Iter")

        # --- Assertions ---

        # 1. Prompt Builder Calls
        mock_user_prompt_builder.assert_called_once_with(orchestrator.prompt_inputs)
        mock_primary_builder.assert_not_called()
        mock_moderator_builder.assert_not_called()
        # The regular debater prompt builder is NOT called for the initial turn
        mock_debater_builder.assert_not_called()

        # 2. LLM Query Calls
        # Should only be called once for the initial debater
        mock_debater_query.assert_called_once()
        # Extract arguments passed to the actual mock_debater_query call
        # The `query` method expects specific arguments like prompt, use_conversation, conversation_history
        # We need to assert based on how _query_agent_and_update_logs calls it.
        # Since _query_agent_and_update_logs prepares history and calls query(prompt="", use_conversation=False, ...)
        # let's check the conversation history passed implicitly via _prepare_history_for_llm
        # For simplicity, let's just assert it was called once.

        mock_primary_query.assert_not_called()
        mock_moderator_query.assert_not_called()

        # 3. Verify log state
        actual_messages = orchestrator.main_conversation_log.get_messages()
        assert len(actual_messages) == 2  # user + init_debater
        assert actual_messages[0]["role"] == "user"
        assert actual_messages[0]["content"] == test_initial_user_prompt
        assert actual_messages[1]["role"] == orchestrator.debater_name
        assert actual_messages[1]["content"] == "Init Debater Only"
        assert debate_history == actual_messages

    def test_debate_flow_error_propagation(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test that an error during a turn propagates correctly."""
        mocker.resetall()  # Explicitly reset mocks for isolation
        orchestrator = orchestrator_instance
        orchestrator.config.debate_iterations = 2  # Set iterations > 0
        orchestrator.prompt_inputs = default_prompt_inputs

        # Revert to patching instance methods
        error_message = "Primary turn failed!"
        mock_init_debater = mocker.patch.object(
            orchestrator, "_run_initial_debater_turn", return_value="Initial Debater Response"
        )
        mock_primary = mocker.patch.object(
            orchestrator, "_run_primary_turn", side_effect=LLMResponseError(error_message)
        )
        mock_moderator = mocker.patch.object(
            orchestrator, "_run_moderator_turn"
        )  # Should not be called
        mock_debater = mocker.patch.object(
            orchestrator, "_run_debater_turn"
        )  # Should not be called

        # Mock initial user prompt builder
        mock_user_prompt_builder = mocker.patch.object(
            orchestrator, "build_debate_user_prompt", return_value="User Prompt Error Test"
        )

        # Execute and expect the error
        with pytest.raises(LLMResponseError) as excinfo:
            orchestrator._run_moderated_debate("Initial Context Error Test")

        assert error_message in str(excinfo.value)

        # Verify calls up to the point of error
        mock_user_prompt_builder.assert_called_once()
        mock_init_debater.assert_called_once()
        mock_primary.assert_called_once()  # Failed on first call
        mock_moderator.assert_not_called()
        mock_debater.assert_not_called()


# --- Phase 8: Summary and Improvement Prompt Tests ---


class TestSummaryAndImprovement:
    """Tests for summary synthesis and improvement prompt building."""

    # Sample debate history for testing summary
    sample_debate_history = [
        {"role": "user", "content": "Start prompt"},
        {"role": "Debater", "content": "Initial critique"},
        {"role": "Primary", "content": "Response to critique"},
        {"role": "Moderator", "content": "Guidance"},
    ]

    def test_synthesize_summary_success(self, orchestrator_instance, default_prompt_inputs, mocker):
        """Test successful summary synthesis by the moderator."""
        orchestrator = orchestrator_instance
        orchestrator.prompt_inputs = default_prompt_inputs
        mock_moderator_llm = orchestrator.moderator_llm

        # Mock the summary prompt builder
        summary_prompt_content = "Please summarize the preceding debate."
        mock_summary_builder = mocker.patch.object(
            orchestrator, "build_summary_prompt", return_value=summary_prompt_content
        )

        # Mock history preparation (return a simplified history for verification)
        prepared_history_for_llm = self.sample_debate_history + [
            {"role": "user", "content": summary_prompt_content}
        ]
        mock_prepare_history = mocker.patch.object(
            orchestrator, "_prepare_history_for_llm", return_value=prepared_history_for_llm
        )

        # Mock the moderator LLM query
        expected_summary = "This is the synthesized summary of the debate."
        mock_moderator_llm.query.return_value = expected_summary

        # Execute the method
        summary = orchestrator._synthesize_summary(self.sample_debate_history)

        # Assertions
        assert summary == expected_summary
        mock_summary_builder.assert_called_once_with(orchestrator.prompt_inputs)
        mock_prepare_history.assert_called_once_with(
            llm_service=mock_moderator_llm,
            messages=self.sample_debate_history
            + [{"role": "user", "content": summary_prompt_content}],
            context="Moderator Summary Synthesis",
        )
        mock_moderator_llm.query.assert_called_once_with(
            prompt="", use_conversation=False, conversation_history=prepared_history_for_llm
        )
        # Check prompt_inputs was updated
        assert orchestrator.prompt_inputs.moderator_summary == expected_summary

    def test_synthesize_summary_llm_error(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test error propagation when moderator LLM fails during summary."""
        orchestrator = orchestrator_instance
        orchestrator.prompt_inputs = default_prompt_inputs
        mock_moderator_llm = orchestrator.moderator_llm

        # Mock the summary prompt builder
        mock_summary_builder = mocker.patch.object(
            orchestrator, "build_summary_prompt", return_value="Summarize prompt"
        )
        # Mock history prep
        mocker.patch.object(orchestrator, "_prepare_history_for_llm", return_value=[])

        # Mock the moderator LLM query to raise error
        error_message = "Failed to summarize!"
        mock_moderator_llm.query.side_effect = LLMResponseError(error_message)

        # Execute and assert error
        with pytest.raises(LLMResponseError) as excinfo:
            orchestrator._synthesize_summary(self.sample_debate_history)

        assert error_message in str(excinfo.value)
        mock_summary_builder.assert_called_once()
        mock_moderator_llm.query.assert_called_once()  # Ensure query was attempted

    def test_build_improvement_prompt_success(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test successful building of the solution improvement prompt."""
        orchestrator = orchestrator_instance
        orchestrator.prompt_inputs = default_prompt_inputs
        moderator_summary = "Summary to use for improvement."

        # Mock the improvement prompt builder
        expected_improvement_prompt = "Based on the summary, improve the solution like this..."
        mock_improvement_builder = mocker.patch.object(
            orchestrator, "build_improvement_prompt", return_value=expected_improvement_prompt
        )

        # Execute the method
        improvement_prompt = orchestrator._build_improvement_prompt(moderator_summary)

        # Assertions
        assert improvement_prompt == expected_improvement_prompt
        # Verify prompt_inputs was updated BEFORE calling the builder
        assert orchestrator.prompt_inputs.moderator_summary == moderator_summary
        mock_improvement_builder.assert_called_once_with(orchestrator.prompt_inputs)

    def test_build_improvement_prompt_builder_error(
        self, orchestrator_instance, default_prompt_inputs, mocker
    ):
        """Test error handling when the improvement prompt builder fails."""
        orchestrator = orchestrator_instance
        orchestrator.prompt_inputs = default_prompt_inputs
        moderator_summary = "Summary causing builder error."
        error_message = "Improvement builder failed!"

        # Mock the improvement prompt builder to raise an error
        mock_improvement_builder = mocker.patch.object(
            orchestrator, "build_improvement_prompt", side_effect=ValueError(error_message)
        )

        # Execute and assert error
        with pytest.raises(LLMOrchestrationError) as excinfo:
            orchestrator._build_improvement_prompt(moderator_summary)

        assert "Failed to build solution improvement prompt" in str(excinfo.value)
        assert error_message in str(excinfo.value)
        # Verify prompt_inputs was updated even though builder failed
        assert orchestrator.prompt_inputs.moderator_summary == moderator_summary
        mock_improvement_builder.assert_called_once()  # Ensure builder was attempted


# --- Phase 9: End-to-End Orchestrator Run Tests ---


class TestEndToEndRun:
    """Tests for the main run() method orchestrating the workflow."""

    @pytest.fixture(autouse=True)
    def patch_internal_methods(self, orchestrator_instance, mocker):
        """Automatically mock all internal step methods for end-to-end tests."""
        # Mock methods called by run()
        self.mock_prepare_inputs = mocker.patch.object(
            orchestrator_instance, "_prepare_prompt_inputs", return_value=None
        )
        self.mock_build_initial = mocker.patch.object(
            orchestrator_instance,
            "_build_initial_prompt_content",
            return_value="Initial Prompt Content",
        )
        self.mock_run_debate = mocker.patch.object(
            orchestrator_instance,
            "_run_moderated_debate",
            return_value=[{"role": "user", "content": "Debate finished"}],
        )
        self.mock_synthesize = mocker.patch.object(
            orchestrator_instance, "_synthesize_summary", return_value="Moderation Summary"
        )
        self.mock_build_improve = mocker.patch.object(
            orchestrator_instance, "_build_improvement_prompt", return_value="Improvement Prompt"
        )
        self.mock_generate_final = mocker.patch.object(
            orchestrator_instance,
            "_generate_final_solution",
            return_value={"final_result": "success"},
        )
        # Store orchestrator instance for use in tests
        self.orchestrator = orchestrator_instance

    def test_run_success_path(self):
        """Test the successful end-to-end execution flow of the run() method."""
        # Expected final result comes from the mock _generate_final_solution
        expected_result = {"final_result": "success"}

        # Execute the main run method
        final_solution = self.orchestrator.run()

        # Assert the final result is correct
        assert final_solution == expected_result

        # Assert all internal methods were called once in order
        call_order = [
            self.mock_prepare_inputs,
            self.mock_build_initial,
            self.mock_run_debate,
            self.mock_synthesize,
            self.mock_build_improve,
            self.mock_generate_final,
        ]
        for mock_method in call_order:
            mock_method.assert_called_once()

        # More specific argument checks if needed (already tested in previous phases)
        # e.g., self.mock_run_debate.assert_called_once_with("Initial Prompt Content")
        # e.g., self.mock_synthesize.assert_called_once_with([{"role": "user", "content": "Debate finished"}])
        # e.g., self.mock_build_improve.assert_called_once_with("Moderation Summary")
        # e.g., self.mock_generate_final.assert_called_once_with("Improvement Prompt")

    @pytest.mark.parametrize(
        "failing_method_name, raised_exception",
        [
            ("_prepare_prompt_inputs", LLMOrchestrationError("Prepare failed")),
            ("_build_initial_prompt_content", LLMOrchestrationError("Build initial failed")),
            ("_run_moderated_debate", LLMResponseError("Debate failed")),
            ("_synthesize_summary", LLMResponseError("Summary failed")),
            ("_build_improvement_prompt", LLMOrchestrationError("Build improve failed")),
            ("_generate_final_solution", MaxIterationsExceededError("Generate failed")),
        ],
    )
    def test_run_error_propagation(self, failing_method_name, raised_exception):
        """Test that errors from internal steps are propagated correctly by run()."""
        # Verify calls up to the point of failure
        all_mocks = {
            "_prepare_prompt_inputs": self.mock_prepare_inputs,
            "_build_initial_prompt_content": self.mock_build_initial,
            "_run_moderated_debate": self.mock_run_debate,
            "_synthesize_summary": self.mock_synthesize,
            "_build_improvement_prompt": self.mock_build_improve,
            "_generate_final_solution": self.mock_generate_final,
        }

        # Get the specific mock method that should fail using the pre-defined mapping
        failing_mock = all_mocks[failing_method_name]
        failing_mock.side_effect = raised_exception

        # Execute run() and expect the exception
        with pytest.raises(type(raised_exception)) as excinfo:
            self.orchestrator.run()

        assert str(raised_exception) in str(excinfo.value)

        # Verify calls up to the point of failure (moved the all_mocks definition earlier)
        encountered_failure = False
        for method_name, mock_method in all_mocks.items():
            if method_name == failing_method_name:
                mock_method.assert_called_once()  # The failing method itself was called
                encountered_failure = True
            elif not encountered_failure:
                mock_method.assert_called_once()  # Methods before failure were called
            else:
                mock_method.assert_not_called()  # Methods after failure were not called

    def test_run_unexpected_error_wrapping(self):
        """Test that unexpected errors are wrapped in LLMOrchestrationError."""
        # Configure one of the mocks to raise a generic Exception
        self.mock_run_debate.side_effect = ValueError("Unexpected issue during debate")

        with pytest.raises(LLMOrchestrationError) as excinfo:
            self.orchestrator.run()

        assert "Unexpected error during orchestration" in str(excinfo.value)
        assert "Unexpected issue during debate" in str(excinfo.value.__cause__)


# --- Phase 10: Public API Wrapper Test ---

# Import the function under test
from convorator.conversations.conversation_orchestrator import improve_solution_with_moderation


class TestPublicAPI:
    """Tests for the improve_solution_with_moderation wrapper function."""

    def test_wrapper_calls_orchestrator_run(self, valid_orchestrator_config, mocker):
        """Verify the wrapper initializes Orchestrator and calls its run method."""
        # Mock the SolutionImprovementOrchestrator class within its module
        mock_orchestrator_class = mocker.patch(
            "convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator"
        )

        # Configure the mock instance that the class constructor will return
        mock_instance = mock_orchestrator_class.return_value
        expected_final_result = {"result": "from mock run"}
        mock_instance.run.return_value = expected_final_result

        # Prepare arguments for the wrapper function
        config = valid_orchestrator_config
        initial_solution = {"id": 999}
        requirements = "Test Requirements"
        assessment_criteria = "Test Criteria"

        # Call the wrapper function
        actual_result = improve_solution_with_moderation(
            config=config,
            initial_solution=initial_solution,
            requirements=requirements,
            assessment_criteria=assessment_criteria,
        )

        # Assertions
        # 1. Check Orchestrator was initialized correctly
        mock_orchestrator_class.assert_called_once_with(
            config=config,
            initial_solution=initial_solution,
            requirements=requirements,
            assessment_criteria=assessment_criteria,
        )

        # 2. Check the run method was called on the instance
        mock_instance.run.assert_called_once_with()

        # 3. Check the result from run() was returned
        assert actual_result == expected_final_result

    def test_wrapper_propagates_exception(self, valid_orchestrator_config, mocker):
        """Verify the wrapper propagates exceptions from orchestrator.run()."""
        # Mock the SolutionImprovementOrchestrator class
        mock_orchestrator_class = mocker.patch(
            "convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator"
        )

        # Configure the mock instance's run method to raise an error
        mock_instance = mock_orchestrator_class.return_value
        error_to_raise = LLMOrchestrationError("Error during run")
        mock_instance.run.side_effect = error_to_raise

        # Prepare arguments
        config = valid_orchestrator_config
        initial_solution = {"id": 999}
        requirements = "Test Requirements"
        assessment_criteria = "Test Criteria"

        # Call the wrapper and assert the exception is raised
        with pytest.raises(LLMOrchestrationError) as excinfo:
            improve_solution_with_moderation(
                config=config,
                initial_solution=initial_solution,
                requirements=requirements,
                assessment_criteria=assessment_criteria,
            )

        assert str(error_to_raise) in str(excinfo.value)

        # Verify constructor and run were still called
        mock_orchestrator_class.assert_called_once()
        mock_instance.run.assert_called_once()


# --- Phase 11: Cleanup and Finalization ---
# No specific cleanup needed for this test suite.
# All resources are managed by pytest fixtures and mocks.
# --- End of Test Suite ---


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
