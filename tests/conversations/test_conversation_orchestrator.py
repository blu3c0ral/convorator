import unittest
from unittest.mock import MagicMock, patch, call, Mock
from datetime import datetime, timezone
import uuid
from convorator.conversations.events import (
    EnhancedMessageMetadata,
    EventType,
    OrchestrationStage,
    MessageEntityType,
    MessagePayloadType,
)
import logging

from convorator.conversations.conversation_orchestrator import (
    SolutionImprovementOrchestrator,
    improve_solution_with_moderation,
)
from convorator.conversations.configurations import OrchestratorConfig
from convorator.conversations.types import SolutionLLMGroup
from convorator.client.llm_client import LLMInterface, Conversation, Message
from convorator.conversations.state import MultiAgentConversation
from convorator.conversations.prompts import PromptBuilder, PromptBuilderInputs
from convorator.exceptions import (
    LLMOrchestrationError,
    LLMResponseError,
    SchemaValidationError,
    MaxIterationsExceededError,
    LLMClientError,
)


class TestSolutionImprovementOrchestrator(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock(spec=OrchestratorConfig)

        # Mock LLMInterface instances
        self.mock_primary_llm = MagicMock(spec=LLMInterface)
        self.mock_primary_llm.get_system_message.return_value = "Primary System Message"
        self.mock_primary_llm.get_role_name.return_value = "TestPrimaryAgent"

        self.mock_debater_llm = MagicMock(spec=LLMInterface)
        self.mock_debater_llm.get_system_message.return_value = "Debater System Message"
        self.mock_debater_llm.get_role_name.return_value = "TestDebaterAgent"

        self.mock_moderator_llm = MagicMock(spec=LLMInterface)
        self.mock_moderator_llm.get_system_message.return_value = "Moderator System Message"
        self.mock_moderator_llm.get_role_name.return_value = "TestModeratorAgent"

        self.mock_solution_llm = MagicMock(spec=LLMInterface)
        self.mock_solution_llm.get_system_message.return_value = "Solution System Message"
        self.mock_solution_llm.get_role_name.return_value = "TestSolutionAgent"

        # Mock SolutionLLMGroup (as an attribute of config)
        self.mock_llm_group = MagicMock(spec=SolutionLLMGroup)
        self.mock_llm_group.primary_llm = self.mock_primary_llm
        self.mock_llm_group.debater_llm = self.mock_debater_llm
        self.mock_llm_group.moderator_llm = self.mock_moderator_llm
        self.mock_llm_group.solution_generation_llm = self.mock_solution_llm

        self.mock_config.llm_group = self.mock_llm_group

        self.mock_config.logger = MagicMock(spec=logging.Logger)
        self.mock_config.messaging_callback = MagicMock()
        self.mock_config.debate_iterations = 3
        self.mock_config.improvement_iterations = 2
        self.mock_config.moderator_instructions_override = "Test Moderator Instructions"
        self.mock_config.debate_context_override = "Test Debate Context"

        self.mock_prompt_builder = MagicMock(spec=PromptBuilder)
        self.mock_config.prompt_builder = self.mock_prompt_builder

        # Values that *can* be overridden by direct args to SolutionImprovementOrchestrator constructor
        self.mock_config.initial_solution = {"config_initial": "solution"}
        self.mock_config.requirements = "Config requirements"
        self.mock_config.assessment_criteria = "Config assessment criteria"

        # Other OrchestratorConfig fields that might be accessed
        self.mock_config.topic = "Test Topic From Config"
        self.mock_config.solution_schema = {"type": "object"}
        self.mock_config.expect_json_output = True
        self.mock_config.session_id = "test_session_123"

        # Direct arguments that will be used to initialize self.orchestrator, overriding config values above
        self.initial_solution_arg = {"problem": "test direct arg"}
        self.requirements_arg = "Test requirements direct arg"
        self.assessment_criteria_arg = "Test assessment criteria direct arg"

        # Reset call counts for LLM mocks before self.orchestrator is created
        # as __init__ will call _reset_conversations, which calls get_system_message
        self.mock_primary_llm.get_system_message.reset_mock()
        self.mock_debater_llm.get_system_message.reset_mock()
        self.mock_moderator_llm.get_system_message.reset_mock()

        self.orchestrator = SolutionImprovementOrchestrator(
            config=self.mock_config,
            initial_solution=self.initial_solution_arg,
            requirements=self.requirements_arg,
            assessment_criteria=self.assessment_criteria_arg,
        )
        # Note: __init__ sets current_iteration_num to 0, so we don't need to override it here

    def test_init_attributes_from_config_and_direct_args(self):
        orchestrator = self.orchestrator  # from setUp

        # 1. Config and its direct attributes
        self.assertIs(orchestrator.config, self.mock_config)
        self.assertIs(orchestrator.llm_group, self.mock_llm_group)
        self.assertIs(orchestrator.logger, self.mock_config.logger)
        self.assertIs(orchestrator.messaging_callback, self.mock_config.messaging_callback)
        self.assertEqual(orchestrator.debate_iterations, self.mock_config.debate_iterations)
        self.assertEqual(
            orchestrator.improvement_iterations, self.mock_config.improvement_iterations
        )
        self.assertEqual(
            orchestrator.moderator_instructions, self.mock_config.moderator_instructions_override
        )
        self.assertEqual(orchestrator.debate_context, self.mock_config.debate_context_override)
        self.assertIs(orchestrator.prompt_builder, self.mock_config.prompt_builder)

        # 2. Direct argument overrides (these should take precedence over config values)
        self.assertEqual(orchestrator.initial_solution, self.initial_solution_arg)
        self.assertEqual(orchestrator.requirements, self.requirements_arg)
        self.assertEqual(orchestrator.assessment_criteria, self.assessment_criteria_arg)

        # 3. LLM Extraction from llm_group
        self.assertIs(orchestrator.primary_llm, self.mock_primary_llm)
        self.assertIs(orchestrator.debater_llm, self.mock_debater_llm)
        self.assertIs(orchestrator.moderator_llm, self.mock_moderator_llm)
        self.assertIs(orchestrator.solution_generation_llm, self.mock_solution_llm)

        # 4. Role Name Assignment (using names from setUp mocks)
        self.assertEqual(orchestrator.primary_name, "TestPrimaryAgent")
        self.assertEqual(orchestrator.debater_name, "TestDebaterAgent")
        self.assertEqual(orchestrator.moderator_name, "TestModeratorAgent")

        # 5. llm_dict - should only contain the 3 debate participants (not solution_generation_llm)
        expected_dict = {
            "TestPrimaryAgent": self.mock_primary_llm,
            "TestDebaterAgent": self.mock_debater_llm,
            "TestModeratorAgent": self.mock_moderator_llm,
        }
        self.assertEqual(orchestrator.llm_dict, expected_dict)

        # 6. State Initialization
        self.assertEqual(orchestrator.current_iteration_num, 0)  # __init__ sets this to 0
        self.assertIsNone(
            orchestrator.prompt_inputs
        )  # _prepare_prompt_inputs is not called in __init__

        # 7. Verify _reset_conversations was called (implicitly by __init__)
        # These were reset in setUp before orchestrator creation, so assert_called_once is appropriate.
        self.mock_primary_llm.get_system_message.assert_called_once()
        self.mock_debater_llm.get_system_message.assert_called_once()
        self.mock_moderator_llm.get_system_message.assert_called_once()

        # 8. Verify conversation objects were created properly (side effect of _reset_conversations)
        self.assertIsInstance(orchestrator.primary_conv, Conversation)
        self.assertIsInstance(orchestrator.debater_conv, Conversation)
        self.assertIsInstance(orchestrator.moderator_conv, Conversation)
        self.assertIsInstance(orchestrator.main_conversation_log, MultiAgentConversation)

    def test_init_attributes_fallback_to_config(self):
        # Modify config to have values we expect to be used when direct args are omitted
        self.mock_config.initial_solution = {"from_config": "initial"}
        self.mock_config.requirements = "Requirements from config"
        self.mock_config.assessment_criteria = "Assessment from config"

        # Reset call counts for LLM mocks before new orchestrator instance is created for this test
        self.mock_primary_llm.get_system_message.reset_mock()
        self.mock_debater_llm.get_system_message.reset_mock()
        self.mock_moderator_llm.get_system_message.reset_mock()

        # Instantiate orchestrator WITHOUT direct args for these fields
        orchestrator = SolutionImprovementOrchestrator(
            config=self.mock_config
            # initial_solution, requirements, assessment_criteria are omitted, should use config's
        )

        # Verify fallback values are used
        self.assertEqual(orchestrator.initial_solution, {"from_config": "initial"})
        self.assertEqual(orchestrator.requirements, "Requirements from config")
        self.assertEqual(orchestrator.assessment_criteria, "Assessment from config")

        # Verify _reset_conversations was called for this instance too
        self.mock_primary_llm.get_system_message.assert_called_once()
        self.mock_debater_llm.get_system_message.assert_called_once()
        self.mock_moderator_llm.get_system_message.assert_called_once()

    def test_init_validation_missing_initial_solution(self):
        # Both direct arg and config are None
        self.mock_config.initial_solution = None
        with self.assertRaisesRegex(ValueError, "initial_solution must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=None,  # Explicitly None direct arg
                requirements=self.requirements_arg,
                assessment_criteria=self.assessment_criteria_arg,
            )

    def test_init_validation_missing_requirements(self):
        # Test 1: Both direct arg and config are None
        self.mock_config.requirements = None
        with self.assertRaisesRegex(ValueError, "requirements must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements=None,  # Direct arg is None
                assessment_criteria=self.assessment_criteria_arg,
            )

        # Test 2: Direct arg is None, config has empty string
        self.mock_config.requirements = ""  # Config has empty string
        with self.assertRaisesRegex(ValueError, "requirements must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements=None,  # Direct arg is None, fallback to config's empty string
                assessment_criteria=self.assessment_criteria_arg,
            )

        # Test 3: Direct arg is empty string (overrides valid config)
        self.mock_config.requirements = "Valid From Config"  # Config is valid
        with self.assertRaisesRegex(ValueError, "requirements must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements="",  # Direct arg is empty, overriding valid config
                assessment_criteria=self.assessment_criteria_arg,
            )

    def test_init_validation_missing_assessment_criteria(self):
        # Test 1: Both direct arg and config are None
        self.mock_config.assessment_criteria = None
        with self.assertRaisesRegex(ValueError, "assessment_criteria must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements=self.requirements_arg,
                assessment_criteria=None,
            )

        # Test 2: Direct arg is None, config has empty string
        self.mock_config.assessment_criteria = ""
        with self.assertRaisesRegex(ValueError, "assessment_criteria must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements=self.requirements_arg,
                assessment_criteria=None,  # Fallback to config's empty string
            )

        # Test 3: Direct arg is empty string (overrides valid config)
        self.mock_config.assessment_criteria = "Valid From Config"
        with self.assertRaisesRegex(ValueError, "assessment_criteria must be provided"):
            SolutionImprovementOrchestrator(
                config=self.mock_config,
                initial_solution=self.initial_solution_arg,
                requirements=self.requirements_arg,
                assessment_criteria="",  # Direct arg is empty, overriding valid config
            )

    def test_init_role_name_fallback(self):
        # Set LLM role names to None to test fallback behavior
        self.mock_primary_llm.get_role_name.return_value = None
        self.mock_debater_llm.get_role_name.return_value = None
        self.mock_moderator_llm.get_role_name.return_value = None

        # Reset call counts for LLM mocks before new orchestrator instance
        self.mock_primary_llm.get_system_message.reset_mock()
        self.mock_debater_llm.get_system_message.reset_mock()
        self.mock_moderator_llm.get_system_message.reset_mock()
        self.mock_primary_llm.get_role_name.reset_mock()
        self.mock_debater_llm.get_role_name.reset_mock()
        self.mock_moderator_llm.get_role_name.reset_mock()

        orchestrator = SolutionImprovementOrchestrator(
            config=self.mock_config,
            initial_solution=self.initial_solution_arg,
            requirements=self.requirements_arg,
            assessment_criteria=self.assessment_criteria_arg,
        )

        # Verify fallback to default names
        self.assertEqual(orchestrator.primary_name, "Primary")
        self.assertEqual(orchestrator.debater_name, "Debater")
        self.assertEqual(orchestrator.moderator_name, "Moderator")

        # Verify get_role_name was called for each
        self.mock_primary_llm.get_role_name.assert_called_once()
        self.mock_debater_llm.get_role_name.assert_called_once()
        self.mock_moderator_llm.get_role_name.assert_called_once()

        # Verify llm_dict uses the fallback names
        expected_dict = {
            "Primary": self.mock_primary_llm,
            "Debater": self.mock_debater_llm,
            "Moderator": self.mock_moderator_llm,
        }
        self.assertEqual(orchestrator.llm_dict, expected_dict)

    @patch.object(SolutionImprovementOrchestrator, "_reset_conversations", autospec=True)
    def test_init_explicitly_calls_reset_conversations(self, mock_reset_conversations_method):
        # Create new instance and verify _reset_conversations is called
        SolutionImprovementOrchestrator(
            config=self.mock_config,
            initial_solution=self.initial_solution_arg,
            requirements=self.requirements_arg,
            assessment_criteria=self.assessment_criteria_arg,
        )
        # Verify the method was called exactly once, with the instance as first argument (due to autospec=True)
        mock_reset_conversations_method.assert_called_once()

    def test_current_iteration_num_getter(self):
        # Directly set the backing attribute for testing the getter
        self.orchestrator._current_iteration_num = 5
        self.assertEqual(self.orchestrator.current_iteration_num, 5)

    def test_current_iteration_num_setter_prompt_inputs_none(self):
        # prompt_inputs is None by default after __init__ because _prepare_prompt_inputs isn't called by __init__
        self.assertIsNone(self.orchestrator.prompt_inputs)
        self.orchestrator.current_iteration_num = 10
        self.assertEqual(self.orchestrator._current_iteration_num, 10)
        # No attempt to set on prompt_inputs should occur or raise error

    def test_current_iteration_num_setter_with_prompt_inputs(self):
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        self.orchestrator.current_iteration_num = 7

        self.assertEqual(self.orchestrator._current_iteration_num, 7)
        self.assertEqual(mock_prompt_inputs.current_iteration_num, 7)

    def test_reset_conversations(self):
        # Reset mocks for system message calls because _reset_conversations is called during __init__.
        self.mock_primary_llm.get_system_message.reset_mock()
        self.mock_debater_llm.get_system_message.reset_mock()
        self.mock_moderator_llm.get_system_message.reset_mock()

        initial_primary_conv = self.orchestrator.primary_conv
        initial_main_log = self.orchestrator.main_conversation_log

        self.orchestrator._reset_conversations()

        self.mock_primary_llm.get_system_message.assert_called_once()
        self.mock_debater_llm.get_system_message.assert_called_once()
        self.mock_moderator_llm.get_system_message.assert_called_once()

        self.assertIsInstance(self.orchestrator.primary_conv, Conversation)
        self.assertEqual(self.orchestrator.primary_conv.system_message, "Primary System Message")
        self.assertIsNot(self.orchestrator.primary_conv, initial_primary_conv)

        self.assertIsInstance(self.orchestrator.debater_conv, Conversation)
        self.assertEqual(self.orchestrator.debater_conv.system_message, "Debater System Message")

        self.assertIsInstance(self.orchestrator.moderator_conv, Conversation)
        self.assertEqual(
            self.orchestrator.moderator_conv.system_message, "Moderator System Message"
        )

        self.assertIsInstance(self.orchestrator.main_conversation_log, MultiAgentConversation)
        self.assertEqual(len(self.orchestrator.main_conversation_log.get_messages()), 0)
        self.assertIsNot(self.orchestrator.main_conversation_log, initial_main_log)

        self.assertEqual(
            self.orchestrator.role_conversation[self.orchestrator.primary_name],
            self.orchestrator.primary_conv,
        )
        self.assertEqual(
            self.orchestrator.role_conversation[self.orchestrator.debater_name],
            self.orchestrator.debater_conv,
        )
        self.assertEqual(
            self.orchestrator.role_conversation[self.orchestrator.moderator_name],
            self.orchestrator.moderator_conv,
        )

    @patch("convorator.conversations.conversation_orchestrator.PromptBuilderInputs", autospec=True)
    def test_prepare_prompt_inputs_creates_and_updates_inputs(
        self, mock_prompt_builder_inputs_class
    ):
        # Arrange
        # The mock_prompt_builder_inputs_class is the patched constructor for PromptBuilderInputs
        mock_created_prompt_inputs_instance = MagicMock(spec=PromptBuilderInputs)
        mock_prompt_builder_inputs_class.return_value = mock_created_prompt_inputs_instance

        # Ensure current_iteration_num has a known value before the call
        self.orchestrator.current_iteration_num = 5

        # Act
        self.orchestrator._prepare_prompt_inputs()

        # Assert PromptBuilderInputs constructor call
        mock_prompt_builder_inputs_class.assert_called_once()
        constructor_args, constructor_kwargs = mock_prompt_builder_inputs_class.call_args

        # Assertions on constructor_kwargs (PromptBuilderInputs uses keyword arguments)
        self.assertEqual(constructor_kwargs.get("topic"), self.mock_config.topic)  # From config
        self.assertIs(
            constructor_kwargs.get("logger"), self.orchestrator.logger
        )  # Should be same object
        self.assertIs(
            constructor_kwargs.get("llm_group"), self.orchestrator.llm_group
        )  # Should be same object
        self.assertEqual(
            constructor_kwargs.get("solution_schema"), self.mock_config.solution_schema
        )  # From config
        self.assertEqual(
            constructor_kwargs.get("initial_solution"), self.orchestrator.initial_solution
        )  # From direct arg
        self.assertEqual(
            constructor_kwargs.get("requirements"), self.orchestrator.requirements
        )  # From direct arg
        self.assertEqual(
            constructor_kwargs.get("assessment_criteria"), self.orchestrator.assessment_criteria
        )  # From direct arg
        self.assertEqual(
            constructor_kwargs.get("moderator_instructions"),
            self.orchestrator.moderator_instructions,
        )  # From config override
        self.assertEqual(
            constructor_kwargs.get("debate_context"), self.orchestrator.debate_context
        )  # From config override
        self.assertEqual(
            constructor_kwargs.get("primary_role_name"), self.orchestrator.primary_name
        )
        self.assertEqual(
            constructor_kwargs.get("debater_role_name"), self.orchestrator.debater_name
        )
        self.assertEqual(
            constructor_kwargs.get("moderator_role_name"), self.orchestrator.moderator_name
        )
        self.assertEqual(
            constructor_kwargs.get("expect_json_output"), self.mock_config.expect_json_output
        )  # From config
        self.assertIs(
            constructor_kwargs.get("conversation_history"), self.orchestrator.main_conversation_log
        )  # Should be same object
        self.assertEqual(
            constructor_kwargs.get("debate_iterations"), self.orchestrator.debate_iterations
        )
        self.assertEqual(
            constructor_kwargs.get("current_iteration_num"), 5  # Should be 5 as set above
        )

        # Assert self.orchestrator.prompt_inputs is set to the correct instance
        self.assertIs(self.orchestrator.prompt_inputs, mock_created_prompt_inputs_instance)

        # Assert prompt_builder.update_inputs was called with the correct instance
        self.orchestrator.prompt_builder.update_inputs.assert_called_once_with(
            mock_created_prompt_inputs_instance
        )

        # Assert logger calls
        expected_log_calls = [
            call.debug("Preparing inputs for prompt builders."),
            call.debug("Prompt inputs prepared."),
        ]
        self.orchestrator.logger.debug.assert_has_calls(expected_log_calls, any_order=False)

    def test_execute_agent_loop_turn_primary_agent(self):
        """Test _execute_agent_loop_turn for primary agent - verifies correct prompt type selection and method calls."""
        agent_name = self.orchestrator.primary_name
        expected_prompt_type = "primary_prompt"
        expected_llm = self.orchestrator.primary_llm
        mock_built_prompt = "Generated primary prompt"
        mock_final_response = "Primary LLM Response"

        # Configure mocks
        self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value=mock_final_response,
            autospec=True,
        ) as mock_query_method:
            actual_response = self.orchestrator._execute_agent_loop_turn(agent_name)

        # Verify the prompt builder was called with correct prompt type
        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(expected_prompt_type)

        # Verify _query_agent_in_debate_turn was called with correct parameters
        # Note: build_prompt is called as a parameter to _query_agent_in_debate_turn
        mock_query_method.assert_called_once_with(expected_llm, agent_name, mock_built_prompt)

        # Verify return value
        self.assertEqual(actual_response, mock_final_response)

    def test_execute_agent_loop_turn_debater_agent(self):
        """Test _execute_agent_loop_turn for debater agent - verifies correct prompt type selection and method calls."""
        agent_name = self.orchestrator.debater_name
        expected_prompt_type = "debater_prompt"
        expected_llm = self.orchestrator.debater_llm
        mock_built_prompt = "Generated debater prompt"
        mock_final_response = "Debater LLM Response"

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value=mock_final_response,
            autospec=True,
        ) as mock_query_method:
            actual_response = self.orchestrator._execute_agent_loop_turn(agent_name)

        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(expected_prompt_type)
        mock_query_method.assert_called_once_with(expected_llm, agent_name, mock_built_prompt)
        self.assertEqual(actual_response, mock_final_response)

    def test_execute_agent_loop_turn_moderator_agent(self):
        """Test _execute_agent_loop_turn for moderator agent - verifies correct prompt type selection and method calls."""
        agent_name = self.orchestrator.moderator_name
        expected_prompt_type = "moderator_context"
        expected_llm = self.orchestrator.moderator_llm
        mock_built_prompt = "Generated moderator context prompt"
        mock_final_response = "Moderator LLM Response"

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value=mock_final_response,
            autospec=True,
        ) as mock_query_method:
            actual_response = self.orchestrator._execute_agent_loop_turn(agent_name)

        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(expected_prompt_type)
        mock_query_method.assert_called_once_with(expected_llm, agent_name, mock_built_prompt)
        self.assertEqual(actual_response, mock_final_response)

    def test_execute_agent_loop_turn_unknown_agent(self):
        """Test _execute_agent_loop_turn with unknown agent name - should raise LLMOrchestrationError without calling other methods."""
        unknown_agent_name = "UnknownAgentType"

        with self.assertRaisesRegex(
            LLMOrchestrationError, f"Unknown agent name: {unknown_agent_name}"
        ):
            self.orchestrator._execute_agent_loop_turn(unknown_agent_name)

        # Verify that neither prompt_builder nor _query_agent_in_debate_turn were called
        self.orchestrator.prompt_builder.build_prompt.assert_not_called()

    def test_execute_agent_loop_turn_integration_flow(self):
        """Test the complete integration flow of _execute_agent_loop_turn without mocking internal calls."""
        # This test verifies the actual flow: agent_name -> prompt_type -> build_prompt -> _query_agent_in_debate_turn
        agent_name = self.orchestrator.primary_name
        expected_prompt_type = "primary_prompt"
        expected_llm = self.orchestrator.primary_llm
        mock_built_prompt = "Integration test primary prompt"
        mock_final_response = "Integration test response"

        # Only mock _query_agent_in_debate_turn, let build_prompt run through
        self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value=mock_final_response,
        ) as mock_query_method:
            actual_response = self.orchestrator._execute_agent_loop_turn(agent_name)

        # Verify the integration: prompt was built and query was called with the built prompt
        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(expected_prompt_type)
        mock_query_method.assert_called_once_with(expected_llm, agent_name, mock_built_prompt)
        self.assertEqual(actual_response, mock_final_response)

    def test_execute_agent_loop_turn_llm_dict_lookup(self):
        """Test that _execute_agent_loop_turn correctly looks up LLM from llm_dict."""
        # Test each agent type to ensure llm_dict lookup works correctly
        test_cases = [
            (self.orchestrator.primary_name, self.orchestrator.primary_llm),
            (self.orchestrator.debater_name, self.orchestrator.debater_llm),
            (self.orchestrator.moderator_name, self.orchestrator.moderator_llm),
        ]

        for agent_name, expected_llm in test_cases:
            with self.subTest(agent=agent_name):
                mock_built_prompt = f"Test prompt for {agent_name}"
                mock_response = f"Test response from {agent_name}"

                self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

                with patch.object(
                    self.orchestrator,
                    "_query_agent_in_debate_turn",
                    return_value=mock_response,
                ) as mock_query:
                    result = self.orchestrator._execute_agent_loop_turn(agent_name)

                    # Verify correct LLM was passed (first argument to _query_agent_in_debate_turn)
                    mock_query.assert_called_once()
                    call_args = mock_query.call_args[0]  # Get positional arguments
                    self.assertIs(call_args[0], expected_llm)  # First argument should be the LLM
                    self.assertEqual(
                        call_args[1], agent_name
                    )  # Second argument should be agent_name
                    self.assertEqual(
                        call_args[2], mock_built_prompt
                    )  # Third argument should be built prompt
                    self.assertEqual(result, mock_response)

                # Reset for next iteration
                self.orchestrator.prompt_builder.build_prompt.reset_mock()

    def test_execute_agent_loop_turn_exception_propagation(self):
        """Test that exceptions from _query_agent_in_debate_turn are properly propagated."""
        agent_name = self.orchestrator.primary_name
        mock_built_prompt = "Test prompt that will cause an error"
        test_exception = LLMResponseError("Test LLM error")

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_built_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            side_effect=test_exception,
        ):
            with self.assertRaises(LLMResponseError) as context:
                self.orchestrator._execute_agent_loop_turn(agent_name)

            # Verify the same exception is raised
            self.assertIs(context.exception, test_exception)

    def test_synthesize_summary_happy_path(self):
        """Test _synthesize_summary normal execution flow."""
        # Ensure prompt_inputs is set (usually done by _prepare_prompt_inputs)
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        mock_summary_prompt = "Generated summary prompt"
        mock_llm_summary = "This is the LLM summary."

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_summary_prompt

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value=mock_llm_summary,
            autospec=True,
        ) as mock_query_method:
            self.orchestrator._synthesize_summary()

        # Verify prompt_builder was called correctly
        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with("summary_prompt")

        # Verify _query_agent_in_debate_turn was called correctly
        mock_query_method.assert_called_once_with(
            self.orchestrator.moderator_llm,  # Expected LLM
            self.orchestrator.moderator_name,  # Expected agent name
            mock_summary_prompt,  # Expected prompt
        )

        # Verify prompt_inputs.moderator_summary was updated
        self.assertEqual(self.orchestrator.prompt_inputs.moderator_summary, mock_llm_summary)

        # Verify logger calls
        expected_log_calls = [
            call.info("Step 3: Synthesizing moderation summary."),
            call.debug("Querying moderator LLM for summary."),
            call.debug(f"Moderation summary received: {mock_llm_summary[:200]}..."),
        ]
        self.orchestrator.logger.info.assert_called_once_with(
            "Step 3: Synthesizing moderation summary."
        )
        self.orchestrator.logger.debug.assert_has_calls(
            [
                call("Querying moderator LLM for summary."),
                call(f"Moderation summary received: {mock_llm_summary[:200]}..."),
            ],
            any_order=False,
        )

    def test_synthesize_summary_prompt_inputs_none(self):
        """Test _synthesize_summary when prompt_inputs is None."""
        self.orchestrator.prompt_inputs = None  # Explicitly set to None

        with self.assertRaisesRegex(
            LLMOrchestrationError, "Prompt inputs not prepared before synthesizing summary."
        ):
            self.orchestrator._synthesize_summary()

        # Ensure logger was called before the exception
        self.orchestrator.logger.info.assert_called_once_with(
            "Step 3: Synthesizing moderation summary."
        )
        self.orchestrator.prompt_builder.build_prompt.assert_not_called()

    def test_synthesize_summary_llm_response_error(self):
        """Test _synthesize_summary when _query_agent_in_debate_turn raises LLMResponseError."""
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.moderator_summary = None  # Initialize to None
        mock_summary_prompt = "Summary prompt for error test"
        test_exception = LLMResponseError("LLM failed to summarize")

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_summary_prompt

        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", side_effect=test_exception
        ) as mock_query_method:
            with self.assertRaises(LLMResponseError) as context_manager:
                self.orchestrator._synthesize_summary()

            self.assertIs(context_manager.exception, test_exception)

        # Verify calls up to the point of failure
        self.orchestrator.prompt_builder.build_prompt.assert_called_once_with("summary_prompt")
        mock_query_method.assert_called_once()
        self.orchestrator.logger.error.assert_called_once_with(
            f"Moderator LLM failed to generate summary: {test_exception}", exc_info=True
        )
        # Ensure summary was not set on prompt_inputs
        self.assertIsNone(self.orchestrator.prompt_inputs.moderator_summary)

    def test_synthesize_summary_unexpected_query_exception(self):
        """Test _synthesize_summary when _query_agent_in_debate_turn raises an unexpected error."""
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.moderator_summary = None  # Initialize to None
        mock_summary_prompt = "Summary prompt for unexpected error"
        original_exception = ValueError("Some unexpected internal error during query")

        self.orchestrator.prompt_builder.build_prompt.return_value = mock_summary_prompt

        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", side_effect=original_exception
        ):
            with self.assertRaises(LLMOrchestrationError) as context_manager:
                self.orchestrator._synthesize_summary()

            # Check that the original exception is the cause of the new one
            self.assertIs(context_manager.exception.__cause__, original_exception)
            self.assertIn(
                f"Moderator LLM failed to generate summary: {original_exception}",
                str(context_manager.exception),
            )

        self.orchestrator.logger.error.assert_called_once_with(
            f"Unexpected error querying moderator for summary: {original_exception}", exc_info=True
        )
        self.assertIsNone(self.orchestrator.prompt_inputs.moderator_summary)

    def test_synthesize_summary_prompt_builder_exception(self):
        """Test _synthesize_summary when prompt_builder.build_prompt raises an exception."""
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.moderator_summary = None  # Initialize to None
        build_exception = RuntimeError("Failed to build summary prompt")

        self.orchestrator.prompt_builder.build_prompt.side_effect = build_exception

        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn"
        ) as mock_query_method:  # mock to ensure it's not called
            with self.assertRaises(LLMOrchestrationError) as context_manager:
                self.orchestrator._synthesize_summary()

            self.assertIs(context_manager.exception.__cause__, build_exception)
            self.assertIn(
                f"Moderator LLM failed to generate summary: {build_exception}",
                str(context_manager.exception),
            )

        mock_query_method.assert_not_called()
        self.orchestrator.logger.debug.assert_called_once_with(
            "Querying moderator LLM for summary."
        )
        self.orchestrator.logger.error.assert_called_once_with(
            f"Unexpected error querying moderator for summary: {build_exception}", exc_info=True
        )
        self.assertIsNone(self.orchestrator.prompt_inputs.moderator_summary)

    def test_generate_final_solution_happy_path(self):
        """Test _generate_final_solution normal execution flow."""
        # Ensure prompt_inputs is set
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        mock_improved_solution = {"final_solution": "all good"}

        with patch.object(
            self.orchestrator,
            "_generate_and_verify_result",
            return_value=mock_improved_solution,
            autospec=True,
        ) as mock_generate_verify:
            result = self.orchestrator._generate_final_solution()

        self.assertEqual(result, mock_improved_solution)
        mock_generate_verify.assert_called_once_with(
            llm_service=self.orchestrator.solution_generation_llm,
            context="Final Solution Generation",
            result_schema=self.orchestrator.config.solution_schema,
            use_conversation=False,  # As per current implementation of _generate_final_solution
            max_improvement_iterations=self.orchestrator.improvement_iterations,
            json_result=self.orchestrator.config.expect_json_output,
        )
        # Verify logger calls in order
        expected_calls = [
            call.info(
                f"Step 4: Generating improved solution (max {self.orchestrator.improvement_iterations} attempts)."
            ),
            call.info("Improved solution generated successfully."),
        ]
        self.orchestrator.logger.info.assert_has_calls(expected_calls, any_order=False)

    def test_generate_final_solution_prompt_inputs_none(self):
        """Test _generate_final_solution when prompt_inputs is None."""
        self.orchestrator.prompt_inputs = None

        with patch.object(self.orchestrator, "_generate_and_verify_result") as mock_generate_verify:
            with self.assertRaisesRegex(
                LLMOrchestrationError,
                "Prompt inputs not prepared before generating final solution.",
            ):
                self.orchestrator._generate_final_solution()

        # Verify that _generate_and_verify_result was never called due to early validation failure
        mock_generate_verify.assert_not_called()
        self.orchestrator.logger.info.assert_called_once_with(
            f"Step 4: Generating improved solution (max {self.orchestrator.improvement_iterations} attempts)."
        )

    def test_generate_final_solution_specific_errors_propagated(self):
        """Test that specific errors from _generate_and_verify_result are re-raised directly."""
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        specific_errors = [
            LLMResponseError("LLM failed during generation"),
            SchemaValidationError("Schema validation failed for final solution"),
            MaxIterationsExceededError("Max attempts reached for final solution"),
        ]

        for error_instance in specific_errors:
            with self.subTest(error=error_instance.__class__.__name__):
                with patch.object(
                    self.orchestrator, "_generate_and_verify_result", side_effect=error_instance
                ) as mock_generate_verify:
                    with self.assertRaises(error_instance.__class__) as context_manager:
                        self.orchestrator._generate_final_solution()
                    self.assertIs(context_manager.exception, error_instance)
                    mock_generate_verify.assert_called_once()
                    self.orchestrator.logger.error.assert_called_once_with(
                        f"Failed to generate final verified solution: {error_instance}",
                        exc_info=True,
                    )
                    # Reset mocks for the next iteration in subtest
                    self.orchestrator.logger.reset_mock()
                    mock_generate_verify.reset_mock()  # Reset this mock specifically

    def test_generate_final_solution_general_exception_wrapped(self):
        """Test that general exceptions from _generate_and_verify_result are wrapped."""
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        original_exception = ValueError("Some other unexpected error")

        with patch.object(
            self.orchestrator, "_generate_and_verify_result", side_effect=original_exception
        ) as mock_generate_verify:
            with self.assertRaises(LLMOrchestrationError) as context_manager:
                self.orchestrator._generate_final_solution()

            self.assertIs(context_manager.exception.__cause__, original_exception)
            self.assertIn(
                f"Unexpected error during final solution generation: {original_exception}",
                str(context_manager.exception),
            )
            mock_generate_verify.assert_called_once()
            self.orchestrator.logger.error.assert_called_once_with(
                f"Unexpected error during final solution generation: {original_exception}",
                exc_info=True,
            )

    # --- Tests for _prepare_history_for_llm ---

    def _setup_llm_service_for_history_test(
        self, context_limit, llm_max_tokens, token_counts_side_effect=None
    ):
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.get_context_limit.return_value = context_limit
        mock_llm.max_tokens = llm_max_tokens  # max_tokens is an attribute on LLMInterface

        if token_counts_side_effect:
            if callable(token_counts_side_effect) and not isinstance(
                token_counts_side_effect, MagicMock
            ):
                mock_llm.count_tokens.side_effect = token_counts_side_effect
            else:  # Assume it's a list of return values or a single value
                mock_llm.count_tokens.side_effect = token_counts_side_effect
        else:
            # Default simple token counter (e.g., len of string for simplicity in some tests)
            # More complex tests will provide their own side_effect list or function.
            mock_llm.count_tokens.side_effect = lambda text: len(text) if text else 0
        return mock_llm

    def test_prepare_history_no_truncation_needed(self):
        """Test _prepare_history_for_llm when history fits and no truncation is needed."""
        # Setup: context_limit = 100, llm_max_tokens (for buffer) = 10, buffer = 20
        # System msg (10) + Hist1 (10) + Hist2 (10) + buffer (20) = 50. Limit = 100. Should fit.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100,
            llm_max_tokens=10,  # response_buffer will be 10 + 10 = 20
            token_counts_side_effect=[10, 10, 10],  # sys, hist1, hist2
        )

        messages = [
            {"role": "system", "content": "SysPrompt"},  # 10 tokens
            {"role": "user", "content": "UserMsg1"},  # 10 tokens
            {"role": "assistant", "content": "AssistMsg1"},  # 10 tokens
        ]

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestNoTruncate"
        )

        self.assertEqual(prepared_messages, messages)  # Should return original messages
        self.assertEqual(mock_llm.count_tokens.call_count, 3)
        # Logger debug message verification
        self.orchestrator.logger.debug.assert_any_call(
            "[TestNoTruncate] History size (30 tokens) + buffer (20) fits within limit (100). No truncation needed."
        )
        self.orchestrator.logger.warning.assert_not_called()  # No truncation warnings

    def test_prepare_history_simple_truncation_with_system_message(self):
        """Test truncation when history (with system message) exceeds limit."""
        # context_limit = 100, llm_max_tokens=10 -> response_buffer = 20
        # Sys (30) + U1 (30) + A1 (30) + U2 (30) = 120. Total needed = 120 + 20 = 140. Limit=100
        # Expected: Sys, A1, U2. (U1 should be removed)
        # Token counts: Sys(30), U1(30), A1(30), U2(30)
        # After U1 removed: Sys(30) + A1(30) + U2(30) = 90. With buffer: 90+20 = 110. Still > 100. Oh, wait.

        # Recalculate: response_buffer is llm.max_tokens (10) + 10 = 20.
        # Max history (excluding system) + system + buffer <= context_limit
        # Max history = context_limit - system_tokens - buffer
        #             = 100 - 30 - 20 = 50 tokens for history messages.
        # Initial: Sys(30), U1(30), A1(30), U2(30). Hist tokens = 30+30+30 = 90.
        # Remove U1 (30 tokens). Hist tokens = 30+30 = 60. Still > 50.
        # Remove A1 (30 tokens). Hist tokens = 30. Fits.
        # Expected final messages: Sys, U2

        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100,
            llm_max_tokens=10,  # response_buffer = 20
            # count_tokens will be called for: system, U1, A1, U2 initially. Then for U1 again when removed.
            # then A1 when removed. Total: 4 + 2 = 6 calls for token counting.
            token_counts_side_effect=[30, 30, 30, 30, 30, 30],
        )

        messages = [
            {"role": "system", "content": "S" * 30},  # 30 tokens
            {"role": "user", "content": "U1" * 15},  # 30 tokens (mocked)
            {"role": "assistant", "content": "A1" * 10},  # 30 tokens (mocked)
            {"role": "user", "content": "U2" * 5},  # 30 tokens (mocked)
        ]
        expected_messages = [
            {"role": "system", "content": "S" * 30},
            {"role": "user", "content": "U2" * 5},  # A1 also removed
        ]

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestSimpleTruncate"
        )

        self.assertEqual(prepared_messages, expected_messages)
        self.assertEqual(
            mock_llm.count_tokens.call_count, 6
        )  # Sys, U1, A1, U2 + U1 (removed), A1 (removed)

        self.orchestrator.logger.warning.assert_any_call(
            "[TestSimpleTruncate] History size (120 tokens) + buffer (20) exceeds limit (100). Truncation required."
        )
        self.orchestrator.logger.debug.assert_any_call(
            "[TestSimpleTruncate] Truncating: Removed message 'user' (30 tokens). Remaining history tokens: 60"
        )
        self.orchestrator.logger.debug.assert_any_call(
            "[TestSimpleTruncate] Truncating: Removed message 'assistant' (30 tokens). Remaining history tokens: 30"
        )
        self.orchestrator.logger.warning.assert_any_call(
            "[TestSimpleTruncate] Truncation complete. Removed 2 messages (60 tokens). Final history size: 60 tokens."
        )

    def test_prepare_history_truncation_no_system_message(self):
        """Test truncation when history (no system message) exceeds limit."""
        # context_limit = 50, llm_max_tokens=5 -> response_buffer = 15
        # U1 (20) + A1 (20) + U2 (20) = 60. Total needed = 60 + 15 = 75. Limit=50
        # Max history = 50 - 0 - 15 = 35 tokens for history.
        # Initial: U1(20), A1(20), U2(20). Hist tokens = 60.
        # Remove U1 (20 tokens). Hist tokens = A1(20)+U2(20) = 40. Still > 35.
        # Remove A1 (20 tokens). Hist tokens = U2(20). Fits.
        # Expected: U2

        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=50,
            llm_max_tokens=5,  # response_buffer = 15
            # U1, A1, U2 (initial sum) + U1 (removed) + A1 (removed)
            token_counts_side_effect=[20, 20, 20, 20, 20],
        )

        messages = [
            {"role": "user", "content": "U1" * 10},  # 20 tokens (mocked)
            {"role": "assistant", "content": "A1" * 5},  # 20 tokens (mocked)
            {"role": "user", "content": "U2" * 20},  # 20 tokens (mocked)
        ]
        expected_messages = [
            {"role": "user", "content": "U2" * 20},
        ]

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestNoSysTruncate"
        )

        self.assertEqual(prepared_messages, expected_messages)
        # 3 for initial sum, 2 for removed messages
        self.assertEqual(mock_llm.count_tokens.call_count, 5)

        self.orchestrator.logger.warning.assert_any_call(
            "[TestNoSysTruncate] History size (60 tokens) + buffer (15) exceeds limit (50). Truncation required."
        )
        self.orchestrator.logger.debug.assert_any_call(
            "[TestNoSysTruncate] Truncating: Removed message 'user' (20 tokens). Remaining history tokens: 40"
        )
        self.orchestrator.logger.debug.assert_any_call(
            "[TestNoSysTruncate] Truncating: Removed message 'assistant' (20 tokens). Remaining history tokens: 20"
        )
        self.orchestrator.logger.warning.assert_any_call(
            "[TestNoSysTruncate] Truncation complete. Removed 2 messages (40 tokens). Final history size: 20 tokens."
        )

    def test_prepare_history_system_message_check_triggers_error(self):
        """Test error when system message + buffer alone exceeds context limit (system check)."""
        # context_limit=50, llm_max_tokens=5 -> response_buffer=15.
        # System message tokens = 40.
        # History message tokens = 10.
        # Initial check: current_history_tokens (10) + system_tokens (40) + buffer (15) = 65. Limit = 50. -> Truncation path.
        # Target check: system_tokens (40) + buffer (15) = 55. Limit = 50. -> This should trigger the specific error.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=50,
            llm_max_tokens=5,  # response_buffer = 15
            token_counts_side_effect=[
                10,
                40,
                99,
                99,
                99,
            ],  # History, System, then extras to prevent StopIteration
        )

        messages = [
            {"role": "system", "content": "S" * 40},  # 40 tokens
            {"role": "user", "content": "U1" * 1},  # 10 tokens (mocked)
        ]

        with self.assertRaisesRegex(
            LLMOrchestrationError, r"System message .* exceeds limit"
        ) as cm:
            self.orchestrator._prepare_history_for_llm(
                mock_llm, messages, context="TestTruncateFail"
            )

        self.assertIn("(40 tokens)", str(cm.exception))
        self.assertIn("buffer (15)", str(cm.exception))
        self.assertIn("exceeds limit (50)", str(cm.exception))

        # Should only call count_tokens twice (history + system) before system check triggers
        self.assertEqual(mock_llm.count_tokens.call_count, 2)
        self.orchestrator.logger.error.assert_called_once_with(
            "[TestTruncateFail] System message (40 tokens) + buffer (15) alone exceeds limit (50). Cannot proceed."
        )

    def test_prepare_history_empty_messages_list(self):
        """Test behavior with an empty messages list."""
        # context_limit=100, llm_max_tokens=10 -> buffer=20
        # Empty messages. current_history_tokens = 0, system_tokens = 0.
        # Total = 0 + 0 + 20 = 20. Fits.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100,
            llm_max_tokens=10,  # buffer = 20
            # No calls to count_tokens are expected if history_messages is empty and no system_message.
            # The sum() over an empty list is 0.
        )

        messages = []
        expected_messages = []

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestEmptyList"
        )

        self.assertEqual(prepared_messages, expected_messages)
        mock_llm.count_tokens.assert_not_called()  # Since history_messages is empty, and system_message is None after pop attempt.
        self.orchestrator.logger.debug.assert_any_call(
            "[TestEmptyList] History size (0 tokens) + buffer (20) fits within limit (100). No truncation needed."
        )
        self.orchestrator.logger.warning.assert_not_called()

    def test_prepare_history_only_system_message_fits(self):
        """Test with only a system message that fits."""
        # context_limit=100, llm_max_tokens=10 -> buffer=20. Sys_tokens=30
        # Sys (30) + buffer (20) = 50. Fits.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100,
            llm_max_tokens=10,  # buffer = 20
            token_counts_side_effect=[30],  # system message
        )

        messages = [
            {"role": "system", "content": "S" * 30},  # 30 tokens
        ]
        expected_messages = messages  # Original should be returned

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestOnlySysFits"
        )

        self.assertEqual(prepared_messages, expected_messages)
        self.assertEqual(mock_llm.count_tokens.call_count, 1)  # System content
        self.orchestrator.logger.debug.assert_any_call(
            "[TestOnlySysFits] History size (30 tokens) + buffer (20) fits within limit (100). No truncation needed."
        )
        self.orchestrator.logger.warning.assert_not_called()

    def test_prepare_history_exception_during_token_counting(self):
        """Test that an exception during llm.count_tokens is wrapped."""
        counting_exception = RuntimeError("Token counting failed")
        # Setup so that token counting will be attempted.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100, llm_max_tokens=10, token_counts_side_effect=counting_exception
        )

        messages = [
            {"role": "system", "content": "SysPrompt"},
            {"role": "user", "content": "UserMsg1"},
        ]

        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator._prepare_history_for_llm(
                mock_llm, messages, context="TestTokenCountEx"
            )

        self.assertIs(cm.exception.__cause__, counting_exception)
        self.assertIn(
            "Failed to prepare history due to token counting/limit error", str(cm.exception)
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"[TestTokenCountEx] Unexpected error during history preparation/token counting: {counting_exception}"
        )

    def test_prepare_history_with_none_or_empty_content_messages(self):
        """Test that messages with None or empty string content are handled."""
        # context_limit=100, llm_max_tokens=10 -> buffer=20
        # Actual call order for count_tokens:
        #   1. history_messages[0] (role:user, content:None)       -> count_tokens(None) -> 0
        #   2. history_messages[1] (role:assistant, content:"")    -> count_tokens("") -> 0
        #   3. history_messages[2] (role:user, content:"User2")   -> count_tokens("User2") -> 5
        #   (current_history_tokens = 0+0+5 = 5)
        #   4. system_message (role:system, content:"Sys")          -> count_tokens("Sys") -> 10
        #   (system_tokens = 10)
        # Total tokens for check = history (5) + system (10) + buffer (20) = 35. Limit is 100. Fits.
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=100,
            llm_max_tokens=10,
            token_counts_side_effect=[0, 0, 5, 10],  # For U1(None), A1(""), U2, then Sys
        )

        messages = [
            {"role": "system", "content": "Sys"},  # Mocked as 10 tokens by side_effect[3]
            {"role": "user", "content": None},  # Mocked as 0 tokens by side_effect[0]
            {"role": "assistant", "content": ""},  # Mocked as 0 tokens by side_effect[1]
            {"role": "user", "content": "User2"},  # Mocked as 5 tokens by side_effect[2]
        ]
        expected_messages = messages  # No truncation expected

        prepared_messages = self.orchestrator._prepare_history_for_llm(
            mock_llm, messages, context="TestEmptyContent"
        )

        self.assertEqual(prepared_messages, expected_messages)
        # count_tokens calls should match the side_effect order and content
        self.assertEqual(mock_llm.count_tokens.call_count, 4)
        self.orchestrator.logger.debug.assert_any_call(
            "[TestEmptyContent] History size (15 tokens) + buffer (20) fits within limit (100). No truncation needed."
        )
        # Verification of total tokens in log: History sum (0+0+5=5) + System (10) = 15.
        self.orchestrator.logger.warning.assert_not_called()

    def test_prepare_history_post_truncation_check_failure_real_scenario(self):
        """Test the actual post-truncation check failure: misconfigured max_tokens too high."""
        # This tests the scenario where response_buffer alone exceeds context_limit
        # context_limit=50, llm_max_tokens=45 -> response_buffer=55
        # No system message, so system check is skipped
        # After truncating all history: 0 + 0 + 55 > 50, triggers post-truncation failure
        mock_llm = self._setup_llm_service_for_history_test(
            context_limit=50,
            llm_max_tokens=45,  # response_buffer = 45 + 10 = 55 > 50
            token_counts_side_effect=[
                10,
                20,
                10,
                20,
            ],  # Initial U1, U2, then U1 removal, U2 removal
        )

        messages = [
            {"role": "user", "content": "U1"},  # 10 tokens
            {"role": "user", "content": "U2"},  # 20 tokens
        ]

        with self.assertRaisesRegex(
            LLMOrchestrationError, r"Failed to truncate history sufficiently"
        ) as cm:
            self.orchestrator._prepare_history_for_llm(
                mock_llm, messages, context="TestPostTruncationFail"
            )

        # Verify the exception details
        self.assertIn("Remaining tokens (0)", str(cm.exception))
        self.assertIn("buffer (55)", str(cm.exception))
        self.assertIn("exceed limit (50)", str(cm.exception))

        # Should call count_tokens for: U1, U2, U1 (removal), U2 (removal) = 4 calls
        self.assertEqual(mock_llm.count_tokens.call_count, 4)
        self.orchestrator.logger.error.assert_called_once_with(
            "[TestPostTruncationFail] Truncation failed! Final history tokens (0) + system tokens (0) + buffer (55) still exceed limit (50)."
        )

    # --- Tests for _query_llm_core ---

    def test_query_llm_core_happy_path_no_callback(self):
        """Test _query_llm_core successful execution without messaging_callback."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = "TestLLM"
        mock_llm_service.model_name = "test-model-v1"

        raw_messages = [{"role": "user", "content": "Hello"}]
        prepared_messages = [{"role": "user", "content": "Hello Prepared"}]  # Simulate preparation
        current_prompt_text = "Hello"
        llm_response_content = "Hi there!"
        query_kwargs_to_pass = {"temperature": 0.5}

        self.orchestrator.messaging_callback = None  # Ensure no callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ) as mock_prepare_history:
            mock_llm_service.query.return_value = llm_response_content

            actual_response = self.orchestrator._query_llm_core(
                llm_service=mock_llm_service,
                messages_to_send_raw=raw_messages,
                current_prompt_text_for_callback=current_prompt_text,
                stage=OrchestrationStage.DEBATE_TURN,
                step_description="Test Step No Callback",
                iteration_num=1,
                prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                prompt_source_entity_name="user",
                prompt_metadata_data={"custom_key": "custom_value"},
                query_kwargs=query_kwargs_to_pass,
            )

        self.assertEqual(actual_response, llm_response_content)
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="DEBATE_TURN - Test Step No Callback - History Prep",
        )
        mock_llm_service.query.assert_called_once_with(
            prompt="",
            use_conversation=False,
            conversation_history=prepared_messages,
            temperature=0.5,  # from query_kwargs
        )
        self.orchestrator.logger.debug.assert_any_call(
            "_query_llm_core: Stage='DEBATE_TURN', Step='Test Step No Callback', LLM='TestLLM'"
        )
        self.orchestrator.logger.debug.assert_any_call(
            f"_query_llm_core: LLM response received. Length: {len(llm_response_content)}"
        )

    @patch("uuid.uuid4")
    @patch(
        "convorator.conversations.conversation_orchestrator.datetime"
    )  # Patch datetime used in _query_llm_core
    def test_query_llm_core_happy_path_with_callback(self, mock_datetime, mock_uuid):
        """Test _query_llm_core successful execution with messaging_callback, verifying metadata."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = "CallbackLLM"
        mock_llm_service.model_name = "callback-model-v1"

        raw_messages = [{"role": "user", "content": "Test prompt"}]
        prepared_messages = [{"role": "user", "content": "Test prompt prepared"}]
        current_prompt_text = "Test prompt"
        llm_response_content = "Callback response."
        test_session_id = "session-for-callback"
        self.orchestrator.config.session_id = test_session_id  # Set for metadata

        # Mock datetime and uuid for predictable metadata
        mock_now = datetime(2023, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        # Use side_effect to provide different UUIDs for prompt and response events
        mock_uuid.side_effect = ["prompt-uuid-123", "response-uuid-456"]

        mock_messaging_callback = MagicMock()
        self.orchestrator.messaging_callback = mock_messaging_callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ) as mock_prepare_history:
            mock_llm_service.query.return_value = llm_response_content

            actual_response = self.orchestrator._query_llm_core(
                llm_service=mock_llm_service,
                messages_to_send_raw=raw_messages,
                current_prompt_text_for_callback=current_prompt_text,
                stage=OrchestrationStage.SOLUTION_GENERATION_INITIAL,  # Fixed enum value
                step_description="Test Step With Callback",
                iteration_num=2,
                prompt_source_entity_type=MessageEntityType.ORCHESTRATOR_INTERNAL,
                prompt_source_entity_name="OrchestratorLogic",
                prompt_metadata_data={"info": "extra prompt data"},
                query_kwargs=None,
            )

        self.assertEqual(actual_response, llm_response_content)
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="SOLUTION_GENERATION_INITIAL - Test Step With Callback - History Prep",
        )
        mock_llm_service.query.assert_called_once_with(
            prompt="", use_conversation=False, conversation_history=prepared_messages
        )

        # Verify callback calls
        self.assertEqual(mock_messaging_callback.call_count, 2)
        prompt_call_args, response_call_args = mock_messaging_callback.call_args_list

        # Prompt Callback Verification
        self.assertEqual(prompt_call_args[0][0], EventType.PROMPT)
        self.assertEqual(prompt_call_args[0][1], current_prompt_text)
        prompt_meta = prompt_call_args[0][2]  # This is a dict (TypedDict)
        self.assertEqual(prompt_meta["event_id"], "prompt-uuid-123")
        self.assertEqual(prompt_meta["timestamp"], mock_now.isoformat())
        self.assertEqual(prompt_meta["session_id"], test_session_id)
        self.assertEqual(prompt_meta["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL)
        self.assertEqual(prompt_meta["step_description"], "Test Step With Callback - Prompt")
        self.assertEqual(prompt_meta["iteration_num"], 2)
        self.assertEqual(prompt_meta["source_entity_type"], MessageEntityType.ORCHESTRATOR_INTERNAL)
        self.assertEqual(prompt_meta["source_entity_name"], "OrchestratorLogic")
        self.assertEqual(prompt_meta["target_entity_type"], MessageEntityType.LLM_AGENT)
        self.assertEqual(prompt_meta["target_entity_name"], "CallbackLLM")
        self.assertEqual(prompt_meta["llm_service_details"], {"model_name": "callback-model-v1"})
        self.assertEqual(prompt_meta["payload_type"], MessagePayloadType.TEXT_CONTENT)
        self.assertEqual(prompt_meta["data"], {"info": "extra prompt data"})

        # Response Callback Verification
        self.assertEqual(response_call_args[0][0], EventType.RESPONSE)
        self.assertEqual(response_call_args[0][1], llm_response_content)
        response_meta = response_call_args[0][2]  # This is a dict (TypedDict)
        self.assertEqual(response_meta["event_id"], "response-uuid-456")
        self.assertEqual(response_meta["timestamp"], mock_now.isoformat())
        self.assertEqual(response_meta["session_id"], test_session_id)
        self.assertEqual(response_meta["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL)
        self.assertEqual(response_meta["step_description"], "Test Step With Callback - Response")
        self.assertEqual(response_meta["iteration_num"], 2)
        self.assertEqual(response_meta["source_entity_type"], MessageEntityType.LLM_AGENT)
        self.assertEqual(response_meta["source_entity_name"], "CallbackLLM")
        self.assertEqual(
            response_meta["target_entity_type"], MessageEntityType.ORCHESTRATOR_INTERNAL
        )
        self.assertEqual(response_meta["target_entity_name"], "OrchestratorLogic")
        self.assertEqual(response_meta["llm_service_details"], {"model_name": "callback-model-v1"})
        self.assertEqual(response_meta["payload_type"], MessagePayloadType.TEXT_CONTENT)
        self.assertIsNone(response_meta["data"])

    def test_query_llm_core_prepare_history_raises_exception(self):
        """Test _query_llm_core when _prepare_history_for_llm raises an exception."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = None  # Will become 'UnnamedLLM' in debug log

        raw_messages = [{"role": "user", "content": "Test prompt"}]
        current_prompt_text = "Test prompt"
        history_error = LLMOrchestrationError("History preparation failed!")

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            side_effect=history_error,
            autospec=True,
        ) as mock_prepare_history:
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._query_llm_core(
                    llm_service=mock_llm_service,
                    messages_to_send_raw=raw_messages,
                    current_prompt_text_for_callback=current_prompt_text,
                    stage=OrchestrationStage.DEBATE_TURN,
                    step_description="Test History Prep Error",
                    iteration_num=None,
                    prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                    prompt_source_entity_name="user",
                    prompt_metadata_data=None,
                    query_kwargs=None,
                )

        self.assertEqual(str(cm.exception), "History preparation failed!")
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="DEBATE_TURN - Test History Prep Error - History Prep",
        )
        # Verify that no LLM query was attempted
        mock_llm_service.query.assert_not_called()

        # Verify logging
        self.orchestrator.logger.debug.assert_any_call(
            "_query_llm_core: Stage='DEBATE_TURN', Step='Test History Prep Error', LLM='UnnamedLLM'"  # Fixed expectation
        )

    @patch("uuid.uuid4")
    @patch("convorator.conversations.conversation_orchestrator.datetime")
    def test_query_llm_core_llm_query_raises_llm_response_error(self, mock_datetime, mock_uuid):
        """Test _query_llm_core when llm_service.query raises LLMResponseError, with callback."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = "ResponseErrorLLM"
        mock_llm_service.model_name = "error-model"

        raw_messages = [{"role": "user", "content": "Trigger response error"}]
        prepared_messages = [{"role": "user", "content": "Prepared error"}]
        current_prompt_text = "Response error prompt"
        response_error = LLMResponseError("API rate limit exceeded")

        # Mock datetime and uuid for metadata
        mock_now = datetime(2023, 10, 26, 14, 30, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        # Use side_effect to provide different UUIDs for prompt and error events
        mock_uuid.side_effect = ["prompt-uuid-err", "error-uuid-err"]

        mock_messaging_callback = MagicMock()
        self.orchestrator.messaging_callback = mock_messaging_callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ) as mock_prepare_history:
            mock_llm_service.query.side_effect = response_error

            with self.assertRaises(LLMResponseError) as cm:
                self.orchestrator._query_llm_core(
                    llm_service=mock_llm_service,
                    messages_to_send_raw=raw_messages,
                    current_prompt_text_for_callback=current_prompt_text,
                    stage=OrchestrationStage.DEBATE_TURN,
                    step_description="Test LLM Response Error",
                    iteration_num=2,
                    prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                    prompt_source_entity_name="user",
                    prompt_metadata_data=None,
                    query_kwargs=None,
                )

        self.assertIs(cm.exception, response_error)
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="DEBATE_TURN - Test LLM Response Error - History Prep",
        )
        mock_llm_service.query.assert_called_once_with(
            prompt="", use_conversation=False, conversation_history=prepared_messages
        )

        # Verify 2 callback calls: prompt event and error event
        self.assertEqual(mock_messaging_callback.call_count, 2)
        prompt_call, error_call = mock_messaging_callback.call_args_list

        # Verify prompt callback
        self.assertEqual(prompt_call[0][0], EventType.PROMPT)
        self.assertEqual(prompt_call[0][1], current_prompt_text)
        prompt_meta = prompt_call[0][2]  # This is a dict (TypedDict)
        self.assertEqual(prompt_meta["event_id"], "prompt-uuid-err")
        self.assertEqual(prompt_meta["stage"], OrchestrationStage.DEBATE_TURN)

        # Verify error callback
        self.assertEqual(error_call[0][0], EventType.LLM_ERROR)
        self.assertEqual(error_call[0][1], "API rate limit exceeded")
        error_meta = error_call[0][2]  # This is a dict (TypedDict)
        self.assertEqual(error_meta["event_id"], "error-uuid-err")
        self.assertEqual(error_meta["stage"], OrchestrationStage.DEBATE_TURN)
        self.assertEqual(
            error_meta["step_description"], "Test LLM Response Error - LLM Query Error"
        )
        self.assertEqual(error_meta["source_entity_type"], MessageEntityType.LLM_AGENT)
        self.assertEqual(error_meta["source_entity_name"], "ResponseErrorLLM")
        self.assertEqual(error_meta["payload_type"], MessagePayloadType.ERROR_DETAILS_STR)
        self.assertEqual(
            error_meta["data"],
            {
                "error_type": "LLMResponseError",
                "error_message": "API rate limit exceeded",
            },
        )

        # Verify error logging
        self.orchestrator.logger.error.assert_any_call(
            "_query_llm_core: LLM query failed. Stage='DEBATE_TURN', Step='Test LLM Response Error'. Error: API rate limit exceeded",
            exc_info=True,
        )

    def test_query_llm_core_llm_query_raises_llm_client_error(self):
        """Test _query_llm_core when llm_service.query raises LLMClientError, no error callback if callback is None."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = "ClientErrorLLM"

        raw_messages = [{"role": "user", "content": "Trigger client error"}]
        prepared_messages = [{"role": "user", "content": "Prepared"}]
        current_prompt_text = "Client error prompt"
        client_error = LLMClientError("Network connection failed")

        # No messaging callback set (None)
        self.orchestrator.messaging_callback = None

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ) as mock_prepare_history:
            mock_llm_service.query.side_effect = client_error

            with self.assertRaises(LLMClientError) as cm:
                self.orchestrator._query_llm_core(
                    llm_service=mock_llm_service,
                    messages_to_send_raw=raw_messages,
                    current_prompt_text_for_callback=current_prompt_text,
                    stage=OrchestrationStage.MODERATION_SUMMARY,  # Fixed enum value
                    step_description="Test LLM Client Error",
                    iteration_num=None,
                    prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                    prompt_source_entity_name="user",
                    prompt_metadata_data=None,
                    query_kwargs=None,
                )

        self.assertIs(cm.exception, client_error)
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="MODERATION_SUMMARY - Test LLM Client Error - History Prep",
        )
        mock_llm_service.query.assert_called_once_with(
            prompt="", use_conversation=False, conversation_history=prepared_messages
        )

        # Verify error logged
        self.orchestrator.logger.error.assert_any_call(
            "_query_llm_core: LLM query failed. Stage='MODERATION_SUMMARY', Step='Test LLM Client Error'. Error: Network connection failed",
            exc_info=True,
        )

    def test_query_llm_core_prompt_callback_raises_exception(self):
        """Test _query_llm_core when the prompt callback itself raises an exception."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_role_name.return_value = "CallbackFailLLM"

        raw_messages = [{"role": "user", "content": "Test prompt CB fail"}]
        prepared_messages = [{"role": "user", "content": "Prepared CB fail"}]
        current_prompt_text = "Test prompt CB fail"
        llm_response = "LLM Response after prompt CB error"
        callback_exception = Exception("Prompt callback failed!")

        # Set up callback that only fails for PROMPT events
        def failing_prompt_callback(event_type, content, metadata):
            if event_type == EventType.PROMPT:
                raise callback_exception
            # For other events (like RESPONSE), don't raise exception
            return None

        self.orchestrator.messaging_callback = failing_prompt_callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ) as mock_prepare_history:
            mock_llm_service.query.return_value = llm_response

            # The prompt callback exception should be caught and logged, but execution continues
            actual_response = self.orchestrator._query_llm_core(
                llm_service=mock_llm_service,
                messages_to_send_raw=raw_messages,
                current_prompt_text_for_callback=current_prompt_text,
                stage=OrchestrationStage.DEBATE_TURN,
                step_description="Test PromptCBException",
                iteration_num=1,
                prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                prompt_source_entity_name="user",
                prompt_metadata_data=None,
                query_kwargs=None,
            )

        # Verify the main flow continues despite callback failure
        self.assertEqual(actual_response, llm_response)
        mock_prepare_history.assert_called_once_with(
            mock_llm_service,
            raw_messages,
            context="DEBATE_TURN - Test PromptCBException - History Prep",
        )
        mock_llm_service.query.assert_called_once_with(
            prompt="", use_conversation=False, conversation_history=prepared_messages
        )

        # Verify prompt callback error was logged
        self.orchestrator.logger.error.assert_any_call(
            "Messaging callback for prompt failed: Prompt callback failed!", exc_info=True
        )

    def test_query_llm_core_response_callback_raises_exception(self):
        """Test _query_llm_core when the response callback raises an exception."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        prepared_messages = [{"role": "user", "content": "Prepared for resp CB fail"}]
        llm_response_content = "LLM Response, but resp CB will fail"

        prompt_callback_mock = MagicMock()
        response_callback_exception = KeyError("Response callback failed!")
        # Callback will be called twice: once for prompt (MagicMock), once for response (side_effect)
        mock_messaging_callback = MagicMock(
            side_effect=[
                None,  # Prompt callback succeeds
                response_callback_exception,  # Response callback fails
            ]
        )
        self.orchestrator.messaging_callback = mock_messaging_callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ):
            mock_llm_service.query.return_value = llm_response_content
            actual_response = self.orchestrator._query_llm_core(
                llm_service=mock_llm_service,
                messages_to_send_raw=[{"role": "user", "content": "Test"}],
                current_prompt_text_for_callback="Test response CB fail",
                stage=OrchestrationStage.DEBATE_TURN,
                step_description="Test ResponseCBException",
                iteration_num=1,
                prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                prompt_source_entity_name="user",
            )

        self.assertEqual(actual_response, llm_response_content)  # Still returns LLM response
        self.assertEqual(mock_messaging_callback.call_count, 2)  # Both callbacks attempted
        mock_llm_service.query.assert_called_once()
        self.orchestrator.logger.error.assert_called_once_with(
            f"Messaging callback for response failed: {response_callback_exception}", exc_info=True
        )

    def test_query_llm_core_error_callback_raises_exception(self):
        """Test _query_llm_core when the error callback itself raises an exception."""
        mock_llm_service = MagicMock(spec=LLMInterface)
        prepared_messages = [{"role": "user", "content": "Prepared for error CB fail"}]
        llm_error = LLMResponseError("Original LLM error")
        error_callback_exception = TypeError("Error callback failed!")

        # Prompt callback succeeds, LLM query fails, then Error callback fails
        mock_messaging_callback = MagicMock(
            side_effect=[None, error_callback_exception]  # Prompt callback  # Error callback
        )
        self.orchestrator.messaging_callback = mock_messaging_callback

        with patch.object(
            self.orchestrator,
            "_prepare_history_for_llm",
            return_value=prepared_messages,
            autospec=True,
        ):
            mock_llm_service.query.side_effect = llm_error
            with self.assertRaises(
                LLMResponseError
            ) as cm:  # Original LLM error should still propagate
                self.orchestrator._query_llm_core(
                    llm_service=mock_llm_service,
                    messages_to_send_raw=[{"role": "user", "content": "Test"}],
                    current_prompt_text_for_callback="Test error CB fail",
                    stage=OrchestrationStage.DEBATE_TURN,
                    step_description="Test ErrorCBException",
                    iteration_num=1,
                    prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                    prompt_source_entity_name="user",
                )

        self.assertIs(cm.exception, llm_error)  # Original error propagates
        self.assertEqual(mock_messaging_callback.call_count, 2)  # Prompt and Error CBs attempted
        self.orchestrator.logger.error.assert_any_call(  # Log for original LLM error
            f"_query_llm_core: LLM query failed. Stage='DEBATE_TURN', Step='Test ErrorCBException'. Error: {llm_error}",
            exc_info=True,
        )
        self.orchestrator.logger.error.assert_any_call(  # Log for the callback's own error
            f"Messaging callback for LLM error event failed: {error_callback_exception}",
            exc_info=True,
        )

    # --- Tests for improve_solution_with_moderation (module-level function) ---

    @patch("convorator.conversations.conversation_orchestrator.logger")
    @patch("convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator")
    def test_improve_solution_successful_orchestration(
        self, MockSolutionImprovementOrchestrator, mock_module_logger
    ):
        """Test improve_solution_with_moderation successful flow."""
        mock_orchestrator_instance = MagicMock(spec=SolutionImprovementOrchestrator)
        mock_run_result = {"success": True, "solution": "perfect"}
        mock_orchestrator_instance.run.return_value = mock_run_result
        MockSolutionImprovementOrchestrator.return_value = mock_orchestrator_instance

        # Args to be passed to improve_solution_with_moderation
        direct_initial_solution = {"direct_init": "sol"}
        direct_requirements = "Direct requirements"
        direct_assessment_criteria = "Direct assessment"

        result = improve_solution_with_moderation(
            config=self.mock_config,
            initial_solution=direct_initial_solution,
            requirements=direct_requirements,
            assessment_criteria=direct_assessment_criteria,
        )

        # Verify Orchestrator instantiation
        MockSolutionImprovementOrchestrator.assert_called_once_with(
            config=self.mock_config,
            initial_solution=direct_initial_solution,
            requirements=direct_requirements,
            assessment_criteria=direct_assessment_criteria,
        )

        # Verify orchestrator.run() was called
        mock_orchestrator_instance.run.assert_called_once_with()

        # Verify the result
        self.assertEqual(result, mock_run_result)

        # Verify no top-level error logging
        mock_module_logger.error.assert_not_called()

    @patch("convorator.conversations.conversation_orchestrator.logger")
    @patch("convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator")
    def test_improve_solution_orchestrator_init_raises_value_error(
        self, MockSolutionImprovementOrchestrator, mock_module_logger
    ):
        """Test improve_solution_with_moderation when Orchestrator init fails."""
        init_exception = ValueError("Missing critical config for orchestrator")
        MockSolutionImprovementOrchestrator.side_effect = init_exception

        with self.assertRaises(ValueError) as context_manager:
            improve_solution_with_moderation(config=self.mock_config)

        self.assertIs(context_manager.exception, init_exception)
        mock_module_logger.error.assert_called_once_with(
            f"Top-level execution via improve_solution_with_moderation failed: {init_exception}",
            exc_info=True,
        )

    @patch("convorator.conversations.conversation_orchestrator.logger")
    @patch("convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator")
    def test_improve_solution_orchestrator_run_raises_llm_orchestration_error(
        self, MockSolutionImprovementOrchestrator, mock_module_logger
    ):
        """Test improve_solution_with_moderation when orchestrator.run() fails."""
        mock_orchestrator_instance = MagicMock(spec=SolutionImprovementOrchestrator)
        run_exception = LLMOrchestrationError("Orchestrator run failed badly")
        mock_orchestrator_instance.run.side_effect = run_exception
        MockSolutionImprovementOrchestrator.return_value = mock_orchestrator_instance

        with self.assertRaises(LLMOrchestrationError) as context_manager:
            improve_solution_with_moderation(config=self.mock_config)

        self.assertIs(context_manager.exception, run_exception)
        MockSolutionImprovementOrchestrator.assert_called_once_with(
            config=self.mock_config,
            initial_solution=None,  # Direct args are None in this call
            requirements=None,
            assessment_criteria=None,
        )
        mock_orchestrator_instance.run.assert_called_once_with()
        mock_module_logger.error.assert_called_once_with(
            f"Top-level execution via improve_solution_with_moderation failed: {run_exception}",
            exc_info=True,
        )

    @patch("convorator.conversations.conversation_orchestrator.logger")
    @patch("convorator.conversations.conversation_orchestrator.SolutionImprovementOrchestrator")
    def test_improve_solution_orchestrator_run_raises_generic_exception(
        self, MockSolutionImprovementOrchestrator, mock_module_logger
    ):
        """Test improve_solution_with_moderation when orchestrator.run() fails with a generic Exception."""
        mock_orchestrator_instance = MagicMock(spec=SolutionImprovementOrchestrator)
        run_exception = Exception("Some generic unexpected failure in run")  # Generic Exception
        mock_orchestrator_instance.run.side_effect = run_exception
        MockSolutionImprovementOrchestrator.return_value = mock_orchestrator_instance

        with self.assertRaises(Exception) as context_manager:  # Expect the generic Exception
            improve_solution_with_moderation(config=self.mock_config)

        self.assertIs(context_manager.exception, run_exception)
        MockSolutionImprovementOrchestrator.assert_called_once_with(
            config=self.mock_config,
            initial_solution=None,
            requirements=None,
            assessment_criteria=None,
        )
        mock_orchestrator_instance.run.assert_called_once_with()
        mock_module_logger.error.assert_called_once_with(
            f"Top-level execution via improve_solution_with_moderation failed: {run_exception}",
            exc_info=True,
        )

    # --- Tests for _query_agent_in_debate_turn ---

    def test_query_agent_in_debate_turn_happy_path_defaults(self):
        """Test _query_agent_in_debate_turn happy path with default add_prompt_to_main_log=True."""
        # Arrange
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm
        test_prompt = "This is a test prompt for the primary agent."
        llm_core_response = "LLM core response for primary."
        self.orchestrator.current_iteration_num = 3

        # Ensure the agent's conversation is in a valid state (e.g., empty or ends with assistant)
        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        agent_conversation.add_assistant_message("Previous assistant response")

        # Set up spies on the conversation methods
        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user

        spy_agent_conv_add_assistant = MagicMock(wraps=agent_conversation.add_assistant_message)
        agent_conversation.add_assistant_message = spy_agent_conv_add_assistant

        spy_main_log_add_message = MagicMock(
            wraps=self.orchestrator.main_conversation_log.add_message
        )
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add_message

        # Expected messages to be sent to _query_llm_core
        expected_messages_for_core = [
            {"role": "system", "content": "Primary System Message"},  # Fixed system message
            {"role": "assistant", "content": "Previous assistant response"},
            {"role": "user", "content": test_prompt},
        ]

        # Mock _query_llm_core
        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            response = self.orchestrator._query_agent_in_debate_turn(
                llm_to_query=mock_llm_to_query,
                queried_role_name=queried_role_name,
                prompt=test_prompt,
                # Default: add_prompt_to_main_log=True
            )

        # Assert
        self.assertEqual(response, llm_core_response)

        # Verify conversation updates
        spy_agent_conv_add_user.assert_called_once_with(test_prompt)
        spy_agent_conv_add_assistant.assert_called_once_with(llm_core_response)

        # Verify main log updates (both prompt and response should be added)
        expected_main_log_calls = [
            call(role="user", content=test_prompt),
            call(role=queried_role_name, content=llm_core_response),
        ]
        spy_main_log_add_message.assert_has_calls(expected_main_log_calls, any_order=False)

        # Verify _query_llm_core was called correctly
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=3,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

    def test_query_agent_in_debate_turn_add_prompt_to_main_log_false(self):
        """Test _query_agent_in_debate_turn when add_prompt_to_main_log is False."""
        # Arrange
        queried_role_name = self.orchestrator.debater_name
        mock_llm_to_query = self.orchestrator.debater_llm
        test_prompt = "Debater prompt, not for main log."
        llm_core_response = "Debater response."
        self.orchestrator.current_iteration_num = 1

        # Set up the debater conversation with a previous assistant message
        debater_conv = self.orchestrator.role_conversation[queried_role_name]
        debater_conv.add_assistant_message("Previous debater response")

        # Set up spies
        spy_agent_conv_add_user = MagicMock(wraps=debater_conv.add_user_message)
        debater_conv.add_user_message = spy_agent_conv_add_user

        spy_agent_conv_add_assistant = MagicMock(wraps=debater_conv.add_assistant_message)
        debater_conv.add_assistant_message = spy_agent_conv_add_assistant

        spy_main_log_add_message = MagicMock(
            wraps=self.orchestrator.main_conversation_log.add_message
        )
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add_message

        expected_messages_for_core = [
            {"role": "system", "content": "Debater System Message"},  # Fixed system message
            {"role": "assistant", "content": "Previous debater response"},
            {"role": "user", "content": test_prompt},
        ]

        # Mock _query_llm_core
        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            response = self.orchestrator._query_agent_in_debate_turn(
                llm_to_query=mock_llm_to_query,
                queried_role_name=queried_role_name,
                prompt=test_prompt,
                add_prompt_to_main_log=False,  # Key difference
            )

        # Assert
        self.assertEqual(response, llm_core_response)

        # Verify agent conversation updates
        spy_agent_conv_add_user.assert_called_once_with(test_prompt)
        spy_agent_conv_add_assistant.assert_called_once_with(llm_core_response)

        # Verify main log updates (only response should be added, not the prompt)
        expected_main_log_calls = [
            call(role=queried_role_name, content=llm_core_response),
        ]
        spy_main_log_add_message.assert_has_calls(expected_main_log_calls, any_order=False)

        # Verify _query_llm_core was called correctly
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=1,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

    def test_query_agent_in_debate_turn_empty_prompt(self):
        """Test _query_agent_in_debate_turn with an empty prompt."""
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user

        spy_main_log_add = MagicMock(wraps=self.orchestrator.main_conversation_log.add_message)
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add

        with patch.object(self.orchestrator, "_query_llm_core") as mock_query_llm_core:
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._query_agent_in_debate_turn(
                    llm_to_query=mock_llm_to_query,
                    queried_role_name=queried_role_name,
                    prompt="",  # Empty prompt
                )

        self.assertEqual(str(cm.exception), "Prompt cannot be empty.")
        spy_agent_conv_add_user.assert_not_called()
        spy_main_log_add.assert_not_called()
        mock_query_llm_core.assert_not_called()
        self.orchestrator.logger.info.assert_any_call(f"Try to query {queried_role_name}...")

    def test_query_agent_in_debate_turn_history_ends_with_user(self):
        """Test _query_agent_in_debate_turn when agent history already ends with a user message."""
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        # Set up history to end with a user message
        agent_conversation.add_assistant_message("Previous assistant msg")
        agent_conversation.add_user_message("Existing user message")

        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user
        spy_main_log_add = MagicMock(wraps=self.orchestrator.main_conversation_log.add_message)
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add

        with patch.object(self.orchestrator, "_query_llm_core") as mock_query_llm_core:
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._query_agent_in_debate_turn(
                    llm_to_query=mock_llm_to_query,
                    queried_role_name=queried_role_name,
                    prompt="New prompt that should fail validation",
                )

        expected_error_msg = f"{queried_role_name}'s last message must be an assistant message. Since prompt is going to be the user message"
        self.assertEqual(str(cm.exception), expected_error_msg)
        # add_user_message on agent_conversation might be called before validation in some structures,
        # but the core logic shouldn't proceed. In this case, the check is before add.
        spy_agent_conv_add_user.assert_not_called()
        spy_main_log_add.assert_not_called()
        mock_query_llm_core.assert_not_called()
        self.orchestrator.logger.info.assert_any_call(f"Try to query {queried_role_name}...")

    def test_query_agent_in_debate_turn_core_raises_llm_response_error(self):
        """Test when _query_llm_core raises LLMResponseError."""
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm
        test_prompt = "Prompt leading to LLMResponseError"
        core_error = LLMResponseError("Core LLM response failed")

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        # Start with empty history (system message will be there)
        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user

        spy_main_log_add_message = MagicMock(
            wraps=self.orchestrator.main_conversation_log.add_message
        )
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add_message

        expected_messages_for_core = [
            {"role": "system", "content": "Primary System Message"},  # Fixed system message
            {"role": "user", "content": test_prompt},
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=core_error, autospec=True
        ) as mock_query_llm_core:
            with self.assertRaises(LLMResponseError) as cm:
                self.orchestrator._query_agent_in_debate_turn(
                    llm_to_query=mock_llm_to_query,
                    queried_role_name=queried_role_name,
                    prompt=test_prompt,
                )

        # Verify the original error was propagated
        self.assertIs(cm.exception, core_error)

        # Verify _query_llm_core was called with expected arguments
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=0,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

        # Verify prompt was added to agent conversation before the error
        spy_agent_conv_add_user.assert_called_once_with(test_prompt)
        # Verify prompt was added to main log before the error
        spy_main_log_add_message.assert_called_once_with(role="user", content=test_prompt)

    def test_query_agent_in_debate_turn_core_raises_llm_client_error(self):
        """Test when _query_llm_core raises LLMClientError."""
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm
        test_prompt = "Prompt leading to LLMClientError"
        core_error = LLMClientError("Core LLM client failed")

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        # Start with empty history (system message will be there)
        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user

        spy_main_log_add_message = MagicMock(
            wraps=self.orchestrator.main_conversation_log.add_message
        )
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add_message

        expected_messages_for_core = [
            {"role": "system", "content": "Primary System Message"},  # Fixed system message
            {"role": "user", "content": test_prompt},
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=core_error, autospec=True
        ) as mock_query_llm_core:
            with self.assertRaises(LLMClientError) as cm:
                self.orchestrator._query_agent_in_debate_turn(
                    llm_to_query=mock_llm_to_query,
                    queried_role_name=queried_role_name,
                    prompt=test_prompt,
                )

        # Verify the original error was propagated
        self.assertIs(cm.exception, core_error)

        # Verify _query_llm_core was called with expected arguments
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=0,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

        # Verify prompt was added to agent conversation before the error
        spy_agent_conv_add_user.assert_called_once_with(test_prompt)
        # Verify prompt was added to main log before the error
        spy_main_log_add_message.assert_called_once_with(role="user", content=test_prompt)

    def test_query_agent_in_debate_turn_core_raises_generic_exception(self):
        """Test when _query_llm_core raises a generic exception."""
        queried_role_name = self.orchestrator.primary_name
        mock_llm_to_query = self.orchestrator.primary_llm
        test_prompt = "Prompt leading to generic error"
        core_error = ValueError("Some unexpected error")

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        spy_agent_conv_add_user = MagicMock(wraps=agent_conversation.add_user_message)
        agent_conversation.add_user_message = spy_agent_conv_add_user

        spy_main_log_add_message = MagicMock(
            wraps=self.orchestrator.main_conversation_log.add_message
        )
        self.orchestrator.main_conversation_log.add_message = spy_main_log_add_message

        expected_messages_for_core = [
            {"role": "system", "content": "Primary System Message"},  # Fixed system message
            {"role": "user", "content": test_prompt},
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=core_error, autospec=True
        ) as mock_query_llm_core:
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._query_agent_in_debate_turn(
                    llm_to_query=mock_llm_to_query,
                    queried_role_name=queried_role_name,
                    prompt=test_prompt,
                )

        # Verify the exception was wrapped appropriately
        self.assertIn(
            "Unexpected error during TestPrimaryAgent query/log update", str(cm.exception)
        )
        self.assertIs(cm.exception.__cause__, core_error)

        # Verify _query_llm_core was called with expected arguments
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=0,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

        # Verify prompt was added to agent conversation before the error
        spy_agent_conv_add_user.assert_called_once_with(test_prompt)
        # Verify prompt was added to main log before the error
        spy_main_log_add_message.assert_called_once_with(role="user", content=test_prompt)

    def test_query_agent_in_debate_turn_kwargs_passed_through(self):
        """Test that query_kwargs are correctly passed to _query_llm_core."""
        queried_role_name = self.orchestrator.moderator_name
        mock_llm_to_query = self.orchestrator.moderator_llm
        test_prompt = "Kwargs test prompt"
        llm_core_response = "Kwargs test response"
        custom_kwargs = {"temperature": 0.9, "max_tokens": 200}

        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        expected_messages_for_core = [
            {"role": "system", "content": "Moderator System Message"},  # Fixed system message
            {"role": "user", "content": test_prompt},
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            self.orchestrator._query_agent_in_debate_turn(
                llm_to_query=mock_llm_to_query,
                queried_role_name=queried_role_name,
                prompt=test_prompt,
                query_kwargs=custom_kwargs,
            )

        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=0,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=custom_kwargs,
        )

    def test_query_agent_in_debate_turn_agent_history_empty(self):
        """Test _query_agent_in_debate_turn when agent conversation history is initially empty."""
        queried_role_name = self.orchestrator.debater_name
        mock_llm_to_query = self.orchestrator.debater_llm
        test_prompt = "Empty history prompt"
        llm_core_response = "Empty history response"

        # Ensure the agent conversation is empty (except for system message)
        agent_conversation = self.orchestrator.role_conversation[queried_role_name]
        agent_conversation.clear(keep_system=True)

        expected_messages_for_core = [
            {"role": "system", "content": "Debater System Message"},  # Fixed system message
            {"role": "user", "content": test_prompt},
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            response = self.orchestrator._query_agent_in_debate_turn(
                llm_to_query=mock_llm_to_query,
                queried_role_name=queried_role_name,
                prompt=test_prompt,
            )

        self.assertEqual(response, llm_core_response)

        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_to_query,
            messages_to_send_raw=expected_messages_for_core,
            current_prompt_text_for_callback=test_prompt,
            stage=OrchestrationStage.DEBATE_TURN,
            step_description=f"{queried_role_name} Agent Response",
            iteration_num=0,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

    # --- Tests for _query_for_iterative_generation ---

    def test_qfig_stateful_happy_path(self):
        """Test _query_for_iterative_generation in stateful mode, happy path.
        Uses an initially empty generation_conversation_obj.
        """
        # Arrange
        mock_llm_service = self.mock_solution_llm  # Use one of the LLMs from setUp
        mock_llm_service.get_system_message.return_value = "Iterative Gen System Message"

        current_prompt_content = "Initial prompt for iterative generation."
        # Create a real Conversation object to be mutated
        generation_conv_obj = Conversation(system_message=mock_llm_service.get_system_message())

        base_context = "Test Stateful Gen"
        attempt_num = 1
        stage = OrchestrationStage.SOLUTION_GENERATION_INITIAL  # Corrected Enum
        llm_core_response = "Successful response from LLM core."

        # Spy on the actual methods of the real Conversation object
        # We need to spy *after* initial system message is potentially added by Conversation.__init__
        spy_add_user = MagicMock(wraps=generation_conv_obj.add_user_message)
        generation_conv_obj.add_user_message = spy_add_user
        spy_add_assistant = MagicMock(wraps=generation_conv_obj.add_assistant_message)
        generation_conv_obj.add_assistant_message = spy_add_assistant
        # get_messages doesn't need a spy if we verify its content via _query_llm_core's call

        # Expected messages for _query_llm_core *after* add_user_message has been called
        expected_messages_to_core = [
            Message(role="system", content="Iterative Gen System Message").to_dict(),
            Message(role="user", content=current_prompt_content).to_dict(),
        ]

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            actual_response = self.orchestrator._query_for_iterative_generation(
                llm_service=mock_llm_service,
                current_prompt_content=current_prompt_content,
                generation_conversation_obj=generation_conv_obj,
                base_context_for_logging=base_context,
                attempt_num=attempt_num,
                stage=stage,
                prompt_metadata_for_llm_core=None,
                query_kwargs=None,
            )

        # Assert
        self.assertEqual(actual_response, llm_core_response)

        # 1. Verify Conversation object interactions
        spy_add_user.assert_called_once_with(current_prompt_content)
        spy_add_assistant.assert_called_once_with(llm_core_response)

        # Verify final state of generation_conv_obj (optional, but good for sanity)
        final_gen_conv_messages = generation_conv_obj.get_messages()
        self.assertEqual(len(final_gen_conv_messages), 3)  # system, user, assistant
        self.assertEqual(
            final_gen_conv_messages[0],
            {"role": "system", "content": "Iterative Gen System Message"},
        )
        self.assertEqual(
            final_gen_conv_messages[1], {"role": "user", "content": current_prompt_content}
        )
        self.assertEqual(
            final_gen_conv_messages[2], {"role": "assistant", "content": llm_core_response}
        )

        # 2. Verify _query_llm_core call
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_service,
            messages_to_send_raw=expected_messages_to_core,
            current_prompt_text_for_callback=current_prompt_content,
            stage=stage,
            step_description=f"{base_context} - Iteration Attempt {attempt_num}",
            iteration_num=attempt_num,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

        # 3. Verify logging
        log_msg1_expected = f"_query_for_iterative_generation: BaseContext='{base_context}', Attempt={attempt_num}, Stage='{stage.value}'"
        log_msg2_expected = f"Added iterative generation response to its dedicated conversation. Context: {base_context}, Attempt: {attempt_num}"

        # Check that the calls were made and get the actual calls
        actual_log_calls = self.orchestrator.logger.debug.call_args_list
        self.assertEqual(len(actual_log_calls), 2)  # Ensure exactly two calls

        # Compare the arguments of each call
        # call_args_list stores calls as call(args, kwargs)
        # so actual_log_calls[0] is the first call object
        # actual_log_calls[0][0] is the args tuple of the first call
        # actual_log_calls[0][0][0] is the first argument of the first call
        self.assertEqual(actual_log_calls[0][0][0], log_msg1_expected)
        self.assertEqual(actual_log_calls[1][0][0], log_msg2_expected)

    def test_qfig_stateless_happy_path(self):
        """Test _query_for_iterative_generation in stateless mode (no conversation object)."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = "System message for stateless"

        current_prompt_content = "Stateless prompt for generation."
        base_context = "Test Stateless Gen"
        attempt_num = 2
        stage = OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT
        llm_core_response = "Stateless response from LLM core."

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            result = self.orchestrator._query_for_iterative_generation(
                llm_service=mock_llm_service,
                current_prompt_content=current_prompt_content,
                generation_conversation_obj=None,  # Stateless mode
                base_context_for_logging=base_context,
                attempt_num=attempt_num,
                stage=stage,
                prompt_metadata_for_llm_core={"test": "metadata"},
                query_kwargs={"temperature": 0.5},
            )

        # Assert
        self.assertEqual(result, llm_core_response)

        # Verify _query_llm_core was called with correct stateless messages
        expected_messages = [
            {"role": "system", "content": "System message for stateless"},
            {"role": "user", "content": current_prompt_content},
        ]
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_service,
            messages_to_send_raw=expected_messages,
            current_prompt_text_for_callback=current_prompt_content,
            stage=stage,
            step_description=f"{base_context} - Iteration Attempt {attempt_num}",
            iteration_num=attempt_num,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data={"test": "metadata"},
            query_kwargs={"temperature": 0.5},
        )

        # Verify logging
        log_msg1_expected = f"_query_for_iterative_generation: BaseContext='{base_context}', Attempt={attempt_num}, Stage='{stage.value}'"
        self.orchestrator.logger.debug.assert_any_call(log_msg1_expected)

        # Should NOT log the "Added iterative generation response" message in stateless mode
        debug_calls = [call.args[0] for call in self.orchestrator.logger.debug.call_args_list]
        self.assertNotIn(
            f"Added iterative generation response to its dedicated conversation. Context: {base_context}, Attempt: {attempt_num}",
            debug_calls,
        )

    def test_qfig_stateless_no_system_message(self):
        """Test _query_for_iterative_generation in stateless mode when LLM has no system message."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = None  # No system message

        current_prompt_content = "Prompt without system message."
        base_context = "No System Test"
        attempt_num = 1
        stage = OrchestrationStage.SOLUTION_GENERATION_INITIAL
        llm_core_response = "Response without system."

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            result = self.orchestrator._query_for_iterative_generation(
                llm_service=mock_llm_service,
                current_prompt_content=current_prompt_content,
                generation_conversation_obj=None,
                base_context_for_logging=base_context,
                attempt_num=attempt_num,
                stage=stage,
            )

        # Assert
        self.assertEqual(result, llm_core_response)

        # Verify _query_llm_core was called with only user message (no system)
        expected_messages = [{"role": "user", "content": current_prompt_content}]
        mock_query_llm_core.assert_called_once_with(
            llm_service=mock_llm_service,
            messages_to_send_raw=expected_messages,
            current_prompt_text_for_callback=current_prompt_content,
            stage=stage,
            step_description=f"{base_context} - Iteration Attempt {attempt_num}",
            iteration_num=attempt_num,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
            prompt_source_entity_name="user",
            prompt_metadata_data=None,
            query_kwargs=None,
        )

    def test_qfig_error_propagation_llm_response_error(self):
        """Test that LLMResponseError from _query_llm_core is properly propagated."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = "System message"

        current_prompt_content = "Error prompt"
        base_context = "Error Test"
        attempt_num = 1
        stage = OrchestrationStage.SOLUTION_GENERATION_INITIAL

        llm_error = LLMResponseError("API rate limit exceeded")

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=llm_error, autospec=True
        ) as mock_query_llm_core:
            # Act & Assert
            with self.assertRaises(LLMResponseError) as cm:
                self.orchestrator._query_for_iterative_generation(
                    llm_service=mock_llm_service,
                    current_prompt_content=current_prompt_content,
                    generation_conversation_obj=None,
                    base_context_for_logging=base_context,
                    attempt_num=attempt_num,
                    stage=stage,
                )

        self.assertEqual(str(cm.exception), "API rate limit exceeded")
        mock_query_llm_core.assert_called_once()

    def test_qfig_error_propagation_llm_client_error(self):
        """Test that LLMClientError from _query_llm_core is properly propagated."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = "System message"

        current_prompt_content = "Client error prompt"
        base_context = "Client Error Test"
        attempt_num = 2
        stage = OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT

        client_error = LLMClientError("Network connection failed")

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=client_error, autospec=True
        ):
            # Act & Assert
            with self.assertRaises(LLMClientError) as cm:
                self.orchestrator._query_for_iterative_generation(
                    llm_service=mock_llm_service,
                    current_prompt_content=current_prompt_content,
                    generation_conversation_obj=None,
                    base_context_for_logging=base_context,
                    attempt_num=attempt_num,
                    stage=stage,
                )

        self.assertEqual(str(cm.exception), "Network connection failed")

    def test_qfig_error_propagation_llm_orchestration_error(self):
        """Test that LLMOrchestrationError from _query_llm_core is properly propagated."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = "System message"

        current_prompt_content = "Orchestration error prompt"
        base_context = "Orchestration Error Test"
        attempt_num = 1
        stage = OrchestrationStage.SOLUTION_GENERATION_INITIAL

        orchestration_error = LLMOrchestrationError("History preparation failed")

        with patch.object(
            self.orchestrator, "_query_llm_core", side_effect=orchestration_error, autospec=True
        ):
            # Act & Assert
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._query_for_iterative_generation(
                    llm_service=mock_llm_service,
                    current_prompt_content=current_prompt_content,
                    generation_conversation_obj=None,
                    base_context_for_logging=base_context,
                    attempt_num=attempt_num,
                    stage=stage,
                )

        self.assertEqual(str(cm.exception), "History preparation failed")

    def test_qfig_stateful_conversation_state_management(self):
        """Test that stateful mode properly manages conversation state."""
        # Arrange
        mock_llm_service = self.mock_solution_llm
        mock_llm_service.get_system_message.return_value = "Conversation system message"

        # Create a real Conversation object to test state management
        generation_conversation = Conversation(system_message="Conversation system message")
        # Add some existing history
        generation_conversation.add_assistant_message("Previous assistant response")

        current_prompt_content = "New user prompt for conversation"
        base_context = "Stateful Conversation Test"
        attempt_num = 3
        stage = OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT
        llm_core_response = "New assistant response"

        # Spy on the conversation methods
        spy_add_user = MagicMock(wraps=generation_conversation.add_user_message)
        spy_add_assistant = MagicMock(wraps=generation_conversation.add_assistant_message)
        spy_get_messages = MagicMock(wraps=generation_conversation.get_messages)

        generation_conversation.add_user_message = spy_add_user
        generation_conversation.add_assistant_message = spy_add_assistant
        generation_conversation.get_messages = spy_get_messages

        with patch.object(
            self.orchestrator, "_query_llm_core", return_value=llm_core_response, autospec=True
        ) as mock_query_llm_core:
            # Act
            result = self.orchestrator._query_for_iterative_generation(
                llm_service=mock_llm_service,
                current_prompt_content=current_prompt_content,
                generation_conversation_obj=generation_conversation,
                base_context_for_logging=base_context,
                attempt_num=attempt_num,
                stage=stage,
            )

        # Assert
        self.assertEqual(result, llm_core_response)

        # Verify conversation state management
        spy_add_user.assert_called_once_with(current_prompt_content)
        spy_get_messages.assert_called_once()
        spy_add_assistant.assert_called_once_with(llm_core_response)

        # Verify the conversation now has the expected messages
        final_messages = generation_conversation.get_messages()
        self.assertEqual(
            len(final_messages), 4
        )  # system + previous assistant + new user + new assistant
        self.assertEqual(final_messages[0]["role"], "system")
        self.assertEqual(final_messages[1]["role"], "assistant")
        self.assertEqual(final_messages[1]["content"], "Previous assistant response")
        self.assertEqual(final_messages[2]["role"], "user")
        self.assertEqual(final_messages[2]["content"], current_prompt_content)
        self.assertEqual(final_messages[3]["role"], "assistant")
        self.assertEqual(final_messages[3]["content"], llm_core_response)

        # Verify _query_llm_core was called with the full conversation history
        mock_query_llm_core.assert_called_once()
        call_args = mock_query_llm_core.call_args
        messages_sent = call_args[1]["messages_to_send_raw"]
        self.assertEqual(
            len(messages_sent), 3
        )  # system + previous assistant + new user (before LLM response)

        # Verify logging includes the stateful-specific log
        log_msg2_expected = f"Added iterative generation response to its dedicated conversation. Context: {base_context}, Attempt: {attempt_num}"
        self.orchestrator.logger.debug.assert_any_call(log_msg2_expected)

    # --- Tests for _generate_and_verify_result ---

    def test_gavr_happy_path_json_with_schema_first_attempt(self):
        """Test _generate_and_verify_result: Happy path with JSON result, schema validation, succeeds on first attempt."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Test system message"

        context = "Test Final Solution Generation"
        test_schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        use_conversation = True
        max_iterations = 3
        expect_json_output = True

        llm_response_content = '{"result": "success"}'
        expected_parsed_result = {"result": "success"}
        initial_improvement_prompt = "Build the initial JSON solution."

        # Set up the orchestrator state
        self.orchestrator.prompt_inputs = None  # Trigger fallback to _prepare_prompt_inputs

        def custom_prepare_inputs_side_effect():
            # Simulate _prepare_prompt_inputs creating and setting prompt_inputs
            mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
            self.orchestrator.prompt_inputs = mock_prompt_inputs
            return None  # _prepare_prompt_inputs returns None

        # Mock _query_for_iterative_generation and parse_json_response
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            return_value=llm_response_content,
            autospec=True,
        ) as mock_qfig, patch(
            "convorator.conversations.conversation_orchestrator.parse_json_response",
            return_value=expected_parsed_result,
            autospec=True,
        ) as mock_parse_json, patch.object(
            self.orchestrator,
            "_prepare_prompt_inputs",
            side_effect=custom_prepare_inputs_side_effect,
            autospec=True,
        ) as mock_prepare_inputs:

            # Mock the prompt builder to return our expected prompt
            self.orchestrator.prompt_builder.build_prompt.return_value = initial_improvement_prompt

            # Act
            result = self.orchestrator._generate_and_verify_result(
                llm_service=mock_llm_service,
                context=context,
                result_schema=test_schema,
                use_conversation=use_conversation,
                max_improvement_iterations=max_iterations,
                json_result=expect_json_output,
            )

            # Assert
            # 1. Verify the result
            self.assertEqual(result, expected_parsed_result)

            # 2. Verify _prepare_prompt_inputs was called due to prompt_inputs being None
            mock_prepare_inputs.assert_called_once()

            # 3. Verify prompt_builder.build_prompt was called for initial prompt
            self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(
                "improvement_prompt"
            )

            # 4. Verify _query_for_iterative_generation call
            mock_qfig.assert_called_once()
            call_args = mock_qfig.call_args
            self.assertEqual(call_args[1]["llm_service"], mock_llm_service)
            self.assertEqual(call_args[1]["current_prompt_content"], initial_improvement_prompt)
            self.assertEqual(call_args[1]["base_context_for_logging"], context)
            self.assertEqual(call_args[1]["attempt_num"], 1)
            self.assertEqual(call_args[1]["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL)
            # Verify conversation object was created
            self.assertIsNotNone(call_args[1]["generation_conversation_obj"])
            self.assertEqual(
                call_args[1]["generation_conversation_obj"].system_message, "Test system message"
            )

            # 5. Verify parse_json_response call
            mock_parse_json.assert_called_once_with(
                self.orchestrator.logger, llm_response_content, context, schema=test_schema
            )

            # 6. Verify logging
            self.orchestrator.logger.info.assert_any_call(
                f"Generating result for context: '{context}'. Max iterations: {max_iterations}. Expect JSON: {expect_json_output}."
            )
            self.orchestrator.logger.warning.assert_any_call(
                f"Context '{context}': PromptBuilderInputs was not prepared. Preparing now."
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generating result (Attempt 1/{max_iterations})"
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generation and JSON validation successful (Attempt 1)."
            )

    def test_gavr_happy_path_non_json_first_attempt(self):
        """Test _generate_and_verify_result: Happy path with non-JSON result, succeeds on first attempt."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Test system message"

        context = "Test Non-JSON Generation"
        use_conversation = True
        max_iterations = 2
        expect_json_output = False

        llm_response_content = "This is a plain text response from the LLM."
        initial_improvement_prompt = "Generate a text solution."

        # Set up the orchestrator state - prompt_inputs already prepared
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation (no need to mock parse_json_response for non-JSON)
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            return_value=llm_response_content,
            autospec=True,
        ) as mock_qfig:

            # Mock the prompt builder to return our expected prompt
            self.orchestrator.prompt_builder.build_prompt.return_value = initial_improvement_prompt

            # Act
            result = self.orchestrator._generate_and_verify_result(
                llm_service=mock_llm_service,
                context=context,
                result_schema=None,
                use_conversation=use_conversation,
                max_improvement_iterations=max_iterations,
                json_result=expect_json_output,
            )

            # Assert
            # 1. Verify the result is the raw response
            self.assertEqual(result, llm_response_content)

            # 2. Verify _prepare_prompt_inputs was NOT called since prompt_inputs was already set
            # (We don't patch it, so if it were called, it would use the real method)

            # 3. Verify prompt_builder.build_prompt was called for initial prompt
            self.orchestrator.prompt_builder.build_prompt.assert_called_once_with(
                "improvement_prompt"
            )

            # 4. Verify _query_for_iterative_generation call
            mock_qfig.assert_called_once()
            call_args = mock_qfig.call_args
            self.assertEqual(call_args[1]["llm_service"], mock_llm_service)
            self.assertEqual(call_args[1]["current_prompt_content"], initial_improvement_prompt)
            self.assertEqual(call_args[1]["base_context_for_logging"], context)
            self.assertEqual(call_args[1]["attempt_num"], 1)
            self.assertEqual(call_args[1]["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL)

            # 5. Verify logging for non-JSON success
            self.orchestrator.logger.info.assert_any_call(
                f"Generating result for context: '{context}'. Max iterations: {max_iterations}. Expect JSON: {expect_json_output}."
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generating result (Attempt 1/{max_iterations})"
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generation successful (non-JSON) (Attempt 1)."
            )

    def test_gavr_stateless_mode_json_success(self):
        """Test _generate_and_verify_result: Stateless mode (use_conversation=False) with JSON success."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Stateless system message"

        context = "Test Stateless Generation"
        test_schema = {"type": "object", "properties": {"data": {"type": "string"}}}
        use_conversation = False  # Key difference: stateless mode
        max_iterations = 3
        expect_json_output = True

        llm_response_content = '{"data": "stateless result"}'
        expected_parsed_result = {"data": "stateless result"}
        initial_improvement_prompt = "Generate JSON in stateless mode."

        # Set up the orchestrator state
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation and parse_json_response
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            return_value=llm_response_content,
            autospec=True,
        ) as mock_qfig, patch(
            "convorator.conversations.conversation_orchestrator.parse_json_response",
            return_value=expected_parsed_result,
            autospec=True,
        ) as mock_parse_json:

            # Mock the prompt builder to return our expected prompt
            self.orchestrator.prompt_builder.build_prompt.return_value = initial_improvement_prompt

            # Act
            result = self.orchestrator._generate_and_verify_result(
                llm_service=mock_llm_service,
                context=context,
                result_schema=test_schema,
                use_conversation=use_conversation,
                max_improvement_iterations=max_iterations,
                json_result=expect_json_output,
            )

            # Assert
            # 1. Verify the result
            self.assertEqual(result, expected_parsed_result)

            # 2. Verify _query_for_iterative_generation call with stateless mode
            mock_qfig.assert_called_once()
            call_args = mock_qfig.call_args
            self.assertEqual(call_args[1]["llm_service"], mock_llm_service)
            self.assertEqual(call_args[1]["current_prompt_content"], initial_improvement_prompt)
            self.assertEqual(call_args[1]["base_context_for_logging"], context)
            self.assertEqual(call_args[1]["attempt_num"], 1)
            self.assertEqual(call_args[1]["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL)
            # Key assertion: generation_conversation_obj should be None for stateless mode
            self.assertIsNone(call_args[1]["generation_conversation_obj"])

            # 3. Verify parse_json_response call
            mock_parse_json.assert_called_once_with(
                self.orchestrator.logger, llm_response_content, context, schema=test_schema
            )

    def test_gavr_llm_response_error_retry_then_success(self):
        """Test _generate_and_verify_result: LLMResponseError on first attempt, success on second attempt."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Retry test system message"

        context = "Test Retry After LLMResponseError"
        test_schema = {"type": "object", "properties": {"retry": {"type": "boolean"}}}
        use_conversation = True
        max_iterations = 3
        expect_json_output = True

        # First attempt fails, second succeeds
        first_attempt_error = LLMResponseError("Rate limit exceeded")
        second_attempt_response = '{"retry": true}'
        expected_parsed_result = {"retry": True}

        initial_improvement_prompt = "Generate initial solution."
        fix_prompt = "Fix the previous error: Rate limit exceeded"

        # Set up the orchestrator state
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation to fail first, succeed second
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            side_effect=[first_attempt_error, second_attempt_response],
            autospec=True,
        ) as mock_qfig, patch(
            "convorator.conversations.conversation_orchestrator.parse_json_response",
            return_value=expected_parsed_result,
            autospec=True,
        ) as mock_parse_json:

            # Mock the prompt builder to return different prompts for initial vs fix
            self.orchestrator.prompt_builder.build_prompt.side_effect = [
                initial_improvement_prompt,
                fix_prompt,
            ]

            # Act
            result = self.orchestrator._generate_and_verify_result(
                llm_service=mock_llm_service,
                context=context,
                result_schema=test_schema,
                use_conversation=use_conversation,
                max_improvement_iterations=max_iterations,
                json_result=expect_json_output,
            )

            # Assert
            # 1. Verify the result
            self.assertEqual(result, expected_parsed_result)

            # 2. Verify _query_for_iterative_generation was called twice
            self.assertEqual(mock_qfig.call_count, 2)

            # First call (initial attempt)
            first_call_args = mock_qfig.call_args_list[0]
            self.assertEqual(first_call_args[1]["attempt_num"], 1)
            self.assertEqual(
                first_call_args[1]["stage"], OrchestrationStage.SOLUTION_GENERATION_INITIAL
            )
            self.assertEqual(
                first_call_args[1]["current_prompt_content"], initial_improvement_prompt
            )

            # Second call (fix attempt)
            second_call_args = mock_qfig.call_args_list[1]
            self.assertEqual(second_call_args[1]["attempt_num"], 2)
            self.assertEqual(
                second_call_args[1]["stage"], OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT
            )
            self.assertEqual(second_call_args[1]["current_prompt_content"], fix_prompt)

            # 3. Verify prompt_builder.build_prompt was called twice
            self.assertEqual(self.orchestrator.prompt_builder.build_prompt.call_count, 2)
            self.orchestrator.prompt_builder.build_prompt.assert_any_call("improvement_prompt")
            self.orchestrator.prompt_builder.build_prompt.assert_any_call("fix_prompt")

            # 4. Verify parse_json_response was called only once (for the successful attempt)
            mock_parse_json.assert_called_once_with(
                self.orchestrator.logger, second_attempt_response, context, schema=test_schema
            )

            # 5. Verify error handling and fix prompt setup
            # The mock_prompt_inputs should have been updated with error details
            self.assertEqual(mock_prompt_inputs.errors_to_fix, "Rate limit exceeded")
            # response_to_fix should be "No response content received from LLM." since the first attempt raised an exception
            self.assertEqual(
                mock_prompt_inputs.response_to_fix, "No response content received from LLM."
            )

            # 6. Verify logging
            self.orchestrator.logger.warning.assert_any_call(
                f"Context '{context}': Attempt 1/{max_iterations} failed. Error: LLMResponseError: Rate limit exceeded"
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generated fix prompt for next attempt."
            )
            self.orchestrator.logger.info.assert_any_call(
                f"Context '{context}': Generation and JSON validation successful (Attempt 2)."
            )

    def test_gavr_max_iterations_exceeded_error(self):
        """Test _generate_and_verify_result: All attempts fail, MaxIterationsExceededError raised."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Max iterations test"

        context = "Test Max Iterations Exceeded"
        max_iterations = 2  # Small number for easier testing
        expect_json_output = True

        # All attempts will fail with different errors
        first_attempt_error = LLMResponseError("First error")
        second_attempt_error = SchemaValidationError("Schema validation failed")

        initial_improvement_prompt = "Generate initial solution."
        fix_prompt = "Fix the previous error."

        # Set up the orchestrator state
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation to always fail
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            side_effect=[first_attempt_error, second_attempt_error],
            autospec=True,
        ) as mock_qfig:

            # Mock the prompt builder
            self.orchestrator.prompt_builder.build_prompt.side_effect = [
                initial_improvement_prompt,
                fix_prompt,
            ]

            # Act & Assert
            with self.assertRaises(MaxIterationsExceededError) as cm:
                self.orchestrator._generate_and_verify_result(
                    llm_service=mock_llm_service,
                    context=context,
                    result_schema=None,
                    use_conversation=True,
                    max_improvement_iterations=max_iterations,
                    json_result=expect_json_output,
                )

            # Verify the exception message
            expected_error_msg = f"Context '{context}': Failed to produce a valid result after {max_iterations} attempts. Last error: SchemaValidationError: Schema validation failed"
            self.assertEqual(str(cm.exception), expected_error_msg)

            # Verify the exception chain
            self.assertIsInstance(cm.exception.__cause__, SchemaValidationError)

            # Verify both attempts were made
            self.assertEqual(mock_qfig.call_count, max_iterations)

            # Verify final error logging
            self.orchestrator.logger.error.assert_any_call(expected_error_msg)

    def test_gavr_llm_client_error_immediate_propagation(self):
        """Test _generate_and_verify_result: LLMClientError is immediately propagated without retry."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Client error test"

        context = "Test LLMClientError Propagation"
        max_iterations = 3
        expect_json_output = True

        # First attempt fails with LLMClientError (should not retry)
        client_error = LLMClientError("Network connection failed")
        initial_improvement_prompt = "Generate initial solution."

        # Set up the orchestrator state
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation to fail with LLMClientError
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            side_effect=client_error,
            autospec=True,
        ) as mock_qfig:

            # Mock the prompt builder
            self.orchestrator.prompt_builder.build_prompt.return_value = initial_improvement_prompt

            # Act & Assert
            with self.assertRaises(LLMClientError) as cm:
                self.orchestrator._generate_and_verify_result(
                    llm_service=mock_llm_service,
                    context=context,
                    result_schema=None,
                    use_conversation=True,
                    max_improvement_iterations=max_iterations,
                    json_result=expect_json_output,
                )

            # Verify the same exception is propagated
            self.assertEqual(str(cm.exception), "Network connection failed")

            # Verify only one attempt was made (no retry for LLMClientError)
            mock_qfig.assert_called_once()

            # Verify error logging
            self.orchestrator.logger.error.assert_any_call(
                f"Context '{context}': Non-recoverable LLM client error during generation (Attempt 1): Network connection failed",
                exc_info=True,
            )

    def test_gavr_fix_prompt_building_failure(self):
        """Test _generate_and_verify_result: Fix prompt building fails, LLMOrchestrationError raised."""
        # Arrange
        mock_llm_service = MagicMock(spec=LLMInterface)
        mock_llm_service.get_system_message.return_value = "Fix prompt failure test"

        context = "Test Fix Prompt Building Failure"
        max_iterations = 3
        expect_json_output = True

        # First attempt fails, then fix prompt building fails
        first_attempt_error = LLMResponseError("First attempt failed")
        fix_prompt_error = Exception("Prompt builder crashed")
        initial_improvement_prompt = "Generate initial solution."

        # Set up the orchestrator state
        mock_prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs = mock_prompt_inputs

        # Mock _query_for_iterative_generation to fail on first attempt
        with patch.object(
            self.orchestrator,
            "_query_for_iterative_generation",
            side_effect=first_attempt_error,
            autospec=True,
        ) as mock_qfig:

            # Mock the prompt builder to succeed first, then fail on fix_prompt
            self.orchestrator.prompt_builder.build_prompt.side_effect = [
                initial_improvement_prompt,
                fix_prompt_error,
            ]

            # Act & Assert
            with self.assertRaises(LLMOrchestrationError) as cm:
                self.orchestrator._generate_and_verify_result(
                    llm_service=mock_llm_service,
                    context=context,
                    result_schema=None,
                    use_conversation=True,
                    max_improvement_iterations=max_iterations,
                    json_result=expect_json_output,
                )

            # Verify the exception message includes context about fix prompt failure
            expected_error_msg = f"Context '{context}': Failed to build fix prompt after error: Prompt builder crashed"
            self.assertEqual(str(cm.exception), expected_error_msg)

            # Verify the exception chain
            self.assertIsInstance(cm.exception.__cause__, Exception)
            self.assertEqual(str(cm.exception.__cause__), "Prompt builder crashed")

            # Verify only one LLM attempt was made (failed, then fix prompt building failed)
            mock_qfig.assert_called_once()

            # Verify error logging for fix prompt failure
            self.orchestrator.logger.error.assert_any_call(
                f"Context '{context}': Failed to build fix prompt: Prompt builder crashed",
                exc_info=True,
            )

    # --- Tests for _run_moderated_debate ---

    def test_rmd_prompt_inputs_none(self):
        """Test _run_moderated_debate when self.prompt_inputs is None."""
        # Arrange
        # Ensure prompt_inputs is None. The setUp might have prepared it via _prepare_prompt_inputs
        # if orchestrator was created differently, but for this unit test, directly set it.
        self.orchestrator.prompt_inputs = None
        self.orchestrator.prompt_builder = MagicMock(spec=PromptBuilder)  # Need a mock builder

        # Act & Assert
        with self.assertRaisesRegex(
            LLMOrchestrationError, "Prompt inputs not prepared before running debate."
        ):
            self.orchestrator._run_moderated_debate()

        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for {self.orchestrator.debate_iterations} iterations."
        )

    def test_rmd_prompt_builder_inputs_not_defined(self):
        """Test _run_moderated_debate when prompt_builder.is_inputs_defined() is False."""
        # Arrange
        # We need a valid prompt_inputs object for the first part of the condition
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)

        # Mock prompt_builder and its is_inputs_defined method
        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = False
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Act & Assert
        with self.assertRaisesRegex(
            LLMOrchestrationError, "Prompt inputs not prepared before running debate."
        ):
            self.orchestrator._run_moderated_debate()

        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for {self.orchestrator.debate_iterations} iterations."
        )

    def test_rmd_initial_prompt_build_fails(self):
        """Test _run_moderated_debate when prompt_builder.build_prompt('initial_prompt') fails."""
        # Arrange
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.side_effect = ValueError("Failed to build initial prompt")
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Act & Assert
        with self.assertRaisesRegex(ValueError, "Failed to build initial prompt"):
            self.orchestrator._run_moderated_debate()

        # Verify build_prompt was called for 'initial_prompt'
        mock_prompt_builder.build_prompt.assert_called_once_with("initial_prompt")
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for {self.orchestrator.debate_iterations} iterations."
        )

    def test_rmd_zero_iterations(self):
        """Test _run_moderated_debate with debate_iterations = 0."""
        # Arrange
        self.orchestrator.debate_iterations = 0
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Initial prompt for debater"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock the methods that would make LLM calls
        # _query_agent_in_debate_turn is called directly for the initial debater turn
        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value="Debater initial response",
        ) as mock_query_initial_debater:
            with patch.object(
                self.orchestrator, "_execute_agent_loop_turn"
            ) as mock_execute_loop_turn:
                # Act
                self.orchestrator._run_moderated_debate()

        # Assert
        # Initial logger call
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 0 iterations."
        )
        # Prompt builder for initial prompt
        mock_prompt_builder.build_prompt.assert_called_once_with("initial_prompt")
        # initial_prompt_content should be set on prompt_inputs
        self.assertEqual(
            self.orchestrator.prompt_inputs.initial_prompt_content, "Initial prompt for debater"
        )

        # Initial debater turn executed
        mock_query_initial_debater.assert_called_once_with(
            self.orchestrator.debater_llm,
            self.orchestrator.debater_name,
            "Initial prompt for debater",
        )

        # Loop turns should NOT be executed
        mock_execute_loop_turn.assert_not_called()

        # Final log message
        self.orchestrator.logger.info.assert_any_call("Moderated debate phase completed.")

    def test_rmd_one_iteration(self):
        """Test _run_moderated_debate with debate_iterations = 1."""
        # Arrange
        self.orchestrator.debate_iterations = 1
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.current_iteration_num = 0  # Initial state

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        # Simulate different prompts being built
        prompt_values = {
            "initial_prompt": "Initial prompt for debater",
            "primary_prompt": "Prompt for Primary",
            "moderator_context": "Context for Moderator",
        }
        mock_prompt_builder.build_prompt.side_effect = lambda P_type: prompt_values.get(
            P_type, f"Unknown prompt {P_type}"
        )
        self.orchestrator.prompt_builder = mock_prompt_builder

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value="Debater initial response",
        ) as mock_query_initial_debater:
            with patch.object(
                self.orchestrator, "_execute_agent_loop_turn"
            ) as mock_execute_loop_turn:
                # Make _execute_agent_loop_turn return different values based on agent
                def loop_turn_side_effect(agent_name):
                    if agent_name == self.orchestrator.primary_name:
                        return "Primary response in loop"
                    elif agent_name == self.orchestrator.moderator_name:
                        return "Moderator response in loop"
                    return "Unknown loop response"

                mock_execute_loop_turn.side_effect = loop_turn_side_effect

                # Act
                self.orchestrator._run_moderated_debate()

        # Assert
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 1 iterations."
        )
        self.assertEqual(
            self.orchestrator.prompt_inputs.initial_prompt_content, "Initial prompt for debater"
        )

        # Initial Debater turn
        mock_query_initial_debater.assert_called_once_with(
            self.orchestrator.debater_llm,
            self.orchestrator.debater_name,
            "Initial prompt for debater",
        )

        # Loop turns assertions
        self.assertEqual(mock_execute_loop_turn.call_count, 2)  # Primary, Moderator
        calls = mock_execute_loop_turn.call_args_list

        # Round 1: Primary
        calls[0].assert_called_with(self.orchestrator.primary_name)
        self.orchestrator.logger.info.assert_any_call(f"Starting Round 1/1.")
        # Check that current_iteration_num was set to 1 for prompt_inputs for this round
        # This is a bit tricky as the property setter updates prompt_inputs directly.
        # We can check the logger message that implies current_iteration_num was 1.
        # A more direct check could be to mock the setter or inspect prompt_inputs if it has a history of values.
        # For now, relying on the logger and correct execution order.

        # Round 1: Moderator
        calls[1].assert_called_with(self.orchestrator.moderator_name)

        # current_iteration_num should be 1 at the end of the loop for this test
        self.assertEqual(self.orchestrator.current_iteration_num, 1)
        # And by extension, prompt_inputs.current_iteration_num should also be 1
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 1)

        self.orchestrator.logger.info.assert_any_call(
            "Max iterations reached. Ending debate after moderator feedback."
        )
        self.orchestrator.logger.info.assert_any_call("Moderated debate phase completed.")

    def test_rmd_multiple_iterations(self):
        """Test _run_moderated_debate with debate_iterations = 2."""
        # Arrange
        self.orchestrator.debate_iterations = 2
        # Initialize prompt_inputs and set its current_iteration_num attribute
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        # We need to ensure prompt_inputs has current_iteration_num attribute for the setter
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0  # Initial state for the orchestrator property

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        prompt_values = {
            "initial_prompt": "Initial prompt for debater",
            "primary_prompt": "Prompt for Primary",
            "moderator_context": "Context for Moderator",
            "debater_prompt": "Prompt for Debater",
        }
        mock_prompt_builder.build_prompt.side_effect = lambda p_type: prompt_values.get(
            p_type, f"Unknown prompt {p_type}"
        )
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Store calls to current_iteration_num setter via prompt_inputs mock
        iteration_updates_to_prompt_inputs = []

        def mock_setter(value):
            iteration_updates_to_prompt_inputs.append(value)
            # Actually set the value on the mock so it can be read by other parts of the code if necessary
            self.orchestrator.prompt_inputs._current_iteration_num_val = value

        # Make prompt_inputs.current_iteration_num a property mock to track updates
        # This requires a bit more elaborate mocking for the property setter on a MagicMock
        # We will mock the actual `prompt_inputs` object's `current_iteration_num` attribute directly if it were a real object.
        # Since `prompt_inputs` is a MagicMock, we can observe changes to `orchestrator.current_iteration_num`
        # and verify `prompt_inputs.current_iteration_num` is also updated by the property setter in the orchestrator.

        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            return_value="Debater initial response",
        ) as mock_query_initial_debater:
            with patch.object(
                self.orchestrator, "_execute_agent_loop_turn"
            ) as mock_execute_loop_turn:

                def loop_turn_side_effect(agent_name):
                    # Simulate that prompt_inputs.current_iteration_num was set correctly by the orchestrator's property
                    iteration_updates_to_prompt_inputs.append(
                        self.orchestrator.prompt_inputs.current_iteration_num
                    )
                    if agent_name == self.orchestrator.primary_name:
                        return f"Primary response loop {self.orchestrator.current_iteration_num}"
                    elif agent_name == self.orchestrator.moderator_name:
                        return f"Moderator response loop {self.orchestrator.current_iteration_num}"
                    elif agent_name == self.orchestrator.debater_name:
                        return f"Debater response loop {self.orchestrator.current_iteration_num}"
                    return "Unknown loop response"

                mock_execute_loop_turn.side_effect = loop_turn_side_effect

                # Act
                self.orchestrator._run_moderated_debate()

        # Assert
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 2 iterations."
        )
        self.assertEqual(
            self.orchestrator.prompt_inputs.initial_prompt_content, "Initial prompt for debater"
        )
        mock_query_initial_debater.assert_called_once_with(
            self.orchestrator.debater_llm,
            self.orchestrator.debater_name,
            "Initial prompt for debater",
        )

        # Loop turns: Primary, Mod, Debater (Round 1) + Primary, Mod (Round 2) = 5 calls
        self.assertEqual(mock_execute_loop_turn.call_count, 5)
        calls = mock_execute_loop_turn.call_args_list

        # Round 1
        self.orchestrator.logger.info.assert_any_call(f"Starting Round 1/2.")
        calls[0].assert_called_with(self.orchestrator.primary_name)  # Primary R1
        calls[1].assert_called_with(self.orchestrator.moderator_name)  # Moderator R1
        calls[2].assert_called_with(self.orchestrator.debater_name)  # Debater R1

        # Round 2
        self.orchestrator.logger.info.assert_any_call(f"Starting Round 2/2.")
        calls[3].assert_called_with(self.orchestrator.primary_name)  # Primary R2
        calls[4].assert_called_with(self.orchestrator.moderator_name)  # Moderator R2

        # Check iteration updates recorded via prompt_inputs mock
        # The orchestrator.current_iteration_num setter should have updated prompt_inputs.current_iteration_num
        # The side_effect for mock_execute_loop_turn records the state of prompt_inputs.current_iteration_num *at the time of the call*
        expected_iteration_sequence_in_prompt_inputs = [
            1,
            1,
            1,
            2,
            2,
        ]  # Corresponds to P1, M1, D1, P2, M2
        self.assertEqual(
            iteration_updates_to_prompt_inputs, expected_iteration_sequence_in_prompt_inputs
        )

        # Orchestrator's current_iteration_num should be 2 at the end
        self.assertEqual(self.orchestrator.current_iteration_num, 2)
        # And prompt_inputs.current_iteration_num should reflect the last update
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 2)

        self.orchestrator.logger.info.assert_any_call(
            "Max iterations reached. Ending debate after moderator feedback."
        )
        self.orchestrator.logger.info.assert_any_call("Moderated debate phase completed.")

    def test_rmd_initial_debater_turn_fails(self):
        """Test _run_moderated_debate when the initial debater turn fails."""
        # Arrange
        self.orchestrator.debate_iterations = (
            1  # Needs at least one iteration to reach the loop if initial fails
        )
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Initial prompt for debater"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn to raise an error for the initial call
        with patch.object(
            self.orchestrator,
            "_query_agent_in_debate_turn",
            side_effect=LLMResponseError("Initial Debater Failed"),
        ) as mock_query_initial_debater:
            with patch.object(
                self.orchestrator, "_execute_agent_loop_turn"
            ) as mock_execute_loop_turn:
                # Act & Assert
                with self.assertRaisesRegex(LLMResponseError, "Initial Debater Failed"):
                    self.orchestrator._run_moderated_debate()

        # Assertions
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for {self.orchestrator.debate_iterations} iterations."
        )
        mock_prompt_builder.build_prompt.assert_called_once_with("initial_prompt")
        self.assertEqual(
            self.orchestrator.prompt_inputs.initial_prompt_content, "Initial prompt for debater"
        )

        # Initial debater turn was attempted
        mock_query_initial_debater.assert_called_once_with(
            self.orchestrator.debater_llm,
            self.orchestrator.debater_name,
            "Initial prompt for debater",
        )

        # Loop turns should NOT be executed because the initial turn failed
        mock_execute_loop_turn.assert_not_called()

        # Error should be logged
        self.orchestrator.logger.error.assert_any_call(
            "Error during moderated debate: Initial Debater Failed", exc_info=True
        )

    def test_rmd_primary_turn_fails(self):
        """Test _run_moderated_debate when the Primary agent's turn fails in the loop."""
        # Arrange
        self.orchestrator.debate_iterations = 1
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Some prompt"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn for the initial debater call
        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="Debater initial OK"
        ) as mock_query_initial_debater:
            # Mock _execute_agent_loop_turn to fail for Primary
            with patch.object(
                self.orchestrator,
                "_execute_agent_loop_turn",
                side_effect=LLMOrchestrationError("Primary Turn Failed"),
            ) as mock_execute_loop_turn:
                # Act & Assert
                with self.assertRaisesRegex(LLMOrchestrationError, "Primary Turn Failed"):
                    self.orchestrator._run_moderated_debate()

        # Assertions
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 1 iterations."
        )
        mock_query_initial_debater.assert_called_once()  # Initial debater turn should have happened

        # _execute_agent_loop_turn should have been called for Primary, which then raised the error
        mock_execute_loop_turn.assert_called_once_with(self.orchestrator.primary_name)
        self.orchestrator.logger.info.assert_any_call(f"Starting Round 1/1.")
        self.assertEqual(
            self.orchestrator.current_iteration_num, 1
        )  # Iteration number updated before failure
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 1)

        # Error should be logged
        self.orchestrator.logger.error.assert_any_call(
            "Error during moderated debate: Primary Turn Failed", exc_info=True
        )

    def test_rmd_moderator_turn_fails(self):
        """Test _run_moderated_debate when the Moderator's turn fails in the loop."""
        # Arrange
        self.orchestrator.debate_iterations = 1
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Some prompt"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn for the initial debater call
        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="Debater initial OK"
        ) as mock_query_initial_debater:
            # Mock _execute_agent_loop_turn to succeed for Primary, then fail for Moderator
            def execute_loop_turn_side_effect(agent_name):
                if agent_name == self.orchestrator.primary_name:
                    return "Primary OK"
                elif agent_name == self.orchestrator.moderator_name:
                    raise LLMResponseError("Moderator Turn Failed")
                return "Should not be called for Debater in this test setup"

            with patch.object(
                self.orchestrator,
                "_execute_agent_loop_turn",
                side_effect=execute_loop_turn_side_effect,
            ) as mock_execute_loop_turn:
                # Act & Assert
                with self.assertRaisesRegex(LLMResponseError, "Moderator Turn Failed"):
                    self.orchestrator._run_moderated_debate()

        # Assertions
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 1 iterations."
        )
        mock_query_initial_debater.assert_called_once()  # Initial debater turn

        # _execute_agent_loop_turn called for Primary then Moderator
        self.assertEqual(mock_execute_loop_turn.call_count, 2)
        mock_execute_loop_turn.assert_any_call(self.orchestrator.primary_name)
        mock_execute_loop_turn.assert_any_call(self.orchestrator.moderator_name)

        self.orchestrator.logger.info.assert_any_call(f"Starting Round 1/1.")
        self.assertEqual(self.orchestrator.current_iteration_num, 1)
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 1)

        # Error should be logged
        self.orchestrator.logger.error.assert_any_call(
            "Error during moderated debate: Moderator Turn Failed", exc_info=True
        )

    def test_rmd_loop_debater_turn_fails(self):
        """Test _run_moderated_debate when the Debater's turn fails in the loop."""
        # Arrange
        self.orchestrator.debate_iterations = (
            2  # Need 2 iterations for debater to have a turn in the loop
        )
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Some prompt"
        self.orchestrator.prompt_builder = mock_prompt_builder

        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="Debater initial OK"
        ) as mock_query_initial_debater:

            def execute_loop_turn_side_effect(agent_name):
                if (
                    agent_name == self.orchestrator.primary_name
                    and self.orchestrator.current_iteration_num == 1
                ):
                    return "Primary R1 OK"
                elif (
                    agent_name == self.orchestrator.moderator_name
                    and self.orchestrator.current_iteration_num == 1
                ):
                    return "Moderator R1 OK"
                elif (
                    agent_name == self.orchestrator.debater_name
                    and self.orchestrator.current_iteration_num == 1
                ):
                    raise LLMResponseError("Loop Debater Turn Failed")
                # Should not reach Primary/Moderator R2 if Debater R1 fails
                return f"Unexpected call to {agent_name} in iter {self.orchestrator.current_iteration_num}"

            with patch.object(
                self.orchestrator,
                "_execute_agent_loop_turn",
                side_effect=execute_loop_turn_side_effect,
            ) as mock_execute_loop_turn:
                with self.assertRaisesRegex(LLMResponseError, "Loop Debater Turn Failed"):
                    self.orchestrator._run_moderated_debate()

        # Assertions
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 2 iterations."
        )
        mock_query_initial_debater.assert_called_once()

        # Loop turns: Primary R1, Moderator R1, Debater R1 (fails)
        self.assertEqual(mock_execute_loop_turn.call_count, 3)
        mock_execute_loop_turn.assert_any_call(self.orchestrator.primary_name)
        mock_execute_loop_turn.assert_any_call(self.orchestrator.moderator_name)
        mock_execute_loop_turn.assert_any_call(self.orchestrator.debater_name)

        self.orchestrator.logger.info.assert_any_call(f"Starting Round 1/2.")
        self.assertEqual(self.orchestrator.current_iteration_num, 1)  # Failed in R1
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 1)

        self.orchestrator.logger.error.assert_any_call(
            "Error during moderated debate: Loop Debater Turn Failed", exc_info=True
        )

    def test_rmd_unexpected_exception_in_loop(self):
        """Test _run_moderated_debate wraps unexpected exceptions in LLMOrchestrationError."""
        # Arrange
        self.orchestrator.debate_iterations = 1
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Some prompt"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn for the initial debater call
        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="Debater initial OK"
        ) as mock_query_initial_debater:
            # Mock _execute_agent_loop_turn to raise a generic TypeError for Primary
            with patch.object(
                self.orchestrator,
                "_execute_agent_loop_turn",
                side_effect=TypeError("Unexpected generic error"),
            ) as mock_execute_loop_turn:
                # Act & Assert
                with self.assertRaisesRegex(
                    LLMOrchestrationError,
                    "Unexpected error during moderated debate: Unexpected generic error",
                ):
                    self.orchestrator._run_moderated_debate()

        # Assertions
        mock_execute_loop_turn.assert_called_once_with(self.orchestrator.primary_name)
        self.orchestrator.logger.error.assert_any_call(
            "Unexpected error during moderated debate: Unexpected generic error", exc_info=True
        )

    def test_rmd_correct_prompt_types_built(self):
        """Test that prompt_builder.build_prompt is called with correct types in sequence."""
        # Arrange
        self.orchestrator.debate_iterations = 2
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        # This is the mock we will inspect for build_prompt calls
        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True
        mock_prompt_builder.build_prompt.return_value = "Generated prompt content"
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn, which is called by the initial setup
        # AND by _execute_agent_loop_turn (which we are NOT mocking directly here).
        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="LLM Agent Response"
        ) as mock_query_agent_turn:

            # Act
            self.orchestrator._run_moderated_debate()

        # Assert
        expected_prompt_build_calls = [
            call("initial_prompt"),  # For initial debater query in _run_moderated_debate
            call("primary_prompt"),  # R1 Primary (via _execute_agent_loop_turn)
            call("moderator_context"),  # R1 Moderator (via _execute_agent_loop_turn)
            call("debater_prompt"),  # R1 Debater (via _execute_agent_loop_turn)
            call("primary_prompt"),  # R2 Primary (via _execute_agent_loop_turn)
            call("moderator_context"),  # R2 Moderator (via _execute_agent_loop_turn)
        ]

        actual_calls = mock_prompt_builder.build_prompt.call_args_list
        self.assertEqual(actual_calls, expected_prompt_build_calls)

        # Sanity check number of LLM calls (via _query_agent_in_debate_turn)
        # Initial Debater + (Primary, Mod, Debater for R1) + (Primary, Mod for R2) = 1 + 3 + 2 = 6 calls
        self.assertEqual(mock_query_agent_turn.call_count, 6)

    def test_rmd_prompt_build_fails_during_loop(self):
        """Test _run_moderated_debate when prompt building fails during loop execution."""
        # Arrange
        self.orchestrator.debate_iterations = 1
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        self.orchestrator.prompt_inputs.current_iteration_num = 0
        self.orchestrator.current_iteration_num = 0

        mock_prompt_builder = MagicMock(spec=PromptBuilder)
        mock_prompt_builder.is_inputs_defined.return_value = True

        # Set up build_prompt to succeed for initial_prompt but fail for primary_prompt
        def build_prompt_side_effect(prompt_type):
            if prompt_type == "initial_prompt":
                return "Initial prompt for debater"
            elif prompt_type == "primary_prompt":
                raise ValueError("Failed to build primary prompt during loop")
            else:
                return f"Unexpected prompt type: {prompt_type}"

        mock_prompt_builder.build_prompt.side_effect = build_prompt_side_effect
        self.orchestrator.prompt_builder = mock_prompt_builder

        # Mock _query_agent_in_debate_turn for the initial debater call
        with patch.object(
            self.orchestrator, "_query_agent_in_debate_turn", return_value="Debater initial OK"
        ) as mock_query_initial_debater:
            # Act & Assert
            # The error should be wrapped in LLMOrchestrationError by the generic exception handler
            with self.assertRaisesRegex(
                LLMOrchestrationError,
                "Unexpected error during moderated debate: Failed to build primary prompt during loop",
            ):
                self.orchestrator._run_moderated_debate()

        # Assertions
        self.orchestrator.logger.info.assert_any_call(
            f"Conducting moderated debate for 1 iterations."
        )

        # Initial prompt should have been built successfully
        mock_prompt_builder.build_prompt.assert_any_call("initial_prompt")

        # Initial debater turn should have succeeded
        mock_query_initial_debater.assert_called_once_with(
            self.orchestrator.debater_llm,
            self.orchestrator.debater_name,
            "Initial prompt for debater",
        )

        # primary_prompt should have been attempted during the loop
        mock_prompt_builder.build_prompt.assert_any_call("primary_prompt")

        # Verify the iteration number was updated before the failure
        self.assertEqual(self.orchestrator.current_iteration_num, 1)
        self.assertEqual(self.orchestrator.prompt_inputs.current_iteration_num, 1)

        # Error should be logged
        self.orchestrator.logger.error.assert_any_call(
            "Unexpected error during moderated debate: Failed to build primary prompt during loop",
            exc_info=True,
        )

    def test_run_synthesize_summary_fails(self):
        """Test run method when _synthesize_summary fails with LLMOrchestrationError."""
        # Arrange
        error_message = "Summary synthesis failed"
        # Mock internal methods
        with patch.object(
            self.orchestrator, "_prepare_prompt_inputs", return_value=None
        ) as mock_prepare_inputs:
            with patch.object(
                self.orchestrator, "_run_moderated_debate", return_value=None
            ) as mock_run_debate:
                with patch.object(
                    self.orchestrator,
                    "_synthesize_summary",
                    side_effect=LLMOrchestrationError(error_message),
                ) as mock_synthesize_summary:
                    with patch.object(
                        self.orchestrator, "_generate_final_solution"
                    ) as mock_generate_solution:
                        # Act & Assert
                        with self.assertRaisesRegex(LLMOrchestrationError, error_message) as cm:
                            self.orchestrator.run()

                        self.assertIsInstance(cm.exception, LLMOrchestrationError)

        # Verify calls up to the point of failure
        mock_prepare_inputs.assert_called_once()
        mock_run_debate.assert_called_once()
        mock_synthesize_summary.assert_called_once()
        # Verify that subsequent methods were NOT called
        mock_generate_solution.assert_not_called()

        # Verify logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

    def test_run_generate_final_solution_fails(self):
        """Test run method when _generate_final_solution fails with SchemaValidationError."""
        # Arrange
        error_message = "Final solution schema validation failed"
        # Mock internal methods
        with patch.object(
            self.orchestrator, "_prepare_prompt_inputs", return_value=None
        ) as mock_prepare_inputs:
            with patch.object(
                self.orchestrator, "_run_moderated_debate", return_value=None
            ) as mock_run_debate:
                with patch.object(
                    self.orchestrator, "_synthesize_summary", return_value=None
                ) as mock_synthesize_summary:
                    with patch.object(
                        self.orchestrator,
                        "_generate_final_solution",
                        side_effect=SchemaValidationError(error_message),
                    ) as mock_generate_solution:
                        # Act & Assert
                        with self.assertRaisesRegex(SchemaValidationError, error_message) as cm:
                            self.orchestrator.run()

                        self.assertIsInstance(cm.exception, SchemaValidationError)

        # Verify calls up to the point of failure
        mock_prepare_inputs.assert_called_once()
        mock_run_debate.assert_called_once()
        mock_synthesize_summary.assert_called_once()
        mock_generate_solution.assert_called_once()

        # Verify logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

    def test_run_unexpected_error_in_prepare_prompt_inputs(self):
        """Test run method when _prepare_prompt_inputs fails with an unexpected error."""
        # Arrange
        original_error_message = "Unexpected issue during input prep"
        original_exception = ValueError(original_error_message)  # A generic exception

        # Mock _prepare_prompt_inputs to raise the generic error
        with patch.object(
            self.orchestrator, "_prepare_prompt_inputs", side_effect=original_exception
        ) as mock_prepare_inputs:
            with patch.object(self.orchestrator, "_run_moderated_debate") as mock_run_debate:
                with patch.object(
                    self.orchestrator, "_synthesize_summary"
                ) as mock_synthesize_summary:
                    with patch.object(
                        self.orchestrator, "_generate_final_solution"
                    ) as mock_generate_solution:

                        # Act & Assert
                        with self.assertRaisesRegex(
                            LLMOrchestrationError,
                            f"Unexpected error during orchestration: {original_error_message}",
                        ) as cm:
                            self.orchestrator.run()

                        self.assertIsInstance(cm.exception, LLMOrchestrationError)
                        self.assertIs(
                            cm.exception.__cause__, original_exception
                        )  # Check that the original exception is wrapped

        # Verify that _prepare_prompt_inputs was called
        mock_prepare_inputs.assert_called_once()
        # Verify that subsequent methods were NOT called
        mock_run_debate.assert_not_called()
        mock_synthesize_summary.assert_not_called()
        mock_generate_solution.assert_not_called()

        # Verify logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        # The run method's except block logs the original error message directly
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {original_error_message}"
        )

    def test_run_unexpected_error_in_run_moderated_debate(self):
        """Test run method when _run_moderated_debate fails with an unexpected error."""
        # Arrange
        original_error_message = "Unexpected issue during debate"
        original_exception = TypeError(original_error_message)

        with patch.object(
            self.orchestrator, "_prepare_prompt_inputs", return_value=None
        ) as mock_prepare_inputs:
            with patch.object(
                self.orchestrator, "_run_moderated_debate", side_effect=original_exception
            ) as mock_run_debate:
                with patch.object(
                    self.orchestrator, "_synthesize_summary"
                ) as mock_synthesize_summary:
                    with patch.object(
                        self.orchestrator, "_generate_final_solution"
                    ) as mock_generate_solution:
                        with self.assertRaisesRegex(
                            LLMOrchestrationError,
                            f"Unexpected error during orchestration: {original_error_message}",
                        ) as cm:
                            self.orchestrator.run()

                        self.assertIsInstance(cm.exception, LLMOrchestrationError)
                        self.assertIs(cm.exception.__cause__, original_exception)

        mock_prepare_inputs.assert_called_once()
        mock_run_debate.assert_called_once()
        mock_synthesize_summary.assert_not_called()
        mock_generate_solution.assert_not_called()
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {original_error_message}"
        )

    def test_run_unexpected_error_in_synthesize_summary(self):
        # Test that an unexpected error in _synthesize_summary is caught, logged, and wrapped
        # Arrange
        self.orchestrator._prepare_prompt_inputs = MagicMock()  # Succeeds
        self.orchestrator._run_moderated_debate = MagicMock()  # Succeeds
        original_summary_method = self.orchestrator._synthesize_summary
        self.orchestrator._synthesize_summary = MagicMock(
            side_effect=ValueError("Unexpected problem during summary synthesis")
        )
        self.orchestrator._generate_final_solution = MagicMock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertIn(
            "Unexpected error during orchestration: Unexpected problem during summary synthesis",
            str(cm.exception),
        )

        # Ensure methods before the failing one were called, but the one after was not
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            "Error during orchestrated solution improvement: Unexpected problem during summary synthesis"
        )
        self.orchestrator._synthesize_summary = original_summary_method

    def test_run_unexpected_error_in_generate_final_solution(self):
        """Test run method when _generate_final_solution fails with an unexpected error."""
        # Arrange
        original_error_message = "Unexpected issue during final solution generation"
        original_exception = KeyError(original_error_message)

        with patch.object(
            self.orchestrator, "_prepare_prompt_inputs", return_value=None
        ) as mock_prepare_inputs:
            with patch.object(
                self.orchestrator, "_run_moderated_debate", return_value=None
            ) as mock_run_debate:
                with patch.object(
                    self.orchestrator, "_synthesize_summary", return_value=None
                ) as mock_synthesize_summary:
                    with patch.object(
                        self.orchestrator,
                        "_generate_final_solution",
                        side_effect=original_exception,
                    ) as mock_generate_solution:
                        with self.assertRaisesRegex(
                            LLMOrchestrationError,
                            f"Unexpected error during orchestration: {original_error_message}",
                        ) as cm:
                            self.orchestrator.run()

                        self.assertIsInstance(cm.exception, LLMOrchestrationError)
                        self.assertIs(cm.exception.__cause__, original_exception)

        mock_prepare_inputs.assert_called_once()
        mock_run_debate.assert_called_once()
        mock_synthesize_summary.assert_called_once()
        mock_generate_solution.assert_called_once()
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {original_error_message}"
        )

    def test_generate_final_solution_general_exception_wrapped(self):
        # Arrange
        self.orchestrator.prompt_inputs = MagicMock(spec=PromptBuilderInputs)
        mock_generate_and_verify = MagicMock(side_effect=ValueError("Unexpected problem"))
        self.orchestrator._generate_and_verify_result = mock_generate_and_verify

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator._generate_final_solution()
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertIn("Unexpected error during final solution generation", str(cm.exception))
        mock_generate_and_verify.assert_called_once()
        self.orchestrator.logger.error.assert_called_with(
            "Unexpected error during final solution generation: Unexpected problem", exc_info=True
        )
        self.orchestrator.logger.info.assert_any_call(
            f"Step 4: Generating improved solution (max {self.orchestrator.improvement_iterations} attempts)."
        )

    def test_run_happy_path(self):
        # Test the successful execution path of the run() method
        # Arrange
        # Mock the main internal methods called by run()
        self.orchestrator._prepare_prompt_inputs = MagicMock()
        self.orchestrator._run_moderated_debate = MagicMock()
        self.orchestrator._synthesize_summary = MagicMock()
        mock_final_solution_result = "The final perfect solution"
        self.orchestrator._generate_final_solution = MagicMock(
            return_value=mock_final_solution_result
        )

        # Act
        result = self.orchestrator.run()

        # Assert
        # Check that the result is what _generate_final_solution returned
        self.assertEqual(result, mock_final_solution_result)

        # Check that all methods were called in order
        expected_calls = [
            call._prepare_prompt_inputs(),
            call._run_moderated_debate(),
            call._synthesize_summary(),
            call._generate_final_solution(),
        ]
        # To check the order, we can look at the call order on the orchestrator's mock itself,
        # if we had mocked the orchestrator. Instead, we check calls on individual mocks.
        # A more robust way is to use a Mock manager if call order across different mocks is critical,
        # or ensure logger calls delineate steps. For now, individual checks are fine.

        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Verify log messages indicating start and end
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.info.assert_any_call(
            "Orchestrated solution improvement workflow finished successfully."
        )

        # Check that no error logs were made
        # Filter out error calls that might be part of other mocked methods' side effects if any were complex.
        # For simple MagicMocks, this is straightforward.
        error_log_calls = [
            c
            for c in self.orchestrator.logger.method_calls
            if c[0] == "error" or c[0] == "exception"
        ]
        # Allow specific error logs if they are expected within the mocked methods,
        # but for a pure happy path, we expect none from the `run` method's direct logic.
        # Example: self.orchestrator.logger.error.assert_not_called()
        # However, if mocked methods internally log errors as part of their normal "happy path" (unlikely),
        # this would need refinement. For now, assume happy path implies no errors logged by run itself.
        # Let's check no logger.exception calls specifically.
        self.orchestrator.logger.exception.assert_not_called()

    def test_run_unexpected_error_in_prepare_prompt_inputs(self):
        # Test that an unexpected error in _prepare_prompt_inputs is caught, logged, and wrapped
        # Arrange
        original_prepare_method = self.orchestrator._prepare_prompt_inputs
        self.orchestrator._prepare_prompt_inputs = MagicMock(
            side_effect=ValueError("Unexpected problem during prompt input prep")
        )
        self.orchestrator._run_moderated_debate = MagicMock()
        self.orchestrator._synthesize_summary = MagicMock()
        self.orchestrator._generate_final_solution = MagicMock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertIn(
            "Unexpected error during orchestration: Unexpected problem during prompt input prep",
            str(cm.exception),
        )

        # Ensure methods after the failing one were not called
        self.orchestrator._run_moderated_debate.assert_not_called()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            "Error during orchestrated solution improvement: Unexpected problem during prompt input prep"
        )
        # Restore original method if other tests might use it, though setUp usually handles this isolation
        self.orchestrator._prepare_prompt_inputs = original_prepare_method

    def test_run_unexpected_error_in_run_moderated_debate(self):
        # Test that an unexpected error in _run_moderated_debate is caught, logged, and wrapped
        # Arrange
        self.orchestrator._prepare_prompt_inputs = MagicMock()  # Succeeds
        original_debate_method = self.orchestrator._run_moderated_debate
        self.orchestrator._run_moderated_debate = MagicMock(
            side_effect=ValueError("Unexpected problem during debate")
        )
        self.orchestrator._synthesize_summary = MagicMock()
        self.orchestrator._generate_final_solution = MagicMock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertIn(
            "Unexpected error during orchestration: Unexpected problem during debate",
            str(cm.exception),
        )

        # Ensure _prepare_prompt_inputs was called, but methods after the failing one were not
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            "Error during orchestrated solution improvement: Unexpected problem during debate"
        )
        self.orchestrator._run_moderated_debate = original_debate_method

    def test_run_unexpected_error_in_generate_final_solution(self):
        # Test that an unexpected error in _generate_final_solution is caught, logged, and wrapped
        # Arrange
        self.orchestrator._prepare_prompt_inputs = MagicMock()  # Succeeds
        self.orchestrator._run_moderated_debate = MagicMock()  # Succeeds
        self.orchestrator._synthesize_summary = MagicMock()  # Succeeds
        original_solution_method = self.orchestrator._generate_final_solution
        self.orchestrator._generate_final_solution = MagicMock(
            side_effect=ValueError("Unexpected problem during final solution generation")
        )

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertIn(
            "Unexpected error during orchestration: Unexpected problem during final solution generation",
            str(cm.exception),
        )

        # Ensure all methods before the failing one were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        # This specific error in _generate_final_solution is logged within _generate_final_solution itself,
        # and then re-raised. The run() method's top-level exception handler will then log it again.
        # So we expect two calls: one from _generate_final_solution, one from run().
        # For this test, we are primarily concerned with the run() method's handling.
        self.orchestrator.logger.exception.assert_called_with(  # Use assert_called_with to check the last call if multiple
            "Error during orchestrated solution improvement: Unexpected problem during final solution generation"
        )
        self.orchestrator._generate_final_solution = original_solution_method

    def test_run_prepare_prompt_inputs_raises_llm_orchestration_error(self):
        """
        Tests that if _prepare_prompt_inputs raises LLMOrchestrationError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_prepare_method = self.orchestrator._prepare_prompt_inputs
        error_message = "Deliberate LLMOrchestrationError from prompt prep"
        self.orchestrator._prepare_prompt_inputs = Mock(
            side_effect=LLMOrchestrationError(error_message)
        )

        # Mock other methods to ensure they are not called
        # Ensure these are fresh mocks if they could have been called in setup or other tests
        self.orchestrator._run_moderated_debate = Mock()
        self.orchestrator._synthesize_summary = Mock()
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMOrchestrationError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure only _prepare_prompt_inputs was attempted
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_not_called()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        # The exception is logged by run() method's main try-except
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._prepare_prompt_inputs = original_prepare_method

    def test_run_moderated_debate_raises_llm_orchestration_error(self):
        """
        Tests that if _run_moderated_debate raises LLMOrchestrationError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_debate_method = self.orchestrator._run_moderated_debate
        error_message = "Deliberate LLMOrchestrationError from debate"
        self.orchestrator._run_moderated_debate = Mock(
            side_effect=LLMOrchestrationError(error_message)
        )

        # Mock _prepare_prompt_inputs to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        # Mock other methods to ensure they are not called post-failure
        self.orchestrator._synthesize_summary = Mock()
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMOrchestrationError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._run_moderated_debate = original_debate_method
        # No need to restore _prepare_prompt_inputs if it was only mocked for this test
        # and setUp correctly initializes it or if we restore it like this:
        # self.orchestrator._prepare_prompt_inputs = self.original_attributes["_prepare_prompt_inputs"]
        # For simplicity, if it's just a simple mock for this test, direct restoration of changed method is fine.

    def test_run_synthesize_summary_raises_llm_orchestration_error(self):
        """
        Tests that if _synthesize_summary raises LLMOrchestrationError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_summary_method = self.orchestrator._synthesize_summary
        error_message = "Deliberate LLMOrchestrationError from summary"
        self.orchestrator._synthesize_summary = Mock(
            side_effect=LLMOrchestrationError(error_message)
        )

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(
            return_value=None
        )  # Assuming it returns something or None
        # Mock subsequent method to ensure it's not called
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMOrchestrationError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._synthesize_summary = original_summary_method

    def test_run_generate_final_solution_raises_llm_orchestration_error(self):
        """
        Tests that if _generate_final_solution raises LLMOrchestrationError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_solution_method = self.orchestrator._generate_final_solution
        error_message = "Deliberate LLMOrchestrationError from final solution"
        self.orchestrator._generate_final_solution = Mock(
            side_effect=LLMOrchestrationError(error_message)
        )

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        self.orchestrator._synthesize_summary = Mock(return_value=None)

        # Act & Assert
        with self.assertRaises(LLMOrchestrationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMOrchestrationError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure all methods up to the failure point were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._generate_final_solution = original_solution_method

    def test_run_generate_final_solution_raises_schema_validation_error(self):
        """
        Tests that if _generate_final_solution raises SchemaValidationError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_solution_method = self.orchestrator._generate_final_solution
        error_message = "Deliberate SchemaValidationError from final solution"
        # SchemaValidationError needs at least one argument, typically the message.
        # If it has a more complex constructor, adjust this.
        self.orchestrator._generate_final_solution = Mock(
            side_effect=SchemaValidationError(error_message)
        )

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        self.orchestrator._synthesize_summary = Mock(return_value=None)

        # Act & Assert
        with self.assertRaises(SchemaValidationError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, SchemaValidationError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure all methods up to the failure point were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._generate_final_solution = original_solution_method

    def test_run_generate_final_solution_raises_max_iterations_exceeded_error(self):
        """
        Tests that if _generate_final_solution raises MaxIterationsExceededError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_solution_method = self.orchestrator._generate_final_solution
        error_message = "Deliberate MaxIterationsExceededError from final solution"
        self.orchestrator._generate_final_solution = Mock(
            side_effect=MaxIterationsExceededError(error_message)
        )

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        self.orchestrator._synthesize_summary = Mock(return_value=None)

        # Act & Assert
        with self.assertRaises(MaxIterationsExceededError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, MaxIterationsExceededError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure all methods up to the failure point were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._generate_final_solution = original_solution_method

    def test_run_run_moderated_debate_raises_llm_response_error(self):
        """
        Tests that if _run_moderated_debate raises LLMResponseError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_debate_method = self.orchestrator._run_moderated_debate
        error_message = "Deliberate LLMResponseError from debate"
        self.orchestrator._run_moderated_debate = Mock(side_effect=LLMResponseError(error_message))

        # Mock _prepare_prompt_inputs to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        # Mock other methods to ensure they are not called post-failure
        self.orchestrator._synthesize_summary = Mock()
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMResponseError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMResponseError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._run_moderated_debate = original_debate_method

    def test_run_synthesize_summary_raises_llm_response_error(self):
        """
        Tests that if _synthesize_summary raises LLMResponseError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_summary_method = self.orchestrator._synthesize_summary
        error_message = "Deliberate LLMResponseError from summary"
        self.orchestrator._synthesize_summary = Mock(side_effect=LLMResponseError(error_message))

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        # Mock subsequent method to ensure it's not called
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMResponseError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMResponseError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._synthesize_summary = original_summary_method

    def test_run_generate_final_solution_raises_llm_response_error(self):
        """
        Tests that if _generate_final_solution raises LLMResponseError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_solution_method = self.orchestrator._generate_final_solution
        error_message = "Deliberate LLMResponseError from final solution"
        self.orchestrator._generate_final_solution = Mock(
            side_effect=LLMResponseError(error_message)
        )

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        self.orchestrator._synthesize_summary = Mock(return_value=None)

        # Act & Assert
        with self.assertRaises(LLMResponseError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMResponseError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure all methods up to the failure point were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._generate_final_solution = original_solution_method

    def test_run_llm_client_error_incorrectly_wrapped_bug_demonstration(self):
        """
        BUGFIX VERIFICATION: Tests that LLMClientError is now correctly propagated
        directly by run() instead of being wrapped (bug has been fixed).

        This test originally demonstrated a bug in the production code where LLMClientError
        was not included in the list of specific exceptions that should be propagated
        directly by the run() method. The bug has now been fixed by adding LLMClientError
        to the exception list.
        """
        # Arrange
        original_debate_method = self.orchestrator._run_moderated_debate
        error_message = "Network connection failed"
        client_error = LLMClientError(error_message)
        self.orchestrator._run_moderated_debate = Mock(side_effect=client_error)

        # Mock _prepare_prompt_inputs to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        # Mock other methods to ensure they are not called post-failure
        self.orchestrator._synthesize_summary = Mock()
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert - BUGFIX: Now correctly propagates LLMClientError
        with self.assertRaises(
            LLMClientError
        ) as cm:  # FIXED: Now correctly expects LLMClientError!
            self.orchestrator.run()

        # BUGFIX VERIFICATION: The LLMClientError is now propagated directly
        self.assertIsInstance(cm.exception, LLMClientError)  # Fixed behavior
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Not wrapped anymore

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._run_moderated_debate = original_debate_method

    def test_run_run_moderated_debate_raises_llm_client_error(self):
        """
        Tests that if _run_moderated_debate raises LLMClientError,
        run() propagates it directly and logs appropriately.

        This test verifies the fix for the bug where LLMClientError was incorrectly wrapped.
        """
        # Arrange
        original_debate_method = self.orchestrator._run_moderated_debate
        error_message = "Network connection failed"
        self.orchestrator._run_moderated_debate = Mock(side_effect=LLMClientError(error_message))

        # Mock _prepare_prompt_inputs to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        # Mock other methods to ensure they are not called post-failure
        self.orchestrator._synthesize_summary = Mock()
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMClientError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMClientError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_not_called()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._run_moderated_debate = original_debate_method

    def test_run_synthesize_summary_raises_llm_client_error(self):
        """
        Tests that if _synthesize_summary raises LLMClientError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_summary_method = self.orchestrator._synthesize_summary
        error_message = "Network timeout during summary"
        self.orchestrator._synthesize_summary = Mock(side_effect=LLMClientError(error_message))

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        # Mock subsequent method to ensure it's not called
        self.orchestrator._generate_final_solution = Mock()

        # Act & Assert
        with self.assertRaises(LLMClientError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMClientError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure methods were called up to the point of failure
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_not_called()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._synthesize_summary = original_summary_method

    def test_run_generate_final_solution_raises_llm_client_error(self):
        """
        Tests that if _generate_final_solution raises LLMClientError,
        run() propagates it directly and logs appropriately.
        """
        # Arrange
        original_solution_method = self.orchestrator._generate_final_solution
        error_message = "API authentication failed"
        self.orchestrator._generate_final_solution = Mock(side_effect=LLMClientError(error_message))

        # Mock preceding methods to succeed
        self.orchestrator._prepare_prompt_inputs = Mock(return_value=None)
        self.orchestrator._run_moderated_debate = Mock(return_value=None)
        self.orchestrator._synthesize_summary = Mock(return_value=None)

        # Act & Assert
        with self.assertRaises(LLMClientError) as cm:
            self.orchestrator.run()

        self.assertIsInstance(cm.exception, LLMClientError)
        self.assertEqual(error_message, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)  # Should not be wrapped

        # Ensure all methods up to the failure point were called
        self.orchestrator._prepare_prompt_inputs.assert_called_once()
        self.orchestrator._run_moderated_debate.assert_called_once()
        self.orchestrator._synthesize_summary.assert_called_once()
        self.orchestrator._generate_final_solution.assert_called_once()

        # Check logging
        self.orchestrator.logger.info.assert_any_call(
            "Starting orchestrated solution improvement workflow."
        )
        self.orchestrator.logger.exception.assert_called_once_with(
            f"Error during orchestrated solution improvement: {error_message}"
        )

        # Restore
        self.orchestrator._generate_final_solution = original_solution_method


if __name__ == "__main__":
    unittest.main()
