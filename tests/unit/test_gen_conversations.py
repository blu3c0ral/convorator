import unittest
from unittest.mock import MagicMock, patch, call, ANY
import json
import logging
import re
import jsonschema

from src.utils.gen_conversations import (
    validate_json,
    parse_json_response,
    generate_and_verify_result,
    llms_conversation,
    moderated_conversation,
    moderated_solution_improvement,
    improve_solution_with_moderation,
)
from src.utils.gen_conversations_helpers import (
    SolutionLLMGroup,
    SolutionPacketConfig,
    ModeratedConversationConfig,
    ModeratedSolutionImprovementConfig,
)
from src.utils.llm_client import LLMInterface, LLMInterfaceConfig, Conversation


class TestValidateJSON(unittest.TestCase):
    """Tests for validate_json function"""

    def setUp(self):
        """Set up test environment"""
        self.logger = MagicMock(spec=logging.Logger)

        # Simple schema for testing
        self.schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"},
            },
        }

    def test_valid_json(self):
        """Test validation with valid JSON data"""
        data = {"name": "John", "age": 30, "email": "john@example.com"}

        result = validate_json(self.logger, data, self.schema)

        self.assertEqual(result, "")  # Empty string means no errors
        self.logger.error.assert_not_called()

    def test_valid_json_minimal(self):
        """Test validation with minimal valid JSON data"""
        data = {"name": "John", "age": 30}  # Email is not required

        result = validate_json(self.logger, data, self.schema)

        self.assertEqual(result, "")
        self.logger.error.assert_not_called()

    def test_invalid_json_missing_required(self):
        """Test validation with missing required field"""
        data = {"name": "John"}  # Missing 'age'

        result = validate_json(self.logger, data, self.schema)

        self.assertNotEqual(result, "")  # Should return error message
        self.assertIn("Validation Error", result)
        self.assertIn("age", result)  # Error should mention missing field
        self.logger.error.assert_called_once()

    def test_invalid_json_wrong_type(self):
        """Test validation with incorrect type"""
        data = {"name": "John", "age": "thirty"}  # Age should be integer

        result = validate_json(self.logger, data, self.schema)

        self.assertNotEqual(result, "")
        self.assertIn("Validation Error", result)
        self.logger.error.assert_called_once()

    def test_invalid_json_constraint_violation(self):
        """Test validation with constraint violation"""
        data = {"name": "John", "age": -5}  # Age should be >= 0

        result = validate_json(self.logger, data, self.schema)

        self.assertNotEqual(result, "")
        self.assertIn("Validation Error", result)
        self.logger.error.assert_called_once()

    def test_invalid_schema(self):
        """Test validation with invalid schema"""
        data = {"name": "John", "age": 30}
        invalid_schema = {"type": "invalid_type"}  # Invalid schema

        with self.assertRaises(jsonschema.exceptions.SchemaError):
            validate_json(self.logger, data, invalid_schema)

        self.logger.error.assert_called_once()


class TestParseJSONResponse(unittest.TestCase):
    """Tests for parse_json_response function"""

    def setUp(self):
        """Set up test environment"""
        self.logger = MagicMock(spec=logging.Logger)

        # Simple schema for testing
        self.schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }

    def test_parse_json_with_code_block(self):
        """Test parsing JSON from code block format"""
        response = 'Here is the result:\n```json\n{"name": "John", "age": 30}\n```\nLet me know if you need anything else.'

        result = parse_json_response(self.logger, response, "Test")

        self.assertEqual(result, {"name": "John", "age": 30})
        self.logger.error.assert_not_called()

    def test_parse_json_without_code_block(self):
        """Test parsing JSON without code block"""
        response = 'Here is the result: {"name": "John", "age": 30}'

        result = parse_json_response(self.logger, response, "Test")

        self.assertEqual(result, {"name": "John", "age": 30})
        self.logger.error.assert_not_called()

    def test_parse_json_with_validation(self):
        """Test parsing JSON with schema validation"""
        response = '{"name": "John"}'

        result = parse_json_response(self.logger, response, "Test", self.schema)

        self.assertEqual(result, {"name": "John"})
        self.logger.error.assert_not_called()

    def test_parse_json_failed_validation(self):
        """Test parsing JSON that fails validation"""
        response = '{"age": 30}'  # Missing required 'name' field

        with self.assertRaises(ValueError):
            parse_json_response(self.logger, response, "Test", self.schema)

        self.logger.error.assert_called_once()

    def test_parse_json_no_json_found(self):
        """Test parsing response with no JSON"""
        response = "This is a text response with no JSON."

        with self.assertRaises(ValueError) as context:
            parse_json_response(self.logger, response, "Test")

        self.assertIn("No JSON found", str(context.exception))
        self.logger.error.assert_called_once()

    def test_parse_json_malformed_json(self):
        """Test parsing malformed JSON"""
        response = '{"name": "John", "age": 30'  # Missing closing brace

        with self.assertRaises(ValueError) as context:
            parse_json_response(self.logger, response, "Test")

        self.assertIn("JSON parsing error", str(context.exception))
        self.logger.error.assert_called_once()


class TestGenerateAndVerifyResult(unittest.TestCase):
    """Tests for generate_and_verify_result function"""

    def setUp(self):
        """Set up test environment"""
        self.llm_service = MagicMock(spec=LLMInterface)
        self.logger = MagicMock(spec=logging.Logger)
        self.schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }

    @patch("src.utils.gen_conversations.parse_json_response")
    def test_success_first_attempt(self, mock_parse):
        """Test successful generation on first attempt"""
        # Mock LLM response and parsing
        self.llm_service.query.return_value = '{"name": "John"}'
        mock_parse.return_value = {"name": "John"}

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person",
            result_schema=self.schema,
        )

        self.assertEqual(result, {"name": "John"})
        self.llm_service.query.assert_called_once()
        mock_parse.assert_called_once()

    @patch("src.utils.gen_conversations.parse_json_response")
    def test_success_with_template(self, mock_parse):
        """Test successful generation with template and data"""
        # Mock LLM response and parsing
        self.llm_service.query.return_value = '{"name": "John"}'
        mock_parse.return_value = {"name": "John"}

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person named {person_name}",
            main_promt_data={"person_name": "John"},
            result_schema=self.schema,
        )

        self.assertEqual(result, {"name": "John"})
        self.llm_service.query.assert_called_once_with(
            "Generate a person named John", use_conversation=True
        )

    @patch("src.utils.gen_conversations.parse_json_response")
    def test_multiple_attempts_success(self, mock_parse):
        """Test successful generation after initial failures"""
        # First attempt fails, second succeeds
        mock_parse.side_effect = [ValueError("JSON validation error"), {"name": "John"}]

        self.llm_service.query.side_effect = [
            '{"age": 30}',  # First response (fails validation)
            '{"name": "John"}',  # Second response (succeeds)
        ]

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person",
            result_schema=self.schema,
        )

        self.assertEqual(result, {"name": "John"})
        self.assertEqual(self.llm_service.query.call_count, 2)
        self.assertEqual(mock_parse.call_count, 2)
        self.logger.error.assert_called_once()

    @patch("src.utils.gen_conversations.parse_json_response")
    def test_max_attempts_reached(self, mock_parse):
        """Test reaching maximum improvement attempts"""
        # All attempts fail
        mock_parse.side_effect = ValueError("JSON validation error")

        self.llm_service.query.return_value = '{"age": 30}'  # Always fails validation

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person",
            result_schema=self.schema,
            max_improvement_iterations=2,
        )

        # Should return error dictionary
        self.assertIn("error", result)
        self.assertEqual(self.llm_service.query.call_count, 3)  # Initial + 2 fixes
        self.assertEqual(mock_parse.call_count, 2)
        self.assertEqual(self.logger.error.call_count, 3)

    @patch("src.utils.gen_conversations.parse_json_response")
    def test_custom_fix_template(self, mock_parse):
        """Test using custom fix prompt template"""
        # First attempt fails, second succeeds
        mock_parse.side_effect = [ValueError("JSON validation error"), {"name": "John"}]

        self.llm_service.query.side_effect = [
            '{"age": 30}',  # First response (fails validation)
            '{"name": "John"}',  # Second response (succeeds)
        ]

        custom_template = (
            "Fix this: {errors}\nPrompt: {main_prompt}\nResponse: {response}"
        )

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person",
            fix_prompt_template=custom_template,
            result_schema=self.schema,
        )

        self.assertEqual(result, {"name": "John"})

        # Verify custom template was used
        second_call_args = self.llm_service.query.call_args_list[1][0][0]
        self.assertIn("Fix this:", second_call_args)

    def test_general_exception(self):
        """Test handling of general exceptions"""
        # Mock LLM to raise exception
        self.llm_service.query.side_effect = Exception("Test error")

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a person",
            result_schema=self.schema,
        )

        # Should return error dictionary
        self.assertIn("error", result)
        self.assertIn("Test error", result["error"])
        self.logger.error.assert_called_once()

    def test_no_json_result(self):
        """Test with json_result=False to return raw response"""
        self.llm_service.query.return_value = "This is a text response"

        result = generate_and_verify_result(
            llm_service=self.llm_service,
            logger=self.logger,
            context="Test",
            main_prompt_or_template="Generate a text",
            json_result=False,
        )

        self.assertEqual(result, "This is a text response")
        self.llm_service.query.assert_called_once()


class TestLLMsConversation(unittest.TestCase):
    """Tests for llms_conversation function"""

    def setUp(self):
        """Set up test environment"""
        self.llm_a = MagicMock(spec=LLMInterface)
        self.llm_a_config = LLMInterfaceConfig(provider="openai")
        self.llm_a.get_current_config.return_value = self.llm_a_config

        self.llm_b_config = LLMInterfaceConfig(provider="claude")

        self.logger = MagicMock(spec=logging.Logger)

    @patch("src.utils.gen_conversations.generate_and_verify_result")
    @patch("src.utils.llm_client.LLMInterface.switch_provider")
    def test_successful_conversation_with_iterations(self, mock_switch, mock_generate):
        """Test successful conversation with multiple iterations"""
        # Mock responses
        self.llm_a.query.side_effect = ["A response 1", "A response 2"]
        mock_llm_b = MagicMock(spec=LLMInterface)
        mock_llm_b.query.side_effect = ["B response 1", "B response 2"]
        self.llm_a.switch_provider.return_value = mock_llm_b

        mock_generate.return_value = {"result": "Final result"}

        # Run conversation with 2 iterations
        result, llm_service = llms_conversation(
            llm_service_a=self.llm_a,
            logger=self.logger,
            context="Test",
            llm_b_config=self.llm_b_config,
            initial_prompt="Initial prompt",
            mid_conv_prompt="What do you think?",
            end_prompt="Provide final answer",
            max_iterations=2,
        )

        self.assertEqual(result, {"result": "Final result"})

        # Verify conversation flow (B start, A respond, B continue, A respond, B continue, A final)
        self.assertEqual(mock_llm_b.query.call_count, 3)
        self.assertEqual(self.llm_a.query.call_count, 2)

        # Verify generate_and_verify_result was called for final response
        mock_generate.assert_called_once()

    @patch("src.utils.gen_conversations.generate_and_verify_result")
    @patch("src.utils.llm_client.LLMInterface.switch_provider")
    def test_conversation_with_one_iteration(self, mock_switch, mock_generate):
        """Test conversation with just one iteration"""
        # Mock responses
        self.llm_a.query.return_value = "A response"
        mock_llm_b = MagicMock(spec=LLMInterface)
        mock_llm_b.query.side_effect = ["B response 1", "B response 2"]
        self.llm_a.switch_provider.return_value = mock_llm_b

        mock_generate.return_value = {"result": "Final result"}

        # Run conversation with 1 iteration
        result, llm_service = llms_conversation(
            llm_service_a=self.llm_a,
            logger=self.logger,
            context="Test",
            llm_b_config=self.llm_b_config,
            initial_prompt="Initial prompt",
            mid_conv_prompt="What do you think?",
            end_prompt="Provide final answer",
            max_iterations=1,
        )

        self.assertEqual(result, {"result": "Final result"})

        # Verify conversation flow (B start, A respond, B continue, A final)
        self.assertEqual(mock_llm_b.query.call_count, 2)
        self.assertEqual(self.llm_a.query.call_count, 1)

        # Verify generate_and_verify_result was called for final response
        mock_generate.assert_called_once()

    def test_skip_conversation(self):
        """Test skipping conversation with max_iterations < 1"""
        result = llms_conversation(
            llm_service_a=self.llm_a,
            logger=self.logger,
            context="Test",
            llm_b_config=self.llm_b_config,
            initial_prompt="Initial prompt",
            mid_conv_prompt="What do you think?",
            end_prompt="Provide final answer",
            max_iterations=0,
        )

        self.assertIsNone(result)
        self.logger.warning.assert_called_once()
        self.llm_a.query.assert_not_called()

    @patch("src.utils.gen_conversations.generate_and_verify_result")
    @patch("src.utils.llm_client.LLMInterface")
    def test_conversation_with_config_instead_of_llm(
        self, mock_llm_interface, mock_generate
    ):
        """Test starting conversation with LLMInterfaceConfig instead of LLMInterface"""
        # Create mock for new LLM instance
        mock_llm_instance = MagicMock(spec=LLMInterface)
        mock_llm_interface.return_value = mock_llm_instance
        mock_llm_instance.query.side_effect = ["B response", "B response 2"]
        mock_llm_instance.switch_provider.return_value = mock_llm_instance

        mock_generate.return_value = {"result": "Final result"}

        # Run conversation with config instead of LLM
        result, llm_service = llms_conversation(
            llm_service_a=self.llm_a_config,  # Use config instead of LLM
            logger=self.logger,
            context="Test",
            llm_b_config=self.llm_b_config,
            initial_prompt="Initial prompt",
            mid_conv_prompt="What do you think?",
            end_prompt="Provide final answer",
            max_iterations=1,
        )

        self.assertEqual(result, {"result": "Final result"})

        # Verify new LLM was created
        mock_llm_interface.assert_called_once()

        # Verify conversation flow
        self.assertEqual(mock_llm_instance.query.call_count, 2)
        mock_generate.assert_called_once()


class TestModeratedConversation(unittest.TestCase):
    """Tests for moderated_conversation function"""

    def setUp(self):
        """Set up test environment"""
        self.primary_llm = MagicMock(spec=LLMInterface)
        self.debater_llm = MagicMock(spec=LLMInterface)
        self.moderator_llm = MagicMock(spec=LLMInterface)

        # Mock conversation to avoid actual conversation manipulation
        self.mock_conversation = MagicMock(spec=Conversation)

        # Mock responses
        self.primary_llm.query.return_value = "Primary response"
        self.debater_llm.query.return_value = "Debater response"
        self.moderator_llm.query.return_value = "Moderator feedback"

    @patch("src.utils.gen_conversations.Conversation")
    def test_full_moderated_conversation(self, mock_conversation_class):
        """Test full moderated conversation with multiple iterations"""
        # Setup mock conversation
        mock_conversation_class.return_value = self.mock_conversation

        # Run moderated conversation with 2 iterations
        result = moderated_conversation(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt="Initial prompt",
            max_iterations=2,
            topic="Test topic",
        )

        # Verify results
        self.assertEqual(len(result), 7)  # initial + 2 iterations with 3 messages each

        # Verify conversation flow
        self.assertEqual(self.debater_llm.query.call_count, 3)  # Initial + 2 iterations
        self.assertEqual(self.primary_llm.query.call_count, 2)  # 2 iterations
        self.assertEqual(self.moderator_llm.query.call_count, 2)  # 2 iterations

        # Verify message order and roles
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "debater")
        self.assertEqual(result[2]["role"], "primary")
        self.assertEqual(result[3]["role"], "moderator")
        self.assertEqual(result[4]["role"], "debater")
        self.assertEqual(result[5]["role"], "primary")
        self.assertEqual(result[6]["role"], "moderator")

    @patch("src.utils.gen_conversations.Conversation")
    def test_minimal_moderated_conversation(self, mock_conversation_class):
        """Test moderated conversation with one iteration"""
        # Setup mock conversation
        mock_conversation_class.return_value = self.mock_conversation

        # Run moderated conversation with 1 iteration
        result = moderated_conversation(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt="Initial prompt",
            max_iterations=1,
        )

        # Verify results
        self.assertEqual(len(result), 4)  # initial + 1 iteration with 3 messages

        # Verify conversation flow
        self.assertEqual(self.debater_llm.query.call_count, 1)  # Initial only
        self.assertEqual(self.primary_llm.query.call_count, 1)  # 1 iteration
        self.assertEqual(self.moderator_llm.query.call_count, 1)  # 1 iteration

    @patch("src.utils.gen_conversations.Conversation")
    def test_custom_prompts_and_contexts(self, mock_conversation_class):
        """Test with custom system prompts and context"""
        # Setup mock conversation
        mock_conversation_class.return_value = self.mock_conversation

        # Run moderated conversation with custom settings
        result = moderated_conversation(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt="Initial prompt",
            max_iterations=1,
            topic="Custom topic",
            primary_system_prompt="Primary prompt",
            moderator_system_prompt="Moderator prompt",
            debater_system_prompt="Debater prompt",
            debate_context="Custom context",
            moderator_instructions="Custom instructions",
        )

        # Verify system messages were set
        system_calls = [
            call for call in self.mock_conversation.set_system_message.call_args_list
        ]
        self.assertEqual(len(system_calls), 3)  # Debater, primary, debater again

        # Verify custom prompts and contexts were used
        self.mock_conversation.set_system_message.assert_any_call("Debater prompt")
        self.mock_conversation.set_system_message.assert_any_call("Primary prompt")

        # Verify moderator conversation was created with custom prompt
        moderator_call = mock_conversation_class.call_args_list[1]
        self.assertEqual(moderator_call[0][0], "Moderator prompt")

        # Verify moderator was given custom instructions
        self.moderator_llm.query.assert_called_once()
        moderator_args = self.moderator_llm.query.call_args[1]["conversation_history"]
        self.assertTrue(
            any(
                "Custom instructions" in str(msg.get("content", ""))
                for msg in moderator_args
            )
        )


class TestModeratedSolutionImprovement(unittest.TestCase):
    """Tests for moderated_solution_improvement function"""

    def setUp(self):
        """Set up test environment"""
        self.primary_llm = MagicMock(spec=LLMInterface)
        self.debater_llm = MagicMock(spec=LLMInterface)
        self.moderator_llm = MagicMock(spec=LLMInterface)
        self.solution_generation_llm = MagicMock(spec=LLMInterface)
        self.logger = MagicMock(spec=logging.Logger)

        # Mock conversation to avoid actual conversation manipulation
        self.mock_conversation = MagicMock(spec=Conversation)

        # Schema for solution validation
        self.schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }

    @patch("src.utils.gen_conversations.moderated_conversation")
    @patch("src.utils.gen_conversations.generate_and_verify_result")
    def test_successful_solution_improvement(self, mock_generate, mock_moderated_conv):
        """Test successful solution improvement workflow"""
        # Mock moderated conversation
        mock_moderated_conv.return_value = [
            {"role": "user", "content": "Initial prompt"},
            {"role": "debater", "content": "Debater response"},
            {"role": "primary", "content": "Primary response"},
            {"role": "moderator", "content": "Moderator feedback"},
        ]

        # Mock generate_and_verify_result
        mock_generate.return_value = {"name": "Improved solution"}

        # Mock synthesize_moderation_summary
        mock_synthesize = MagicMock(return_value="Synthesized summary")

        # Run solution improvement
        result = moderated_solution_improvement(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            solution_generation_llm=self.solution_generation_llm,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Criteria",
            logger=self.logger,
            solution_schema=self.schema,
            topic="Test topic",
            debate_iterations=2,
            improvement_iterations=2,
            synthesize_moderation_summary=mock_synthesize,
        )

        self.assertEqual(result, {"name": "Improved solution"})

        # Verify workflow steps
        mock_moderated_conv.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_generate.assert_called_once()

        # Verify logger calls
        self.assertEqual(self.logger.info.call_count, 3)

    @patch("src.utils.gen_conversations.moderated_conversation")
    @patch("src.utils.gen_conversations.generate_and_verify_result")
    def test_solution_improvement_with_custom_functions(
        self, mock_generate, mock_moderated_conv
    ):
        """Test solution improvement with custom helper functions"""
        # Mock moderated conversation
        mock_moderated_conv.return_value = [
            {"role": "user", "content": "Initial prompt"},
            {"role": "debater", "content": "Debater response"},
        ]

        # Mock generate_and_verify_result
        mock_generate.return_value = {"name": "Improved solution"}

        # Custom functions
        mock_create_prompt = MagicMock(return_value="Custom initial prompt")
        mock_synthesize = MagicMock(return_value="Custom synthesis")
        mock_build_prompt = MagicMock(return_value="Custom improvement prompt")

        # Run solution improvement with custom functions
        result = moderated_solution_improvement(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            solution_generation_llm=self.solution_generation_llm,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Criteria",
            logger=self.logger,
            create_initial_prompt=mock_create_prompt,
            synthesize_moderation_summary=mock_synthesize,
            build_solution_improvement_prompt=mock_build_prompt,
        )

        self.assertEqual(result, {"name": "Improved solution"})

        # Verify custom functions were used
        mock_create_prompt.assert_called_once_with(
            "Initial solution", "Requirements", "Criteria"
        )
        mock_synthesize.assert_called_once()
        mock_build_prompt.assert_called_once_with(
            "Initial solution", "Custom synthesis"
        )

        # Verify moderated conversation used custom prompt
        mock_moderated_conv.assert_called_once_with(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt="Custom initial prompt",
            max_iterations=3,  # Default
            topic=None,  # Default
            primary_system_prompt=None,
            moderator_system_prompt=None,
            debater_system_prompt=None,
            solution_generation_system_prompt=None,
            debate_context=None,
            moderator_context_builder=ANY,
            moderator_instructions=None,
        )

    @patch("src.utils.gen_conversations.moderated_conversation")
    def test_solution_improvement_exception_handling(self, mock_moderated_conv):
        """Test handling of exceptions during solution improvement"""
        # Mock moderated conversation to raise exception
        mock_moderated_conv.side_effect = Exception("Test error")

        # Run solution improvement
        with self.assertRaises(Exception) as context:
            moderated_solution_improvement(
                primary_llm=self.primary_llm,
                debater_llm=self.debater_llm,
                moderator_llm=self.moderator_llm,
                solution_generation_llm=self.solution_generation_llm,
                initial_solution="Initial solution",
                requirements="Requirements",
                assessment_criteria="Criteria",
                logger=self.logger,
            )

        self.assertIn("Test error", str(context.exception))
        self.logger.exception.assert_called_once()


class TestImproveSolutionWithModeration(unittest.TestCase):
    """Tests for improve_solution_with_moderation function"""

    def setUp(self):
        """Set up test environment with mock LLMs and configs"""
        # Mock LLMs
        self.mock_primary = MagicMock(spec=LLMInterface)
        self.mock_debater = MagicMock(spec=LLMInterface)
        self.mock_moderator = MagicMock(spec=LLMInterface)
        self.mock_generator = MagicMock(spec=LLMInterface)

        # Create LLM group
        self.llm_group = SolutionLLMGroup(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
        )

        # Create required configs
        self.solution_config = SolutionPacketConfig(
            llm_group=self.llm_group,
            topic="Test topic",
            logger=MagicMock(spec=logging.Logger),
            debate_iterations=2,
            improvement_iterations=2,
            solution_schema={"type": "object"},
            primary_system_prompt="Primary prompt",
            debater_system_prompt="Debater prompt",
            moderator_system_prompt="Moderator prompt",
            solution_generation_system_prompt="Generator prompt",
        )

        self.conversation_config = ModeratedConversationConfig(
            debate_context="Debate context",
            moderator_context_builder=MagicMock(),
            moderator_instructions="Moderator instructions",
        )

    @patch("src.utils.gen_conversations.moderated_solution_improvement")
    def test_improve_solution_with_moderation(self, mock_moderated_improvement):
        """Test improve_solution_with_moderation with full configuration"""
        # Mock moderated_solution_improvement result
        mock_moderated_improvement.return_value = {"name": "Improved solution"}

        # Create improvement config
        improvement_config = ModeratedSolutionImprovementConfig(
            solution_packet_config=self.solution_config,
            moderated_conversation_config=self.conversation_config,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Criteria",
            create_initial_prompt=MagicMock(),
            synthesize_moderation_summary=MagicMock(),
            build_solution_improvement_prompt=MagicMock(),
            fix_prompt_template="Fix template",
        )

        # Run improve_solution_with_moderation
        result = improve_solution_with_moderation(improvement_config)

        self.assertEqual(result, {"name": "Improved solution"})

        # Verify moderated_solution_improvement was called with correct parameters
        mock_moderated_improvement.assert_called_once_with(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Criteria",
            logger=self.solution_config.logger,
            solution_schema=self.solution_config.solution_schema,
            topic=self.solution_config.topic,
            debate_iterations=self.solution_config.debate_iterations,
            improvement_iterations=self.solution_config.improvement_iterations,
            primary_system_prompt=self.solution_config.primary_system_prompt,
            moderator_system_prompt=self.solution_config.moderator_system_prompt,
            debater_system_prompt=self.solution_config.debater_system_prompt,
            solution_generation_system_prompt=self.solution_config.solution_generation_system_prompt,
            debate_context=self.conversation_config.debate_context,
            moderator_context_builder=self.conversation_config.moderator_context_builder,
            moderator_instructions=self.conversation_config.moderator_instructions,
        )

    @patch("src.utils.gen_conversations.moderated_solution_improvement")
    def test_minimal_config(self, mock_moderated_improvement):
        """Test with minimal configuration"""
        # Mock moderated_solution_improvement result
        mock_moderated_improvement.return_value = {"result": "Simple solution"}

        # Create minimal configs
        minimal_solution_config = SolutionPacketConfig(llm_group=self.llm_group)
        minimal_conversation_config = ModeratedConversationConfig()

        # Create minimal improvement config
        minimal_improvement_config = ModeratedSolutionImprovementConfig(
            solution_packet_config=minimal_solution_config,
            moderated_conversation_config=minimal_conversation_config,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Criteria",
        )

        # Run improve_solution_with_moderation
        result = improve_solution_with_moderation(minimal_improvement_config)

        self.assertEqual(result, {"result": "Simple solution"})

        # Verify moderated_solution_improvement was called with minimal parameters
        mock_moderated_improvement.assert_called_once()

        # Verify default values were used for optional parameters
        call_args = mock_moderated_improvement.call_args[1]
        self.assertIsNone(call_args["topic"])
        self.assertEqual(call_args["debate_iterations"], 3)  # Default
        self.assertIsNone(call_args["solution_schema"])


if __name__ == "__main__":
    unittest.main()
