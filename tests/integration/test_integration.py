"""
Integration Tests for LLM Conversation Framework.

This module contains integration tests that verify the correct functioning of the entire
LLM conversation system as an integrated whole, focusing on component interactions,
data flow, error handling, and end-to-end workflows.

Key Aspects Tested:
-----------------
1. End-to-End Workflows:
   - Complete solution improvement workflow from initial prompt to final solution
   - Conversation exchanges between multiple LLMs with proper context preservation
   - Multi-step processes involving moderation, debate, and synthesis

2. Cross-Component Data Flow:
   - Correct propagation of context between conversation stages
   - Preservation of information across LLM switches
   - Proper handling of conversation state during role transitions
   - Accurate transmission of solution requirements and assessment criteria

3. Error Handling and Recovery:
   - System behavior with invalid JSON responses from LLMs
   - Schema validation failures and automatic correction attempts
   - Maximum retry limit behavior
   - Exception propagation from sub-components

4. Configuration-Based Operation:
   - Workflow execution using configuration dataclasses
   - Parameter passing through nested config objects
   - Default value handling in config chains

Test Classes:
-----------
TestFullWorkflowWithMocks:
    Tests the complete solution improvement pipeline from initial solution to
    improved solution, verifying that all components interact correctly and
    produce the expected outcomes. Uses controlled LLM responses to simulate
    a successful workflow.

TestErrorHandlingAndEdgeCases:
    Tests system behavior under non-ideal conditions, including invalid JSON
    responses, schema validation failures, and exceptions. Verifies the system's
    ability to recover from errors and its behavior when recovery isn't possible.

TestDataFlowIntegrity:
    Tests the integrity of information flow throughout the conversation process,
    ensuring that context is properly maintained, responses are correctly influenced
    by previous messages, and the final output properly incorporates all relevant
    information from the conversation history.

Testing Approach:
---------------
These tests use mock LLM interfaces with controlled responses rather than actual API
calls to ensure deterministic behavior and avoid API costs during testing. The mocks
are configured to simulate realistic conversation flows while allowing verification
of internal state and message passing.

Unlike unit tests which focus on isolated component behavior, these integration tests
verify that components work together correctly in the context of complete workflows,
with particular attention to the boundaries between components where data handoffs occur.

These tests are critical to maintaining the reliability of the conversation system
which forms a core part of the business model and directly impacts revenue generation.
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
import logging
import json

from convorator.client.llm_client import LLMInterface, LLMInterfaceConfig, Conversation
from convorator.conversations.conversation_setup import (
    SolutionLLMGroup,
    SolutionPacketConfig,
    ModeratedConversationConfig,
    ModeratedSolutionImprovementConfig,
)
from convorator.conversations.conversation_orchestrator import (
    parse_json_response,
    generate_and_verify_result,
    llms_conversation,
    moderated_conversation,
    moderated_solution_improvement,
    improve_solution_with_moderation,
)


class TestFullWorkflowWithMocks(unittest.TestCase):
    """Integration tests for the full solution improvement workflow using mocks"""

    def setUp(self):
        """Set up test environment with mock LLMs"""
        # Configure logging
        self.logger = logging.getLogger("test_integration")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.NullHandler())

        # Create mock LLMs
        self.primary_llm = MagicMock(spec=LLMInterface)
        self.debater_llm = MagicMock(spec=LLMInterface)
        self.moderator_llm = MagicMock(spec=LLMInterface)
        self.solution_llm = MagicMock(spec=LLMInterface)

        # Set up responses
        self.primary_llm.query.return_value = "Primary LLM response analyzing the solution"
        self.debater_llm.query.return_value = "Debater LLM response with critiques"
        self.moderator_llm.query.return_value = "Moderator summary with improvement suggestions"
        self.solution_llm.query.return_value = json.dumps(
            {"name": "Improved solution", "score": 95}
        )

        # Solution schema
        self.schema = {
            "type": "object",
            "required": ["name", "score"],
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        }

        # Sample solution and requirements
        self.initial_solution = json.dumps({"name": "Initial solution", "score": 70})
        self.requirements = "Create a solution with score > 90"
        self.assessment_criteria = "Solution will be judged based on name clarity and score"

    def test_generate_and_verify_result_workflow(self):
        """Test the generate_and_verify_result workflow"""
        result = generate_and_verify_result(
            llm_service=self.solution_llm,
            logger=self.logger,
            context="Test Context",
            main_prompt_or_template="Improve this solution: {solution} based on {requirements}",
            main_promt_data={
                "solution": self.initial_solution,
                "requirements": self.requirements,
            },
            result_schema=self.schema,
        )

        self.assertEqual(result["name"], "Improved solution")
        self.assertEqual(result["score"], 95)
        self.solution_llm.query.assert_called_once()

    @patch("convorator.conversations.gen_conversations.Conversation")
    def test_llms_conversation_workflow(self, mock_conversation_class):
        """Test the llms_conversation workflow"""
        # Create mock for returned conversation
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation_class.return_value = mock_conversation

        # Create LLM configs
        llm_a_config = LLMInterfaceConfig(provider="openai")
        llm_b_config = LLMInterfaceConfig(provider="claude")

        # Create mock for LLM switching
        mock_switched_llm = MagicMock(spec=LLMInterface)
        mock_switched_llm.query.return_value = "Switched LLM response"
        self.primary_llm.switch_provider.return_value = mock_switched_llm
        self.primary_llm.get_current_config.return_value = llm_a_config

        # Run conversation
        response, service = llms_conversation(
            llm_service_a=self.primary_llm,
            logger=self.logger,
            context="Test Context",
            llm_b_config=llm_b_config,
            initial_prompt="Initial prompt",
            mid_conv_prompt="What do you think?",
            end_prompt="Provide final answer",
            max_iterations=2,
        )

        # Verify conversation flow and switching
        self.primary_llm.switch_provider.assert_called_once()
        self.assertEqual(mock_switched_llm.query.call_count, 3)  # Initial + 2 exchanges
        self.assertEqual(self.primary_llm.query.call_count, 2)  # 2 responses

    @patch("convorator.conversations.gen_conversations.Conversation")
    def test_moderated_conversation_workflow(self, mock_conversation_class):
        """Test the moderated_conversation workflow"""
        # Create mock for returned conversation
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation_class.return_value = mock_conversation

        # Run moderated conversation
        result = moderated_conversation(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt="Initial prompt to analyze solution",
            max_iterations=2,
            topic="Solution Analysis",
        )

        # Verify conversation structure
        self.assertEqual(len(result), 7)  # Initial prompt + 2 full iterations (3 messages each)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "debater")

        # Verify LLM interactions
        self.assertEqual(self.debater_llm.query.call_count, 3)  # Initial + 2 iterations
        self.assertEqual(self.primary_llm.query.call_count, 2)  # 2 iterations
        self.assertEqual(self.moderator_llm.query.call_count, 2)  # 2 moderations

    @patch("convorator.conversations.gen_conversations.moderated_conversation")
    def test_moderated_solution_improvement_workflow(self, mock_moderated_conv):
        """Test the moderated_solution_improvement workflow"""
        # Mock moderated conversation result
        mock_moderated_conv.return_value = [
            {"role": "user", "content": "Initial prompt"},
            {"role": "debater", "content": "Debater analysis"},
            {"role": "primary", "content": "Primary analysis"},
            {"role": "moderator", "content": "Improvement suggestions"},
        ]

        # Run solution improvement
        result = moderated_solution_improvement(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            solution_generation_llm=self.solution_llm,
            initial_solution=self.initial_solution,
            requirements=self.requirements,
            assessment_criteria=self.assessment_criteria,
            logger=self.logger,
            solution_schema=self.schema,
            debate_iterations=2,
        )

        # Verify result and workflow
        self.assertEqual(result["name"], "Improved solution")
        self.assertEqual(result["score"], 95)

        # Verify LLM interactions
        mock_moderated_conv.assert_called_once()
        self.moderator_llm.query.assert_called_once()  # For summary
        self.solution_llm.query.assert_called_once()  # For final solution

    @patch("convorator.conversations.gen_conversations.moderated_solution_improvement")
    def test_improve_solution_with_moderation_workflow(self, mock_improvement):
        """Test the improve_solution_with_moderation workflow using config objects"""
        # Mock improvement result
        mock_improvement.return_value = {"name": "Config-based solution", "score": 98}

        # Create LLM group
        llm_group = SolutionLLMGroup(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            solution_generation_llm=self.solution_llm,
        )

        # Create solution config
        solution_config = SolutionPacketConfig(
            llm_group=llm_group,
            topic="Solution Improvement",
            logger=self.logger,
            debate_iterations=2,
            improvement_iterations=2,
            solution_schema=self.schema,
        )

        # Create conversation config
        conversation_config = ModeratedConversationConfig(
            debate_context="Analyze this solution for improvements"
        )

        # Create improvement config
        improvement_config = ModeratedSolutionImprovementConfig(
            solution_packet_config=solution_config,
            moderated_conversation_config=conversation_config,
            initial_solution=self.initial_solution,
            requirements=self.requirements,
            assessment_criteria=self.assessment_criteria,
        )

        # Run improvement with configs
        result = improve_solution_with_moderation(improvement_config)

        # Verify result
        self.assertEqual(result["name"], "Config-based solution")
        self.assertEqual(result["score"], 98)

        # Verify improvement function was called with correct parameters
        mock_improvement.assert_called_once_with(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            solution_generation_llm=self.solution_llm,
            initial_solution=self.initial_solution,
            requirements=self.requirements,
            assessment_criteria=self.assessment_criteria,
            logger=self.logger,
            solution_schema=self.schema,
            topic="Solution Improvement",
            debate_iterations=2,
            improvement_iterations=2,
            primary_system_prompt=None,
            moderator_system_prompt=None,
            debater_system_prompt=None,
            solution_generation_system_prompt=None,
            debate_context="Analyze this solution for improvements",
            moderator_context_builder=ANY,
            moderator_instructions=None,
        )


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Integration tests for error handling and edge cases"""

    def setUp(self):
        """Set up test environment with mock LLMs"""
        # Configure logging
        self.logger = logging.getLogger("test_integration")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.NullHandler())

        # Create mock LLMs
        self.primary_llm = MagicMock(spec=LLMInterface)
        self.debater_llm = MagicMock(spec=LLMInterface)
        self.moderator_llm = MagicMock(spec=LLMInterface)
        self.solution_llm = MagicMock(spec=LLMInterface)

        # Schema for solution validation
        self.schema = {
            "type": "object",
            "required": ["name", "score"],
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        }

    def test_generate_and_verify_with_invalid_json(self):
        """Test generate_and_verify_result with initially invalid JSON"""
        # First response is invalid JSON, second response is valid
        self.solution_llm.query.side_effect = [
            "This is not JSON",
            json.dumps({"name": "Fixed solution", "score": 80}),
        ]

        result = generate_and_verify_result(
            llm_service=self.solution_llm,
            logger=self.logger,
            context="Test Context",
            main_prompt_or_template="Generate a solution",
            result_schema=self.schema,
        )

        # Verify correct solution was returned after retry
        self.assertEqual(result["name"], "Fixed solution")
        self.assertEqual(result["score"], 80)
        self.assertEqual(self.solution_llm.query.call_count, 2)

    def test_generate_and_verify_with_schema_violation(self):
        """Test generate_and_verify_result with schema validation failure"""
        # First response violates schema, second response is valid
        self.solution_llm.query.side_effect = [
            json.dumps({"name": "Invalid solution"}),  # Missing required 'score'
            json.dumps({"name": "Valid solution", "score": 85}),
        ]

        result = generate_and_verify_result(
            llm_service=self.solution_llm,
            logger=self.logger,
            context="Test Context",
            main_prompt_or_template="Generate a solution",
            result_schema=self.schema,
        )

        # Verify correct solution was returned after retry
        self.assertEqual(result["name"], "Valid solution")
        self.assertEqual(result["score"], 85)
        self.assertEqual(self.solution_llm.query.call_count, 2)

    def test_generate_and_verify_max_retries_exceeded(self):
        """Test generate_and_verify_result with maximum retries exceeded"""
        # All responses are invalid
        self.solution_llm.query.return_value = "This is not JSON"

        result = generate_and_verify_result(
            llm_service=self.solution_llm,
            logger=self.logger,
            context="Test Context",
            main_prompt_or_template="Generate a solution",
            result_schema=self.schema,
            max_improvement_iterations=2,
        )

        # Verify error response
        self.assertIn("error", result)
        self.assertEqual(self.solution_llm.query.call_count, 3)  # Initial + 2 retries

    @patch("convorator.conversations.gen_conversations.moderated_conversation")
    def test_moderated_solution_improvement_with_exception(self, mock_moderated_conv):
        """Test moderated_solution_improvement with exception in moderated conversation"""
        # Mock moderated conversation to raise exception
        mock_moderated_conv.side_effect = Exception("Conversation failed")

        with self.assertRaises(Exception) as context:
            moderated_solution_improvement(
                primary_llm=self.primary_llm,
                debater_llm=self.debater_llm,
                moderator_llm=self.moderator_llm,
                solution_generation_llm=self.solution_llm,
                initial_solution="Initial solution",
                requirements="Requirements",
                assessment_criteria="Criteria",
                logger=self.logger,
            )

        # Verify exception was propagated
        self.assertIn("Conversation failed", str(context.exception))

    def test_parse_json_response_edge_cases(self):
        """Test parse_json_response with various edge cases"""
        # Test with JSON at start of response
        result = parse_json_response(
            self.logger,
            '{"name": "First solution"} and some extra text',
            "Test Context",
        )
        self.assertEqual(result["name"], "First solution")

        # Test with JSON in middle of response
        result = parse_json_response(
            self.logger,
            'Here is the solution: {"name": "Middle solution"} and some extra text',
            "Test Context",
        )
        self.assertEqual(result["name"], "Middle solution")

        # Test with JSON at end of response
        result = parse_json_response(
            self.logger,
            'Here is the solution: {"name": "End solution"}',
            "Test Context",
        )
        self.assertEqual(result["name"], "End solution")

        # Test with multiple JSON objects (should extract first complete one)
        result = parse_json_response(
            self.logger,
            'First: {"name": "First"} Second: {"name": "Second"}',
            "Test Context",
        )
        self.assertEqual(result["name"], "First")

        # Test with multi-line JSON
        result = parse_json_response(
            self.logger,
            'Here is the result:\n{\n  "name": "Multi-line",\n  "score": 90\n}\nMore text',
            "Test Context",
        )
        self.assertEqual(result["name"], "Multi-line")
        self.assertEqual(result["score"], 90)


class TestDataFlowIntegrity(unittest.TestCase):
    """Tests to verify data integrity through the conversation workflow"""

    def setUp(self):
        """Set up test environment with controlled LLM responses"""
        # Configure logging
        self.logger = logging.getLogger("test_integrity")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.NullHandler())

        # Create patchers for conversation
        self.conv_patcher = patch("convorator.conversations.gen_conversations.Conversation")
        self.mock_conversation_class = self.conv_patcher.start()

        # Mock conversation instances
        self.main_conversation = MagicMock(spec=Conversation)
        self.moderator_conversation = MagicMock(spec=Conversation)

        # Set up conversation instances to be returned
        self.mock_conversation_class.side_effect = [
            self.main_conversation,  # First call for main conversation
            self.moderator_conversation,  # Second call for moderator conversation
        ]

        # Create mock LLMs with deterministic responses
        self.primary_llm = MagicMock(spec=LLMInterface)
        self.debater_llm = MagicMock(spec=LLMInterface)
        self.moderator_llm = MagicMock(spec=LLMInterface)

        # Create initial solution and requirements
        self.initial_solution = '{"feature": "Login system", "implementation": "Basic"}'
        self.requirements = "Create a secure login system"
        self.assessment_criteria = "Security, usability, and performance"

    def tearDown(self):
        """Clean up patchers"""
        self.conv_patcher.stop()

    def test_data_flow_in_moderated_conversation(self):
        """Test data flow integrity in moderated conversation"""
        # Configure responses to include and reference previous messages
        self.debater_llm.query.return_value = "First analysis: The login system needs 2FA"
        self.primary_llm.query.return_value = (
            "Response to first analysis: Implementing 2FA would add complexity"
        )
        self.moderator_llm.query.return_value = (
            "Feedback: Both perspectives on 2FA are valid, consider trade-offs"
        )

        # Track conversation messages
        conversation_messages = []

        # Mock conversation get_messages to capture messages
        def capture_messages():
            # Generate mock messages based on conversation state
            messages = [{"role": "user", "content": f"Initial: {self.initial_solution}"}]

            for msg in conversation_messages:
                messages.append(msg)

            return messages

        self.main_conversation.get_messages.side_effect = capture_messages

        # Mock add_assistant_message to capture responses
        def capture_assistant_message(content):
            conversation_messages.append({"role": "assistant", "content": content})

        self.main_conversation.add_assistant_message.side_effect = capture_assistant_message

        # Mock add_user_message to capture user messages
        def capture_user_message(content):
            conversation_messages.append({"role": "user", "content": content})

        self.main_conversation.add_user_message.side_effect = capture_user_message

        # Run moderated conversation
        result = moderated_conversation(
            primary_llm=self.primary_llm,
            debater_llm=self.debater_llm,
            moderator_llm=self.moderator_llm,
            initial_prompt=f"Analyze this solution: {self.initial_solution}",
            max_iterations=1,
            topic="Login System",
        )

        # Verify data flow integrity

        # 1. Check debater received initial prompt
        debater_call = self.debater_llm.query.call_args_list[0]
        self.assertIn(self.initial_solution, str(debater_call))

        # 2. Check primary received debater's analysis
        primary_call = self.primary_llm.query.call_args_list[0]
        primary_call_content = str(primary_call)
        self.assertIn("2FA", primary_call_content)

        # 3. Check moderator received both perspectives
        moderator_call = self.moderator_llm.query.call_args_list[0]
        moderator_call_content = str(moderator_call)
        self.assertIn("2FA", moderator_call_content)
        self.assertIn("complexity", moderator_call_content)

        # 4. Check final result includes all messages in proper order
        self.assertEqual(len(result), 4)  # Initial + debater + primary + moderator
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "debater")
        self.assertEqual(result[2]["role"], "primary")
        self.assertEqual(result[3]["role"], "moderator")

        # 5. Check content references throughout the conversation
        self.assertIn("2FA", result[1]["content"])  # Debater mentioned 2FA
        self.assertIn("complexity", result[2]["content"])  # Primary mentioned complexity
        self.assertIn("trade-offs", result[3]["content"])  # Moderator mentioned trade-offs


if __name__ == "__main__":
    unittest.main()
