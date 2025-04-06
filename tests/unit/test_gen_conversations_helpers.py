import unittest
from unittest.mock import MagicMock, patch
import logging
from src.utils.gen_conversations_helpers import (
    SolutionLLMGroup,
    SolutionPacketConfig,
    ModeratedConversationConfig,
    ModeratedSolutionImprovementConfig,
    moderator_context_builder,
    create_initial_prompt,
    synthesize_moderation_summary,
    build_solution_improvement_prompt,
)
from src.utils.llm_client import LLMInterface


class TestDataclasses(unittest.TestCase):
    """Tests for the dataclasses in gen_conversations_helpers.py"""

    def setUp(self):
        """Set up test environment with mock LLMs"""
        self.mock_primary = MagicMock(spec=LLMInterface)
        self.mock_debater = MagicMock(spec=LLMInterface)
        self.mock_moderator = MagicMock(spec=LLMInterface)
        self.mock_generator = MagicMock(spec=LLMInterface)

    def test_solution_llm_group(self):
        """Test SolutionLLMGroup initialization and properties"""
        group = SolutionLLMGroup(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
        )

        self.assertEqual(group.primary_llm, self.mock_primary)
        self.assertEqual(group.debater_llm, self.mock_debater)
        self.assertEqual(group.moderator_llm, self.mock_moderator)
        self.assertEqual(group.solution_generation_llm, self.mock_generator)

    def test_solution_packet_config_minimal(self):
        """Test SolutionPacketConfig with minimal parameters"""
        group = SolutionLLMGroup(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
        )

        config = SolutionPacketConfig(llm_group=group)

        self.assertEqual(config.llm_group, group)
        self.assertIsNone(config.topic)
        self.assertIsNone(config.logger)
        self.assertEqual(config.debate_iterations, 3)  # Default value
        self.assertEqual(config.improvement_iterations, 3)  # Default value
        self.assertIsNone(config.solution_schema)
        self.assertIsNone(config.primary_system_prompt)
        self.assertIsNone(config.debater_system_prompt)
        self.assertIsNone(config.moderator_system_prompt)
        self.assertIsNone(config.solution_generation_system_prompt)

    def test_solution_packet_config_full(self):
        """Test SolutionPacketConfig with all parameters"""
        group = SolutionLLMGroup(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
        )

        logger = logging.getLogger("test")
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        config = SolutionPacketConfig(
            llm_group=group,
            topic="Test Topic",
            logger=logger,
            debate_iterations=5,
            improvement_iterations=2,
            solution_schema=schema,
            primary_system_prompt="Primary prompt",
            debater_system_prompt="Debater prompt",
            moderator_system_prompt="Moderator prompt",
            solution_generation_system_prompt="Generator prompt",
        )

        self.assertEqual(config.llm_group, group)
        self.assertEqual(config.topic, "Test Topic")
        self.assertEqual(config.logger, logger)
        self.assertEqual(config.debate_iterations, 5)
        self.assertEqual(config.improvement_iterations, 2)
        self.assertEqual(config.solution_schema, schema)
        self.assertEqual(config.primary_system_prompt, "Primary prompt")
        self.assertEqual(config.debater_system_prompt, "Debater prompt")
        self.assertEqual(config.moderator_system_prompt, "Moderator prompt")
        self.assertEqual(config.solution_generation_system_prompt, "Generator prompt")

    def test_moderated_conversation_config(self):
        """Test ModeratedConversationConfig initialization and defaults"""
        # Default initialization
        config = ModeratedConversationConfig()

        self.assertIsNone(config.debate_context)
        self.assertIsNotNone(
            config.moderator_context_builder
        )  # Default builder function
        self.assertIsNone(config.moderator_instructions)

        # With custom values
        mock_builder = MagicMock()

        config = ModeratedConversationConfig(
            debate_context="Debate context",
            moderator_context_builder=mock_builder,
            moderator_instructions="Moderator instructions",
        )

        self.assertEqual(config.debate_context, "Debate context")
        self.assertEqual(config.moderator_context_builder, mock_builder)
        self.assertEqual(config.moderator_instructions, "Moderator instructions")

    def test_moderated_solution_improvement_config(self):
        """Test ModeratedSolutionImprovementConfig initialization and defaults"""
        # Create required configs
        group = SolutionLLMGroup(
            primary_llm=self.mock_primary,
            debater_llm=self.mock_debater,
            moderator_llm=self.mock_moderator,
            solution_generation_llm=self.mock_generator,
        )

        solution_config = SolutionPacketConfig(llm_group=group)
        conversation_config = ModeratedConversationConfig()

        # Test with minimal required parameters
        config = ModeratedSolutionImprovementConfig(
            solution_packet_config=solution_config,
            moderated_conversation_config=conversation_config,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Assessment criteria",
        )

        self.assertEqual(config.solution_packet_config, solution_config)
        self.assertEqual(config.moderated_conversation_config, conversation_config)
        self.assertEqual(config.initial_solution, "Initial solution")
        self.assertEqual(config.requirements, "Requirements")
        self.assertEqual(config.assessment_criteria, "Assessment criteria")
        self.assertIsNone(config.fix_prompt_template)

        # Verify default function references
        self.assertEqual(config.create_initial_prompt, create_initial_prompt)
        self.assertEqual(
            config.synthesize_moderation_summary, synthesize_moderation_summary
        )
        self.assertEqual(
            config.build_solution_improvement_prompt, build_solution_improvement_prompt
        )

        # Test with custom functions
        mock_create_prompt = MagicMock()
        mock_synthesize = MagicMock()
        mock_build_prompt = MagicMock()

        config = ModeratedSolutionImprovementConfig(
            solution_packet_config=solution_config,
            moderated_conversation_config=conversation_config,
            initial_solution="Initial solution",
            requirements="Requirements",
            assessment_criteria="Assessment criteria",
            create_initial_prompt=mock_create_prompt,
            synthesize_moderation_summary=mock_synthesize,
            build_solution_improvement_prompt=mock_build_prompt,
            fix_prompt_template="Fix template",
        )

        self.assertEqual(config.create_initial_prompt, mock_create_prompt)
        self.assertEqual(config.synthesize_moderation_summary, mock_synthesize)
        self.assertEqual(config.build_solution_improvement_prompt, mock_build_prompt)
        self.assertEqual(config.fix_prompt_template, "Fix template")


class TestHelperFunctions(unittest.TestCase):
    """Tests for the helper functions in gen_conversations_helpers.py"""

    def test_moderator_context_builder(self):
        """Test moderator_context_builder formats context correctly"""
        debate_context = "This is a debate"
        initial_prompt = "Initial question"
        conversation_history = [
            {"role": "user", "content": "User message"},
            {"role": "debater", "content": "Debater response"},
            {"role": "primary", "content": "Primary response"},
        ]
        moderator_instructions = "Please provide feedback"

        result = moderator_context_builder(
            debate_context, initial_prompt, conversation_history, moderator_instructions
        )

        # Check that all necessary components are in the result
        self.assertIn(debate_context, result)
        self.assertIn(initial_prompt, result)
        self.assertIn("User said: User message", result)
        self.assertIn("Debater said: Debater response", result)
        self.assertIn("Primary said: Primary response", result)
        self.assertIn(moderator_instructions, result)

        # Test with empty conversation history
        result = moderator_context_builder(
            debate_context, initial_prompt, [], moderator_instructions
        )

        self.assertIn(debate_context, result)
        self.assertIn(initial_prompt, result)
        self.assertIn(moderator_instructions, result)

        # Test with None values for optional parameters
        result = moderator_context_builder(
            None, initial_prompt, conversation_history, moderator_instructions
        )

        self.assertIn(initial_prompt, result)
        self.assertIn("User said: User message", result)
        self.assertIn(moderator_instructions, result)

    def test_create_initial_prompt(self):
        """Test create_initial_prompt formats prompt correctly"""
        solution = "Sample solution"
        requirements = "Sample requirements"
        assessment_criteria = "Sample criteria"

        result = create_initial_prompt(solution, requirements, assessment_criteria)

        # Check that all necessary components are in the result
        self.assertIn(solution, result)
        self.assertIn(requirements, result)
        self.assertIn(assessment_criteria, result)
        self.assertIn("Discuss and critically analyze this solution", result)

        # Test with empty strings
        result = create_initial_prompt("", "", "")

        self.assertIn("Initial solution:\n\n", result)
        self.assertIn("Requirements:\n\n", result)
        self.assertIn("Assessment criteria:\n\n", result)

        # Test with multi-line inputs
        multi_line_solution = "Line 1\nLine 2\nLine 3"

        result = create_initial_prompt(
            multi_line_solution, requirements, assessment_criteria
        )

        self.assertIn(multi_line_solution, result)

    def test_synthesize_moderation_summary(self):
        """Test synthesize_moderation_summary function"""
        mock_moderator = MagicMock(spec=LLMInterface)
        mock_moderator.query.return_value = "Moderation summary"

        debate_history = [
            {"role": "user", "content": "Initial prompt"},
            {"role": "debater", "content": "Debater response"},
        ]

        # Test with logger
        logger = MagicMock(spec=logging.Logger)

        result = synthesize_moderation_summary(mock_moderator, debate_history, logger)

        self.assertEqual(result, "Moderation summary")
        mock_moderator.query.assert_called_once()

        # Check if debate history was included in the query
        call_args = mock_moderator.query.call_args[1]
        history = call_args["conversation_history"]
        self.assertEqual(
            len(history), len(debate_history) + 1
        )  # +1 for the added prompt

        # Verify logger calls
        logger.info.assert_called_once()
        logger.debug.assert_called_once()

        # Test without logger
        mock_moderator.reset_mock()

        result = synthesize_moderation_summary(mock_moderator, debate_history)

        self.assertEqual(result, "Moderation summary")
        mock_moderator.query.assert_called_once()

    def test_build_solution_improvement_prompt(self):
        """Test build_solution_improvement_prompt formats prompt correctly"""
        initial_solution = "Initial solution"
        moderator_summary = "Moderator summary"

        result = build_solution_improvement_prompt(initial_solution, moderator_summary)

        # Check that all necessary components are in the result
        self.assertIn(initial_solution, result)
        self.assertIn(moderator_summary, result)
        self.assertIn("provide an improved solution", result)
        self.assertIn("matching the original format", result)

        # Test with empty strings
        result = build_solution_improvement_prompt("", "")

        self.assertIn("Original solution:\n\n", result)
        self.assertIn("Moderator's summary and instructions:\n\n", result)

        # Test with multi-line inputs
        multi_line_summary = "Point 1\nPoint 2\nPoint 3"

        result = build_solution_improvement_prompt(initial_solution, multi_line_summary)

        self.assertIn(multi_line_summary, result)


if __name__ == "__main__":
    unittest.main()
