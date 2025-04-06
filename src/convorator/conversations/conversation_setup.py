# conversation_setup.py

"""
This module provides functions for generating and verifying results from LLMs,
conducting moderated conversations, and synthesizing improvements for solutions.
It includes functions for validating JSON responses, parsing LLM outputs,
and managing conversations between multiple LLMs.

TODO: Configuration Validation: Add checks at the beginning of major functions or within the __post_init__ of dataclasses to validate the configuration (e.g., ensure required LLMs are provided, callables are actually callable).
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Any

from convorator.client.llm_client import LLMInterface


# --- Custom Exception Classes ---
class LLMOrchestrationError(Exception):
    """Base class for exceptions in this module."""

    pass


class SchemaValidationError(LLMOrchestrationError):
    """Raised when JSON data fails schema validation."""

    def __init__(self, message: str, validation_error: Optional[Exception] = None):
        super().__init__(message)
        self.validation_error = validation_error


class LLMResponseError(LLMOrchestrationError):
    """Raised when an LLM response is invalid or cannot be parsed."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class MaxIterationsExceededError(LLMOrchestrationError):
    """Raised when a process exceeds its maximum allowed iterations."""

    pass


# --- End Custom Exception Classes ---


@dataclass
class SolutionLLMGroup:
    # The LLMs used in the solution packet.
    # Can be the same LLM or different ones.
    # Most of the conversations are handled outside of the LLMInterface instance.
    # Instances should be configured already with the needed system prompts.
    # As well as the role names.
    primary_llm: LLMInterface
    debater_llm: LLMInterface
    moderator_llm: LLMInterface
    solution_generation_llm: LLMInterface


@dataclass
class SolutionPacketConfig:
    """
    These configuration defines the parameters for the solution packet.
    In essence, it defines what the solution should look like, how to generate it,
    and how to validate it.
    """

    # The LLMs used in the solution packet.
    llm_group: SolutionLLMGroup

    # Some general parameters for the solution packet.
    topic: Optional[str] = None  # General topic for the conversation
    logger: Optional[logging.Logger] = None  # Logger for debugging and information
    debate_iterations: int = 3  # Number of iterations for the moderated debate
    improvement_iterations: int = 3  # Number of iterations to improve the solution

    # Should the solution be a JSON object
    # If True, the solution will be a JSON object and the solution will be validated against the schema, if provided.
    # If False, the solution will be a string.
    expect_json_output: bool = True

    # If the solution is a JSON object, this is the schema for the solution, in case you want to validate it.
    # If None, no validation is done.
    solution_schema: Optional[Dict] = None


import logging
from typing import List, Dict, Optional

# Assume logger, primary_role_name, debater_role_name are accessible
# You might need to pass role names in or have them defined globally/class level
logger = logging.getLogger(__name__)


# Input for every prompt_builder function
# topic: str
# logger: logging.Logger
# llm_group: SolutionLLMGroup
# solution_schema: Optional[Dict]
# conversation_history: MultiAgentConversation
# initial_solution: str
# requirements: str
# assessment_criteria: str


@dataclass
class ModeratedConversationConfig:

    # `debate_context` to be used as the first message in the debate conversation.
    # It's added to the initial prompt for the debate.
    debate_context: Optional[str] = None

    # The moderator conversation is kept separate from the main conversation.
    # This is building the context for the moderator.
    # It should have the signature:
    # def <function_name>(debate_context: str,
    #                     initial_prompt: str,
    #                     last_primary_msg_content: str,
    #                     last_debater_msg_content: str,
    #                     conversation_history: Optional[List[Dict[str, str]]],
    #                     moderator_instructions: str,
    #                     primary_role_name: str,
    #                     debater_role_name: str) -> str
    # The function should return a string that will be used as the context for the moderator.
    # See `default_moderator_context_builder` above for details
    moderator_context_builder: Optional[callable] = default_moderator_context_builder

    # After each context, the moderator will be asked to provide feedback. With these instructions.
    # It'll be used in the `moderator_context_builder` function.
    moderator_instructions: Optional[str] = None

    # The prompt for the debater to respond to the moderator's feedback.
    # It should have the signature:
    # def <function_name>(primary_response: str,
    #                     moderator_feedback: str,
    #                     primary_role_name: str,
    #                     moderator_role_name: str,) -> str:
    # See `default_debater_prompt_with_feedback` above for details
    debater_prompt_with_feedback: Optional[callable] = default_debater_prompt_with_feedback

    # The prompt for the primary to respond to the moderator's and the debater's feedback.
    # It should have the signature:
    # def <function_name>(debater_response: str,
    #                     moderator_feedback: str,
    #                     primary_last_response: str,
    #                     debater_role_name: str,
    #                     moderator_role_name: str,) -> str:
    # See `default_primary_prompt_with_feedback` above for details
    primary_prompt_builder: Optional[callable] = default_primary_prompt_with_feedback


@dataclass
class ModeratedSolutionImprovementConfig:
    # This is the main configuration for the moderated solution improvement process. Defined above.
    solution_packet_config: SolutionPacketConfig

    # The conversation config for the moderated conversation.
    moderated_conversation_config: ModeratedConversationConfig

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Below are the main parameters for the solution improvement process.
    # They define what you want to achieve and how to do it.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The initial solution to be improved. This is the starting point for the debate.
    initial_solution: str

    # The requirements for the solution. This is what the solution should meet.
    requirements: str

    # The assessment criteria for the solution. This is how the solution will be evaluated by the debnater and the moderator.
    assessment_criteria: str

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Below are functions to build the prompts for the conversation.
    # They are used to create the initial prompt for the debate, synthesize the moderation summary,
    # and build the solution improvement prompt.
    # You can change them if you want to use different functions. Each one has a default implementation below.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # This function should build the initial prompt for the debate.
    # It should have the signature: def <function_name>(solution: str, requirements: str, assessment_criteria: str) -> str
    # See `create_initial_prompt` below for an example and more details
    create_initial_prompt: callable = default_create_initial_prompt

    # See synthesize_moderation_summary below for details
    synthesize_moderation_summary: callable = default_synthesize_moderation_summary

    # See build_solution_improvement_prompt below for details
    # It should have the signature: def <function_name>(initial_solution: str, moderator_summary: str) -> str
    # See `build_solution_improvement_prompt` below for an example and more details
    build_solution_improvement_prompt: callable = default_build_solution_improvement_prompt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Below configs are mainly for the generation and verification of the result.
    # For simplicity, leave most of them as is, unless you really want to get into it.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Function to build fixing prompt for fixing errors (if SolutionPacketConfig.expect_json_output is True).
    # It should be with the signature: def <function>(main_prompt: str, response: str, errors: str) -> str:
    # See `default_fix_prompt_builder` below for an example and more details
    fix_prompt_template: callable = default_fix_prompt_builder
