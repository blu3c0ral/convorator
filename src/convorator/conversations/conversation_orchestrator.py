# conversation_orchestrator.py

"""
Multi-LLM Orchestration Framework for Solution Improvement

This module implements a sophisticated orchestration system for conducting moderated
debates between multiple LLMs to incrementally improve solutions through structured
critique, analysis, and refinement.

Key Components:
---------------
1. JSON Validation & Parsing:
   - Robust parsing of LLM-generated JSON with schema validation
   - Error recovery and correction mechanisms

2. LLM Conversation Management:
   - Stateful conversation tracking with role management
   - Turn-taking coordination between multiple LLM participants

3. Moderated Debate System:
   - Three-role framework (Primary, Debater, Moderator)
   - Structured feedback loops for continuous improvement

4. Solution Generation & Verification:
   - Schema-driven solution validation
   - Iterative refinement with error correction

Core Functions:
--------------
- generate_and_verify_result: Produces schema-compliant outputs with retry logic
- llms_conversation: Manages two-party LLM interactions with turn coordination
- moderated_conversation: Orchestrates three-party debates with moderation
- moderated_solution_improvement: End-to-end solution refinement pipeline
- parse_json_response: Extracts and validates JSON from LLM outputs

Usage Patterns:
--------------
This module is typically used with configuration objects from gen_conversations_helpers.py
to create parameterized improvement pipelines. The most common entry point is
the moderated_solution_improvement function or the improve_solution_with_moderation
wrapper.

Dependencies:
------------
- jsonschema: For JSON validation
- src.utils.gen_conversations_helpers: For configuration classes
- src.utils.llm_client: For LLM interface and conversation management

Example:
-------
```python
# Assuming setup is done as in the example in conversation_setup.py
from conversation_orchestrator import moderated_solution_improvement, improve_solution_with_moderation
from conversation_setup import (
    ModeratedSolutionImprovementConfig, SolutionPacketConfig,
    ModeratedConversationConfig, SolutionLLMGroup, LLMInterface,
    default_create_initial_prompt, default_synthesize_moderation_summary,
    default_build_solution_improvement_prompt, default_moderator_context_builder
)
import logging

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure LLMs (using placeholder)
primary_llm = LLMInterface()
debater_llm = LLMInterface()
moderator_llm = LLMInterface()
solution_llm = LLMInterface()

llm_group = SolutionLLMGroup(
    primary_llm=primary_llm,
    debater_llm=debater_llm,
    moderator_llm=moderator_llm,
    solution_generation_llm=solution_llm
)

solution_packet_config = SolutionPacketConfig(
    llm_group=llm_group,
    logger=logger,
    debate_iterations=2,
    improvement_iterations=2
)

moderated_conversation_config = ModeratedConversationConfig()

config = ModeratedSolutionImprovementConfig(
    solution_packet_config=solution_packet_config,
    moderated_conversation_config=moderated_conversation_config,
    initial_solution='{"key": "initial_value"}',
    requirements="Must be a valid JSON object with 'key' and 'improved_key'.",
    assessment_criteria="Accuracy and adherence to requirements.",
    # Add other optional parameters like custom prompt functions if needed
)

try:
    improved_solution = improve_solution_with_moderation(config)
    print(f"Improved Solution: {improved_solution}")
except Exception as e:
    logger.error(f"Orchestration failed: {e}", exc_info=True)
```

TODO: Async Support
TODO: Token Limit Awareness
"""

import json
from typing import Callable, Dict, List, Optional, Union, Any

import jsonschema

# Import exceptions from the new central location
from convorator.exceptions import (
    LLMOrchestrationError,
    LLMResponseError,
    MaxIterationsExceededError,
    SchemaValidationError,
)
from convorator.client.llm_client import Conversation, LLMInterface, Message

from convorator.conversations.utils import parse_json_response
import convorator.utils.logger as setup_logger

# Configure logging
logger = setup_logger.setup_logger(__name__)

# Import the config class
from convorator.conversations.configurations import OrchestratorConfig

# Import the conversation state class
from convorator.conversations.state import MultiAgentConversation

# Import prompt builders and the container class
from convorator.conversations.prompts import (
    PromptBuilderInputs,
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


def improve_solution_with_moderation(
    config: OrchestratorConfig,  # Changed type hint
    initial_solution: Union[Dict, str],
    requirements: str,
    assessment_criteria: str,
) -> Union[Dict, str]:
    """
    Orchestrates the moderated solution improvement process using a configuration object.

    This function serves as a convenient wrapper around the main orchestration logic
    encapsulated in the `SolutionImprovementOrchestrator` class. It instantiates
    the orchestrator with the provided configuration and runs the workflow.

    Args:
        config: An OrchestratorConfig instance containing all necessary configurations.
        initial_solution: The starting solution (string or dictionary).
        requirements: The requirements the solution must meet.
        assessment_criteria: Criteria for evaluating the solution during debate.

    Returns:
        The final improved solution dictionary or string.

    Raises:
        Exceptions propagated from the `SolutionImprovementOrchestrator.run()` method.
    """
    try:
        orchestrator = SolutionImprovementOrchestrator(
            config=config,
            initial_solution=initial_solution,
            requirements=requirements,
            assessment_criteria=assessment_criteria,
        )
        return orchestrator.run()
    except Exception as e:
        # Log top-level failure if desired, but primarily rely on orchestrator logging
        logger.error(
            f"Top-level execution via improve_solution_with_moderation failed: {e}", exc_info=True
        )
        raise  # Re-raise the original exception


class SolutionImprovementOrchestrator:
    """
    Orchestrates a multi-LLM workflow for debating and improving a solution.

    This class encapsulates the logic previously found in the standalone functions
    `moderated_solution_improvement` and `moderated_conversation`. It manages the
    state, configuration, and step-by-step execution of the workflow.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        initial_solution: Union[Dict, str],
        requirements: str,
        assessment_criteria: str,
    ):  # Changed type hint
        """
        Initializes the orchestrator with the provided configuration and task specifics.

        Args:
            config: The OrchestratorConfig object containing general settings.
            initial_solution: The starting solution (string or dictionary).
            requirements: The requirements the solution must meet.
            assessment_criteria: Criteria for evaluating the solution during debate.
        """
        if not isinstance(config, OrchestratorConfig):
            raise TypeError("config must be an instance of OrchestratorConfig.")

        self.config = config
        # Directly access fields from OrchestratorConfig
        self.llm_group = config.llm_group
        self.logger = config.logger
        self.prompt_builders = config.prompt_builders  # Already aligned
        self.debate_iterations = config.debate_iterations
        self.improvement_iterations = config.improvement_iterations
        self.moderator_instructions = config.moderator_instructions
        self.debate_context = config.debate_context  # Added field

        # Store task-specific inputs directly
        self.initial_solution = initial_solution
        self.requirements = requirements
        self.assessment_criteria = assessment_criteria

        # Extract LLMs
        self.primary_llm = self.llm_group.primary_llm
        self.debater_llm = self.llm_group.debater_llm
        self.moderator_llm = self.llm_group.moderator_llm
        self.solution_generation_llm = self.llm_group.solution_generation_llm

        # Role names
        self.primary_name = self.primary_llm.get_role_name() or "Primary"
        self.debater_name = self.debater_llm.get_role_name() or "Debater"
        self.moderator_name = self.moderator_llm.get_role_name() or "Moderator"

        # Initialize state variables
        self.main_conversation_log: MultiAgentConversation = MultiAgentConversation()
        self.prompt_inputs: Optional[PromptBuilderInputs] = None

    def run(self) -> Union[Dict, str]:
        """
        Executes the full solution improvement workflow.

        Returns:
            The final improved solution (string or dictionary).

        Raises:
            Exceptions propagated from internal steps (e.g., LLMResponseError,
            SchemaValidationError, LLMOrchestrationError).
        """
        self.logger.info("Starting orchestrated solution improvement workflow.")
        try:
            # --- Workflow Steps ---
            # 0. Prepare Prompt Inputs
            self._prepare_prompt_inputs()

            # 1. Build Initial Prompt for Debate
            initial_prompt_content = self._build_initial_prompt_content()

            # 2. Run Moderated Debate
            debate_history = self._run_moderated_debate(initial_prompt_content)

            # 3. Synthesize Summary
            moderator_summary = self._synthesize_summary(debate_history)

            # 4. Build Improvement Prompt
            improvement_prompt = self._build_improvement_prompt(moderator_summary)

            # 5. Generate Final Solution
            improved_solution = self._generate_final_solution(improvement_prompt)

            self.logger.info("Orchestrated solution improvement workflow finished successfully.")
            return improved_solution

        except Exception as e:
            self.logger.exception(f"Error during orchestrated solution improvement: {str(e)}")
            # Re-raise the original exception or wrap it if needed
            if isinstance(
                e,
                (
                    LLMResponseError,
                    SchemaValidationError,
                    LLMOrchestrationError,
                    MaxIterationsExceededError,
                ),
            ):
                raise
            else:
                # Wrap unexpected errors
                raise LLMOrchestrationError(f"Unexpected error during orchestration: {e}") from e

    # --- Private Methods for Workflow Steps ---

    def _prepare_prompt_inputs(self) -> None:
        """Populates the PromptBuilderInputs object for use by prompt builders."""
        self.logger.debug("Preparing inputs for prompt builders.")
        self.prompt_inputs = PromptBuilderInputs(
            topic=self.config.topic,
            logger=self.logger,
            llm_group=self.llm_group,
            solution_schema=self.config.solution_schema,
            initial_solution=self.initial_solution,
            requirements=self.requirements,
            assessment_criteria=self.assessment_criteria,
            moderator_instructions=self.moderator_instructions,  # Use attribute set in __init__
            debate_context=self.debate_context,  # Use attribute set in __init__
            primary_role_name=self.primary_name,
            debater_role_name=self.debater_name,
            moderator_role_name=self.moderator_name,
            expect_json_output=self.config.expect_json_output,
            conversation_history=self.main_conversation_log,
        )

    def _build_initial_prompt_content(self) -> str:
        """Builds the core content for the initial debate prompt."""
        self.logger.debug("Step 1: Building initial prompt content.")
        if not self.prompt_inputs:
            raise LLMOrchestrationError(
                "Prompt inputs not prepared before building initial prompt."
            )

        builder = self.prompt_builders.build_initial_prompt or default_build_initial_prompt
        try:
            content = builder(self.prompt_inputs)
            self.prompt_inputs.initial_prompt_content = content  # Store for later use
            self.logger.debug(f"Initial prompt content built: {content[:200]}...")
            return content
        except Exception as e:
            self.logger.error(f"Failed to build initial prompt content: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build initial prompt content: {e}") from e

    def _run_moderated_debate(self, initial_prompt_content: str) -> List[Dict[str, str]]:
        """
        Orchestrates the moderated debate phase.

        Args:
            initial_prompt_content: The core content for the debate's start.

        Returns:
            The complete debate history log.
        """
        self.logger.info(
            f"Step 2: Conducting moderated debate for {self.debate_iterations} iterations."
        )
        if not self.prompt_inputs:
            raise LLMOrchestrationError("Prompt inputs not prepared before running debate.")

        # Initialize separate conversation perspectives for each agent for this run
        primary_conversation = Conversation(system_message=self.primary_llm.get_system_message())
        debater_conversation = Conversation(system_message=self.debater_llm.get_system_message())
        moderator_conversation = Conversation(
            system_message=self.moderator_llm.get_system_message()
        )

        # Reset main log for this debate run
        self.main_conversation_log = MultiAgentConversation()
        self.prompt_inputs.conversation_history = (
            self.main_conversation_log
        )  # Ensure prompt inputs uses the current log

        # --- Initial User Prompt Setup ---
        user_prompt_builder = (
            self.prompt_builders.build_debate_user_prompt or default_build_debate_user_prompt
        )
        try:
            enhanced_prompt = user_prompt_builder(self.prompt_inputs)
        except Exception as e:
            self.logger.error(f"Failed to build initial debate user prompt: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build initial debate user prompt: {e}") from e

        self.main_conversation_log.add_message(role="user", content=enhanced_prompt)
        self.logger.info(
            f"Starting moderated conversation. Max iterations: {self.debate_iterations}"
        )

        # --- Debate Execution ---
        try:
            last_debater_response = self._run_initial_debater_turn(
                enhanced_prompt, debater_conversation, primary_conversation, moderator_conversation
            )
            previous_moderator_feedback = None  # Feedback from the *previous* round for Primary

            # --- Main Debate Rounds Loop ---
            for i in range(self.debate_iterations):
                round_num = i + 1
                self.logger.info(f"Starting Round {round_num}/{self.debate_iterations}.")

                # Primary Turn
                last_primary_response = self._run_primary_turn(
                    round_num,
                    last_debater_response,
                    previous_moderator_feedback,
                    primary_conversation,
                    debater_conversation,
                    moderator_conversation,
                )

                # Moderator Turn
                current_moderator_feedback = self._run_moderator_turn(
                    round_num,
                    last_primary_response,
                    last_debater_response,
                    moderator_conversation,
                    primary_conversation,
                    debater_conversation,
                )
                # Store feedback for the *next* primary turn
                previous_moderator_feedback = current_moderator_feedback

                # End if Last Iteration
                if i == self.debate_iterations - 1:
                    self.logger.info(
                        "Max iterations reached. Ending debate after moderator feedback."
                    )
                    break

                # Debater Turn (with Feedback)
                last_debater_response = self._run_debater_turn_with_feedback(
                    round_num,
                    last_primary_response,
                    current_moderator_feedback,
                    last_debater_response,  # Pass previous debater response for context
                    debater_conversation,
                    primary_conversation,
                    moderator_conversation,
                )

            self.logger.info("Moderated debate phase completed.")
            return self.main_conversation_log.get_messages()

        except (LLMResponseError, LLMOrchestrationError) as e:
            self.logger.error(f"Error during moderated debate: {e}", exc_info=True)
            raise  # Re-raise specific errors
        except Exception as e:
            self.logger.error(f"Unexpected error during moderated debate: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Unexpected error during moderated debate: {e}") from e

    def _run_initial_debater_turn(
        self, enhanced_prompt: str, debater_conv: Conversation, *other_convs: Conversation
    ) -> str:
        """Handles the first turn where the debater responds to the initial prompt."""
        self.logger.info("Round 0: Getting initial Debater response.")
        debater_conv.add_user_message(enhanced_prompt)
        # Use empty prompt content as it's now in history
        response = self._query_agent_and_update_logs(
            llm_to_query=self.debater_llm,
            prompt_content="",
            role_name=self.debater_name,
            conversation_history=debater_conv,
            other_conversations_history=list(other_convs),
        )
        return response

    def _run_primary_turn(
        self,
        round_num: int,
        last_debater_response: str,
        previous_moderator_feedback: Optional[str],
        primary_conv: Conversation,
        *other_convs: Conversation,
    ) -> str:
        """Handles the Primary agent's turn."""
        self.logger.debug(f"Round {round_num}: Getting {self.primary_name} response.")

        # Need primary's *own* previous response for context in the prompt builder
        primary_msgs = self.main_conversation_log.get_messages_by_role(self.primary_name)
        last_primary_response_content = primary_msgs[-1]["content"] if primary_msgs else None

        # Build prompt using the dedicated builder
        builder = self.prompt_builders.build_primary_prompt or default_build_primary_prompt
        try:
            # Update dynamic prompt inputs (if builder uses them, though defaults rely on history)
            # self.prompt_inputs.last_debater_response = last_debater_response # Example if needed

            primary_prompt = builder(self.prompt_inputs)  # Pass the stateful input object
        except Exception as e:
            self.logger.error(f"Failed to build primary prompt: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build primary prompt: {e}") from e

        primary_conv.add_user_message(primary_prompt)  # Add the *built* prompt to agent history
        response = self._query_agent_and_update_logs(
            llm_to_query=self.primary_llm,
            prompt_content="",  # Handled by history
            role_name=self.primary_name,
            conversation_history=primary_conv,
            other_conversations_history=list(other_convs),
        )
        return response

    def _run_moderator_turn(
        self,
        round_num: int,
        last_primary_response: str,
        last_debater_response: str,
        moderator_conv: Conversation,
        *other_convs: Conversation,
    ) -> str:
        """Handles the Moderator agent's assessment turn."""
        self.logger.debug(f"Round {round_num}: Getting {self.moderator_name} feedback.")

        # Ensure moderator instructions are set (use default if necessary)
        mod_instructions = self._get_moderator_instructions()
        self.prompt_inputs.moderator_instructions = mod_instructions  # Update inputs object

        # Build context using the dedicated builder
        builder = self.prompt_builders.build_moderator_context or default_build_moderator_context
        try:
            # Update dynamic prompt inputs (if builder uses them, though defaults rely on history)
            # self.prompt_inputs.last_primary_response = last_primary_response # Example if needed

            moderator_context = builder(self.prompt_inputs)
        except Exception as e:
            self.logger.error(f"Failed to build moderator context: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build moderator context: {e}") from e

        moderator_conv.add_user_message(moderator_context)  # Add *built* context to agent history
        response = self._query_agent_and_update_logs(
            llm_to_query=self.moderator_llm,
            prompt_content="",  # Handled by history
            role_name=self.moderator_name,
            conversation_history=moderator_conv,
            other_conversations_history=list(other_convs),
        )
        return response

    def _run_debater_turn_with_feedback(
        self,
        round_num: int,
        last_primary_response: str,
        current_moderator_feedback: str,
        last_debater_response: Optional[str],
        debater_conv: Conversation,
        *other_convs: Conversation,
    ) -> str:
        """Handles the Debater agent's turn, incorporating moderator feedback."""
        self.logger.debug(
            f"Round {round_num}: Getting {self.debater_name} response (with feedback)."
        )

        # Build prompt using the dedicated builder
        builder = self.prompt_builders.build_debater_prompt or default_build_debater_prompt
        try:
            # Update dynamic prompt inputs (if builder uses them)
            # self.prompt_inputs.last_primary_response = last_primary_response # Example if needed
            # self.prompt_inputs.current_moderator_feedback = current_moderator_feedback # Example if needed

            debater_prompt = builder(self.prompt_inputs)  # Pass stateful object
        except Exception as e:
            self.logger.error(f"Failed to build debater prompt with feedback: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build debater prompt with feedback: {e}") from e

        debater_conv.add_user_message(debater_prompt)  # Add *built* prompt to agent history
        response = self._query_agent_and_update_logs(
            llm_to_query=self.debater_llm,
            prompt_content="",  # Handled by history
            role_name=self.debater_name,
            conversation_history=debater_conv,
            other_conversations_history=list(other_convs),
        )
        return response

    def _get_moderator_instructions(self) -> str:
        """Gets moderator instructions from config or generates default."""
        if self.moderator_instructions:  # Check the instance attribute
            return self.moderator_instructions

        self.logger.debug("No moderator instructions provided, generating default.")
        builder = (
            self.prompt_builders.build_moderator_instructions
            or default_build_moderator_instructions
        )
        try:
            return builder(self.prompt_inputs)
        except Exception as e:
            self.logger.error(f"Failed to build default moderator instructions: {e}", exc_info=True)
            raise LLMOrchestrationError(
                f"Failed to build default moderator instructions: {e}"
            ) from e

    def _synthesize_summary(self, debate_history: List[Dict[str, str]]) -> str:
        """Generates a summary from the moderator using the debate history."""
        self.logger.info("Step 3: Synthesizing moderation summary.")
        if not self.prompt_inputs:
            raise LLMOrchestrationError("Prompt inputs not prepared before synthesizing summary.")

        # Build the prompt *for* the summary
        builder = self.prompt_builders.build_summary_prompt or default_build_summary_prompt
        try:
            # Pass the *actual* debate history to the builder if it needs it
            # Note: The default builder doesn't directly use history content in the prompt string itself,
            # but expects the caller (this method) to prepend the history for the LLM call.
            # We pass prompt_inputs for consistency and access to static config if needed.
            summary_prompt = builder(self.prompt_inputs)
        except Exception as e:
            self.logger.error(f"Failed to build summary prompt: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build summary prompt: {e}") from e

        self.logger.debug("Querying moderator LLM for summary.")
        try:
            # Query the moderator LLM with the *history* plus the *summary prompt*
            summary = self.moderator_llm.query(
                prompt="",  # Prompt is part of the history
                use_conversation=False,  # Use explicit history
                # Ensure history includes system messages if the LLM expects them for context
                conversation_history=debate_history + [{"role": "user", "content": summary_prompt}],
                # Add other query params if needed from config
            )
            self.logger.debug(f"Moderation summary received: {summary[:200]}...")
            self.prompt_inputs.moderator_summary = summary  # Store for potential later use
            return summary
        except LLMResponseError as e:
            self.logger.error(f"Moderator LLM failed to generate summary: {e}", exc_info=True)
            raise  # Re-raise specific LLM errors
        except Exception as e:
            self.logger.error(
                f"Unexpected error querying moderator for summary: {e}", exc_info=True
            )
            raise LLMOrchestrationError(f"Moderator LLM failed to generate summary: {e}") from e

    def _build_improvement_prompt(self, moderator_summary: str) -> str:
        """Builds the prompt for the final solution generation LLM."""
        self.logger.debug("Step 4: Building solution improvement prompt.")
        if not self.prompt_inputs:
            raise LLMOrchestrationError(
                "Prompt inputs not prepared before building improvement prompt."
            )

        self.prompt_inputs.moderator_summary = moderator_summary  # Ensure it's set

        builder = self.prompt_builders.build_improvement_prompt or default_build_improvement_prompt
        try:
            prompt = builder(self.prompt_inputs)
            self.logger.debug(f"Improvement prompt created: {prompt[:200]}...")
            return prompt
        except Exception as e:
            self.logger.error(f"Failed to build solution improvement prompt: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build solution improvement prompt: {e}") from e

    def _generate_final_solution(self, improvement_prompt: str) -> Union[Dict, str]:
        """Generates and verifies the final improved solution."""
        self.logger.info(
            f"Step 5: Generating improved solution (max {self.improvement_iterations} attempts)."
        )
        if not self.prompt_inputs:
            raise LLMOrchestrationError(
                "Prompt inputs not prepared before generating final solution."
            )

        # Determine the correct fix prompt builder
        fix_builder = self.prompt_builders.build_fix_prompt or default_build_fix_prompt

        try:
            # Use the generalized generate_and_verify method
            improved_solution = self._generate_and_verify_result(
                llm_service=self.solution_generation_llm,
                context="Final Solution Generation",
                main_prompt_or_template=improvement_prompt,
                fix_prompt_builder=fix_builder,
                result_schema=self.config.solution_schema,
                use_conversation=False,
                max_improvement_iterations=self.improvement_iterations,
                json_result=self.config.expect_json_output,
            )
            self.logger.info("Improved solution generated successfully.")
            return improved_solution
        except (LLMResponseError, SchemaValidationError, MaxIterationsExceededError) as e:
            self.logger.error(f"Failed to generate final verified solution: {e}", exc_info=True)
            raise  # Re-raise specific verification/generation errors
        except Exception as e:
            self.logger.error(
                f"Unexpected error during final solution generation: {e}", exc_info=True
            )
            raise LLMOrchestrationError(
                f"Unexpected error during final solution generation: {e}"
            ) from e

    # --- Helper Methods (Previously standalone functions) ---

    def _query_agent_and_update_logs(
        self,
        llm_to_query: LLMInterface,
        prompt_content: str,  # Keeping for now, but should be empty if prompt added to history first
        role_name: str,
        conversation_history: Conversation,  # Agent's perspective history
        other_conversations_history: Optional[List[Conversation]] = None,
        query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Queries an LLM agent, updates its perspective history, the main log,
        and optionally other agents' histories.

        Args:
            llm_to_query: The LLMInterface instance to query.
            prompt_content: The prompt content (might be empty if using history).
            role_name: The specific role name for logging.
            conversation_history: The Conversation object holding the agent's view.
            other_conversations_history: List of other Conversation objects to update.
            query_kwargs: Additional arguments for the LLM query.

        Returns:
            The content of the LLM response.

        Raises:
            LLMResponseError: If the query fails.
            LLMOrchestrationError: For unexpected errors during logging/updates.
        """
        try:
            self.logger.info(f"Querying {role_name}...")
            # Ensure prompt is in history if needed
            # If prompt_content is not empty, it implies it wasn't added to history yet.
            # This depends on how the calling functions (_run_xxx_turn) manage history.
            # Assuming history IS managed correctly by callers and prompt_content is redundant/empty.
            if prompt_content:
                self.logger.warning(
                    "prompt_content provided to _query_agent_and_update_logs, but history is expected to contain the prompt."
                )
                # Decide: Add it now? -> conversation_history.add_user_message(prompt_content) Or rely on caller?

            response_content = llm_to_query.query(
                prompt="",  # Assuming prompt is last message in history now
                use_conversation=False,  # Use explicit history provided
                conversation_history=conversation_history.get_messages(),
                **(query_kwargs or {}),
            )
            self.logger.info(f"{role_name} responded successfully.")
            self.logger.debug(f"{role_name} response snippet: {response_content[:100]}...")

            # 2. Add response to the agent's own perspective history
            conversation_history.add_assistant_message(response_content)
            self.logger.debug(f"Added response to {role_name}'s perspective history.")

            # 3. Add response to the main multi-agent log
            if not self.main_conversation_log:
                raise LLMOrchestrationError("Main conversation log not initialized.")
            self.main_conversation_log.add_message(role=role_name, content=response_content)
            self.logger.debug(f"Added {role_name}'s response to the main conversation log.")

            # 4. Update other conversation histories (e.g., add Debater's response to Primary's history)
            if other_conversations_history:
                # The original logic added the *other* agent's response as a 'user' message
                # to the current agent's history. This seems correct for context.
                message_to_add_for_others = f"[{role_name}]: {response_content}"
                for other_conv in other_conversations_history:
                    other_conv.add_user_message(message_to_add_for_others)
                self.logger.debug(
                    f"Updated {len(other_conversations_history)} other conversations with {role_name}'s response."
                )

            return response_content

        except LLMResponseError as e:
            self.logger.error(f"{role_name} query failed: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error querying {role_name} or updating logs: {e}")
            raise LLMOrchestrationError(
                f"Unexpected error during {role_name} query/log update: {e}"
            ) from e

    def _generate_and_verify_result(
        self,
        llm_service: LLMInterface,
        context: str,
        main_prompt_or_template: str,
        fix_prompt_builder: Callable,
        result_schema: Optional[Dict] = None,
        use_conversation: bool = True,
        max_improvement_iterations: int = 3,
        json_result: bool = True,
    ) -> Union[Dict, str]:
        """
        Generates a result from an LLM, verifies (if JSON), and attempts to fix errors.

        Args:
            llm_service: LLM interface to use.
            context: Logging context string.
            main_prompt_or_template: The main prompt content.
            fix_prompt_builder: Callable function to build the fix prompt. It should accept PromptBuilderInputs.
            result_schema: Optional JSON schema for validation.
            use_conversation: Whether to use the LLM service's internal conversation state.
            max_improvement_iterations: Max attempts to fix errors.
            json_result: Whether the expected result is JSON.

        Returns:
            The verified result (Dict if JSON, otherwise str).

        Raises:
            MaxIterationsExceededError: If max iterations are reached without success.
            SchemaValidationError: If JSON validation fails after retries.
            LLMResponseError: If an LLM query fails.
        """
        self.logger.info(
            f"Generating result for context: {context}. Max iterations: {max_improvement_iterations}"
        )
        if not self.prompt_inputs:
            raise LLMOrchestrationError(
                "Prompt inputs not prepared before generating verified result."
            )

        last_response = None
        last_error = None

        for attempt in range(max_improvement_iterations):
            self.logger.info(
                f"Attempt {attempt + 1}/{max_improvement_iterations} for context: {context}"
            )

            if attempt == 0:
                # Initial attempt with the main prompt
                prompt = main_prompt_or_template
            else:
                # Subsequent attempts: use the fix prompt builder
                if last_response is None or last_error is None:
                    # This should not happen if attempt > 0
                    raise LLMOrchestrationError(
                        "Cannot build fix prompt: missing last response or error."
                    )

                # Update prompt_inputs with data needed for the fix prompt
                self.prompt_inputs.response_to_fix = last_response
                self.prompt_inputs.errors_to_fix = str(last_error)  # Pass error as string

                try:
                    prompt = fix_prompt_builder(self.prompt_inputs)
                    self.logger.debug(f"Fix prompt generated for attempt {attempt + 1}.")
                except Exception as e:
                    self.logger.error(f"Failed to build fix prompt: {e}", exc_info=True)
                    raise LLMOrchestrationError(f"Failed to build fix prompt: {e}") from e

            try:
                # Query the LLM
                response = llm_service.query(prompt, use_conversation=use_conversation)
                last_response = response  # Store for potential next fix prompt

                if not json_result:
                    self.logger.info(f"Context '{context}': Non-JSON result obtained successfully.")
                    return response  # Return raw string if no JSON expected

                # Attempt to parse and validate JSON
                parsed_json = parse_json_response(response, self.logger)  # Using util function
                if result_schema:
                    try:
                        jsonschema.validate(instance=parsed_json, schema=result_schema)
                        self.logger.info(
                            f"Context '{context}': JSON result parsed and validated successfully."
                        )
                        return parsed_json  # Success!
                    except jsonschema.exceptions.ValidationError as e:
                        self.logger.warning(
                            f"Context '{context}': JSON validation failed (Attempt {attempt + 1}): {e.message}"
                        )
                        last_error = f"Schema validation failed: {e.message}"
                        # Continue to next iteration to try fixing
                else:
                    self.logger.info(
                        f"Context '{context}': JSON result parsed successfully (no schema validation)."
                    )
                    return parsed_json  # Success (parsed, no schema)

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Context '{context}': JSON parsing failed (Attempt {attempt + 1}): {e}"
                )
                last_error = f"JSON parsing failed: {e}"
                # Continue to next iteration
            except LLMResponseError as e:
                self.logger.error(
                    f"Context '{context}': LLM query failed (Attempt {attempt + 1}): {e}",
                    exc_info=True,
                )
                raise  # Propagate LLM errors immediately
            except Exception as e:
                self.logger.error(
                    f"Context '{context}': Unexpected error during generation/verification (Attempt {attempt+1}): {e}",
                    exc_info=True,
                )
                last_error = f"Unexpected error: {e}"  # Treat as a fixable error for next attempt

        # If loop finishes without returning, max iterations were exceeded
        self.logger.error(
            f"Context '{context}': Failed to generate valid result after {max_improvement_iterations} attempts."
        )
        if isinstance(last_error, str) and "Schema validation failed" in last_error:
            # If the last error was validation, raise specific exception
            raise SchemaValidationError(
                f"Failed to validate result for '{context}' against schema after {max_improvement_iterations} attempts. Last error: {last_error}. Last response: {last_response}",
                schema=result_schema,
                instance=last_response,  # Pass the raw response that failed
            )
        else:
            raise MaxIterationsExceededError(
                f"Failed to generate valid result for '{context}' after {max_improvement_iterations} attempts. Last error: {last_error}. Last response: {last_response}"
            )
