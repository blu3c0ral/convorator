# src/convorator/conversations/conversation_orchestrator.py

"""
Multi-LLM Orchestration Framework for Collaborative Solution Improvement

This module implements a comprehensive orchestration system that leverages multiple
Large Language Models (LLMs) in distinct roles to collaboratively refine and improve
solutions through structured debate, analysis, and iterative enhancement.

Architecture Overview:
---------------------
The orchestrator employs a multi-agent framework where each LLM plays a specialized role:

1. Primary Agent: Responsible for proposing initial solutions and refinements during the debate.
2. Debater Agent: Critiques proposals, identifies weaknesses, and suggests alternatives during the debate.
3. Moderator Agent: Guides the discussion, evaluates arguments, and synthesizes feedback from the debate.
4. Solution Generation Agent: Constructs the final, improved solution based on the debate summary and requirements.

The solution improvement process follows a structured workflow:

1. Initial Setup: Configuration, loading LLMs, preparing initial inputs.
2. Moderated Debate: Multi-turn discussion between the Primary and Debater agents,
   with the Moderator providing guidance and evaluation based on assessment criteria.
3. Summary Synthesis: The Moderator consolidates insights and key feedback from the debate.
4. Solution Generation: The Solution Generation Agent creates an improved solution incorporating the debate summary.
5. Verification & Correction: The generated solution is validated against schema requirements.
   If validation fails, the orchestrator attempts correction by feeding errors back to the Solution Generation Agent.

Key Features:
------------
1. Stateful Conversation Management:
   - Maintains separate conversation contexts for each agent during the debate.
   - Coordinates turns and ensures proper information flow between agents.
   - Tracks the overall conversation history for context and logging.

2. Robust Error Handling & Verification:
   - Schema validation for JSON outputs with descriptive error messages.
   - Automatic retry loop for solution generation with intelligent error correction feedback.
   - Token limit awareness and automatic history truncation to prevent API errors.

3. Configurable Orchestration:
   - Customizable prompt generation logic via pluggable builder functions.
   - Adjustable iteration counts for both the debate and the solution improvement/correction loop.
   - Flexible LLM assignment to different roles.

4. Schema-Driven Solution Refinement:
   - Ensures final output adheres to a specified JSON schema (if provided).
   - Focuses the improvement process on meeting structural and content requirements.

Implementation Details:
---------------------
The core of the system is the SolutionImprovementOrchestrator class, which:

1. Manages the multi-step workflow through its `run()` method.
2. Coordinates the debate phase via `_run_moderated_debate()`.
3. Handles individual agent turns using the `_execute_agent_turn()` helper.
4. Generates and iteratively corrects the final solution via `_generate_and_verify_result()`.

The orchestrator maintains conversation state separately for each agent involved in the
debate, allowing them distinct perspectives while tracking the unified conversation flow.
Error recovery in `_generate_and_verify_result()` is crucial, providing specific error
details back to the Solution Generation Agent to guide the correction attempts.

Usage:
-----
The primary entry point is the `improve_solution_with_moderation()` function, which accepts:
- A configuration object (`OrchestratorConfig`) detailing LLMs, iterations, etc.
- An initial solution (string or dictionary).
- Requirements the final solution must meet.
- Assessment criteria to guide the debate phase.

Example:
```python
from convorator.conversations.conversation_orchestrator import improve_solution_with_moderation
from convorator.conversations.configurations import OrchestratorConfig
from convorator.client.llm_client import LLMInterface

# 1. Configure LLMs for each role
primary_llm = LLMInterface(...) # E.g., model tuned for generation
debater_llm = LLMInterface(...) # E.g., model tuned for critique
moderator_llm = LLMInterface(...) # E.g., model tuned for summarization/evaluation
solution_llm = LLMInterface(...) # E.g., model tuned for final generation/correction

# 2. Group LLMs
llm_group = SolutionLLMGroup(
    primary_llm=primary_llm,
    debater_llm=debater_llm,
    moderator_llm=moderator_llm,
    solution_generation_llm=solution_llm
)

# 3. Create Orchestrator Configuration
config = OrchestratorConfig(
    llm_group=llm_group,
    topic="Improve Product Description",
    requirements=requirements,
    assessment_criteria=assessment_criteria,
    debate_iterations=3,
    improvement_iterations=2
    # Add other configurations like custom prompt builders or schema if needed
)

# 4. Define the task
initial_solution = {'product_id': 123, 'status': 'draft'}
requirements = "The final solution must be a JSON object containing 'product_id' (int), 'status' (string: 'active' or 'inactive'), and 'description' (string, max 100 chars)."
assessment_criteria = "Focus on correctness of status and conciseness/relevance of description."

# 5. Run the improvement process
try:
    improved_solution = improve_solution_with_moderation(
        config=config,
        initial_solution=initial_solution,
        requirements=requirements,
        assessment_criteria=assessment_criteria
    )
    print("Successfully generated improved solution:", improved_solution)
except Exception as e:
    print(f"Orchestration failed: {e}")

```

Dependencies:
------------
- jsonschema: For JSON schema validation.
- convorator.exceptions: For specialized exception types.
- convorator.client.llm_client: For LLM communication and conversation management.
- convorator.conversations.configurations: For orchestration configuration (`OrchestratorConfig`).
- convorator.conversations.state: For conversation state management (`MultiAgentConversation`).
- convorator.conversations.prompts: For default prompt builders and input structure (`PromptBuilderInputs`).
- convorator.conversations.utils: For utility functions like JSON parsing (`parse_json_response`).
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
    LLMClientError,
    LLMConfigurationError,
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

# Import shared types
from convorator.conversations.types import SolutionLLMGroup

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
        # self.prompt_builders = config.prompt_builders  # Already aligned - REMOVED
        self.debate_iterations = config.debate_iterations
        self.improvement_iterations = config.improvement_iterations
        # Get moderator instructions and context
        self.moderator_instructions = config.moderator_instructions_override
        self.debate_context = config.debate_context_override
        # Access individual prompt builder functions directly from config
        self.build_initial_prompt = config.build_initial_prompt
        self.build_debate_user_prompt = config.build_debate_user_prompt
        self.build_moderator_context = config.build_moderator_context
        self.build_moderator_role_instructions = config.build_moderator_role_instructions
        self.build_primary_prompt = config.build_primary_prompt
        self.build_debater_prompt = config.build_debater_prompt
        self.build_summary_prompt = config.build_summary_prompt
        self.build_improvement_prompt = config.build_improvement_prompt
        self.build_fix_prompt = config.build_fix_prompt

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

        builder = self.build_initial_prompt or default_build_initial_prompt
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
        # Assign to instance attributes to fix the bug
        self.primary_conv = primary_conversation
        self.debater_conv = debater_conversation
        self.moderator_conv = moderator_conversation

        # Reset main log for this debate run
        self.main_conversation_log = MultiAgentConversation()
        self.prompt_inputs.conversation_history = (
            self.main_conversation_log
        )  # Ensure prompt inputs uses the current log

        # --- Initial User Prompt Setup ---
        user_prompt_builder = self.build_debate_user_prompt or default_build_debate_user_prompt
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
                enhanced_prompt, self.debater_conv, self.primary_conv, self.moderator_conv
            )
            previous_moderator_feedback = None  # Feedback from the *previous* round for Primary

            # --- Main Debate Rounds Loop ---
            for i in range(self.debate_iterations):
                round_num = i + 1
                self.logger.info(f"Starting Round {round_num}/{self.debate_iterations}.")

                # Primary Turn
                last_primary_response = self._run_primary_turn(round_num)

                # Moderator Turn
                current_moderator_feedback = self._run_moderator_turn(round_num)
                # Store feedback for the *next* primary turn
                previous_moderator_feedback = current_moderator_feedback

                # End if Last Iteration
                if i == self.debate_iterations - 1:
                    self.logger.info(
                        "Max iterations reached. Ending debate after moderator feedback."
                    )
                    break

                # Debater Turn (with Feedback)
                last_debater_response = self._run_debater_turn(round_num)

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
            role_name=self.debater_name,
            conversation_history=debater_conv,
            other_conversations_history=list(other_convs),
        )
        return response

    def _run_primary_turn(self, round_num: int) -> str:
        """Executes one turn of the Primary agent."""
        self.logger.debug(f"Round {round_num}: Getting {self.primary_name} response.")

        return self._execute_agent_turn(
            role_name=self.primary_name,
            llm=self.primary_llm,
            prompt_builder=self.build_primary_prompt,
            agent_conv=self.primary_conv,
            other_convs=[self.debater_conv, self.moderator_conv],
        )

    def _run_moderator_turn(self, round_num: int) -> str:
        """Executes one turn of the Moderator agent."""
        self.logger.debug(f"Round {round_num}: Getting {self.moderator_name} response.")

        # Ensure moderator instructions are set (use default if necessary)
        mod_instructions = self._get_moderator_instructions()
        self.prompt_inputs.moderator_instructions = mod_instructions  # Update inputs object

        return self._execute_agent_turn(
            role_name=self.moderator_name,
            llm=self.moderator_llm,
            prompt_builder=self.build_moderator_context,
            agent_conv=self.moderator_conv,
            other_convs=[self.primary_conv, self.debater_conv],
        )

    def _run_debater_turn(self, round_num: int) -> str:
        """Executes one turn of the Debater agent."""
        self.logger.debug(f"Round {round_num}: Getting {self.debater_name} response.")

        return self._execute_agent_turn(
            role_name=self.debater_name,
            llm=self.debater_llm,
            prompt_builder=self.build_debater_prompt,
            agent_conv=self.debater_conv,
            other_convs=[self.primary_conv, self.moderator_conv],
        )

    def _execute_agent_turn(
        self,
        role_name: str,
        llm: LLMInterface,
        prompt_builder: Callable[[PromptBuilderInputs], str],
        agent_conv: Conversation,
        other_convs: List[Conversation],
    ) -> str:
        """
        Executes a single turn for an agent: builds prompt, adds to history, queries, updates logs.

        Args:
            role_name: The name of the role executing the turn.
            llm: The LLMInterface for the agent.
            prompt_builder: The function to build the prompt content.
            agent_conv: The Conversation object for the agent.
            other_convs: List of Conversation objects for other agents.

        Returns:
            The content of the LLM response.

        Raises:
            LLMOrchestrationError: If prompt building fails.
            LLMResponseError: If the LLM query fails.
        """
        if not self.prompt_inputs:
            # This should ideally not happen if run() is called first
            raise LLMOrchestrationError(
                "Prompt inputs not initialized before executing agent turn."
            )

        # Build the prompt using the dedicated builder
        try:
            prompt_content = prompt_builder(self.prompt_inputs)
            self.logger.debug(f"Built prompt for {role_name}: {prompt_content[:100]}...")
        except Exception as e:
            self.logger.error(f"Failed to build prompt for {role_name}: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Failed to build prompt for {role_name}: {e}") from e

        # Add the *built* prompt to the agent's perspective history
        agent_conv.add_user_message(prompt_content)

        # Query the agent and update all relevant logs
        response = self._query_agent_and_update_logs(
            llm_to_query=llm,
            role_name=role_name,
            conversation_history=agent_conv,
            other_conversations_history=other_convs,
        )
        return response

    def _get_moderator_instructions(self) -> str:
        """Gets moderator instructions from config or generates default."""
        if self.moderator_instructions:  # Check the instance attribute
            return self.moderator_instructions

        self.logger.debug("No moderator instructions provided, generating default.")
        builder = self.build_moderator_role_instructions or default_build_moderator_instructions
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
        builder = self.build_summary_prompt or default_build_summary_prompt
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
            # Prepare history + prompt for the summary call
            full_history_for_summary = debate_history + [
                {"role": "user", "content": summary_prompt}
            ]

            # Prepare history, handling token limits
            messages_to_send = self._prepare_history_for_llm(
                llm_service=self.moderator_llm,
                messages=full_history_for_summary,
                context="Moderator Summary Synthesis",
            )

            # Query the moderator LLM with the prepared history
            summary = self.moderator_llm.query(
                prompt="",  # Prompt is part of the history
                use_conversation=False,  # Use explicit history
                conversation_history=messages_to_send,
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

        builder = self.build_improvement_prompt or default_build_improvement_prompt
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
        fix_builder = self.build_fix_prompt or default_build_fix_prompt

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

    def _prepare_history_for_llm(
        self,
        llm_service: LLMInterface,
        messages: List[Dict[str, str]],
        context: str = "LLM Query",
    ) -> List[Dict[str, str]]:
        """
        Prepares a list of messages for an LLM call, ensuring it fits within the context limit.

        Args:
            llm_service: The LLMInterface instance being used.
            messages: The list of messages (dictionaries) intended for the LLM.
            context: A string describing the context for logging purposes.

        Returns:
            The potentially truncated list of messages ready for the API call.

        Raises:
            LLMOrchestrationError: If token counting fails unexpectedly.
        """
        try:
            max_limit = llm_service.get_context_limit()
            response_buffer = llm_service.max_tokens + 10

            system_message = None
            history_messages = list(messages)  # Work on a copy
            if history_messages and history_messages[0].get("role") == "system":
                system_message = history_messages.pop(0)

            current_history_tokens = sum(
                llm_service.count_tokens(msg.get("content", "")) for msg in history_messages
            )
            system_tokens = (
                llm_service.count_tokens(system_message.get("content", "")) if system_message else 0
            )
            total_tokens_with_buffer = current_history_tokens + system_tokens + response_buffer

            if total_tokens_with_buffer <= max_limit:
                self.logger.debug(
                    f"[{context}] History size ({current_history_tokens + system_tokens} tokens) + buffer ({response_buffer}) fits within limit ({max_limit}). No truncation needed."
                )
                return messages  # Return original if no truncation needed

            self.logger.warning(
                f"[{context}] History size ({current_history_tokens + system_tokens} tokens) + buffer ({response_buffer}) exceeds limit ({max_limit}). Truncation required."
            )

            # Check if system message + buffer alone exceeds limit
            if system_tokens + response_buffer > max_limit and system_message:
                self.logger.error(
                    f"[{context}] System message ({system_tokens} tokens) + buffer ({response_buffer}) alone exceeds limit ({max_limit}). Cannot proceed."
                )
                raise LLMOrchestrationError(
                    f"[{context}] System message ({system_tokens} tokens) + buffer ({response_buffer}) exceeds limit ({max_limit})."
                )

            removed_count = 0
            removed_tokens_count = 0
            # Loop while the current history + system + buffer is too large
            while (
                current_history_tokens + system_tokens + response_buffer > max_limit
                and history_messages
            ):
                msg_to_remove = history_messages.pop(0)  # Remove oldest message from history
                tokens_removed = llm_service.count_tokens(msg_to_remove.get("content", ""))
                current_history_tokens -= tokens_removed  # Decrement history token count
                removed_count += 1
                removed_tokens_count += tokens_removed
                self.logger.debug(
                    f"[{context}] Truncating: Removed message '{msg_to_remove.get('role')}' ({tokens_removed} tokens). Remaining history tokens: {current_history_tokens}"
                )

            # After loop, check if the remaining history + system + buffer fits
            if current_history_tokens + system_tokens + response_buffer > max_limit:
                # This check is crucial. It catches cases where the loop finished because
                # history_messages became empty, but the remaining (potentially just system message)
                # still exceeds the limit, OR if the last message couldn't be removed and still violates the limit.
                self.logger.error(
                    f"[{context}] Truncation failed! Final history tokens ({current_history_tokens}) + system tokens ({system_tokens}) + buffer ({response_buffer}) still exceed limit ({max_limit})."
                )
                raise LLMOrchestrationError(
                    f"[{context}] Failed to truncate history sufficiently. Remaining tokens ({current_history_tokens + system_tokens}) + buffer ({response_buffer}) exceed limit ({max_limit})."
                )

            self.logger.warning(
                f"[{context}] Truncation complete. Removed {removed_count} messages ({removed_tokens_count} tokens). Final history size: {current_history_tokens + system_tokens} tokens."
            )

            # Construct the final list
            final_messages = ([system_message] if system_message else []) + history_messages
            return final_messages

        except Exception as e:
            self.logger.exception(
                f"[{context}] Unexpected error during history preparation/token counting: {e}"
            )
            # Wrap unexpected errors during token handling
            raise LLMOrchestrationError(
                f"Failed to prepare history due to token counting/limit error: {e}"
            ) from e

    def _query_agent_and_update_logs(
        self,
        llm_to_query: LLMInterface,
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
            # The calling function is responsible for adding the prompt/context
            # as the last user message in conversation_history before calling this.
            # if prompt_content:
            #     self.logger.warning(
            #         "prompt_content provided to _query_agent_and_update_logs, but history is expected to contain the prompt."
            #     )

            # Prepare history, handling token limits
            messages_to_send = self._prepare_history_for_llm(
                llm_service=llm_to_query,
                messages=conversation_history.get_messages(),
                context=f"{role_name} Query",
            )

            response_content = llm_to_query.query(
                prompt="",  # Assuming prompt is last message in history now
                use_conversation=False,  # Use explicit history provided
                conversation_history=messages_to_send,  # Use the prepared list
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
        Generates a result using an LLM, verifies it (parsing, schema validation),
        and attempts to fix errors iteratively up to a maximum number of attempts.

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

        response: Optional[str] = None
        last_error_str: str = (
            "No error encountered"  # Keep track of the string representation for logging
        )
        last_error_obj: Optional[Exception] = None  # Store the last actual exception object

        # Initialize conversation history for the generation LLM if needed
        generation_conversation = Conversation(system_message=llm_service.get_system_message())

        current_prompt = main_prompt_or_template

        for attempt in range(1, max_improvement_iterations + 1):
            self.logger.info(
                f"Context '{context}': Generating result (Attempt {attempt}/{max_improvement_iterations})"
            )
            response_content = None
            response_to_fix = None  # Store raw response for potential fix prompt

            try:
                # === Check Token Limits Before Query (Stateless Case) ===
                if not use_conversation:
                    required_buffer = (
                        llm_service.max_tokens + 10 if llm_service.max_tokens else 250 + 10
                    )  # Add buffer
                    context_limit = llm_service.get_context_limit()
                    history_token_limit = (
                        context_limit - required_buffer if context_limit else float("inf")
                    )

                    try:
                        prompt_tokens = llm_service.count_tokens(current_prompt)
                    except Exception as token_error:
                        self.logger.warning(
                            f"Context '{context}': Token counting failed for stateless prompt on attempt {attempt}: {token_error}"
                        )
                        err_msg = f"Token counting failed for stateless prompt: {token_error}"
                        last_error_str = err_msg
                        # Fix syntax error - store the exception directly
                        last_error_obj = token_error
                        if attempt == max_improvement_iterations:
                            break  # Break to raise MaxIterationsExceededError outside the loop
                        continue  # Try again if iterations remain

                    if prompt_tokens > history_token_limit:
                        error_msg = (
                            f"Context '{context}': Single prompt exceeded token limit "
                            f"(Prompt: {prompt_tokens}, Limit: {history_token_limit}) "
                            f"on attempt {attempt}. Skipping generation."
                        )
                        self.logger.warning(error_msg)
                        last_error_str = error_msg
                        last_error_obj = LLMOrchestrationError(error_msg)
                        if attempt == max_improvement_iterations:
                            break  # Break to raise MaxIterationsExceededError outside the loop
                        continue  # Try again if iterations remain

                # === Perform LLM Query ===
                if use_conversation:
                    generation_conversation.add_user_message(current_prompt)
                    prepared_history = self._prepare_history_for_llm(
                        llm_service,
                        generation_conversation.get_messages(),
                        context=f"{context} - Gen Prep",
                    )
                    response_content = llm_service.query(
                        prompt="", use_conversation=False, conversation_history=prepared_history
                    )
                    generation_conversation.add_assistant_message(response_content)
                else:
                    response_content = llm_service.query(current_prompt, use_conversation=False)

                response_to_fix = response_content

                # === Verification ===
                if json_result:
                    result: Union[Dict, str] = parse_json_response(
                        self.logger, response_content, context, schema=result_schema
                    )
                    self.logger.info(
                        f"Context '{context}': Generation and validation successful (Attempt {attempt})."
                    )
                    return result
                else:
                    self.logger.info(
                        f"Context '{context}': Generation successful (non-JSON) (Attempt {attempt})."
                    )
                    return response_content

            except (LLMResponseError, SchemaValidationError) as e:
                self.logger.warning(
                    f"Context '{context}': Generation/Verification failed (Attempt {attempt}/{max_improvement_iterations}): {type(e).__name__}: {e}"
                )
                last_error_str = f"{type(e).__name__}: {e}"  # Keep string for logging
                last_error_obj = e  # Store the actual exception object

                if attempt < max_improvement_iterations:
                    try:
                        if not self.prompt_inputs:
                            self._prepare_prompt_inputs()
                        self.prompt_inputs.response_to_fix = (
                            response_to_fix if response_to_fix else "No response received."
                        )
                        self.prompt_inputs.errors_to_fix = str(e)
                        fix_prompt = fix_prompt_builder(self.prompt_inputs)
                        current_prompt = fix_prompt
                        self.logger.info(
                            f"Context '{context}': Generated fix prompt for next attempt."
                        )
                    except Exception as builder_error:
                        raise LLMOrchestrationError(
                            f"Context '{context}': Failed to build fix prompt: {builder_error}"
                        ) from builder_error
                # If last attempt, loop exits and raises MaxIterationsExceededError below

            except LLMClientError as e:
                self.logger.error(
                    f"Context '{context}': Non-recoverable LLM client error during generation: {e}",
                    exc_info=True,
                )
                raise  # Propagate immediately

            except Exception as e:
                self.logger.error(
                    f"Context '{context}': Unexpected error during generation/verification attempt {attempt}: {e}",
                    exc_info=True,
                )
                last_error_str = f"Unexpected error: {e}"  # Keep string for logging
                last_error_obj = e  # Store the actual exception object
                break  # Exit loop immediately on unexpected errors to raise MaxIterationsExceededError below

        # === Loop finished without success ===
        final_error_message = (
            f"Context '{context}': Failed to produce a valid result after {max_improvement_iterations} attempts. "
            f"Last error: {last_error_str}"  # Log the string representation
        )
        self.logger.error(final_error_message)

        # Raise MaxIterationsExceededError, chaining the *actual* last specific error encountered
        if last_error_obj:
            raise MaxIterationsExceededError(final_error_message) from last_error_obj
        else:
            # This case should be rare (e.g., max_improvement_iterations <= 0)
            # but raise the generic error if no specific exception was captured.
            raise MaxIterationsExceededError(final_error_message)
