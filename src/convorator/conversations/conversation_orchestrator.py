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

from typing import Dict, List, Optional, Union, Any


# Import exceptions from the new central location
from convorator.exceptions import (
    LLMOrchestrationError,
    LLMResponseError,
    MaxIterationsExceededError,
    SchemaValidationError,
    LLMClientError,
)
from convorator.client.llm_client import Conversation, LLMInterface

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
)

# Added for EnhancedMessageMetadata and related types
from datetime import datetime, timezone
import uuid
from .events import (
    EnhancedMessageMetadata,
    EventType,
    OrchestrationStage,
    MessageEntityType,
    MessagePayloadType,
)


def improve_solution_with_moderation(
    config: OrchestratorConfig,  # Changed type hint
    initial_solution: Optional[Union[Dict[str, object], str]] = None,
    requirements: Optional[str] = None,
    assessment_criteria: Optional[str] = None,
) -> Union[Dict[str, object], str]:
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
        initial_solution: Optional[Union[Dict[str, object], str]] = None,
        requirements: Optional[str] = None,
        assessment_criteria: Optional[str] = None,
    ):  # Changed type hint
        """
        Initializes the orchestrator with the provided configuration and task specifics.

        Args:
            config: The OrchestratorConfig object containing general settings.
            initial_solution: The starting solution (string or dictionary).
            requirements: The requirements the solution must meet.
            assessment_criteria: Criteria for evaluating the solution during debate.
        """
        self.config = config

        # Directly access fields from OrchestratorConfig
        self.llm_group = config.llm_group
        self.logger = config.logger
        self.messaging_callback = config.messaging_callback
        self.debate_iterations = config.debate_iterations
        self.improvement_iterations = config.improvement_iterations

        # Get moderator instructions and context
        self.moderator_instructions = config.moderator_instructions_override
        self.debate_context = config.debate_context_override

        # Access individual prompt builder functions directly from config
        self.prompt_builder = config.prompt_builder

        # Store task-specific inputs directly
        # Prioritize direct __init__ arguments, then fall back to config
        self.initial_solution = (
            initial_solution if initial_solution is not None else config.initial_solution
        )
        self.requirements = requirements if requirements is not None else config.requirements
        self.assessment_criteria = (
            assessment_criteria if assessment_criteria is not None else config.assessment_criteria
        )

        # Validation: Ensure critical parameters are now set from either source
        # Assuming initial_solution is always needed for this orchestrator. Adjust if it can be truly optional.
        if self.initial_solution is None:
            raise ValueError(
                "initial_solution must be provided either directly to SolutionImprovementOrchestrator "
                "or in OrchestratorConfig."
            )
        if not self.requirements:  # Catches None or empty string
            raise ValueError(
                "requirements must be provided either directly to SolutionImprovementOrchestrator "
                "or in OrchestratorConfig, and cannot be empty."
            )
        if not self.assessment_criteria:  # Catches None or empty string
            raise ValueError(
                "assessment_criteria must be provided either directly to SolutionImprovementOrchestrator "
                "or in OrchestratorConfig, and cannot be empty."
            )

        # Extract LLMs
        self.primary_llm = self.llm_group.primary_llm
        self.debater_llm = self.llm_group.debater_llm
        self.moderator_llm = self.llm_group.moderator_llm
        self.solution_generation_llm = self.llm_group.solution_generation_llm

        # Role names
        self.primary_name = self.primary_llm.get_role_name() or "Primary"
        self.debater_name = self.debater_llm.get_role_name() or "Debater"
        self.moderator_name = self.moderator_llm.get_role_name() or "Moderator"

        # LLM dict
        self.llm_dict = {
            self.primary_name: self.primary_llm,
            self.debater_name: self.debater_llm,
            self.moderator_name: self.moderator_llm,
        }

        # Initialize state variables
        self._reset_conversations()
        self.current_iteration_num = 0
        self.prompt_inputs: Optional[PromptBuilderInputs] = None

    def _reset_conversations(self) -> None:
        self.primary_conv = Conversation(system_message=self.primary_llm.get_system_message())
        self.debater_conv = Conversation(system_message=self.debater_llm.get_system_message())
        self.moderator_conv = Conversation(system_message=self.moderator_llm.get_system_message())
        self.main_conversation_log = MultiAgentConversation()

        self.role_conversation = {
            self.primary_name: self.primary_conv,
            self.debater_name: self.debater_conv,
            self.moderator_name: self.moderator_conv,
        }

    def run(self) -> Union[Dict[str, object], str]:
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
            # 1. Prepare Prompt Inputs
            self._prepare_prompt_inputs()

            # 2. Run Moderated Debate
            self._run_moderated_debate()

            # 3. Synthesize Summary
            self._synthesize_summary()

            # 4. Generate Final Solution
            improved_solution = self._generate_final_solution()

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
            debate_iterations=self.debate_iterations,
            current_iteration_num=self.current_iteration_num,
        )
        # Update prompt inputs
        self.prompt_builder.update_inputs(self.prompt_inputs)
        self.logger.debug("Prompt inputs prepared.")

    def _run_moderated_debate(self):
        """
        Orchestrates the moderated debate phase.

        Args:
            initial_prompt_content: The core content for the debate's start.

        Returns:
            The complete debate history log.
        """
        self.logger.info(f"Conducting moderated debate for {self.debate_iterations} iterations.")
        if not self.prompt_builder.is_inputs_defined():
            raise LLMOrchestrationError("Prompt inputs not prepared before running debate.")

        # --- Initial User Prompt Setup ---
        initial_prompt = self.prompt_builder.build_prompt("initial_prompt")

        self.prompt_inputs.initial_prompt_content = initial_prompt

        self.logger.info(
            f"Starting moderated conversation. Max iterations: {self.debate_iterations}"
        )

        # --- Debate Execution ---
        try:
            _ = self._query_agent_in_debate_turn(
                self.debater_llm,
                self.debater_name,
                initial_prompt,
            )

            # --- Main Debate Rounds Loop ---
            for i in range(self.debate_iterations):
                round_num = i + 1
                self.prompt_inputs.current_iteration_num = round_num
                self.logger.info(f"Starting Round {round_num}/{self.debate_iterations}.")

                # Primary Turn
                _ = self._execute_agent_loop_turn(
                    self.primary_name,
                )

                # Moderator Turn
                _ = self._execute_agent_loop_turn(
                    self.moderator_name,
                )

                # End if Last Iteration
                if i == self.debate_iterations - 1:
                    self.logger.info(
                        "Max iterations reached. Ending debate after moderator feedback."
                    )
                    break

                # Debater Turn (with Feedback)
                _ = self._execute_agent_loop_turn(
                    self.debater_name,
                )

            self.logger.info("Moderated debate phase completed.")

        except (LLMResponseError, LLMOrchestrationError) as e:
            self.logger.error(f"Error during moderated debate: {e}", exc_info=True)
            raise  # Re-raise specific errors
        except Exception as e:
            self.logger.error(f"Unexpected error during moderated debate: {e}", exc_info=True)
            raise LLMOrchestrationError(f"Unexpected error during moderated debate: {e}") from e

    def _execute_agent_loop_turn(self, agent_name: str) -> str:
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
        if agent_name == self.primary_name:
            build_prompt = "primary_prompt"
        elif agent_name == self.debater_name:
            build_prompt = "debater_prompt"
        elif agent_name == self.moderator_name:
            build_prompt = "moderator_context"

        response = self._query_agent_in_debate_turn(
            self.llm_dict[agent_name],
            agent_name,
            self.prompt_builder.build_prompt(build_prompt),
        )
        return response

    def _synthesize_summary(self):
        """Generates a summary from the moderator using the debate history."""
        self.logger.info("Step 3: Synthesizing moderation summary.")
        if not self.prompt_inputs:
            raise LLMOrchestrationError("Prompt inputs not prepared before synthesizing summary.")

        self.logger.debug("Querying moderator LLM for summary.")
        try:
            summary_prompt = self.prompt_builder.build_prompt("summary_prompt")
            summary = self._query_agent_in_debate_turn(
                self.moderator_llm,
                self.moderator_name,
                summary_prompt,
            )
            self.prompt_inputs.moderator_summary = summary
            self.logger.debug(f"Moderation summary received: {summary[:200]}...")
        except LLMResponseError as e:
            self.logger.error(f"Moderator LLM failed to generate summary: {e}", exc_info=True)
            raise  # Re-raise specific LLM errors
        except Exception as e:
            self.logger.error(
                f"Unexpected error querying moderator for summary: {e}", exc_info=True
            )
            raise LLMOrchestrationError(f"Moderator LLM failed to generate summary: {e}") from e

    def _generate_final_solution(self) -> Union[Dict[str, object], str]:
        """Generates and verifies the final improved solution."""
        self.logger.info(
            f"Step 4: Generating improved solution (max {self.improvement_iterations} attempts)."
        )
        if not self.prompt_inputs:
            raise LLMOrchestrationError(
                "Prompt inputs not prepared before generating final solution."
            )

        try:
            # Use the generalized generate_and_verify method
            improved_solution = self._generate_and_verify_result(
                llm_service=self.solution_generation_llm,
                context="Final Solution Generation",
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

    # --- Helper Methods ---

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

    def _query_llm_core(
        self,
        llm_service: LLMInterface,
        messages_to_send_raw: List[Dict[str, str]],
        current_prompt_text_for_callback: str,
        stage: OrchestrationStage,
        step_description: str,
        iteration_num: Optional[int],
        prompt_source_entity_type: MessageEntityType,
        prompt_source_entity_name: Optional[str],
        prompt_metadata_data: Optional[Any] = None,
        query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Core private method to query an LLM, including history preparation and messaging callbacks.

        This method encapsulates the fundamental sequence of operations for an LLM call:
        1. Prepares the message history using `_prepare_history_for_llm` (handles token limits).
        2. Invokes the `messaging_callback` with detailed metadata before sending the prompt.
        3. Executes the `llm_service.query()` call.
        4. Invokes the `messaging_callback` with detailed metadata after receiving the response.
        5. Returns the raw response content from the LLM.

        This method does NOT:
        - Manage `Conversation` objects directly (e.g., adding messages to them).
        - Update the main conversation log (`self.main_conversation_log`) or agent-specific logs (`self.role_conversation`).
        - Implement retry logic (errors from LLM or history preparation are propagated).

        Args:
            llm_service: The LLMInterface instance to query.
            messages_to_send_raw: The complete list of messages (system, user, assistant roles)
                                  to be potentially sent to the LLM, prior to truncation.
            current_prompt_text_for_callback: The specific text of the current prompt message being sent.
                                            This is used for the 'content' field in the prompt callback.
            stage: The high-level operational stage (e.g., DEBATE_TURN) for metadata.
            step_description: A detailed description of the current step for metadata (e.g., "Primary Agent Round 1").
            iteration_num: Optional iteration number (e.g., debate round, improvement attempt) for metadata.
            prompt_source_entity_type: The type of entity initiating the prompt (for prompt metadata).
            prompt_source_entity_name: The name of the entity initiating the prompt (for prompt metadata).
            prompt_metadata_data: Optional additional structured data for the prompt's metadata 'data' field.
            query_kwargs: Optional additional arguments to pass to `llm_service.query()`.

        Returns:
            The string content of the LLM's response.

        Raises:
            LLMOrchestrationError: If `_prepare_history_for_llm` fails (e.g., token limit issues).
            LLMResponseError: If `llm_service.query()` fails due to issues with the LLM's response.
            LLMClientError: If `llm_service.query()` fails due to client-side issues (e.g., network, auth).
        """
        self.logger.debug(
            f"_query_llm_core: Stage='{stage.value}', Step='{step_description}', LLM='{llm_service.get_role_name() or 'UnnamedLLM'}'"
        )

        # 1. Prepare history for the LLM (handles token limits)
        # The context for _prepare_history_for_llm should be descriptive
        prepare_history_context = f"{stage.value} - {step_description} - History Prep"
        prepared_messages = self._prepare_history_for_llm(
            llm_service, messages_to_send_raw, context=prepare_history_context
        )

        # 2. Messaging callback for the prompt being sent
        if self.messaging_callback:
            prompt_meta = EnhancedMessageMetadata(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=self.config.session_id if hasattr(self.config, "session_id") else None,
                stage=stage,
                step_description=f"{step_description} - Prompt",
                iteration_num=iteration_num,
                source_entity_type=prompt_source_entity_type,
                source_entity_name=prompt_source_entity_name,
                target_entity_type=MessageEntityType.LLM_AGENT,
                target_entity_name=llm_service.get_role_name() or "TargetLLM",
                llm_service_details={"model_name": getattr(llm_service, "model_name", "N/A")},
                payload_type=MessagePayloadType.TEXT_CONTENT,
                data=prompt_metadata_data,
            )
            try:
                self.messaging_callback(
                    EventType.PROMPT, current_prompt_text_for_callback, prompt_meta
                )
            except Exception as cb_ex:
                self.logger.error(f"Messaging callback for prompt failed: {cb_ex}", exc_info=True)
                # Decide if this should be fatal or just logged. For now, log and continue.

        # 3. Query the LLM
        try:
            response_content = llm_service.query(
                prompt="",  # Prompt is part of prepared_messages
                use_conversation=False,  # Explicit history management
                conversation_history=prepared_messages,
                **(query_kwargs or {}),
            )
            self.logger.debug(
                f"_query_llm_core: LLM response received. Length: {len(response_content)}"
            )

            # 4. Messaging callback for the response received
            if self.messaging_callback:
                response_meta = EnhancedMessageMetadata(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    session_id=(
                        self.config.session_id if hasattr(self.config, "session_id") else None
                    ),
                    stage=stage,
                    step_description=f"{step_description} - Response",
                    iteration_num=iteration_num,
                    source_entity_type=MessageEntityType.LLM_AGENT,
                    source_entity_name=llm_service.get_role_name() or "SourceLLM",
                    target_entity_type=MessageEntityType.ORCHESTRATOR_INTERNAL,
                    target_entity_name="OrchestratorLogic",
                    llm_service_details={"model_name": getattr(llm_service, "model_name", "N/A")},
                    payload_type=MessagePayloadType.TEXT_CONTENT,
                    data=None,  # Or e.g. token usage stats if llm_service could provide them
                )
                try:
                    self.messaging_callback(EventType.RESPONSE, response_content, response_meta)
                except Exception as cb_ex:
                    self.logger.error(
                        f"Messaging callback for response failed: {cb_ex}", exc_info=True
                    )
            return response_content

        except (LLMResponseError, LLMClientError) as e:
            self.logger.error(
                f"_query_llm_core: LLM query failed. Stage='{stage.value}', Step='{step_description}'. Error: {e}",
                exc_info=True,
            )
            if self.messaging_callback:
                error_meta = EnhancedMessageMetadata(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    session_id=(
                        self.config.session_id if hasattr(self.config, "session_id") else None
                    ),
                    stage=stage,
                    step_description=f"{step_description} - LLM Query Error",
                    iteration_num=iteration_num,
                    source_entity_type=MessageEntityType.LLM_AGENT,  # The LLM agent is the source of the error event
                    source_entity_name=llm_service.get_role_name() or "ErroredLLM",
                    target_entity_type=MessageEntityType.ORCHESTRATOR_INTERNAL,  # Error is reported to the orchestrator logic
                    target_entity_name="OrchestratorLogic",
                    llm_service_details={"model_name": getattr(llm_service, "model_name", "N/A")},
                    payload_type=MessagePayloadType.ERROR_DETAILS_STR,  # Content will be str(e)
                    data={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },  # Store actual exception details in data
                )
                try:
                    # Assuming EventType.LLM_ERROR or a similar enum member exists
                    self.messaging_callback(EventType.LLM_ERROR, str(e), error_meta)
                except Exception as cb_ex:
                    self.logger.error(
                        f"Messaging callback for LLM error event failed: {cb_ex}", exc_info=True
                    )
            raise  # Re-raise the original exception to be handled by the caller

    def _query_agent_in_debate_turn(
        self,
        llm_to_query: LLMInterface,
        queried_role_name: str,
        prompt: str,
        query_kwargs: Optional[Dict[str, Any]] = None,
        add_prompt_to_main_log: bool = True,
    ) -> str:
        """
        Queries an LLM agent, updates its perspective history, the main log,
        and optionally other agents' histories. This version delegates the core LLM
        interaction to _query_llm_core for better code reuse and standardized handling.

        Args:
            llm_to_query: The LLMInterface instance to query.
            queried_role_name: The specific role name for logging.
            prompt: The prompt to send to the LLM.
            query_kwargs: Optional additional arguments for the LLM query.
            add_prompt_to_main_log: Whether to add the prompt to the main log.

        Returns:
            The content of the LLM response.

        Raises:
            LLMOrchestrationError: For validation issues or token counting failures.
            LLMResponseError: If the LLM query fails.
            LLMClientError: For client-side errors during LLM interaction.
        """
        try:
            self.logger.info(f"Try to query {queried_role_name}...")

            # 1. Validations - these remain the same as in _query_agent
            conversation_history = self.role_conversation[queried_role_name]
            if not isinstance(conversation_history, (Conversation, MultiAgentConversation)):
                raise LLMOrchestrationError(
                    f"{queried_role_name}'s conversation history must be a Conversation or MultiAgentConversation"
                )

            if prompt == "":
                raise LLMOrchestrationError(f"Prompt cannot be empty.")

            messages = conversation_history.get_messages()
            if messages and messages[-1].get("role") == "user":
                raise LLMOrchestrationError(
                    f"{queried_role_name}'s last message must be an assistant message. Since prompt is going to be the user message"
                )

            # 2. Add the prompt to the conversation histories
            conversation_history.add_user_message(prompt)
            if add_prompt_to_main_log:
                self.main_conversation_log.add_message(role="user", content=prompt)

            # 3. Query the LLM using _query_llm_core
            response_content = self._query_llm_core(
                llm_service=llm_to_query,
                messages_to_send_raw=conversation_history.get_messages(),
                current_prompt_text_for_callback=prompt,
                stage=OrchestrationStage.DEBATE_TURN,
                step_description=f"{queried_role_name} Agent Response",
                iteration_num=self.prompt_inputs.current_iteration_num,
                prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,
                prompt_source_entity_name="user",
                prompt_metadata_data=None,
                query_kwargs=query_kwargs,
            )
            self.logger.info(f"{queried_role_name} responded successfully.")
            self.logger.debug(f"{queried_role_name} response snippet: {response_content[:100]}...")

            # 4. Add response to the agent's own perspective history
            conversation_history.add_assistant_message(response_content)
            self.logger.debug(f"Added response to {queried_role_name}'s perspective history.")

            # 5. Add response to the main multi-agent log
            self.main_conversation_log.add_message(role=queried_role_name, content=response_content)
            self.logger.debug(f"Added {queried_role_name}'s response to the main conversation log.")

            return response_content

        except LLMResponseError as e:
            self.logger.error(f"{queried_role_name} query failed: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.exception(
                f"Unexpected error querying {queried_role_name} or updating logs: {e}"
            )
            raise LLMOrchestrationError(
                f"Unexpected error during {queried_role_name} query/log update: {e}"
            ) from e

    def _query_for_iterative_generation(
        self,
        llm_service: LLMInterface,
        current_prompt_content: str,
        generation_conversation_obj: Optional[Conversation],
        base_context_for_logging: str,
        attempt_num: int,
        stage: OrchestrationStage,  # e.g., SOLUTION_GENERATION_INITIAL_ATTEMPT or SOLUTION_GENERATION_FIX_ATTEMPT
        prompt_metadata_for_llm_core: Optional[
            Any
        ] = None,  # Additional data for the prompt callback
        query_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Handles LLM querying specifically for the iterative generation loop
        in _generate_and_verify_result, using _query_llm_core.

        This method prepares messages for stateless or stateful (via generation_conversation_obj)
        calls and invokes _query_llm_core with appropriate metadata for the generation/fix stage.

        Args:
            llm_service: The LLMInterface to use for the query.
            current_prompt_content: The actual prompt string for the current attempt.
            generation_conversation_obj: Optional Conversation object for maintaining state across attempts.
                                       If None, the call is treated as stateless.
            base_context_for_logging: The base context string for logging (e.g., "Final Solution Generation").
            attempt_num: The current attempt number in the iterative loop.
            stage: The specific OrchestrationStage for this attempt (e.g., INITIAL_ATTEMPT, FIX_ATTEMPT).
            prompt_metadata_for_llm_core: Optional additional structured data to be included in the
                                          'data' field of the prompt's EnhancedMessageMetadata via _query_llm_core.
            query_kwargs: Optional dictionary of keyword arguments for the LLM's query method.

        Returns:
            The raw string content of the LLM's response.

        Raises:
            Propagates exceptions from _query_llm_core (e.g., LLMResponseError, LLMClientError, LLMOrchestrationError).
        """
        self.logger.debug(
            f"_query_for_iterative_generation: BaseContext='{base_context_for_logging}', Attempt={attempt_num}, Stage='{stage.value}"
        )

        messages_to_send_raw: List[Dict[str, str]]

        if generation_conversation_obj:
            # Stateful: use the provided conversation object
            generation_conversation_obj.add_user_message(current_prompt_content)
            messages_to_send_raw = generation_conversation_obj.get_messages()
        else:
            # Stateless: construct messages from scratch
            messages_to_send_raw = []
            system_msg_content = llm_service.get_system_message()
            if system_msg_content:
                messages_to_send_raw.append({"role": "system", "content": system_msg_content})
            messages_to_send_raw.append({"role": "user", "content": current_prompt_content})

        response_content = self._query_llm_core(
            llm_service=llm_service,
            messages_to_send_raw=messages_to_send_raw,
            current_prompt_text_for_callback=current_prompt_content,
            stage=stage,
            step_description=f"{base_context_for_logging} - Iteration Attempt {attempt_num}",
            iteration_num=attempt_num,
            prompt_source_entity_type=MessageEntityType.USER_PROMPT_SOURCE,  # Conceptual user need driving generation
            prompt_source_entity_name="user",
            prompt_metadata_data=prompt_metadata_for_llm_core,
            query_kwargs=query_kwargs,
        )

        if generation_conversation_obj:
            generation_conversation_obj.add_assistant_message(response_content)
            self.logger.debug(
                f"Added iterative generation response to its dedicated conversation. Context: {base_context_for_logging}, Attempt: {attempt_num}"
            )

        return response_content

    def _generate_and_verify_result(
        self,
        llm_service: LLMInterface,
        context: str,
        result_schema: Optional[Dict[str, object]] = None,
        use_conversation: bool = True,
        max_improvement_iterations: int = 3,
        json_result: bool = True,
    ) -> Union[Dict[str, object], str]:
        """
        Generates a result using an LLM, verifies it (parsing, schema validation),
        and attempts to fix errors iteratively up to a maximum number of attempts.
        This version leverages other internal methods for history preparation and prompt building.

        Args:
            llm_service: LLM interface to use.
            context: Logging context string (e.g., "Final Solution Generation").
            result_schema: Optional JSON schema for validation if json_result is True.
            use_conversation: If True, maintains a separate Conversation object for this
                              generation task, allowing the LLM to build on its previous
                              attempts within this specific generation loop. If False,
                              each attempt is more stateless from the LLM's perspective,
                              though error context is still provided in the prompt.
            max_improvement_iterations: Max attempts to generate and fix errors.
            json_result: Whether the expected result is JSON.

        Returns:
            The verified result (Dict if JSON, otherwise str).

        Raises:
            MaxIterationsExceededError: If max iterations are reached without success.
            SchemaValidationError: If JSON validation fails after retries (and json_result is True).
            LLMResponseError: If an LLM query fails.
            LLMOrchestrationError: For other orchestration issues (e.g., prompt building failure, token limit issues).
        """
        self.logger.info(
            f"Generating result for context: '{context}'. Max iterations: {max_improvement_iterations}. Expect JSON: {json_result}."
        )
        if not self.prompt_inputs:
            # This should ideally be prepared before calling this function,
            # but as a fallback, ensure it's populated.
            self.logger.warning(
                f"Context '{context}': PromptBuilderInputs was not prepared. Preparing now."
            )
            self._prepare_prompt_inputs()
            if not self.prompt_inputs:  # Should not happen if _prepare_prompt_inputs is correct
                raise LLMOrchestrationError("Critical error: Prompt inputs could not be prepared.")

        last_error_str: str = "No error encountered during generation."
        last_error_obj: Optional[Exception] = None

        # Initialize a dedicated conversation for this generation task if use_conversation is True
        generation_conversation: Optional[Conversation] = None
        if use_conversation:
            generation_conversation = Conversation(
                system_message=llm_service.get_system_message()  # Use system message from the specific LLM service
            )

        # Initial prompt for the first attempt
        current_prompt_content = self.prompt_builder.build_prompt("improvement_prompt")

        for attempt in range(1, max_improvement_iterations + 1):
            self.logger.info(
                f"Context '{context}': Generating result (Attempt {attempt}/{max_improvement_iterations})"
            )
            raw_response_for_fix_prompt: Optional[str] = (
                None  # Store the raw response for the fix prompt
            )
            current_stage: OrchestrationStage.SOLUTION_GENERATION_INITIAL_ATTEMPT
            prompt_data_for_callback: Optional[Dict[str, Any]] = None

            if attempt > 1:
                current_stage = OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT
                # For fix attempts, we can pass the error and previous response in the metadata for the prompt callback
                if last_error_obj:
                    prompt_data_for_callback = {
                        "error_type": type(last_error_obj).__name__,
                        "error_message": str(last_error_obj),
                        "previous_response_to_fix": (
                            self.prompt_inputs.response_to_fix if self.prompt_inputs else "N/A"
                        ),
                    }

            try:
                response_content = self._query_for_iterative_generation(
                    llm_service=llm_service,
                    current_prompt_content=current_prompt_content,
                    generation_conversation_obj=(
                        generation_conversation if use_conversation else None
                    ),
                    base_context_for_logging=context,
                    attempt_num=attempt,
                    stage=current_stage,
                    prompt_metadata_for_llm_core=prompt_data_for_callback,  # Pass error details for fix prompts
                )
                raw_response_for_fix_prompt = response_content

                # --- Verification ---
                if json_result:
                    result = parse_json_response(
                        self.logger, response_content, context, schema=result_schema
                    )
                    self.logger.info(
                        f"Context '{context}': Generation and JSON validation successful (Attempt {attempt})."
                    )
                    return result
                else:
                    self.logger.info(
                        f"Context '{context}': Generation successful (non-JSON) (Attempt {attempt})."
                    )
                    return response_content

            except (LLMResponseError, SchemaValidationError, LLMOrchestrationError) as e:
                # LLMOrchestrationError can also be raised by _query_for_iterative_generation (from _prepare_history_for_llm)
                self.logger.warning(
                    f"Context '{context}': Attempt {attempt}/{max_improvement_iterations} failed. Error: {type(e).__name__}: {e}"
                )
                last_error_str = f"{type(e).__name__}: {e}"
                last_error_obj = e

                # If it's not the last attempt, build a fix prompt
                if attempt < max_improvement_iterations:
                    try:
                        if not self.prompt_inputs:
                            self._prepare_prompt_inputs()
                            if not self.prompt_inputs:
                                raise LLMOrchestrationError(
                                    "Critical: Prompt inputs disappeared during fix prompt generation."
                                )

                        self.prompt_inputs.response_to_fix = (
                            raw_response_for_fix_prompt
                            if raw_response_for_fix_prompt
                            else "No response content received from LLM."
                        )
                        self.prompt_inputs.errors_to_fix = str(e)

                        current_prompt_content = self.prompt_builder.build_prompt("fix_prompt")
                        self.logger.info(
                            f"Context '{context}': Generated fix prompt for next attempt."
                        )
                    except Exception as builder_error:
                        self.logger.error(
                            f"Context '{context}': Failed to build fix prompt: {builder_error}",
                            exc_info=True,
                        )
                        raise LLMOrchestrationError(
                            f"Context '{context}': Failed to build fix prompt after error: {builder_error}"
                        ) from builder_error
                # If it's the last attempt, the loop will exit and MaxIterationsExceededError will be raised after the loop

            except LLMClientError as e:
                self.logger.error(
                    f"Context '{context}': Non-recoverable LLM client error during generation (Attempt {attempt}): {e}",
                    exc_info=True,
                )
                # last_error_obj and last_error_str are already set if the error came from _query_for_iterative_generation
                # If it happened before that call somehow, set them here.
                if not last_error_obj:
                    last_error_str = f"{type(e).__name__}: {e}"
                    last_error_obj = e
                raise  # Propagate immediately, MaxIterationsExceededError will be raised by the caller if appropriate

            except Exception as e:
                self.logger.error(
                    f"Context '{context}': Unexpected error during generation/verification (Attempt {attempt}): {e}",
                    exc_info=True,
                )
                last_error_str = f"Unexpected error: {e}"
                last_error_obj = e
                break

        # --- Loop finished without a successful return ---
        final_error_message = (
            f"Context '{context}': Failed to produce a valid result after {max_improvement_iterations} attempts. "
            f"Last error: {last_error_str}"
        )
        self.logger.error(final_error_message)

        if last_error_obj:
            raise MaxIterationsExceededError(final_error_message) from last_error_obj
        else:
            # This case implies max_improvement_iterations might be 0 or negative, or an issue before first attempt.
            raise MaxIterationsExceededError(final_error_message)
