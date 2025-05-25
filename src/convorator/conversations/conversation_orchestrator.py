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

1. Initial Setup: Configuration, loading LLMs, preparing initial inputs via `__init__` and `_prepare_prompt_inputs`.
2. Moderated Debate: Multi-turn discussion between the Primary and Debater agents,
   with the Moderator providing guidance and evaluation based on assessment criteria. Managed by `_run_moderated_debate`.
3. Summary Synthesis: The Moderator consolidates insights and key feedback from the debate, managed by `_synthesize_summary`.
4. Solution Generation: The Solution Generation Agent creates an improved solution incorporating the debate summary. Managed by `_generate_final_solution` which utilizes `_generate_and_verify_result`.
5. Verification & Correction: The generated solution is validated against schema requirements (if any).
   If validation fails, the orchestrator attempts correction by feeding errors back to the Solution Generation Agent within `_generate_and_verify_result`.

Key Features:
------------
1. Stateful Conversation Management:
   - Maintains separate `Conversation` contexts for each agent (Primary, Debater, Moderator) during the debate.
   - Coordinates turns and ensures proper information flow between agents during the debate.
   - Tracks the overall conversation history in a `MultiAgentConversation` log for context and logging.

2. Robust Error Handling & Verification:
   - Schema validation for JSON outputs using `parse_json_response`, with descriptive error messages.
   - Automatic retry loop within `_generate_and_verify_result` for solution generation, providing intelligent error correction feedback to the LLM.
   - Token limit awareness and automatic history truncation via `_prepare_history_for_llm` to prevent API errors.

3. Configurable Orchestration:
   - Customizable prompt generation logic via a pluggable `PromptBuilder` system.
   - Adjustable iteration counts for both the debate (`debate_iterations`) and the solution improvement/correction loop (`improvement_iterations`).
   - Flexible LLM assignment to different roles through `OrchestratorConfig` and `SolutionLLMGroup`.

4. Schema-Driven Solution Refinement:
   - Ensures final output adheres to a specified JSON schema (if `config.solution_schema` is provided).
   - Focuses the improvement process on meeting structural and content requirements for JSON outputs.

5. Event-Driven Messaging Callback System:
   - Provides detailed event notifications via a `messaging_callback` function (if configured).
   - Uses `EnhancedMessageMetadata` and specific `EventType`, `OrchestrationStage`, `MessageEntityType`,
     and `MessagePayloadType` enums to describe events like prompts, responses, and errors.
   - Facilitates observability, logging, and integration with external monitoring or data collection systems.

Implementation Details:
---------------------
The core of the system is the `SolutionImprovementOrchestrator` class, which:

1. Manages the multi-step workflow through its `run()` method. This involves calling:
   - `_prepare_prompt_inputs()` to set up data for prompt builders.
   - `_run_moderated_debate()` to coordinate the debate phase.
   - `_synthesize_summary()` to have the Moderator produce a debate summary.
   - `_generate_final_solution()` to produce the final output.
2. The debate phase (`_run_moderated_debate`) orchestrates turns:
   - Agent turns within the debate loop are managed by `_execute_agent_loop_turn`.
   - `_execute_agent_loop_turn` builds the appropriate prompt using `PromptBuilder` and then calls `_query_agent_in_debate_turn`.
   - `_query_agent_in_debate_turn` handles adding messages to the respective agent's conversation log and the main log, then delegates to `_query_llm_core`.
   - `_query_llm_core` is the central private method for LLM interaction. It prepares history (including truncation via `_prepare_history_for_llm`),
     invokes messaging callbacks, and executes the LLM query.
3. The final solution generation (`_generate_final_solution`) calls `_generate_and_verify_result`.
   - `_generate_and_verify_result` iteratively prompts the Solution Generation LLM, validates the output (parsing, schema checks),
     and retries with error feedback if necessary, using `_query_for_iterative_generation` (which also calls `_query_llm_core`).

The orchestrator maintains separate `Conversation` states for each debating agent, allowing them distinct
perspectives, while `MultiAgentConversation` tracks the unified conversation flow. Error recovery in
`_generate_and_verify_result()` is crucial, providing specific error details back to the
Solution Generation Agent to guide correction attempts.

Usage:
-----
The primary entry point is the `improve_solution_with_moderation()` function, which accepts:
- A configuration object (`OrchestratorConfig`) detailing LLMs, iterations, topic, initial solution, requirements, assessment criteria, schema, callback, etc.
- Optionally, `initial_solution`, `requirements`, and `assessment_criteria` can be passed directly to override or supplement values in the `OrchestratorConfig`.

Example:
```python
from convorator.conversations.conversation_orchestrator import improve_solution_with_moderation
from convorator.conversations.configurations import OrchestratorConfig, SolutionLLMGroup
from convorator.client.llm_client import LLMInterface # Assuming LLMInterface is the base/actual class
from convorator.conversations.prompts import PromptBuilder # If custom prompt builders are needed

# 1. Configure LLMs for each role (replace ... with actual LLM client instantiations)
primary_llm = LLMInterface(...)
debater_llm = LLMInterface(...)
moderator_llm = LLMInterface(...)
solution_llm = LLMInterface(...)

# 2. Group LLMs
llm_group = SolutionLLMGroup(
    primary_llm=primary_llm,
    debater_llm=debater_llm,
    moderator_llm=moderator_llm,
    solution_generation_llm=solution_llm
)

# 3. Define the task inputs
topic = "Improve Product Description for a New Smartwatch"
initial_solution_dict = {'product_id': 'SW-001', 'name': 'Chronos', 'status': 'beta', 'description': 'A watch.'}
requirements_str = "Final JSON: product_id (str), name (str), status ('active'|'discontinued'), description (str, <50 words, engaging), features (list of str)."
assess_criteria_str = "Clarity of description, accuracy of status, completeness of features, adherence to requirements."
json_schema = {
    "type": "object",
    "properties": {
        "product_id": {"type": "string"},
        "name": {"type": "string"},
        "status": {"type": "string", "enum": ["active", "discontinued"]},
        "description": {"type": "string", "maxLength": 250}, # Example: longer than 50 words for demo
        "features": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["product_id", "name", "status", "description", "features"]
}

# 4. Create Orchestrator Configuration
# (PromptBuilder can be customized and passed if defaults are not sufficient)
config = OrchestratorConfig(
    llm_group=llm_group,
    topic=topic,
    initial_solution=initial_solution_dict,
    requirements=requirements_str,
    assessment_criteria=assess_criteria_str,
    solution_schema=json_schema, # Ensure output is validated JSON
    expect_json_output=True,     # Critical if schema is provided
    debate_iterations=2,
    improvement_iterations=2,
    # messaging_callback=my_custom_callback_function, # Optional
    # prompt_builder=custom_prompt_builder_instance, # Optional
)

# 5. Run the improvement process
try:
    improved_solution = improve_solution_with_moderation(
        config=config
        # Direct overrides for initial_solution, requirements, assessment_criteria are also possible here
    )
    print("Successfully generated improved solution:", improved_solution)
except Exception as e:
    print(f"Orchestration failed: {e}")

```

Dependencies:
------------
Core:
- `typing`: For type hinting.
- `datetime`, `uuid` (from standard library): Used by the event messaging system.

Internal `convorator` modules:
- `convorator.exceptions`: For specialized exception types (`LLMOrchestrationError`, etc.).
- `convorator.client.llm_client`: For LLM communication (`LLMInterface`, `Conversation`).
- `convorator.conversations.configurations`: For orchestration configuration (`OrchestratorConfig`, `SolutionLLMGroup`).
- `convorator.conversations.state`: For multi-agent conversation state (`MultiAgentConversation`).
- `convorator.conversations.prompts`: For prompt construction (`PromptBuilderInputs`, `PromptBuilder`).
- `convorator.conversations.utils`: For utility functions (e.g., `parse_json_response`).
- `convorator.conversations.events`: For the event messaging system (`EnhancedMessageMetadata`, enums).
- `convorator.utils.logger`: For structured logging.

Third-party:
- `jsonschema`: For JSON schema validation (used within `parse_json_response`).
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
        self.current_iteration_num = 0  # This will now use the property setter
        self.prompt_inputs: Optional[PromptBuilderInputs] = None

    # Class level annotation for the instance variable that backs the property
    _current_iteration_num: int

    @property
    def current_iteration_num(self) -> int:
        """The current iteration number of a loop (e.g., debate round)."""
        return self._current_iteration_num

    @current_iteration_num.setter
    def current_iteration_num(self, value: int) -> None:
        self._current_iteration_num = value
        if hasattr(self, "prompt_inputs") and self.prompt_inputs is not None:
            self.prompt_inputs.current_iteration_num = value

    def _reset_conversations(self) -> None:
        """
        Initializes or resets all conversation state variables.

        This method sets up fresh `Conversation` objects for the primary, debater,
        and moderator agents, using their respective system messages. It also
        initializes an empty `MultiAgentConversation` log to track the overall
        dialogue and a dictionary to map agent role names to their dedicated
        conversation objects. This is typically called during orchestrator
        initialization or if a full reset of the conversation state is required.
        """
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
            SchemaValidationError, LLMOrchestrationError, MaxIterationsExceededError,
            LLMClientError).
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
                    LLMClientError,
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

        if (self.prompt_inputs is None) or (not self.prompt_builder.is_inputs_defined()):
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
                # Update current_iteration_num through the property, which will sync prompt_inputs
                self.current_iteration_num = round_num
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
        else:
            raise LLMOrchestrationError(f"Unknown agent name: {agent_name}")

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
        Prepares and potentially truncates a list of messages for an LLM call.

        This method ensures that the total number of tokens in the message history,
        including any system message and a buffer for the LLM's response, does not
        exceed the LLM's context limit. If the history is too long, it removes
        the oldest messages (excluding the system message, if present) until
        the history fits within the allowed token limit.

        Args:
            llm_service: The `LLMInterface` instance being used for the upcoming call.
                         This is used to access token counting methods and context limits.
            messages: The list of message dictionaries (each with 'role' and 'content')
                      to be sent to the LLM.
            context: A descriptive string for logging purposes, indicating the operational
                     context of this history preparation (e.g., "Primary Agent Turn").

        Returns:
            A list of message dictionaries, potentially truncated, ready for the API call.

        Raises:
            LLMOrchestrationError: If the system message alone (plus buffer) exceeds
                                   the token limit, or if truncation fails to reduce
                                   the history to a valid size. Also raised for
                                   unexpected errors during token counting.
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
        3. Executes the `llm_service.query()` call using the prepared messages.
        4. Invokes the `messaging_callback` with detailed metadata after receiving the response.
        5. Returns the raw response content from the LLM.

        This method does NOT:
        - Manage `Conversation` objects directly (e.g., adding messages to them).
        - Update the main conversation log (`self.main_conversation_log`) or agent-specific logs (`self.role_conversation`).
        - Implement retry logic; errors from LLM interaction or history preparation are propagated.

        Args:
            llm_service: The `LLMInterface` instance to query.
            messages_to_send_raw: The complete list of messages (potentially including system,
                                  user, and assistant roles) that will be processed by
                                  `_prepare_history_for_llm` before being sent to the LLM.
            current_prompt_text_for_callback: The specific text of the current prompt message
                                            being sent. This is used for the 'content' field
                                            in the `messaging_callback` for the prompt event.
            stage: The high-level operational stage (e.g., `OrchestrationStage.DEBATE_TURN`)
                   for creating `EnhancedMessageMetadata` for callbacks.
            step_description: A detailed human-readable description of the current step
                              (e.g., "Primary Agent Round 1 Response") for metadata.
            iteration_num: Optional iteration number (e.g., debate round number,
                           solution improvement attempt number) for metadata.
            prompt_source_entity_type: The type of entity initiating the prompt (e.g.,
                                       `MessageEntityType.USER_PROMPT_SOURCE`) for prompt metadata.
            prompt_source_entity_name: The name of the entity initiating the prompt (e.g., "user",
                                       or an agent role name if one agent is prompting another
                                       through the orchestrator) for prompt metadata.
            prompt_metadata_data: Optional additional structured data to be included in the
                                  `EnhancedMessageMetadata.data` field for the prompt event callback.
            query_kwargs: Optional dictionary of additional keyword arguments to pass to
                          `llm_service.query()` (e.g., specific model parameters not
                          covered by default settings like temperature or max_tokens).

        Returns:
            The string content of the LLM's response.

        Raises:
            LLMOrchestrationError: If `_prepare_history_for_llm` fails (e.g., due to token
                                   limit issues that cannot be resolved by truncation).
            LLMResponseError: If `llm_service.query()` fails due to issues with the LLM's
                              response (e.g., API errors, rate limits, content filtering).
            LLMClientError: If `llm_service.query()` fails due to client-side issues
                            (e.g., network problems, authentication failures).
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
        Queries an LLM agent during a debate turn, updating relevant conversation logs.

        This method orchestrates a single agent's turn within the debate. It performs
        the following steps:
        1. Validates that the prompt is not empty and that the agent's conversation
           history is in a valid state (last message is not from 'user').
        2. Adds the provided `prompt` as a user message to the specified agent's
           dedicated `Conversation` object (`self.role_conversation[queried_role_name]`).
        3. Optionally adds the `prompt` to the main multi-agent conversation log
           (`self.main_conversation_log`).
        4. Invokes `_query_llm_core` to execute the actual LLM call, passing the
           agent's updated conversation history and relevant metadata.
        5. Adds the LLM's response as an assistant message to the agent's dedicated
           `Conversation` object.
        6. Adds the LLM's response to the `main_conversation_log`, attributed to the
           `queried_role_name`.

        Args:
            llm_to_query: The `LLMInterface` instance for the agent to be queried.
            queried_role_name: The role name (e.g., "Primary", "Debater") of the agent
                               executing this turn. This is used for logging and to
                               access the correct conversation history.
            prompt: The prompt content to be sent to the LLM agent.
            query_kwargs: Optional dictionary of additional keyword arguments to pass to
                          the `_query_llm_core` method, and subsequently to the
                          `llm_service.query()` call.
            add_prompt_to_main_log: If True (default), the `prompt` is added as a 'user'
                                    message to the `main_conversation_log` before the
                                    LLM call. Set to False if the prompt source is not
                                    conceptually a user input to the main debate flow.

        Returns:
            The string content of the LLM agent's response.

        Raises:
            LLMOrchestrationError: If prompt validation fails, or if there are unexpected
                                   errors during the query or logging process.
            LLMResponseError: Propagated from `_query_llm_core` if the LLM query fails
                              due to issues with the LLM's response.
            LLMClientError: Propagated from `_query_llm_core` for client-side errors
                            during LLM interaction (e.g., network, authentication).
        """
        try:
            self.logger.info(f"Try to query {queried_role_name}...")

            # 1. Validations - these remain the same as in _query_agent
            conversation_history = self.role_conversation[queried_role_name]

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
                iteration_num=self.current_iteration_num,
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

        except (LLMResponseError, LLMClientError) as e:
            self.logger.error(f"{queried_role_name} query failed: {e}", exc_info=True)
            raise
        except LLMOrchestrationError as e:
            # Don't wrap LLMOrchestrationError, just propagate
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
        Handles LLM querying for iterative generation and correction loops.

        This method is a specialized wrapper around `_query_llm_core` tailored for
        the iterative process of generating a solution and then attempting to fix it
        if errors occur. It manages message preparation for either stateful (using
        a dedicated `Conversation` object) or stateless LLM calls.

        Key steps:
        1. If a `generation_conversation_obj` is provided (stateful mode), adds the
           `current_prompt_content` as a user message to it and prepares the full
           history from this object.
        2. If `generation_conversation_obj` is None (stateless mode), constructs a new
           message list containing the LLM's system message (if any) and the
           `current_prompt_content` as a user message.
        3. Calls `_query_llm_core` with these messages, the `current_prompt_content`
           (for callback purposes), and specific metadata including the `stage`
           (e.g., `SOLUTION_GENERATION_INITIAL` or `SOLUTION_GENERATION_FIX_ATTEMPT`),
           `base_context_for_logging`, and `attempt_num`.
        4. If in stateful mode, adds the LLM's response as an assistant message to the
           `generation_conversation_obj`.
        5. Returns the raw LLM response content.

        Args:
            llm_service: The `LLMInterface` instance to use for the query.
            current_prompt_content: The actual prompt string for the current generation
                                    or correction attempt.
            generation_conversation_obj: Optional `Conversation` object. If provided,
                                       this conversation's history is used and updated,
                                       allowing the LLM to maintain context across
                                       multiple generation/fix attempts. If None, each
                                       call is stateless from the LLM's perspective.
            base_context_for_logging: A string describing the broader context of this
                                      iterative generation (e.g., "Final Solution Generation").
                                      Used for logging and metadata.
            attempt_num: The current attempt number within the iterative loop (e.g., 1
                         for initial attempt, 2 for first fix attempt).
            stage: The specific `OrchestrationStage` for this attempt (e.g.,
                   `OrchestrationStage.SOLUTION_GENERATION_INITIAL_ATTEMPT` or
                   `OrchestrationStage.SOLUTION_GENERATION_FIX_ATTEMPT`).
            prompt_metadata_for_llm_core: Optional additional structured data to be passed
                                          to `_query_llm_core` and included in the
                                          `EnhancedMessageMetadata.data` field for the
                                          prompt event callback. Useful for passing, e.g.,
                                          error details during fix attempts.
            query_kwargs: Optional dictionary of keyword arguments to pass to the
                          `llm_service.query()` method via `_query_llm_core`.

        Returns:
            The raw string content of the LLM's response.

        Raises:
            Propagates exceptions from `_query_llm_core`, which include:
            - `LLMOrchestrationError`: For history preparation failures.
            - `LLMResponseError`: For LLM API errors or problematic responses.
            - `LLMClientError`: For client-side LLM interaction issues.
        """
        self.logger.debug(
            f"_query_for_iterative_generation: BaseContext='{base_context_for_logging}', Attempt={attempt_num}, Stage='{stage.value}'"
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
        Generates, verifies, and iteratively corrects a result using an LLM.

        This method implements a loop to produce a result that meets specified
        criteria (e.g., parsing as JSON, validating against a schema). It first
        prompts the LLM for an initial result. If `json_result` is True, it attempts
        to parse the response as JSON and, if `result_schema` is provided, validates
        it against the schema. If any of these steps fail, or if an `LLMResponseError`
        or `LLMOrchestrationError` occurs during generation, it captures the error,
        constructs a "fix prompt" (including the error and the faulty response),
        and re-queries the LLM. This process repeats up to `max_improvement_iterations`.

        The method can operate in two modes regarding conversation history for this loop:
        - Stateful (`use_conversation=True`): A dedicated `Conversation` object is
          maintained for this generation task. The LLM sees the history of its previous
          attempts and the corresponding fix prompts, allowing it to learn from errors.
        - Stateless (`use_conversation=False`): Each attempt is made with a fresh context,
          though the fix prompt still provides error details from the last attempt.

        Args:
            llm_service: The `LLMInterface` instance to use for generating the result.
            context: A descriptive string for logging and metadata, indicating the purpose
                     of this generation task (e.g., "Final Solution Generation").
            result_schema: Optional JSON schema (as a dictionary) to validate against if
                           `json_result` is True. If None, only JSON parsing is checked.
            use_conversation: If True (default), a dedicated `Conversation` object is used
                              to maintain state across iterative attempts within this method.
                              If False, each attempt is more stateless for the LLM.
            max_improvement_iterations: The maximum number of attempts (including the initial
                                      one) to generate and correct the result.
            json_result: If True (default), the expected result is a JSON object. The method
                         will attempt to parse the LLM response as JSON and validate against
                         `result_schema` if provided. If False, the raw string response is
                         returned after the first successful generation.

        Returns:
            The verified result. This will be a dictionary if `json_result` is True and
            parsing/validation succeeds, otherwise it's the raw string response from the LLM
            (if `json_result` is False and generation succeeds).

        Raises:
            MaxIterationsExceededError: If a valid result cannot be produced within
                                      `max_improvement_iterations`.
            SchemaValidationError: If `json_result` is True and the final successfully parsed
                                   JSON response fails schema validation (and no more retries).
                                   This is typically the error from the last attempt.
            LLMResponseError: If an LLM query fails irrecoverably (e.g., client error not
                              related to content, or an error occurs on the last attempt).
            LLMOrchestrationError: For other orchestration issues, such as failure to prepare
                                   prompt inputs, or if a fix prompt cannot be built.
            LLMClientError: If a non-recoverable LLM client error occurs during generation.
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
        current_stage: OrchestrationStage = OrchestrationStage.SOLUTION_GENERATION_INITIAL
        for attempt in range(1, max_improvement_iterations + 1):
            self.logger.info(
                f"Context '{context}': Generating result (Attempt {attempt}/{max_improvement_iterations})"
            )
            raw_response_for_fix_prompt: Optional[str] = (
                None  # Store the raw response for the fix prompt
            )
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
