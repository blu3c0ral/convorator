# prompts.py
from dataclasses import dataclass, field
import logging
from typing import Callable, Dict, List, Optional

from convorator.conversations.conversation_orchestrator import MultiAgentConversation
from convorator.conversations.conversation_setup import SolutionLLMGroup


@dataclass
class PromptBuilderInputs:
    """Holds static configuration and the conversation history."""

    # Static Configuration & Context (Set once by orchestrator)
    topic: Optional[str] = None
    logger: logging.Logger
    llm_group: Optional[SolutionLLMGroup] = None  # Provides access to LLMs if needed
    solution_schema: Optional[Dict] = None  # e.g., for format instructions
    initial_solution: Optional[str] = None
    requirements: Optional[str] = None
    assessment_criteria: Optional[str] = None
    moderator_instructions: Optional[str] = None  # Core instructions for moderator role
    debate_context: Optional[str] = None  # Overall context description for the debate
    primary_role_name: Optional[str] = "Primary"  # Role names are likely static
    debater_role_name: Optional[str] = "Debater"
    moderator_role_name: Optional[str] = "Moderator"
    expect_json_output: bool = False  # Hint for formatting instructions

    # Conversation History (Updated by orchestrator, passed to builders)
    conversation_history: Optional[MultiAgentConversation] = None

    # --- Dynamic State (Set by orchestrator before specific calls if needed) ---
    initial_prompt_content: Optional[str] = None  # Result of build_initial_prompt
    moderator_summary: Optional[str] = None  # Result of summary generation
    response_to_fix: Optional[str] = None  # Response that needs fixing
    errors_to_fix: Optional[str] = None  # Errors identified in the response


def _get_last_message_content_by_role(
    history: Optional[MultiAgentConversation], role_name: str, logger: logging.Logger
) -> Optional[str]:
    """Helper to safely get the content of the last message by a specific role."""
    if not history:
        logger.warning(f"Cannot get message for role '{role_name}': Conversation history is None.")
        return None
    message = history.get_last_message_by_role(role_name)
    if message:
        return message.get("content")
    logger.debug(f"No message found for role '{role_name}' in history.")
    return None


def _get_nth_last_message_content_by_role(
    history: Optional[MultiAgentConversation], role_name: str, n: int, logger: logging.Logger
) -> Optional[str]:
    """Helper to safely get the content of the nth last message by a specific role."""
    if not history:
        logger.warning(f"Cannot get message for role '{role_name}': Conversation history is None.")
        return None
    messages = history.get_messages_by_role(role_name)
    if messages and len(messages) >= n:
        # Messages are typically appended, so the nth last is len - n
        return messages[len(messages) - n].get("content")
    logger.debug(f"Could not find the {n}th last message for role '{role_name}'.")
    return None


def default_moderator_context_builder(inputs: PromptBuilderInputs) -> str:
    """
    Revised default builder for the Moderator's context prompt using PromptBuilderInputs.
    Includes the most recent exchange by retrieving messages from history.
    """
    if not inputs.conversation_history:
        inputs.logger.warning("Moderator context builder called with no conversation history.")
        # Fallback or raise error? For now, provide minimal context.
        recent_exchange_str = "[Conversation history not available]"
    else:
        # Get last messages from history
        last_debater_msg_content = _get_last_message_content_by_role(
            inputs.conversation_history, inputs.debater_role_name, inputs.logger
        )
        last_primary_msg_content = _get_last_message_content_by_role(
            inputs.conversation_history, inputs.primary_role_name, inputs.logger
        )

        exchange_parts = []
        # Order: Typically Debater spoke, then Primary, then Moderator reviews
        if last_debater_msg_content:
            exchange_parts.append(f"[{inputs.debater_role_name}]:\n{last_debater_msg_content}")
        if last_primary_msg_content:
            exchange_parts.append(f"[{inputs.primary_role_name}]:\n{last_primary_msg_content}")

        if exchange_parts:
            recent_exchange_str = "\n\n".join(exchange_parts)
        else:
            # Handle case where messages might be missing (e.g., first round)
            recent_exchange_str = "[No preceding messages found in history]"
            inputs.logger.info(
                "Moderator context builder: No Primary/Debater messages in history yet."
            )

    # Start with background info
    moderator_full_context = f"BACKGROUND CONTEXT:\n{inputs.debate_context}\n\n"
    moderator_full_context += (
        f"INITIAL TASK:\n{inputs.initial_prompt_content}\n\n"  # Use initial_prompt_content
    )

    moderator_full_context += f"RECENT EXCHANGE TO MODERATE:\n{recent_exchange_str}\n\n"

    # Add the specific instructions
    moderator_full_context += f"YOUR INSTRUCTIONS AS MODERATOR:\n{inputs.moderator_instructions}"

    return moderator_full_context


def default_debater_prompt_with_feedback(inputs: PromptBuilderInputs) -> str:
    """
    Default prompt builder for the Debater LLM turn AFTER moderation, using PromptBuilderInputs.
    Retrieves necessary context from conversation history.
    """
    prompt = ""

    # Get Debater's own last response (which would be the 2nd last debater message overall)
    debater_last_response = _get_nth_last_message_content_by_role(
        inputs.conversation_history, inputs.debater_role_name, 2, inputs.logger
    )
    if debater_last_response:
        prompt += (
            f'PREVIOUSLY, YOU ({inputs.debater_role_name}) SAID:\n"{debater_last_response}"\n\n'
        )

    # Get the Primary's last response
    primary_response = _get_last_message_content_by_role(
        inputs.conversation_history, inputs.primary_role_name, inputs.logger
    )
    if primary_response:
        prompt += f'THE {inputs.primary_role_name.upper()} RESPONDED:\n"{primary_response}"\n\n'
    else:
        inputs.logger.warning("Debater prompt builder: Could not find Primary's last response.")
        prompt += f"THE {inputs.primary_role_name.upper()} RESPONSE:\n[Not found in history]\n\n"

    # Get the Moderator's last feedback/instructions
    moderator_feedback = _get_last_message_content_by_role(
        inputs.conversation_history, inputs.moderator_role_name, inputs.logger
    )
    if moderator_feedback:
        prompt += f'THE {inputs.moderator_role_name.upper()} PROVIDED FEEDBACK/INSTRUCTIONS:\n"{moderator_feedback}"\n\n'
    else:
        # This might be expected if moderation is skipped or it's the first round post-initial
        inputs.logger.info(
            "Debater prompt builder: No Moderator feedback found in history (may be expected)."
        )
        prompt += f"THE {inputs.moderator_role_name.upper()} FEEDBACK/INSTRUCTIONS:\n[Not provided or found in history]\n\n"

    prompt += f"YOUR TASK AS {inputs.debater_role_name.upper()}:\nPlease provide your next response, addressing the points above and incorporating the moderator's feedback (if any)."
    return prompt


def default_primary_prompt_with_feedback(inputs: PromptBuilderInputs) -> str:
    """
    Default prompt builder for the Primary LLM turn using PromptBuilderInputs.
    Includes optional feedback from the previous round's Moderator as context.
    Responds primarily to the Debater's most recent message.
    Retrieves necessary context from conversation history.
    """
    prompt = ""

    # Context: Primary's own previous response (2nd last Primary message)
    primary_last_response = _get_nth_last_message_content_by_role(
        inputs.conversation_history, inputs.primary_role_name, 2, inputs.logger
    )
    if primary_last_response:
        prompt += (
            f'PREVIOUSLY, YOU ({inputs.primary_role_name}) SAID:\n"{primary_last_response}"\n\n'
        )

    # Context: Moderator feedback from the round that just ended (last Moderator message)
    # This feedback was directed at the Debater, but Primary needs it for context.
    moderator_feedback = _get_last_message_content_by_role(
        inputs.conversation_history, inputs.moderator_role_name, inputs.logger
    )
    if moderator_feedback:
        prompt += f'CONTEXT FROM LAST ROUND\'S {inputs.moderator_role_name.upper()} MODERATION (which the {inputs.debater_role_name} should have addressed):\n"{moderator_feedback}"\n\n'
    else:
        inputs.logger.info(
            "Primary prompt builder: No prior Moderator feedback found in history (may be expected)."
        )

    # The immediate trigger for this turn: the Debater's last response
    debater_response = _get_last_message_content_by_role(
        inputs.conversation_history, inputs.debater_role_name, inputs.logger
    )
    if debater_response:
        prompt += f'THE {inputs.debater_role_name.upper()} NOW RESPONDS:\n"{debater_response}"\n\n'
    else:
        inputs.logger.error(
            "Primary prompt builder: Could not find Debater's last response in history!"
        )
        # This is likely an error state if Primary is expected to respond
        prompt += f"THE {inputs.debater_role_name.upper()} RESPONSE:\n[Critical Error: Not found in history]\n\n"

    # The task instruction
    prompt += f"YOUR TASK AS {inputs.primary_role_name.upper()}:\nPlease analyze the {inputs.debater_role_name}'s latest response above, considering the previous moderator feedback (if provided), and provide your counter-argument, refinement, or agreement."
    return prompt


def default_build_initial_prompt(inputs: PromptBuilderInputs) -> str:
    """Builds the initial prompt string from inputs."""
    if not all([inputs.initial_solution, inputs.requirements, inputs.assessment_criteria]):
        # Log a warning or error if essential parts are missing
        inputs.logger.warning(
            "Building initial prompt with missing solution, requirements, or criteria."
        )

    return (
        f"Initial solution:\n{inputs.initial_solution or '[Not provided]'}\n\n"
        f"Requirements:\n{inputs.requirements or '[Not provided]'}\n\n"
        f"Assessment criteria:\n{inputs.assessment_criteria or '[Not provided]'}\n\n"
        "Discuss and critically analyze this solution. Identify strengths, weaknesses, and suggest improvements."
    )


# Renamed from default_synthesize_moderation_summary
def default_build_summary_prompt(inputs: PromptBuilderInputs) -> str:
    """
    Builds the prompt FOR the moderator LLM to synthesize a summary, using PromptBuilderInputs.
    Does NOT call the LLM itself. Requires the conversation history to be passed within inputs.
    """
    if inputs.logger:
        inputs.logger.info("Building moderation summary prompt.")

    # The prompt itself doesn't need the history *content* directly,
    # but the calling code will pass history + this prompt to the LLM.
    prompt = (
        "Given the preceding debate history, provide:\n"
        "1. A concise summary of strengths and weaknesses discussed.\n"
        "2. Specific actionable improvements suggested or confirm no improvements required.\n"
        "3. Clear instructions for generating an improved final solution, ensuring alignment with the original requirements and assessment criteria."
    )
    return prompt


def default_build_solution_improvement_prompt(inputs: PromptBuilderInputs) -> str:
    """Builds the prompt for generating an improved solution using PromptBuilderInputs."""
    if not inputs.moderator_summary:
        inputs.logger.warning("Building solution improvement prompt without moderator summary.")

    return (
        f"Original solution:\n{inputs.initial_solution or '[Not Provided]'}\n\n"
        f"Moderator's summary and instructions:\n{inputs.moderator_summary or '[Not Provided]'}\n\n"
        "Using these insights, provide an improved solution. Ensure the output strictly matches the format and structure of the original solution."
    )


def default_fix_prompt_builder(inputs: PromptBuilderInputs) -> str:
    """
    Default fix prompt template using PromptBuilderInputs.
    """
    if not all([inputs.requirements, inputs.response_to_fix, inputs.errors_to_fix]):
        inputs.logger.warning("Building fix prompt with missing requirements, response, or errors.")

    return (
        f"Requirements:\n{inputs.requirements or '[Not Provided]'}\n\n"
        f"Response with errors:\n{inputs.response_to_fix or '[Not Provided]'}\n\n"
        f"Identified Errors:\n{inputs.errors_to_fix or '[Not Provided]'}\n\n"
        "Please fix the above response to meet the requirements and address the identified errors. Output only the corrected response."
    )


def default_build_debate_user_prompt(inputs: PromptBuilderInputs) -> str:
    """Builds the default initial user prompt for the moderated debate."""
    topic_context = f" on the topic of '{inputs.topic}'" if inputs.topic else ""
    # TODO: Add max_iterations to PromptBuilderInputs if needed dynamically here
    # Assuming max_iterations isn't easily available here, using a generic phrasing
    if inputs.debate_context:
        # If user provided context, prepend it to the initial prompt content
        debate_context_str = inputs.debate_context
    else:
        # Otherwise, build the default context string
        debate_context_str = (
            f"This is a moderated conversation between {inputs.primary_role_name} and {inputs.debater_role_name}{topic_context}. "
            f"{inputs.moderator_role_name} will guide the discussion after exchanges. "
            f"Please respond thoughtfully to the following:"
        )

    # Combine context and the core initial prompt content (which should be pre-built)
    if not inputs.initial_prompt_content:
        inputs.logger.error(
            "Cannot build debate user prompt: initial_prompt_content is missing in inputs."
        )
        return "[Error: Initial prompt content missing]"

    return f"{debate_context_str}\n\n{inputs.initial_prompt_content}"


def default_build_moderator_instructions(inputs: PromptBuilderInputs) -> str:
    """Builds the default instructions for the Moderator role."""
    # Note: inputs.moderator_instructions would contain user-provided instructions.
    # This function generates the *default* if none were provided.
    return (
        f"Moderator, analyze the last exchange ({inputs.debater_role_name} -> {inputs.primary_role_name}). Provide:\n"
        f"1. Assessment of reasoning, clarity, and adherence to instructions/requirements.\n"
        f"2. Specific suggestions or questions for the next {inputs.debater_role_name}'s response.\n"
        f"3. Corrections for any factual errors or logical fallacies observed."
    )


@dataclass
class PromptBuilderFunctions:
    """Holds references to the functions used for building prompts."""

    build_initial_prompt: Callable[[PromptBuilderInputs], str] = default_build_initial_prompt
    # Added builder for the initial user message in the debate
    build_debate_user_prompt: Callable[[PromptBuilderInputs], str] = (
        default_build_debate_user_prompt
    )
    build_moderator_context: Callable[[PromptBuilderInputs], str] = (
        default_moderator_context_builder
    )
    # Added builder for the default moderator instructions
    build_moderator_instructions: Callable[[PromptBuilderInputs], str] = (
        default_build_moderator_instructions
    )
    build_primary_prompt: Callable[[PromptBuilderInputs], str] = (
        default_primary_prompt_with_feedback  # Renamed original function
    )
    build_debater_prompt: Callable[[PromptBuilderInputs], str] = (
        default_debater_prompt_with_feedback  # Renamed original function
    )
    build_summary_prompt: Callable[[PromptBuilderInputs], str] = (
        default_build_summary_prompt  # Renamed original function
    )
    build_improvement_prompt: Callable[[PromptBuilderInputs], str] = (
        default_build_solution_improvement_prompt  # Renamed original function
    )
    build_fix_prompt: Callable[[PromptBuilderInputs], str] = default_fix_prompt_builder
