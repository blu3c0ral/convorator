# prompts.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
import logging
from typing import Callable, Dict, Optional

from convorator.conversations.state import MultiAgentConversation

# Import shared types
from convorator.conversations.types import PromptBuilderInputs, LoggerProtocol


def _get_last_message_content_by_role(
    history: Optional[MultiAgentConversation], role_name: str, logger: LoggerProtocol
) -> Optional[str]:
    """Helper to safely get the content of the last message by a specific role."""
    if not history:
        logger.warning(f"Cannot get message for role '{role_name}': Conversation history is None.")
        return None
    messages = history.get_messages_by_role(role_name)
    if messages:
        return messages[-1].get("content")
    logger.debug(f"No message found for role '{role_name}' in history.")
    return None


def _get_nth_last_message_content_by_role(
    history: Optional[MultiAgentConversation], role_name: str, n: int, logger: LoggerProtocol
) -> Optional[str]:
    """Helper to safely get the content of the nth last message by a specific role."""
    if not history:
        logger.warning(f"Cannot get message for role '{role_name}': Conversation history is None.")
        return None
    messages = history.get_messages_by_role(role_name)
    if messages and len(messages) >= n:
        return messages[-n].get("content")
    logger.debug(f"Could not find the {n}th last message for role '{role_name}'.")
    return None


def default_build_moderator_context(inputs: PromptBuilderInputs) -> str:
    """
    Default builder for the Moderator's context prompt using PromptBuilderInputs.
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
    moderator_full_context = f"BACKGROUND CONTEXT:\n{inputs.debate_context or '[Not Provided]'}\n\n"
    moderator_full_context += f"INITIAL TASK:\n{inputs.initial_prompt_content or '[Not Provided]'}\n\n"  # Use initial_prompt_content

    moderator_full_context += f"RECENT EXCHANGE TO MODERATE:\n{recent_exchange_str}\n\n"

    # Add the specific instructions
    moderator_full_context += (
        f"YOUR INSTRUCTIONS AS MODERATOR:\n{inputs.moderator_instructions or '[Not Provided]'}"
    )

    return moderator_full_context


def default_build_debater_prompt(inputs: PromptBuilderInputs) -> str:
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


def default_build_primary_prompt(inputs: PromptBuilderInputs) -> str:
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
    if not all(
        [inputs.topic, inputs.initial_solution, inputs.requirements, inputs.assessment_criteria]
    ):
        # Log a warning or error if essential parts are missing
        inputs.logger.warning(
            "Building initial prompt with missing topic, solution, requirements, or criteria."
        )

    topic = inputs.topic or "[Not Provided]"
    initial_solution = inputs.initial_solution or "[Not Provided]"
    requirements = inputs.requirements or "[Not Provided]"
    assessment_criteria = inputs.assessment_criteria or "[Not Provided]"

    # Explicitly join parts to avoid f-string parsing issues
    parts = [
        f"Critically analyze the proposed initial solution for the topic '{topic}' provided below.",
        f"Evaluate it against the requirements and assessment criteria.",
        f"Identify strengths, weaknesses, potential issues, and areas for improvement.",
        "\n",
        "INITIAL SOLUTION:",
        "```",
        initial_solution,
        "```",
        "\n",
        "REQUIREMENTS:",
        requirements,
        "\n",
        "ASSESSMENT CRITERIA:",
        assessment_criteria,
    ]
    full_prompt = "\n".join(parts)

    return full_prompt


# Renamed from default_synthesize_moderation_summary
def default_build_summary_prompt(inputs: PromptBuilderInputs) -> str:
    """
    Builds the prompt FOR the moderator LLM to synthesize a summary, using PromptBuilderInputs.
    Does NOT call the LLM itself. Requires the conversation history to be passed within inputs.
    """
    inputs.logger.info("Building moderation summary prompt.")

    # The prompt itself doesn't need the history *content* directly,
    # but the calling code will pass history + this prompt to the LLM.
    # Using a triple-quoted string for clarity
    prompt = f"""Given the preceding debate history, provide:

1. A concise summary of strengths and weaknesses discussed.
2. Specific actionable improvements suggested or confirm no improvements required.
3. Clear instructions for generating an improved final solution, ensuring alignment with the original requirements and assessment criteria."""
    return prompt


def default_build_improvement_prompt(inputs: PromptBuilderInputs) -> str:
    """Builds the prompt for generating an improved solution using PromptBuilderInputs."""
    if not inputs.moderator_summary:
        inputs.logger.warning("Building solution improvement prompt without moderator summary.")

    # Start with core instructions
    prompt_parts = [
        f"Original solution:\n{inputs.initial_solution or '[Not Provided]'}",
        f"Moderator's summary and instructions:\n{inputs.moderator_summary or '[Not Provided]'}",
        "Please generate an improved solution based *only* on the moderator's summary and instructions.",
        f"Adhere strictly to the requirements and assessment criteria mentioned earlier.",
    ]

    # Add format instructions based on expect_json_output
    if inputs.expect_json_output:
        prompt_parts.append("Format your response as a valid JSON object.")
        if inputs.solution_schema:
            schema_str = json.dumps(inputs.solution_schema, indent=2)
            prompt_parts.append(f" matching this schema:\n```json\n{schema_str}\n```")
        else:
            prompt_parts.append(" Use appropriate keys and values based on the context.")
    else:
        prompt_parts.append("Format your response as a plain text string.")

    return "\n\n".join(prompt_parts)  # Join with double newline for clarity


def default_build_fix_prompt(inputs: PromptBuilderInputs) -> str:
    """Builds a prompt to ask the LLM to fix its previous response."""
    if not inputs.response_to_fix or not inputs.errors_to_fix:
        inputs.logger.error("Fix prompt builder called without response or errors to fix.")
        return "[Error: Missing input for fix prompt]"

    # Start with core instructions
    prompt_parts = [
        f"The following response you provided resulted in errors:\n---\n{inputs.response_to_fix}\n---",
        f"The errors encountered were:\n{inputs.errors_to_fix}",
        f"Please regenerate the response, correcting these errors.",
        f"Ensure the corrected response still meets the original requirements.",
    ]

    # Add format instructions again for clarity
    if inputs.expect_json_output:
        prompt_parts.append("Format your corrected response as a valid JSON object.")
        if inputs.solution_schema:
            # Schema assumed to be available from previous context
            prompt_parts.append(" matching the schema provided previously.")
        # No else needed here? If no schema, just ask for JSON.
    else:
        prompt_parts.append("Format your corrected response as a plain text string.")

    return "\n\n".join(prompt_parts)  # Join with double newline for clarity


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
    """Builds the default instructions FOR the Moderator role (system message)."""
    # User-provided instructions are in inputs.moderator_instructions and used in the context prompt builder.
    # This function defines the *role* of the moderator.
    return (
        f"You are the Moderator. Your role is to objectively analyze the exchange between {inputs.primary_role_name} and {inputs.debater_role_name}. "
        f"Focus on the quality of arguments based on the initial requirements and assessment criteria. "
        f"Provide clear, actionable feedback to guide the {inputs.debater_role_name} for their next response. "
        f"Do not introduce new opinions or solutions yourself. Focus on facilitating improvement."
    )
