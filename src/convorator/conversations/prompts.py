# src/convorator/conversations/prompts.py
from __future__ import annotations

# Default System Messages for Orchestrator Roles

DEFAULT_PRIMARY_AGENT_SYSTEM_MESSAGE = (
    "You are the Primary Agent. Your main role is to propose, refine, and defend solutions. "
    "You will engage in a debate with a Debater Agent, and a Moderator will oversee the discussion. "
    "When the Debater Agent criticizes your proposal or offers alternatives, analyze their points carefully. "
    "When the Moderator provides feedback or instructions, adhere to them. "
    "Your responses should be constructive, aiming to improve the solution based on the dialogue. "
    "You will receive user messages that consist of the Debater's arguments and/or the Moderator's feedback. "
    "Respond by clearly articulating your counter-arguments, revised proposals, or points of agreement. "
    "Focus on addressing the specific points raised and iteratively improving the solution. "
    "Keep your responses concise and to the point, avoiding unnecessary verbosity. "
    "Strive to be proactive and insightful in your proposals, anticipating potential problems and aiming for the highest quality solution based on a deep understanding of the requirements and the ongoing discussion."
)

DEFAULT_DEBATER_AGENT_SYSTEM_MESSAGE = (
    "You are the Debater Agent. Your primary role is to critically evaluate proposals made by the Primary Agent. "
    "Identify weaknesses, logical flaws, unmet requirements, or areas for improvement. Suggest alternatives if appropriate. "
    "A Moderator will oversee the discussion and may provide you with feedback or instructions. "
    "You will receive user messages containing the Primary Agent's proposal and, at times, feedback from the Moderator. "
    "Your response should be a specific and constructive critique of the Primary Agent's proposal, "
    "or a response to the Moderator's guidance. Clearly explain your reasoning and any alternative suggestions. "
    "Do not generate full solutions yourself; your focus is on critique and identifying areas for enhancement. "
    "Be proactive, insightful, and detail-obsessed in your evaluations. Question assumptions, anticipate potential problems in the proposed solutions, and strive to deliver the highest quality critique to guide the process towards an optimal outcome."
    "Strive for clarity and conciseness in your critiques. "
)

DEFAULT_MODERATOR_AGENT_SYSTEM_MESSAGE = (
    "You are the Moderator. Your role is to facilitate a productive and focused debate between a Primary Agent and a Debater Agent. "
    "You must remain objective and impartial. Do NOT take sides or offer your own solutions to the problem being discussed. "
    "Your tasks include: "
    "1. After an exchange between the Primary and Debater, analyze their arguments based on the original requirements and assessment criteria. "
    "2. Provide concise, actionable feedback and clear instructions, typically to the Debater Agent, to guide their next response and ensure the debate stays on track towards improving the solution. "
    "3. At the end of the debate, you will be asked to synthesize a comprehensive summary of the entire discussion, highlighting key strengths, weaknesses, and specific, actionable recommendations for generating an improved final solution. "
    "User messages will present you with the conversational exchange to moderate or a request to summarize the debate. "
    "Your responses should be structured, clear, and focused solely on your moderating duties. "
    "Be insightful and detail-obsessed in your analysis of the debate. Anticipate potential ambiguities or sticking points in the discussion. Your aim is to deliver the highest quality guidance and summary, fostering a robust and effective improvement process based on deep understanding and careful consideration of the arguments presented. "
    "Ensure your feedback and summaries are concise and directly actionable. "
)

DEFAULT_SOLUTION_GENERATION_AGENT_SYSTEM_MESSAGE = (
    "You are the Solution Generation Agent. Your sole responsibility is to generate a complete and refined solution based on provided inputs. "
    "You will receive user messages containing: "
    "1. An initial request: This will include the original problem requirements, assessment criteria, the initial (potentially flawed) solution, and a summary from a Moderator detailing insights and instructions from a debate phase. "
    "2. A correction request (if your previous attempt failed): This will include your previously generated solution, the specific errors or validation failures encountered, and a request to fix these issues. "
    "You must strictly adhere to all provided requirements, instructions, and (if applicable) a JSON schema. "
    "If a JSON output is expected, ensure your response is a single, valid JSON object, without any additional explanatory text before or after it, unless explicitly instructed otherwise. "
    "Focus on producing the requested artifact accurately and completely. "
    "Adopt a proactive and detail-obsessed approach to your generation task. Anticipate and meticulously address all aspects of the provided instructions, requirements, and schema to deliver the highest quality, fully compliant solution. "
    "Be precise and avoid any extraneous information in your output. "
)

from functools import wraps
import json
from typing import Callable, Dict, Optional

from convorator.conversations.state import MultiAgentConversation

# Import shared types
from convorator.conversations.types import PromptBuilderInputs, LoggerProtocol


def get_last_message_content_by_role(
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


def get_nth_last_message_content_by_role(
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
        last_debater_msg_content = get_last_message_content_by_role(
            inputs.conversation_history, inputs.debater_role_name, inputs.logger
        )
        last_primary_msg_content = get_last_message_content_by_role(
            inputs.conversation_history, inputs.primary_role_name, inputs.logger
        )

        exchange_parts: list[str] = []
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
    debater_last_response = get_nth_last_message_content_by_role(
        inputs.conversation_history, inputs.debater_role_name, 2, inputs.logger
    )
    if debater_last_response:
        prompt += (
            f'PREVIOUSLY, YOU ({inputs.debater_role_name}) SAID:\n"{debater_last_response}"\n\n'
        )

    # Get the Primary's last response
    primary_response = get_last_message_content_by_role(
        inputs.conversation_history, inputs.primary_role_name, inputs.logger
    )
    if primary_response:
        prompt += f'THE {inputs.primary_role_name.upper()} RESPONDED:\n"{primary_response}"\n\n'
    else:
        inputs.logger.warning("Debater prompt builder: Could not find Primary's last response.")
        prompt += f"THE {inputs.primary_role_name.upper()} RESPONSE:\n[Not found in history]\n\n"

    # Get the Moderator's last feedback/instructions
    moderator_feedback = get_last_message_content_by_role(
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
    primary_last_response = get_nth_last_message_content_by_role(
        inputs.conversation_history, inputs.primary_role_name, 2, inputs.logger
    )
    if primary_last_response:
        prompt += (
            f'PREVIOUSLY, YOU ({inputs.primary_role_name}) SAID:\n"{primary_last_response}"\n\n'
        )

    # Context: Moderator feedback from the round that just ended (last Moderator message)
    # This feedback was directed at the Debater, but Primary needs it for context.
    moderator_feedback = get_last_message_content_by_role(
        inputs.conversation_history, inputs.moderator_role_name, inputs.logger
    )
    if moderator_feedback:
        prompt += f'CONTEXT FROM LAST ROUND\'S {inputs.moderator_role_name.upper()} MODERATION (which the {inputs.debater_role_name} should have addressed):\n"{moderator_feedback}"\n\n'
    else:
        inputs.logger.info(
            "Primary prompt builder: No prior Moderator feedback found in history (may be expected)."
        )

    # The immediate trigger for this turn: the Debater's last response
    debater_response = get_last_message_content_by_role(
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
        str(initial_solution),
        "```",
        "\n",
        "REQUIREMENTS:",
        str(requirements),
        "\n",
        "ASSESSMENT CRITERIA:",
        str(assessment_criteria),
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


"""
1. **"initial_prompt"** - `_run_moderated_debate`
   - Purpose: Constructs the first prompt that kicks off the debate process
   - Usage: Sent to the debater agent at the beginning of the debate to initiate critical analysis of the proposed solution
   - Content: Contains the topic, initial solution, requirements, and assessment criteria

2. **"primary_prompt"** - `_execute_agent_loop_turn`
   - Purpose: Builds prompt for the Primary agent's turns
   - Usage: When the Primary agent needs to respond during the debate
   - Content: Includes Primary's previous response, Moderator's feedback, and Debater's latest response

3. **"debater_prompt"** - `_execute_agent_loop_turn`
   - Purpose: Builds prompt for the Debater agent's turns
   - Usage: When the Debater needs to respond to the Primary and Moderator's feedback
   - Content: Includes Debater's previous response, Primary's response, and Moderator's feedback

4. **"moderator_context"** - `_execute_agent_loop_turn`
   - Purpose: Builds prompt for the Moderator agent's turns
   - Usage: When the Moderator needs to guide the discussion after Primary and Debater exchanges
   - Content: Includes debate context, recent exchange history, and instructions for moderation

5. **"summary_prompt"** - `_synthesize_summary`
   - Purpose: Creates prompt for the Moderator to generate a final summary
   - Usage: After the debate iterations are complete
   - Content: Instructions for the Moderator to synthesize findings from the debate into actionable improvements

6. **"improvement_prompt"** - `_generate_and_verify_result`
   - Purpose: Builds prompt to generate improved solution based on debate
   - Usage: When creating the final solution after debate summary
   - Content: Original solution, Moderator's summary, and format requirements (JSON or text)

7. **"fix_prompt"** - `_generate_and_verify_result`
   - Purpose: Creates prompt to fix errors in generated solution
   - Usage: When the solution generation fails validation or has errors
   - Content: Previous failed response, error details, and instructions to correct issues

"""


class PromptBuilder:
    _prompt_builders: Dict[str, Callable[[PromptBuilderInputs], str]] = {}

    @classmethod
    def register_prompt_builder(
        cls, prompt_type: str
    ) -> Callable[[Callable[[PromptBuilderInputs], str]], Callable[[PromptBuilderInputs], str]]:
        """
        Decorator to register a prompt builder function for a specific prompt type.

        Args:
            prompt_type (str): The type of prompt this builder handles

        Returns:
            Callable: The decorator function

        Example:
            @PromptBuilder.register_prompt_builder("custom_prompt")
            def build_custom_prompt(inputs: PromptBuilderInputs) -> str:
                return "Custom prompt content"
        """

        def decorator(
            func: Callable[[PromptBuilderInputs], str],
        ) -> Callable[[PromptBuilderInputs], str]:
            @wraps(func)
            def wrapper(inputs: PromptBuilderInputs) -> str:
                return func(inputs)

            cls._prompt_builders[prompt_type] = wrapper
            return wrapper

        return decorator

    def __init__(self, inputs: Optional[PromptBuilderInputs] = None):
        """Initialize the PromptBuilder with registered builders."""
        # Register default builders if not already registered
        self._register_default_builders()
        self._inputs = inputs

    def build_prompt(self, prompt_type: str, inputs: Optional[PromptBuilderInputs] = None) -> str:
        """
        Build a prompt using the registered builder for the given prompt type.

        Args:
            inputs (PromptBuilderInputs): The inputs for building the prompt
            prompt_type (str): The type of prompt to build

        Returns:
            str: The built prompt

        Raises:
            ValueError: If no builder is registered for the given prompt type
        """
        if prompt_type not in self._prompt_builders:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        if inputs is None:
            if self._inputs is None:
                raise ValueError(
                    "PromptBuilderInputs type inputs must be provided. Both the inputs argument and the property were None."
                )
            inputs = self._inputs
        return self._prompt_builders[prompt_type](inputs)

    def update_inputs(self, inputs: PromptBuilderInputs):
        self._inputs = inputs

    def is_inputs_defined(self):
        return self._inputs is not None

    def is_defined(self, prompt_type: str) -> bool:
        """Check if a prompt builder is defined for the given prompt type."""
        return prompt_type in self._prompt_builders

    @classmethod
    def _register_default_builders(cls) -> None:
        """Register the default prompt builders if they haven't been overridden."""
        default_builders = {
            "moderator_instructions": default_build_moderator_instructions,
            "moderator_context": default_build_moderator_context,
            "debater_prompt": default_build_debater_prompt,
            "primary_prompt": default_build_primary_prompt,
            "initial_prompt": default_build_initial_prompt,
            "summary_prompt": default_build_summary_prompt,
            "improvement_prompt": default_build_improvement_prompt,
            "fix_prompt": default_build_fix_prompt,
            "debate_user_prompt": default_build_debate_user_prompt,
        }

        for prompt_type, builder in default_builders.items():
            if prompt_type not in cls._prompt_builders:
                cls._prompt_builders[prompt_type] = builder
