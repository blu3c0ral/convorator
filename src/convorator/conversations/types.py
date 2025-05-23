from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from convorator.client.llm_client import LLMInterface
from convorator.conversations.state import MultiAgentConversation

# --- Protocols / Type Aliases ---


class LoggerProtocol(Protocol):
    """Defines the expected interface for a logger."""

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


@dataclass
class PromptBuilderInputs:
    """Holds static configuration and dynamic state needed by prompt building functions.

    Moved here from prompts.py to break circular dependency.
    """

    # --- Fields WITHOUT defaults first ---
    logger: LoggerProtocol

    # --- Fields WITH defaults follow ---
    topic: Optional[str] = None
    llm_group: Optional[SolutionLLMGroup] = None
    solution_schema: Optional[Dict[str, Any]] = None
    initial_solution: Optional[str] = None
    requirements: Optional[str] = None
    assessment_criteria: Optional[str] = None
    moderator_instructions: Optional[str] = None  # User-provided instructions
    debate_context: Optional[str] = None  # User-provided context
    primary_role_name: str = "Primary"
    debater_role_name: str = "Debater"
    moderator_role_name: str = "Moderator"
    expect_json_output: bool = False
    conversation_history: Optional[MultiAgentConversation] = None
    initial_prompt_content: Optional[str] = None  # Content built by initial prompt builder
    moderator_summary: Optional[str] = None  # Output of the summary step
    response_to_fix: Optional[str] = None  # Last LLM response if it failed validation
    errors_to_fix: Optional[str] = None  # Validation errors for the failed response
    debate_iterations: int = 3
    current_iteration_num: int = 0


# Define the type alias for prompt building functions AFTER PromptBuilderInputs is defined
PromptBuilderFunction = Callable[[PromptBuilderInputs], str]


# --- Data Classes ---


@dataclass
class SolutionLLMGroup:
    """Groups the different LLMs required for the solution improvement process.

    Moved here from configurations.py to break circular dependency.
    """

    primary_llm: LLMInterface
    debater_llm: LLMInterface
    moderator_llm: LLMInterface
    solution_generation_llm: LLMInterface
