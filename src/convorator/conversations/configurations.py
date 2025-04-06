from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Optional

from convorator.client.llm_client import LLMInterface

from convorator.utils.logger import setup_logger


@dataclass
class SolutionLLMGroup:
    """Specifies the LLM clients for different roles in the process."""

    primary_llm: LLMInterface
    debater_llm: LLMInterface
    moderator_llm: LLMInterface
    solution_generation_llm: LLMInterface


# --- Core Process Configuration ---
@dataclass
class OrchestratorConfig:
    """Main configuration for the solution improvement process."""

    # --- LLMs ---
    llm_group: SolutionLLMGroup

    # --- Process Control & Task Definition ---
    debate_iterations: int = 3
    improvement_iterations: int = 3
    expect_json_output: bool = True
    solution_schema: Optional[Dict[str, Any]] = None
    topic: Optional[str] = None
    # Task specifics often provided at runtime or via a separate object
    initial_solution: str = ""
    requirements: str = ""
    assessment_criteria: str = ""

    # --- Prompt Builders ---
    # Users configure this object with desired functions
    prompt_builders: PromptBuilderFunctions = field(default_factory=PromptBuilderFunctions)

    # --- Logging ---
    logger: logging.Logger = field(default_factory=setup_logger)

    # Debate context (moved here from ModeratedConversationConfig)
    debate_context: Optional[str] = None

    # Potential place for moderator_instructions if needed by default builder funcs
    moderator_instructions: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.llm_group, SolutionLLMGroup):
            raise TypeError("llm_group must be an instance of SolutionLLMGroup")
        if not isinstance(self.prompt_builders, PromptBuilderFunctions):
            raise TypeError("prompt_builders must be an instance of PromptBuilderFunctions")
        # Add more validation checks as needed
        self.logger.info("OrchestratorConfig initialized.")
