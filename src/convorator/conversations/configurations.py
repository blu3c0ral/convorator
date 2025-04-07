from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Optional, Protocol, Union

from convorator.client.llm_client import LLMInterface

from convorator.conversations.prompts import PromptBuilderFunctions
from convorator.utils.logger import setup_logger


# Define a protocol for the logger if needed, or use logging.Logger
class LoggerProtocol(Protocol):
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


@dataclass
class SolutionLLMGroup:
    """Groups the different LLMs required for the solution improvement process."""

    primary_llm: LLMInterface
    debater_llm: LLMInterface
    moderator_llm: LLMInterface
    solution_generation_llm: LLMInterface


# --- Core Process Configuration ---
@dataclass
class OrchestratorConfig:
    """Configuration for the SolutionImprovementOrchestrator."""

    # Core Components
    llm_group: SolutionLLMGroup
    topic: str  # Added topic

    # --- Inputs moved to Orchestrator.__init__ ---
    # initial_solution: str = ""
    # requirements: str = ""
    # assessment_criteria: str = ""

    # --- Orchestration Parameters ---
    debate_iterations: int = 2
    improvement_iterations: int = 3
    expect_json_output: bool = True  # Expect final solution to be JSON by default
    solution_schema: Optional[Dict[str, Any]] = None  # Optional JSON schema for validation

    # --- Customization ---
    # Optional logger, defaults to standard logging if not provided
    logger: LoggerProtocol = field(default_factory=lambda: logging.getLogger(__name__))
    # Prompt builders - uses defaults if not provided
    prompt_builders: PromptBuilderFunctions = field(default_factory=PromptBuilderFunctions)
    # Optional: Specific instructions for the moderator beyond the basic role
    moderator_instructions: Optional[str] = None
    # Optional: Additional context for the entire debate/process
    debate_context: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.debate_iterations < 1:
            raise ValueError("debate_iterations must be at least 1.")
        if self.improvement_iterations < 1:
            raise ValueError("improvement_iterations must be at least 1.")
        if not isinstance(self.llm_group, SolutionLLMGroup):
            raise TypeError("llm_group must be an instance of SolutionLLMGroup.")
        # Add more validation as needed
        self.logger.info("OrchestratorConfig initialized and validated.")
