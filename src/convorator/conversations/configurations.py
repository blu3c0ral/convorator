from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from convorator.client.llm_client import LLMInterface
from convorator.conversations.events import MessagingCallback
from convorator.conversations.state import MultiAgentConversation

from convorator.utils.logger import setup_logger

# Import shared types from types.py
from convorator.conversations.types import (
    LoggerProtocol,
    SolutionLLMGroup,  # Moved to types.py
    PromptBuilderFunction,  # Type alias for prompt functions
)

# Import default prompt function implementations for default_factory
from convorator.conversations.prompts import PromptBuilder

# Import default callback
from .callbacks import DefaultFileResponseLogger


# --- Core Process Configuration ---
@dataclass
class OrchestratorConfig:
    """Configuration for the SolutionImprovementOrchestrator."""

    # --- Core Components (Required) ---
    llm_group: SolutionLLMGroup
    topic: str
    requirements: str
    assessment_criteria: str

    # --- Initial State ---
    initial_solution: Optional[str] = None  # Can be optional if initial prompt generates it

    # --- Orchestration Parameters ---
    debate_iterations: int = 2
    improvement_iterations: int = 3
    expect_json_output: bool = True
    solution_schema: Optional[Dict[str, Any]] = None

    # --- Customization ---
    logger: LoggerProtocol = field(default_factory=lambda: setup_logger(__name__))
    messaging_callback: Optional[MessagingCallback] = field(
        default_factory=DefaultFileResponseLogger
    )

    # User-provided context/instructions (Optional)
    moderator_instructions_override: Optional[str] = None  # For the moderator *context* prompt
    debate_context_override: Optional[str] = None  # Background context

    # --- Prompt Builder Functions (with defaults) ---
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
    session_id: Optional[str] = None  # Added for session tracking

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.debate_iterations < 1:
            raise ValueError("debate_iterations must be at least 1.")
        if self.improvement_iterations < 1:
            raise ValueError("improvement_iterations must be at least 1.")
        if not isinstance(self.llm_group, SolutionLLMGroup):
            raise TypeError("llm_group must be an instance of SolutionLLMGroup.")
        # Add validation for required strings
        if not self.topic:
            raise ValueError("Configuration requires a 'topic'.")
        if not self.requirements:
            raise ValueError("Configuration requires 'requirements'.")
        if not self.assessment_criteria:
            raise ValueError("Configuration requires 'assessment_criteria'.")

        self.logger.info("OrchestratorConfig initialized and validated.")
