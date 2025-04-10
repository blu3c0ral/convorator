from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Optional, Union

from convorator.client.llm_client import LLMInterface
from convorator.conversations.state import MultiAgentConversation

# Import shared types from types.py
from convorator.conversations.types import (
    LoggerProtocol,
    SolutionLLMGroup,  # Moved to types.py
    PromptBuilderInputs,  # Needed for PromptBuilderFunction type hint indirectly
    PromptBuilderFunction,  # Type alias for prompt functions
)

# Import default prompt function implementations for default_factory
from convorator.conversations.prompts import (
    default_build_initial_prompt,
    default_build_debate_user_prompt,
    default_build_moderator_context,
    default_build_moderator_instructions,  # Note: This is the *role* instruction now
    default_build_primary_prompt,
    default_build_debater_prompt,
    default_build_summary_prompt,
    default_build_improvement_prompt,
    default_build_fix_prompt,
)

# Removed SolutionLLMGroup definition (moved to types.py)
# Removed PromptBuilderFunctions definition (no longer needed)


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
    logger: LoggerProtocol = field(default_factory=lambda: logging.getLogger(__name__))

    # User-provided context/instructions (Optional)
    moderator_instructions_override: Optional[str] = None  # For the moderator *context* prompt
    debate_context_override: Optional[str] = None  # Background context

    # --- Prompt Builder Functions (with defaults) ---
    # Renamed build_moderator_instructions to build_moderator_role_instructions
    # Added build_moderator_context_instructions to differentiate
    build_initial_prompt: PromptBuilderFunction = field(default=default_build_initial_prompt)
    build_debate_user_prompt: PromptBuilderFunction = field(
        default=default_build_debate_user_prompt
    )
    build_moderator_context: PromptBuilderFunction = field(default=default_build_moderator_context)
    build_moderator_role_instructions: PromptBuilderFunction = field(
        default=default_build_moderator_instructions
    )  # System message for moderator
    build_primary_prompt: PromptBuilderFunction = field(default=default_build_primary_prompt)
    build_debater_prompt: PromptBuilderFunction = field(default=default_build_debater_prompt)
    build_summary_prompt: PromptBuilderFunction = field(default=default_build_summary_prompt)
    build_improvement_prompt: PromptBuilderFunction = field(
        default=default_build_improvement_prompt
    )
    build_fix_prompt: PromptBuilderFunction = field(default=default_build_fix_prompt)

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
