import pytest
from unittest.mock import MagicMock

from convorator.conversations.configurations import OrchestratorConfig
from convorator.conversations.types import SolutionLLMGroup


@pytest.fixture
def mock_llm():
    """Create a mock LLM interface."""
    return MagicMock()


@pytest.fixture
def valid_llm_group(mock_llm):
    """Create a valid SolutionLLMGroup with mock LLMs."""
    return SolutionLLMGroup(
        primary_llm=mock_llm,
        debater_llm=mock_llm,
        moderator_llm=mock_llm,
        solution_generation_llm=mock_llm,
    )


@pytest.fixture
def valid_config_params(valid_llm_group):
    """Create valid parameters for OrchestratorConfig."""
    return {
        "llm_group": valid_llm_group,
        "topic": "Test Topic",
        "requirements": "Test Requirements",
        "assessment_criteria": "Test Assessment Criteria",
    }


def test_orchestrator_config_initialization(valid_config_params):
    """Test successful initialization of OrchestratorConfig with valid parameters."""
    config = OrchestratorConfig(**valid_config_params)

    assert config.llm_group == valid_config_params["llm_group"]
    assert config.topic == valid_config_params["topic"]
    assert config.requirements == valid_config_params["requirements"]
    assert config.assessment_criteria == valid_config_params["assessment_criteria"]
    assert config.debate_iterations == 2  # Default value
    assert config.improvement_iterations == 3  # Default value
    assert config.expect_json_output is True  # Default value


def test_orchestrator_config_with_optional_params(valid_config_params):
    """Test initialization with optional parameters."""
    optional_params = {
        "initial_solution": "Test Solution",
        "debate_iterations": 5,
        "improvement_iterations": 7,
        "expect_json_output": False,
        "solution_schema": {"type": "object"},
        "moderator_instructions_override": "Custom instructions",
        "debate_context_override": "Custom context",
    }

    config = OrchestratorConfig(**valid_config_params, **optional_params)

    assert config.initial_solution == optional_params["initial_solution"]
    assert config.debate_iterations == optional_params["debate_iterations"]
    assert config.improvement_iterations == optional_params["improvement_iterations"]
    assert config.expect_json_output == optional_params["expect_json_output"]
    assert config.solution_schema == optional_params["solution_schema"]
    assert (
        config.moderator_instructions_override == optional_params["moderator_instructions_override"]
    )
    assert config.debate_context_override == optional_params["debate_context_override"]


def test_orchestrator_config_validation_errors(valid_config_params):
    """Test that validation errors are raised for invalid configurations."""
    # Test invalid debate_iterations
    with pytest.raises(ValueError, match="debate_iterations must be at least 1"):
        OrchestratorConfig(**valid_config_params, debate_iterations=0)

    # Test invalid improvement_iterations
    with pytest.raises(ValueError, match="improvement_iterations must be at least 1"):
        OrchestratorConfig(**valid_config_params, improvement_iterations=0)

    # Test invalid llm_group type
    with pytest.raises(TypeError, match="llm_group must be an instance of SolutionLLMGroup"):
        OrchestratorConfig(**{**valid_config_params, "llm_group": "not a SolutionLLMGroup"})

    # Test missing topic
    with pytest.raises(ValueError, match="Configuration requires a 'topic'"):
        OrchestratorConfig(**{**valid_config_params, "topic": ""})

    # Test missing requirements
    with pytest.raises(ValueError, match="Configuration requires 'requirements'"):
        OrchestratorConfig(**{**valid_config_params, "requirements": ""})

    # Test missing assessment_criteria
    with pytest.raises(ValueError, match="Configuration requires 'assessment_criteria'"):
        OrchestratorConfig(**{**valid_config_params, "assessment_criteria": ""})


def test_orchestrator_config_logger(valid_config_params):
    """Test that logger is properly initialized and used."""
    mock_logger = MagicMock()
    config = OrchestratorConfig(**valid_config_params, logger=mock_logger)

    assert config.logger == mock_logger
    mock_logger.info.assert_called_once_with("OrchestratorConfig initialized and validated.")


def test_orchestrator_config_custom_prompt_builders(valid_config_params):
    """Test that custom prompt builder functions can be provided."""
    mock_prompt_builder = MagicMock(return_value="Custom prompt")

    config = OrchestratorConfig(
        **valid_config_params,
        build_initial_prompt=mock_prompt_builder,
        build_debate_user_prompt=mock_prompt_builder,
        build_moderator_context=mock_prompt_builder,
        build_moderator_role_instructions=mock_prompt_builder,
        build_primary_prompt=mock_prompt_builder,
        build_debater_prompt=mock_prompt_builder,
        build_summary_prompt=mock_prompt_builder,
        build_improvement_prompt=mock_prompt_builder,
        build_fix_prompt=mock_prompt_builder,
    )

    assert config.build_initial_prompt == mock_prompt_builder
    assert config.build_debate_user_prompt == mock_prompt_builder
    assert config.build_moderator_context == mock_prompt_builder
    assert config.build_moderator_role_instructions == mock_prompt_builder
    assert config.build_primary_prompt == mock_prompt_builder
    assert config.build_debater_prompt == mock_prompt_builder
    assert config.build_summary_prompt == mock_prompt_builder
    assert config.build_improvement_prompt == mock_prompt_builder
    assert config.build_fix_prompt == mock_prompt_builder
