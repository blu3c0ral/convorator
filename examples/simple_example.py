# examples/simple_example.py

import os

from convorator.conversations.conversation_orchestrator import improve_solution_with_moderation
from convorator.conversations.configurations import OrchestratorConfig
from convorator.client.llm_client import create_llm_client, LLMClientConfig
from convorator.conversations.prompts import (
    DEFAULT_DEBATER_AGENT_SYSTEM_MESSAGE,
    DEFAULT_MODERATOR_AGENT_SYSTEM_MESSAGE,
    DEFAULT_PRIMARY_AGENT_SYSTEM_MESSAGE,
    DEFAULT_SOLUTION_GENERATION_AGENT_SYSTEM_MESSAGE,
)
from convorator.conversations.types import SolutionLLMGroup


def main():
    # Step 0: Configure LLMs for each role
    openai_llm_config = LLMClientConfig(
        client_type="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.5,
        max_tokens=16384,
    )

    # Step 1: Configure LLMs for each role
    openai_llm_config.role_name = "Primary"
    primary_llm = create_llm_client(
        **vars(openai_llm_config), system_message=DEFAULT_PRIMARY_AGENT_SYSTEM_MESSAGE
    )
    openai_llm_config.role_name = "Debater"
    debater_llm = create_llm_client(
        **vars(openai_llm_config), system_message=DEFAULT_DEBATER_AGENT_SYSTEM_MESSAGE
    )
    openai_llm_config.role_name = "Moderator"
    moderator_llm = create_llm_client(
        **vars(openai_llm_config), system_message=DEFAULT_MODERATOR_AGENT_SYSTEM_MESSAGE
    )
    openai_llm_config.role_name = "Solution"
    solution_llm = create_llm_client(
        **vars(openai_llm_config), system_message=DEFAULT_SOLUTION_GENERATION_AGENT_SYSTEM_MESSAGE
    )

    # Step 2: Group LLMs
    llm_group = SolutionLLMGroup(
        primary_llm=primary_llm,
        debater_llm=debater_llm,
        moderator_llm=moderator_llm,
        solution_generation_llm=solution_llm,
    )

    # Step 3: Create Orchestrator Configuration
    config = OrchestratorConfig(
        llm_group=llm_group,
        topic="Two Sum Problem",
        requirements="The final solution must return the indices of the two numbers that add up to the target.",
        assessment_criteria="The solution must be correct and efficient.",
        debate_iterations=3,  # Default value
        improvement_iterations=2,  # Default value
    )

    # Step 4: Define the task
    initial_solution = {
        "nums": [2, 7, 11, 15],
        "target": 9,
        "solution": "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return []",  # Faulty solution
    }

    # Step 5: Run the improvement process
    try:
        print("Starting the solution improvement process...")
        improved_solution = improve_solution_with_moderation(
            config=config,
            initial_solution=initial_solution,
            requirements=config.requirements,
            assessment_criteria=config.assessment_criteria,
        )
        print("Successfully generated improved solution:", improved_solution)
    except Exception as e:
        print(f"Orchestration failed: {e}")


if __name__ == "__main__":
    main()
