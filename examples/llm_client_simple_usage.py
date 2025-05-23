import os
from convorator.client.llm_client import LLMClientConfig, create_llm_client


# llm_config = LLMClientConfig(
#     client_type="openai",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model=os.getenv("OPENAI_MODEL"),
#     temperature=0.5,
#     max_tokens=1000,
# )

llm_config = LLMClientConfig(
    client_type="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model=os.getenv("ANTHROPIC_MODEL"),
    temperature=0.5,
    max_tokens=1000,
)


# llm_config = LLMClientConfig(
#     client_type="gemini",
#     api_key=os.getenv("GEMINI_API_KEY"),
#     model=os.getenv("GEMINI_MODEL"),
#     temperature=0.5,
#     max_tokens=1000,
# )

llm = create_llm_client(**vars(llm_config))

response = llm.query(
    "What's the answer to the Ultimate Question of Life, the Universe, and Everything?"
)

print(response)
