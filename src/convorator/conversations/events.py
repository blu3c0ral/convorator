# src/convorator/conversations/events.py
from typing import TypedDict, Optional, Any, Dict, Callable, Union
from enum import Enum  # Recommended for stages, payload_types, etc.


# --- Enums for controlled vocabularies ---
class OrchestrationStage(Enum):
    DEBATE_TURN = "DEBATE_TURN"
    MODERATION_SUMMARY = "MODERATION_SUMMARY"
    SOLUTION_GENERATION_INITIAL = "SOLUTION_GENERATION_INITIAL"
    SOLUTION_GENERATION_FIX_ATTEMPT = "SOLUTION_GENERATION_FIX_ATTEMPT"
    # Add other relevant stages
    ORCHESTRATION_START = "ORCHESTRATION_START"
    ORCHESTRATION_END = "ORCHESTRATION_END"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class MessageEntityType(Enum):
    USER_PROMPT_SOURCE = "USER_PROMPT_SOURCE"  # e.g., the ultimate user of the orchestrator
    LLM_AGENT = "LLM_AGENT"
    ORCHESTRATOR_INTERNAL = "ORCHESTRATOR_INTERNAL"
    CALLBACK_CONSUMER = "CALLBACK_CONSUMER"  # The callback itself


class MessagePayloadType(Enum):
    TEXT_CONTENT = "TEXT_CONTENT"  # For prompts, responses
    ERROR_DETAILS_STR = "ERROR_DETAILS_STR"  # str(error)
    STRUCTURED_ERROR = "STRUCTURED_ERROR"  # If 'data' field holds a dict representation of an error
    STRUCTURED_RESULT = "STRUCTURED_DATA"  # If 'data' field holds a dict representation of a result
    # Add other payload types as needed


class EventType(Enum):
    PROMPT = "PROMPT"
    RESPONSE = "RESPONSE"
    ORCHESTRATOR_LOG = "ORCHESTRATOR_LOG"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    SOLUTION_GENERATION_INITIAL_ATTEMPT = "SOLUTION_GENERATION_INITIAL_ATTEMPT"
    SOLUTION_GENERATION_FIX_ATTEMPT = "SOLUTION_GENERATION_FIX_ATTEMPT"


# --- TypedDict for the metadata structure ---
class EnhancedMessageMetadata(
    TypedDict, total=False
):  # total=False allows for optional keys not explicitly Optional[]
    event_id: str  # REQUIRED: A unique ID for this specific message event (e.g., UUID)
    timestamp: str  # REQUIRED: ISO 8601 timestamp
    session_id: Optional[str]  # An ID for the overall orchestration session

    stage: OrchestrationStage  # REQUIRED: High-level operational stage
    step_description: str  # REQUIRED: More detailed description of the current step

    iteration_num: Optional[int]  # e.g., debate round, improvement attempt

    source_entity_type: MessageEntityType  # REQUIRED
    source_entity_name: Optional[str]

    target_entity_type: Optional[MessageEntityType]
    target_entity_name: Optional[str]

    llm_service_details: Optional[
        Dict[str, Any]
    ]  # e.g., {"name": "gpt-4", "provider": "openai", "model_used": "..."}

    payload_type: MessagePayloadType  # REQUIRED: Clarifies what `content` represents

    # 'data' can be used for more complex, structured information
    # beyond the string 'content'. E.g., the actual exception object,
    # token counts, a dict representation of a JSON parsing error, etc.
    data: Optional[Any]


# --- Callback Function Signature ---
# This defines what the user needs to implement
MessagingCallback = Callable[[EventType, Union[str, Dict[str, Any]], EnhancedMessageMetadata], None]
# Args: event_type (EventType),
#       content (Union[str, Dict[str, Any]], e.g. a structured result or a string),
#       metadata (EnhancedMessageMetadata)
