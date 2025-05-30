import os
from typing import Any, Dict, List, Optional

from convorator.client.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Conversation,
    LLMInterface,
)
from convorator.conversations.types import LoggerProtocol
from convorator.exceptions import LLMClientError, LLMConfigurationError, LLMResponseError


ANTHROPIC_CONTEXT_LIMITS: Dict[str, Optional[int]] = {
    # Claude models generally have large context windows.
    # Values represent total context window size.
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3.5-sonnet-20240620": 200000,  # Official ID for Claude 3.5 Sonnet
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}
# Smallest documented context limit for readily available Claude models.
DEFAULT_ANTHROPIC_CONTEXT_LIMIT = 100000


class AnthropicLLM(LLMInterface):
    """Concrete implementation for Anthropic's API (Claude models)."""

    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(
        self,
        logger: LoggerProtocol,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",  # Default to fast/cheap Haiku
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
    ):
        """Initializes the Anthropic client.

        Requires the 'anthropic' package to be installed (`pip install anthropic`).

        Note: Anthropic context limits are currently hardcoded as the API/SDK
        does not provide a standard way to fetch them dynamically.

        Raises:
            LLMConfigurationError: If API key is missing or client initialization fails.
        """
        super().__init__(model=model, max_tokens=max_tokens)
        try:
            import anthropic

            self.anthropic = anthropic
        except ImportError as e:
            raise LLMConfigurationError(
                "Anthropic Python package not found. Please install it using 'pip install anthropic'."
            ) from e
        self.logger = logger
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "Anthropic API key not provided. Set the ANTHROPIC_API_KEY environment variable or pass it during initialization."
            )

        # self.model = model # This is now self._model, set by superclass
        self.temperature = temperature
        # Get context limit from hardcoded dict
        self._context_limit = (
            ANTHROPIC_CONTEXT_LIMITS.get(self._model, DEFAULT_ANTHROPIC_CONTEXT_LIMIT)
            or DEFAULT_ANTHROPIC_CONTEXT_LIMIT
        )
        if self._model not in ANTHROPIC_CONTEXT_LIMITS:
            self.logger.warning(
                f"Context limit not defined for Anthropic model '{self._model}'. Using default: {self._context_limit}"
            )

        try:
            self.client = self.anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            raise LLMConfigurationError(f"Failed to initialize Anthropic client: {e}") from e

        # Check max_tokens against the known limit
        if self.max_tokens > self._context_limit:
            self.logger.warning(
                f"Requested max_tokens ({self.max_tokens}) exceeds the known context limit ({self._context_limit}) for model '{self._model}'. API calls might fail."
            )
        elif self.max_tokens > self._context_limit * 0.8:
            self.logger.info(
                f"Requested max_tokens ({self.max_tokens}) is close to the context limit ({self._context_limit}). Ensure input + output fits."
            )
        else:
            self.logger.warning("No context limit defined for Anthropic API.")

        self._system_message = system_message
        self._role_name = role_name or "Assistant"
        # Conversation object doesn't need the system message initially for Anthropic
        self.conversation = Conversation()
        self.logger.info(
            f"AnthropicLLM initialized. Model: {self._model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Context Limit: {self._context_limit} (Hardcoded)"
        )

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Calls the Anthropic Messages API."""

        # --- System Message Consolidation ---
        # Anthropic API expects a single system parameter, so we need to consolidate
        # system messages from multiple sources with proper prioritization
        system_messages = []

        # 1. Collect system messages from the conversation history (in order)
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_messages.append(msg["content"].strip())

        # 2. Add the interface's system message (highest priority - goes last)
        if self._system_message:
            system_messages.append(self._system_message.strip())

        # 3. Consolidate into a single system prompt
        if system_messages:
            # Remove duplicates while preserving order
            unique_system_messages = []
            seen = set()
            for msg in system_messages:
                if msg and msg not in seen:
                    unique_system_messages.append(msg)
                    seen.add(msg)

            # Join with double newlines for clear separation
            consolidated_system_prompt = "\n\n".join(unique_system_messages)
            self.logger.debug(
                f"Consolidated {len(unique_system_messages)} unique system messages for Anthropic API."
            )
        else:
            consolidated_system_prompt = None
            self.logger.debug("No system messages found for Anthropic API call.")

        # --- Filter and validate non-system messages ---
        filtered_messages: List[Dict[str, str]] = []
        approx_input_tokens = 0  # Estimate input tokens

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                # System messages are now handled above, skip them here
                continue
            elif role in ["user", "assistant"]:
                if content is None:
                    self.logger.warning(
                        f"Skipping message with role '{role}' because content is None."
                    )
                    continue
                filtered_messages.append({"role": role, "content": content})
                approx_input_tokens += self.count_tokens(content)  # Add to token estimate
            else:
                self.logger.warning(
                    f"Skipping message with unsupported role '{role}' for Anthropic API."
                )

        if not filtered_messages:
            raise LLMClientError(
                "Cannot call Anthropic API with an empty message list (after filtering)."
            )

        # Add system prompt tokens to estimate if present
        if consolidated_system_prompt:
            approx_input_tokens += self.count_tokens(consolidated_system_prompt)

        # Check estimated tokens against limit
        if approx_input_tokens + self.max_tokens > self._context_limit:
            self.logger.warning(
                f"Estimated input tokens (~{approx_input_tokens}) + max_tokens ({self.max_tokens}) exceeds context limit ({self._context_limit}). API call might fail."
            )
        elif approx_input_tokens > self._context_limit * 0.95:
            self.logger.info(
                f"Estimated input tokens (~{approx_input_tokens}) is very close to context limit ({self._context_limit})."
            )

        # --- Anthropic API Constraints Validation ---
        if filtered_messages[0]["role"] != "user":
            self.logger.error(
                f"Anthropic message list must start with 'user' role. First role found: '{filtered_messages[0]['role']}'."
            )
            # Attempt to fix by removing leading non-user messages? Or just error out?
            # Let's error out for now to be explicit.
            raise LLMClientError("Anthropic requires messages to start with the 'user' role.")

        # Check for alternating roles (simple check)
        for i in range(len(filtered_messages) - 1):
            if filtered_messages[i]["role"] == filtered_messages[i + 1]["role"]:
                self.logger.warning(
                    f"Consecutive messages with role '{filtered_messages[i]['role']}' found at index {i}. Anthropic API requires alternating roles."
                )
                # Consider raising LLMClientError here if strict adherence is needed.

        self.logger.debug(
            f"Calling Anthropic API ({self._model}) with {len(filtered_messages)} messages. "
            f"System prompt: {'Yes' if consolidated_system_prompt else 'No'} "
            f"(Length: {len(consolidated_system_prompt) if consolidated_system_prompt else 0} chars)"
        )

        try:
            # Prepare the API call parameters
            api_params = {
                "model": self._model,
                "messages": filtered_messages,  # type: ignore
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            # Only include system parameter if we have a consolidated system prompt
            # This avoids sending empty strings which might not be optimal
            if consolidated_system_prompt:
                api_params["system"] = consolidated_system_prompt

            response = self.client.messages.create(**api_params)

            # --- Response Validation ---
            if not response.content:
                stop_reason = getattr(response, "stop_reason", "Unknown")
                self.logger.error(
                    f"Anthropic response missing 'content'. Stop Reason: {stop_reason}. Response: {response}"
                )
                # Check if stop reason indicates max tokens
                if stop_reason == "max_tokens":
                    raise LLMResponseError(
                        f"Anthropic response likely stopped due to reaching max_tokens ({self.max_tokens}) or context limit ({self._context_limit})."
                    )
                else:
                    raise LLMResponseError(
                        f"Invalid response structure from Anthropic: No 'content'. Stop Reason: {stop_reason}."
                    )

            # Content is a list of blocks, usually one text block for simple chat
            if len(response.content) == 0:
                self.logger.error(
                    f"Anthropic response 'content' is not a non-empty list. Type: {type(response.content)}. Response: {response}"
                )
                raise LLMResponseError(
                    f"Invalid response structure from Anthropic: 'content' is not a non-empty list."
                )

            # Extract text from the first text block if available
            content_text: Optional[str] = None
            if hasattr(response.content[0], "type") and (response.content[0].type == "text"):
                content_text = response.content[0].text

            if content_text is None:
                stop_reason = getattr(response, "stop_reason", "Unknown")
                self.logger.error(
                    f"Anthropic response content block missing 'text'. Stop Reason: {stop_reason}. Response: {response}"
                )
                # Check stop reason for clues
                if stop_reason == "max_tokens":
                    raise LLMResponseError(
                        f"Anthropic response truncated due to max_tokens ({self.max_tokens}) or context limit ({self._context_limit})."
                    )
                # Add other specific stop reason checks if needed (e.g., 'tool_use' if tools were involved)
                else:
                    raise LLMResponseError(
                        f"Anthropic response content block missing 'text'. Stop Reason: {stop_reason}."
                    )

            content = content_text.strip()
            stop_reason = getattr(response, "stop_reason", "Unknown")
            self.logger.debug(
                f"Received content from Anthropic API. Length: {len(content)}. Stop Reason: {stop_reason}"
            )

            if not content and stop_reason == "stop_sequence":
                self.logger.warning(
                    f"Received empty content string from Anthropic API, but stop reason was 'stop_sequence'. Response: {response}"
                )
                # Similar to OpenAI, decide if this is valid or an error. Return empty for now.

            # Check for error stop reasons even if content exists (e.g., partial output before error)
            if stop_reason == "error":
                self.logger.error(
                    f"Anthropic response indicates an error stop reason. Response: {response}"
                )
                # Potentially check if error type indicates overload_error (related to context)
                error_type = getattr(response, "error", {}).get("type")
                if error_type == "overload_error":
                    raise LLMResponseError(
                        "Anthropic API Error: Overloaded, potentially due to exceeding context limits."
                    )
                else:
                    raise LLMResponseError(
                        f"Anthropic API reported an error during generation (Type: {error_type})."
                    )

            return content

        # --- Error Handling ---
        except self.anthropic.APIConnectionError as e:
            self.logger.error(f"Anthropic API connection error: {e}")
            raise LLMClientError(f"Network error connecting to Anthropic API: {e}") from e
        except self.anthropic.RateLimitError as e:
            self.logger.error(f"Anthropic API rate limit exceeded: {e}")
            raise LLMResponseError(
                f"Anthropic API rate limit exceeded. Check your plan and usage limits. Error: {e}"
            ) from e
        except self.anthropic.AuthenticationError as e:
            self.logger.error(f"Anthropic API authentication error: {e}")
            raise LLMConfigurationError(
                f"Anthropic API authentication failed. Check your API key. Error: {e}"
            ) from e
        except self.anthropic.PermissionDeniedError as e:
            self.logger.error(f"Anthropic API permission denied: {e}")
            raise LLMConfigurationError(
                f"Anthropic API permission denied. Check API key permissions. Error: {e}"
            ) from e
        except self.anthropic.NotFoundError as e:
            self.logger.error(f"Anthropic API resource not found (e.g., model): {e}")
            raise LLMConfigurationError(
                f"Anthropic API resource not found (check model name '{self._model}'?). Error: {e}"
            ) from e
        except (
            self.anthropic.BadRequestError
        ) as e:  # Covers invalid request structure, role issues etc.
            self.logger.error(f"Anthropic API bad request error: {e}")
            # Try to get more details from the error if possible
            error_detail = str(e)
            error_type = None
            if hasattr(e, "body") and isinstance(e.body, Dict):
                if "error" in e.body:  # type: ignore
                    error_type = e.body["error"].get("type")  # type: ignore
                if "message" in e.body["error"]:  # type: ignore
                    error_detail = e.body["error"]["message"]  # type: ignore

            # Check if the error indicates context length issues
            if error_type == "invalid_request_error" and (
                "prompt is too long" in error_detail or "exceeds the context window" in error_detail
            ):
                raise LLMResponseError(
                    f"Anthropic API Error: Request likely exceeded context limit ({self._context_limit}). Details: {error_detail}"
                ) from e
            elif error_type == "overload_error":  # Sometimes context issues manifest as overload
                raise LLMResponseError(
                    f"Anthropic API Error: Overloaded, potentially due to context limit ({self._context_limit}). Details: {error_detail}"
                ) from e
            else:
                raise LLMResponseError(
                    f"Anthropic API reported a bad request (check message format/roles?). Type: {error_type}. Error: {error_detail}"
                ) from e
        except self.anthropic.APIStatusError as e:  # Catch other 4xx/5xx
            self.logger.error(f"Anthropic API status error: {e.status_code} - {e.response}")
            error_detail = str(e)
            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", str(e))
            except:
                pass
            raise LLMResponseError(
                f"Anthropic API returned an error (Status {e.status_code}): {error_detail}"
            ) from e
        # Catch other potential Anthropic client errors
        except self.anthropic.AnthropicError as e:
            self.logger.error(f"An unexpected Anthropic client error occurred: {e}")
            raise LLMClientError(f"An unexpected Anthropic client error occurred: {e}") from e
        # --- Allow specific LLMResponseErrors from validation to pass through ---
        except LLMResponseError as e:
            # This catches errors raised explicitly in the validation logic above
            # (e.g., missing content)
            self.logger.error(f"LLM Response Error: {e}")  # Log it, but let it propagate
            raise
        # General exception catcher
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during Anthropic API call: {e}")
            raise LLMClientError(f"Unexpected error during Anthropic API call: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using the Anthropic SDK."""
        fallback_reason = None
        try:
            # Use the client instance's count_tokens method (now GA, not beta)
            messages: List[Dict[str, str]] = [{"role": "user", "content": text}]
            token_count_response = self.client.messages.count_tokens(messages=messages, model=self._model)  # type: ignore
            # No debug log here to avoid noise, success is implicit by returning
            return token_count_response.input_tokens
        except AttributeError:
            self.logger.warning(
                "Anthropic client does not have 'count_tokens' method (older SDK version?). Falling back to approximation."
            )
            fallback_reason = "Anthropic SDK 'count_tokens' method not found (old version?)"
        # Handling specific Anthropic API errors that might occur during token counting
        except self.anthropic.AuthenticationError as e:
            self.logger.error(
                f"Anthropic API authentication error during token count: {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic API authentication error: {e}"
        except self.anthropic.PermissionDeniedError as e:
            self.logger.error(
                f"Anthropic API permission denied during token count: {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic API permission error: {e}"
        except self.anthropic.RateLimitError as e:
            self.logger.warning(
                f"Anthropic API rate limit hit during token count: {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic API rate limit: {e}"
        except self.anthropic.APIConnectionError as e:
            self.logger.error(
                f"Anthropic API connection error during token count: {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic API connection error: {e}"
        except self.anthropic.BadRequestError as e:  # E.g. invalid model for token counting
            self.logger.error(
                f"Anthropic API bad request during token count (e.g. invalid model for count_tokens): {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic API bad request: {e}"
        except self.anthropic.AnthropicError as e:  # Catch-all for other Anthropic specific errors
            self.logger.error(
                f"Anthropic SDK error during token count: {e}. Falling back to approximation."
            )
            fallback_reason = f"Anthropic SDK error: {e}"
        except Exception as e:
            self.logger.error(
                f"Unexpected error counting tokens with Anthropic SDK: {e}. Falling back to approximation."
            )
            fallback_reason = f"Unexpected error with Anthropic SDK: {e}"

        # Fallback approximation
        estimated_tokens = len(text) // 4
        self.logger.warning(
            f"Using approximate token count ({estimated_tokens}) for Anthropic due to: {fallback_reason}. "
            f"Ensure 'anthropic' package is up-to-date and API key/model are correct."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) for the configured Anthropic model.

        Note: These limits are currently hardcoded as the Anthropic API/SDK
        does not provide a standard way to fetch them dynamically.
        """
        limit = self._context_limit  # Use the value set in __init__
        self.logger.debug(f"Returning context limit for {self._model}: {limit} (Hardcoded)")
        return limit

    # --- Anthropic-Specific Provider Methods ---

    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Returns Anthropic-specific capabilities."""
        base_capabilities = super().get_provider_capabilities()
        base_capabilities.update(
            {
                "supports_system_parameter": True,  # Anthropic uses system parameter
                "supports_streaming": True,  # Anthropic supports streaming
                "supported_models": self.SUPPORTED_MODELS,
                "context_limits": ANTHROPIC_CONTEXT_LIMITS,
                "requires_alternating_roles": True,  # Anthropic requires user/assistant alternation
                "supports_message_consolidation": True,  # We handle system message consolidation
                "supports_native_token_counting": True,  # Anthropic has native token counting (GA since Dec 2024)
            }
        )
        return base_capabilities

    def get_provider_settings(self) -> Dict[str, Any]:
        """Returns current Anthropic-specific settings."""
        settings = super().get_provider_settings()
        settings.update(
            {
                "api_key_set": bool(self.api_key),
                "context_limit": self._context_limit,
                "context_limit_source": "hardcoded",  # Anthropic limits are hardcoded
            }
        )
        return settings

    def set_provider_setting(self, setting_name: str, value: Any) -> bool:
        """Sets Anthropic-specific settings."""
        if super().set_provider_setting(setting_name, value):
            return True

        # Anthropic-specific settings
        if setting_name == "api_key":
            self.api_key = value
            # Re-initialize client with new API key
            try:
                self.client = self.anthropic.Anthropic(api_key=self.api_key)
                return True
            except Exception as e:
                self.logger.error(f"Failed to update Anthropic client with new API key: {e}")
                return False

        return False

    def supports_feature(self, feature_name: str) -> bool:
        """Checks Anthropic-specific feature support."""
        if super().supports_feature(feature_name):
            return True

        anthropic_features = {
            "system_parameter",
            "streaming",
            "message_consolidation",
            "role_alternation_enforcement",
            "native_token_counting",  # Anthropic has native token counting (GA since Dec 2024)
        }

        return feature_name in anthropic_features

    def get_message_consolidation_info(self) -> Dict[str, Any]:
        """
        Returns information about how system messages are consolidated.
        Anthropic-specific method.

        Returns:
            Dictionary with consolidation information.
        """
        return {
            "consolidates_system_messages": True,
            "deduplication": True,
            "order_preservation": True,
            "interface_message_priority": "highest",
            "separator": "\n\n",
        }
