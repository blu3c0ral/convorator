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


GEMINI_CONTEXT_LIMITS: Dict[str, Optional[int]] = {
    # Values primarily represent input token limits from official Google AI Studio / Vertex AI docs.
    # Dynamic fetching will be preferred, these are fallbacks.
    # Gemini 1.5 Series (Input Token Limits)
    "gemini-1.5-pro-latest": 1048576,  # Alias, typically 1M input
    "gemini-1.5-pro": 1048576,  # Specific model, often 1M input (can be up to 2M total)
    "gemini-1.5-flash-latest": 1048576,  # Alias, 1M input
    "gemini-1.5-flash": 1048576,  # Specific model, 1M input
    "gemini-1.5-flash-8b": 1048576,
    # Gemini 1.0 Series (Input Token Limits)
    "gemini-1.0-pro": 30720,  # Total context often cited as 32k, input is 30720
    "gemini-1.0-pro-001": 30720,
    "gemini-1.0-pro-vision": 12288,  # Input for vision version (total 16384) - this is model, not API endpoint
    # Newer Gemini model variants (often with large context windows, focusing on INPUT limits)
    # These are from ai.google.dev/gemini-api/docs/models page primarily for Gemini API (genai SDK)
    "gemini-2.5-flash-preview-05-20": 1048576,  # Input token limit
    "gemini-2.5-pro-preview-05-06": 1048576,  # Input token limit
    "gemini-2.0-flash": 1048576,  # Input token limit
    "gemini-2.0-flash-lite": 1048576,  # Input token limit
    # Vertex AI specific model names might differ slightly or have different listings.
    # For "models/gemini-1.0-pro-vision-001" on Vertex, input is 12288, output 4096.
    # For "models/gemini-1.0-pro-001" on Vertex, input is 30720, output 2048.
    # Normalizing model names (e.g., removing "models/") is handled in GeminiLLM.__init__
}
# Default for Gemini, often from older models or when dynamic fetch fails.
DEFAULT_GEMINI_CONTEXT_LIMIT = 30720  # Based on gemini-1.0-pro input limit.


class GeminiLLM(LLMInterface):
    """Concrete implementation for Google's Gemini API."""

    SUPPORTED_MODELS = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
    ]
    _input_token_limit: Optional[int] = None  # Stores dynamically fetched input token limit

    def __init__(
        self,
        logger: LoggerProtocol,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash-latest",  # Default to flash for speed/cost
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,  # Gemini uses max_output_tokens
        system_message: Optional[str] = None,
        role_name: Optional[str] = None,
    ):
        """Initializes the Google Gemini client.

        Requires the 'google-generativeai' package (`pip install google-generativeai`).
        Attempts to fetch the model's input/output token limits during initialization.

        Raises:
            LLMConfigurationError: If API key is missing, configuration fails, or client initialization fails.
        """
        super().__init__(model=model, max_tokens=max_tokens)
        try:
            import google.generativeai as genai

            # Import google API core exceptions for specific error handling
            import google.api_core.exceptions as google_exceptions

            self.genai = genai
            self.google_exceptions = google_exceptions
        except ImportError as e:
            raise LLMConfigurationError(
                "Google Generative AI package not found. Please install it using 'pip install google-generativeai'."
            ) from e

        self.logger = logger

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "Google API key not provided. Set the GOOGLE_API_KEY environment variable or pass it during initialization."
            )

        try:
            # Configure the API key globally for the genai module
            self.genai.configure(api_key=self.api_key)  # type: ignore
            self.logger.info("Google Generative AI SDK configured successfully.")
        except Exception as e:
            # Catch potential issues during configure()
            raise LLMConfigurationError(f"Failed to configure Google API key: {e}") from e

        # Normalize model name (remove 'models/' prefix if present)
        # This logic should ideally be in the factory or before calling __init__ if it affects all models.
        # For now, keeping it here, but it means self._model might differ from the 'model' param if 'models/' was stripped.
        # A better approach: LLMInterface.__init__ could store the raw model string, and a property could provide normalized name if needed.
        # Or, normalization happens *before* calling the constructor.
        # Given current structure, self._model from super() will be the potentially un-normalized one.
        # Let's re-assign self._model if normalization occurs here. This makes self._model consistent.
        current_model_name = self._model  # From super().__init__
        if current_model_name.startswith("models/"):
            normalized_model_name = current_model_name.split("/", 1)[1]
            self._model = normalized_model_name  # Update self._model to the normalized version
            self.logger.debug(
                f"Normalized Gemini model name from '{current_model_name}' to '{self._model}'"
            )

        if self._model not in self.SUPPORTED_MODELS:
            self.logger.warning(
                f"Model '{self._model}' is not in the explicitly supported list for GeminiLLM ({self.SUPPORTED_MODELS}). Proceeding, but compatibility is not guaranteed."
            )
        # self.model_name = model # This is now self._model, normalized above
        self.temperature = temperature

        try:
            # Define generation configuration
            self.generation_config = self.genai.types.GenerationConfig(
                # candidate_count=1 # Default is 1, usually no need to change
                # stop_sequences=... # Optional stop sequences
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                # top_p=... # Optional nucleus sampling
                # top_k=... # Optional top-k sampling
            )
        except AttributeError as e:
            # Fallback if self.genai.types.GenerationConfig is not found, try self.genai.GenerationConfig
            self.logger.warning(
                f"Accessing GenerationConfig via self.genai.types.GenerationConfig failed ({e}), trying self.genai.GenerationConfig."
            )
            try:
                self.generation_config = self.genai.GenerationConfig(  # type: ignore
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as e_inner:
                raise LLMConfigurationError(
                    f"Failed to create Gemini GenerationConfig: {e_inner}"
                ) from e_inner
        except Exception as e:
            # Catch potential errors creating GenerationConfig (e.g., invalid values)
            raise LLMConfigurationError(f"Failed to create Gemini GenerationConfig: {e}") from e

        # Define safety settings (adjust as needed)
        # Defaults are generally BLOCK_MEDIUM_AND_ABOVE for most categories.
        # Setting to BLOCK_NONE disables safety filtering for that category (USE WITH CAUTION).
        # Access HarmCategory and HarmBlockThreshold via self.genai.types
        try:
            self.safety_settings: Dict[Any, Any] = {
                # Example: Relax harassment slightly (BLOCK_ONLY_HIGH)
                # self.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: self.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                # Example: Disable hate speech filter (BLOCK_NONE) - Use responsibly!
                # self.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            # Example of correctly accessing an enum if needed for validation or dynamic setup:
            # _ = self.genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
        except AttributeError as e:
            self.logger.warning(
                f"Could not access safety setting enums via self.genai.types (e.g., HarmCategory): {e}. Safety settings might not be configurable with these enums."
            )
            self.safety_settings = {}  # Fallback to empty if enums are not found as expected
        except Exception as e:
            self.logger.error(f"Unexpected error during safety_settings definition: {e}")
            self.safety_settings = {}

        if self.safety_settings:
            self.logger.warning(
                f"Using custom safety settings for Gemini: {self.safety_settings}. Be aware of the implications."
            )

        self._system_message = system_message
        self._role_name = role_name or "Model"  # Gemini uses 'model' role

        try:
            # Initialize the generative model instance
            # Pass system_instruction directly here
            self.generative_model = self.genai.GenerativeModel(
                model_name=self._model,  # Use the potentially normalized self._model
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self._system_message if self._system_message else None,
            )
            self.logger.info(f"Gemini GenerativeModel initialized for '{self._model}'.")
        except self.google_exceptions.NotFound as e:
            self.logger.error(f"Gemini model '{self._model}' not found or access denied: {e}")
            raise LLMConfigurationError(
                f"Gemini model '{self._model}' not found or access denied. Check model name and API key permissions. Error: {e}"
            ) from e
        except Exception as e:
            # Catch other potential errors during model initialization
            self.logger.exception(
                f"Failed to initialize Gemini GenerativeModel '{self._model}': {e}"
            )
            raise LLMConfigurationError(
                f"Failed to initialize Gemini GenerativeModel '{self._model}': {e}"
            ) from e

        # Attempt to dynamically fetch the input token limit for the model
        try:
            model_info = self.genai.get_model(
                f"models/{self._model}"
            )  # Use the normalized self._model
            if hasattr(model_info, "input_token_limit") and model_info.input_token_limit:
                self._input_token_limit = model_info.input_token_limit
                self.logger.info(
                    f"Dynamically fetched input token limit for Gemini model '{self._model}': {self._input_token_limit}"
                )
            else:
                self.logger.warning(
                    f"Could not dynamically fetch 'input_token_limit' for Gemini model '{self._model}'. Will rely on hardcoded values or default."
                )
        except Exception as e_fetch_limit:
            self.logger.warning(
                f"Failed to dynamically fetch model info for Gemini model '{self._model}' to get input_token_limit: {e_fetch_limit}. Will use fallbacks."
            )
            # self._input_token_limit remains None

        # Conversation state for Gemini
        # Internal conversation uses standard roles ('user', 'assistant')
        self.conversation = Conversation()
        # Actual chat session with Gemini API (uses 'user', 'model' roles)
        self.chat: Optional[genai.ChatSession] = None  # Initialize chat session lazily
        self.logger.info(
            f"GeminiLLM initialized. Model: {self._model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}"
        )

    def _translate_roles_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Translates roles from internal standard ('system', 'user', 'assistant')
        to Gemini's required format ('user', 'model') within a 'contents' list.
        System message is handled by model initialization or system_instruction,
        so 'system' roles in the message list are typically skipped.

        Gemini API strictly requires alternating user/model roles. This method
        handles role alternation violations by merging consecutive messages
        with the same role into a single message.
        """
        gemini_history = []
        valid_roles = {"user": "user", "assistant": "model"}

        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")

            if content is None:  # Skip messages without content
                self.logger.warning(
                    f"Skipping message with role '{role}' at index {i} due to None content."
                )
                continue

            if role == "system":
                if i == 0:
                    # Skip initial system message as it's handled by system_instruction
                    self.logger.debug(
                        "Skipping initial system message in history (handled by system_instruction)."
                    )
                    continue
                else:
                    # System messages mid-conversation are not directly supported.
                    # Log a warning and skip. Consider merging with next user message if needed.
                    self.logger.warning(
                        f"System message found mid-conversation at index {i}; skipping for Gemini API call."
                    )
                    continue

            translated_role = valid_roles.get(role)
            if not translated_role:
                self.logger.warning(
                    f"Unsupported role '{role}' encountered at index {i} for Gemini. Skipping message."
                )
                continue

            # --- Role Alternation Handling ---
            # If the history is not empty, check if the current role matches the last one
            if gemini_history and gemini_history[-1]["role"] == translated_role:
                self.logger.info(
                    f"Merging consecutive '{translated_role}' messages for Gemini API compliance. "
                    f"Original message at index {i}: '{content[:50]}...'"
                )
                # Merge with the previous message by concatenating content
                previous_content = gemini_history[-1]["parts"][0]["text"]
                merged_content = f"{previous_content}\n\n{content}"
                gemini_history[-1]["parts"][0]["text"] = merged_content
                self.logger.debug(f"Merged content length: {len(merged_content)} characters")
                continue  # Skip adding as a separate message

            # --- Append to Gemini History ---
            # Gemini expects content in the format: {'role': ..., 'parts': [{'text': ...}]}
            gemini_history.append({"role": translated_role, "parts": [{"text": content}]})

        # --- Final Validation ---
        if gemini_history and gemini_history[0]["role"] != "user":
            self.logger.warning(
                "Gemini history (after translation) does not start with 'user' role. "
                f"First role is '{gemini_history[0]['role']}'. This may cause API errors."
            )
            # For better error handling, we could insert a dummy user message or raise an error
            # For now, we'll let the API handle it and provide a clear error message

        # Log final statistics
        if gemini_history:
            role_sequence = [msg["role"] for msg in gemini_history]
            self.logger.debug(
                f"Final Gemini history: {len(gemini_history)} messages with role sequence: {role_sequence}"
            )

            # Verify alternation in the final result
            for i in range(1, len(gemini_history)):
                if gemini_history[i]["role"] == gemini_history[i - 1]["role"]:
                    # This should not happen with our merging logic, but let's be defensive
                    self.logger.error(
                        f"CRITICAL: Role alternation still violated after processing. "
                        f"Messages at indices {i-1} and {i} both have role '{gemini_history[i]['role']}'. "
                        f"This will cause Gemini API to fail."
                    )
                    # Import the exception at the top of the method to avoid circular imports
                    from convorator.exceptions import LLMClientError

                    raise LLMClientError(
                        f"Failed to resolve role alternation for Gemini API. "
                        f"Consecutive '{gemini_history[i]['role']}' roles remain after processing."
                    )

        return gemini_history

    def _handle_gemini_response(self, response, context: str = "generate_content") -> str:
        """
        Helper to process Gemini response object (from generate_content or chat.send_message),
        extract text content, and handle potential errors like blocking.

        Args:
            response: The response object from the Gemini API call.
            context: String indicating call context ('generate_content' or 'chat') for logging.

        Returns:
            The extracted text content.

        Raises:
            LLMResponseError: If the response is blocked, invalid, empty, or indicates an error.
        """
        self.logger.debug(f"Handling Gemini response from '{context}'.")

        # 1. Check for Prompt Feedback (blocking before generation)
        # Sometimes present even if candidates exist but generation was stopped early.
        if hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason.name  # Get the enum name
            safety_ratings_str = str(getattr(response.prompt_feedback, "safety_ratings", "N/A"))
            self.logger.error(
                f"Gemini prompt blocked in '{context}'. Reason: {block_reason}. Safety Ratings: {safety_ratings_str}. Response: {response}"
            )
            raise LLMResponseError(
                f"Gemini prompt blocked due to {block_reason}. Safety: {safety_ratings_str}"
            )

        # 2. Check for Candidates
        if not response.candidates:
            # This might happen if the prompt itself was blocked, or other issues.
            prompt_feedback_str = str(getattr(response, "prompt_feedback", "N/A"))
            self.logger.error(
                f"Gemini response in '{context}' has no candidates. Prompt Feedback: {prompt_feedback_str}. Response: {response}"
            )
            # Check if prompt feedback gives a reason
            if hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                raise LLMResponseError(
                    f"Gemini response has no candidates, likely due to prompt blocking. Reason: {block_reason}."
                )
            else:
                raise LLMResponseError(
                    f"Gemini response has no candidates. Prompt Feedback: {prompt_feedback_str}"
                )

        # 3. Access Content and Check Finish Reason/Safety (within the first candidate)
        candidate = response.candidates[0]
        content_text = None
        finish_reason = "UNKNOWN"  # Default
        safety_ratings_str = "N/A"  # Default

        try:
            # Finish reason and safety ratings are usually on the candidate
            finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            safety_ratings_str = str(getattr(candidate, "safety_ratings", "N/A"))

            # Content can be inside candidate.content.parts
            if candidate.content and candidate.content.parts:
                # Aggregate text from all parts (usually just one)
                content_text = "".join(
                    part.text for part in candidate.content.parts if hasattr(part, "text")
                )

            # Sometimes, the response object has a direct .text attribute (convenience)
            # Let's prefer the explicit parts extraction but use .text as fallback
            if content_text is None and hasattr(response, "text"):
                content_text = response.text
                self.logger.debug("Extracted content using response.text fallback.")

            if content_text is None:
                # If still no text, something is wrong
                self.logger.error(
                    f"Could not extract text content from Gemini candidate in '{context}'. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}. Candidate: {candidate}"
                )
                raise LLMResponseError(
                    f"Could not extract text content from Gemini response. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
                )

        except AttributeError as e:
            self.logger.error(
                f"AttributeError accessing Gemini response candidate details in '{context}': {e}. Response: {response}",
                exc_info=True,
            )
            raise LLMResponseError(f"Error accessing Gemini response structure: {e}") from e
        except ValueError as e:
            # This might occur if accessing .text fails due to blocking (though prompt_feedback is primary check)
            self.logger.error(
                f"ValueError accessing Gemini response text in '{context}', potentially due to blocking. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}. Error: {e}. Response: {response}",
                exc_info=True,
            )
            raise LLMResponseError(
                f"Gemini response blocked or invalid. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
            ) from e

        # 4. Post-extraction Checks (Finish Reason, Safety, Emptiness)
        content = content_text.strip()
        self.logger.debug(
            f"Received content from Gemini API ({context}). Length: {len(content)}. Finish Reason: {finish_reason}. Safety: {safety_ratings_str}"
        )

        # Check finish reason for potential issues even if some text exists
        if finish_reason == "SAFETY":
            self.logger.error(
                f"Gemini response flagged for SAFETY in '{context}'. Safety Ratings: {safety_ratings_str}. Content (may be partial): '{content[:100]}...'. Response: {response}"
            )
            raise LLMResponseError(
                f"Gemini response blocked or cut short due to SAFETY. Ratings: {safety_ratings_str}"
            )
        elif finish_reason == "RECITATION":
            self.logger.warning(
                f"Gemini response flagged for RECITATION in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )
            # Recitation might be acceptable depending on use case, but log it.
        elif finish_reason == "OTHER":
            self.logger.warning(
                f"Gemini response finished with OTHER reason in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )
            # May indicate unexpected issues.
        elif finish_reason not in [
            "STOP",
            "MAX_TOKENS",
            "UNKNOWN",
        ]:  # Check for unexpected valid reasons
            self.logger.warning(
                f"Gemini response finished with unexpected reason '{finish_reason}' in '{context}'. Content: '{content[:100]}...'. Response: {response}"
            )

        # Check for empty content after stripping
        if not content and finish_reason == "STOP":
            self.logger.warning(
                f"Received empty content string from Gemini API ({context}), but finish reason was 'STOP'. Response: {response}"
            )
            # Return empty string as it might be intentional.
        elif not content and finish_reason != "STOP":
            self.logger.error(
                f"Received empty content string from Gemini API ({context}) with finish reason '{finish_reason}'. Response: {response}"
            )
            raise LLMResponseError(
                f"Received empty content from Gemini API. Finish Reason: {finish_reason}"
            )

        return content

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Internal method for stateless Gemini API calls using generate_content.
        The public 'query' method handles choosing between this and stateful chat.
        """
        self.logger.debug(
            f"Calling Gemini API (stateless generate_content) for model '{self._model}'. Translating {len(messages)} messages."
        )
        gemini_history = self._translate_roles_for_gemini(messages)

        if not gemini_history:
            self.logger.error(
                "Cannot call Gemini API (generate_content) with empty history after role translation."
            )
            # Check if original messages only contained system messages
            if all(m.get("role") == "system" for m in messages):
                raise LLMClientError(
                    "Cannot call Gemini API (generate_content): Input contained only system messages."
                )
            else:
                raise LLMClientError(
                    "No valid messages found for Gemini API call (generate_content) after role translation."
                )

        self.logger.debug(
            f"Calling generate_content with {len(gemini_history)} translated messages."
        )
        try:
            # Use the initialized self.model instance (now self.generative_model)
            response = self.generative_model.generate_content(
                contents=gemini_history,
                # generation_config and safety_settings are part of self.generative_model
                stream=False,  # Use non-streaming for simple query interface
            )
            # Process response using the common handler
            return self._handle_gemini_response(response, context="generate_content")

        # --- Specific Google API Error Handling ---
        except self.google_exceptions.PermissionDenied as e:
            self.logger.error(f"Gemini API permission denied (generate_content): {e}")
            raise LLMConfigurationError(
                f"Gemini API permission denied. Check API key/permissions. Error: {e}"
            ) from e
        except self.google_exceptions.InvalidArgument as e:
            # Often indicates issues with the request structure (e.g., roles, content format)
            self.logger.error(f"Gemini API invalid argument (generate_content): {e}")
            raise LLMResponseError(
                f"Gemini API invalid argument (check message roles/format?). Error: {e}"
            ) from e
        except self.google_exceptions.ResourceExhausted as e:
            self.logger.error(
                f"Gemini API resource exhausted (generate_content - rate limit?): {e}"
            )
            raise LLMResponseError(
                f"Gemini API resource exhausted (likely rate limit). Error: {e}"
            ) from e
        except self.google_exceptions.NotFound as e:
            # Should be caught at init, but maybe model becomes unavailable later?
            self.logger.error(
                f"Gemini API resource not found (generate_content - model '{self._model}'?): {e}"
            )
            raise LLMConfigurationError(
                f"Gemini API resource not found (model '{self._model}'?). Error: {e}"
            ) from e
        except self.google_exceptions.InternalServerError as e:
            self.logger.error(f"Gemini API internal server error (generate_content): {e}")
            raise LLMResponseError(
                f"Gemini API reported an internal server error. Try again later. Error: {e}"
            ) from e
        except self.google_exceptions.GoogleAPIError as e:  # Catch-all for other Google API errors
            self.logger.error(f"Gemini API error (generate_content): {e}")
            raise LLMResponseError(f"Gemini API returned an error: {e}") from e
        # --- General Exceptions ---
        except Exception as e:
            self.logger.exception(
                f"An unexpected error occurred during Gemini API call (generate_content): {e}"
            )
            # Attempt to get a more specific message if available
            error_message = getattr(e, "message", str(e))
            raise LLMClientError(
                f"Unexpected error during Gemini API call (generate_content): {error_message}"
            ) from e

    def query(
        self,
        prompt: str,
        use_conversation: bool = True,
        conversation_history: Optional[
            List[Dict[str, str]]
        ] = None,  # Used only if use_conversation=False
    ) -> str:
        """
        Sends a prompt to the Gemini LLM.

        If use_conversation is True, it utilizes a stateful ChatSession, managing
        history internally (translating roles as needed).
        If use_conversation is False, it performs a stateless generate_content call
        using the logic from the base LLMInterface.query method.

        Args:
            prompt: The user's input prompt.
            use_conversation: If True, use stateful chat session. If False, use stateless call.
            conversation_history: Optional history for stateless calls (ignored if use_conversation=True).

        Returns:
            The LLM's response content as a string.

        Raises:
            LLMConfigurationError: For configuration issues.
            LLMClientError: For client-side issues.
            LLMResponseError: For API errors or problematic responses.
        """
        if use_conversation:
            # --- Stateful Chat Session ---
            self.logger.debug("Using stateful Gemini chat session.")

            # Initialize chat session if it doesn't exist
            if not self.chat:
                self.logger.info("Starting new Gemini chat session.")
                # Translate existing internal history (user/assistant) to Gemini format (user/model)
                # Use the current state of self.conversation
                initial_gemini_history = self._translate_roles_for_gemini(
                    self.conversation.get_messages()
                )
                self.logger.debug(
                    f"Initializing chat with {len(initial_gemini_history)} translated messages."
                )
                try:
                    # Start chat using the initialized self.model (which includes system instruction etc.)
                    # (now self.generative_model)
                    self.chat = self.generative_model.start_chat(history=initial_gemini_history)
                except self.google_exceptions.InvalidArgument as e:
                    self.logger.error(
                        f"Failed to start Gemini chat session due to invalid argument (check history format/roles?): {e}"
                    )
                    raise LLMResponseError(
                        f"Failed to start Gemini chat session (invalid history/roles?): {e}"
                    ) from e
                except Exception as e:
                    self.logger.exception(f"Failed to start Gemini chat session: {e}")
                    raise LLMClientError(f"Failed to start Gemini chat session: {e}") from e

            # --- Send Message via Chat ---
            user_message_added_to_internal = False
            try:
                self.logger.debug(f"Sending message to Gemini chat: '{prompt[:100]}...'")
                # Send the prompt using the chat session
                response = self.chat.send_message(prompt, stream=False)

                # --- Process response ---
                content = self._handle_gemini_response(response, context="chat")

                # --- Update Internal State (on success) ---
                # Add the user prompt that was successfully sent and processed.
                self.conversation.add_user_message(prompt)
                user_message_added_to_internal = True  # Mark success
                # Add the successful assistant/model response.
                self.conversation.add_assistant_message(content)
                self.logger.debug(
                    "Updated internal conversation history after successful chat message."
                )

                # Update the chat object's history (optional, but good practice if reusing chat object elsewhere)
                # Note: The google library might update chat.history automatically, but explicit sync can be safer
                # self.chat.history = self._translate_roles_for_gemini(self.conversation.get_messages())

                return content

            # --- Error Handling for Chat Session ---
            except (LLMClientError, LLMConfigurationError, LLMResponseError) as e:
                # These errors are already logged by _handle_gemini_response or raised directly
                # If the error happened *after* the user message was added internally (shouldn't happen often), log it.
                if user_message_added_to_internal:
                    self.logger.warning(
                        "Error occurred after user message was added to internal state but before assistant response - state might be inconsistent."
                    )
                    # Consider removing the user message here if necessary, though it indicates partial success followed by failure.
                # No need to pop user message here as it wasn't added on the failure path of send_message or _handle_gemini_response
                raise  # Re-raise the specific error
            except self.google_exceptions.PermissionDenied as e:
                self.logger.error(f"Gemini API chat permission denied: {e}")
                raise LLMConfigurationError(f"Gemini chat permission denied: {e}") from e
            except self.google_exceptions.InvalidArgument as e:
                self.logger.error(f"Gemini API chat invalid argument: {e}")
                raise LLMResponseError(f"Gemini chat invalid argument: {e}") from e
            except self.google_exceptions.ResourceExhausted as e:
                self.logger.error(f"Gemini API chat resource exhausted (rate limit?): {e}")
                raise LLMResponseError(f"Gemini chat resource exhausted: {e}") from e
            except self.google_exceptions.InternalServerError as e:
                self.logger.error(f"Gemini API internal server error (chat): {e}")
                raise LLMResponseError(
                    f"Gemini API reported an internal server error during chat. Try again later. Error: {e}"
                ) from e
            except (
                self.google_exceptions.GoogleAPIError
            ) as e:  # Catch-all Google API errors for chat
                self.logger.error(f"Gemini API chat error: {e}")
                raise LLMResponseError(f"Gemini chat API returned an error: {e}") from e
            except Exception as e:
                self.logger.exception(
                    f"An unexpected error occurred during Gemini chat session send_message: {e}"
                )
                error_message = getattr(e, "message", str(e))
                # Don't modify internal conversation state here, as failure happened during API call
                raise LLMClientError(
                    f"Unexpected error during Gemini chat session: {error_message}"
                ) from e

        else:
            # --- Stateless Call ---
            self.logger.debug(
                "Using stateless Gemini API call (generate_content via base class query)."
            )
            # Delegate to the base class query method, which will call our _call_api (stateless version)
            return super().query(
                prompt, use_conversation=False, conversation_history=conversation_history
            )

    def clear_conversation(self, keep_system: bool = True):
        """Clears the internal conversation history and resets the chat session."""
        super().clear_conversation(keep_system=keep_system)
        # Also reset the stateful chat session object
        if self.chat:
            self.logger.debug("Resetting Gemini chat session object.")
            self.chat = None

    def set_system_message(self, message: Optional[str]):
        """Sets the system message and re-initializes the underlying Gemini model if needed.

        This method ensures atomic updates - either the system message is successfully changed
        and the model re-initialized, or the original state is preserved on any failure.

        Args:
            message: The new system message. None to clear the system message.

        Raises:
            LLMConfigurationError: If model re-initialization fails due to configuration issues.
            LLMClientError: If model re-initialization fails due to client-side issues.
        """
        if message == self._system_message:
            self.logger.debug("System message unchanged for Gemini.")
            return

        # Store current state for potential rollback
        previous_system_message = self._system_message
        previous_generative_model = self.generative_model
        previous_chat = self.chat

        self.logger.info(
            f"System message changing for Gemini from '{previous_system_message}' to '{message}'. Re-initializing GenerativeModel."
        )

        try:
            # Step 1: Update the base class state first (this updates conversation object too)
            super().set_system_message(message)

            # Step 2: Create new GenerativeModel with updated system instruction
            new_generative_model = self.genai.GenerativeModel(
                model_name=self._model,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self._system_message if self._system_message else None,
            )

            # Step 3: Only update instance state after successful model creation
            self.generative_model = new_generative_model
            self.chat = None  # Reset chat session as context has changed

            self.logger.info(
                f"Gemini GenerativeModel successfully re-initialized with new system instruction."
            )

        except self.google_exceptions.PermissionDenied as e:
            self.logger.error(f"Permission denied during Gemini model re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Permission denied during Gemini model re-initialization. Check API key permissions. Error: {e}"
            ) from e

        except self.google_exceptions.NotFound as e:
            self.logger.error(f"Model not found during Gemini re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Gemini model '{self._model}' not found during re-initialization. Check model name. Error: {e}"
            ) from e

        except self.google_exceptions.InvalidArgument as e:
            self.logger.error(f"Invalid argument during Gemini model re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Invalid configuration during Gemini model re-initialization. Check system message format or model parameters. Error: {e}"
            ) from e

        except self.google_exceptions.ResourceExhausted as e:
            self.logger.error(f"Resource exhausted during Gemini model re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Resource exhausted during Gemini model re-initialization (rate limit?). Try again later. Error: {e}"
            ) from e

        except self.google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Google API error during Gemini model re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Google API error during Gemini model re-initialization: {e}"
            ) from e

        except Exception as e:
            self.logger.exception(f"Unexpected error during Gemini model re-initialization: {e}")
            self._rollback_system_message_state(
                previous_system_message, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Unexpected error during Gemini model re-initialization: {e}"
            ) from e

    def _rollback_system_message_state(
        self,
        previous_system_message: Optional[str],
        previous_generative_model: Any,
        previous_chat: Optional[Any],
    ) -> None:
        """
        Rolls back the Gemini client state to the previous working configuration.

        This method is called when system message re-initialization fails to ensure
        the client remains in a consistent, working state.

        Args:
            previous_system_message: The previous system message to restore.
            previous_generative_model: The previous GenerativeModel instance to restore.
            previous_chat: The previous chat session to restore.
        """
        try:
            self.logger.warning("Rolling back Gemini state due to re-initialization failure.")

            # Restore the system message in the base class
            # We need to directly update the internal state to avoid triggering another re-initialization
            self._system_message = previous_system_message

            # Update the conversation object to match
            if hasattr(self, "conversation") and self.conversation:
                if previous_system_message:
                    self.conversation.add_message(role="system", content=previous_system_message)
                else:
                    # Remove system message from conversation if rolling back to None
                    if (
                        self.conversation.messages
                        and self.conversation.messages[0].role == "system"
                    ):
                        self.conversation.messages.pop(0)
                    self.conversation.system_message = None

            # Restore the GenerativeModel and chat session
            self.generative_model = previous_generative_model
            self.chat = previous_chat

            self.logger.info(
                "Successfully rolled back Gemini state to previous working configuration."
            )

        except Exception as rollback_error:
            self.logger.critical(
                f"CRITICAL: Failed to rollback Gemini state after re-initialization failure. "
                f"Client may be in an inconsistent state. Rollback error: {rollback_error}"
            )
            # Don't raise here as we're already in an error handling path
            # The original exception will be raised by the caller

    def set_safety_settings(self, safety_settings: Optional[Dict[Any, Any]]):
        """
        Sets or updates the safety settings for the Gemini model.
        This will re-initialize the underlying GenerativeModel and reset any active chat session.

        This method ensures atomic updates - either the safety settings are successfully changed
        and the model re-initialized, or the original state is preserved on any failure.

        Args:
            safety_settings: A dictionary mapping HarmCategory enums to HarmBlockThreshold enums.
                             If None, it defaults to an empty dictionary (provider defaults).

        Raises:
            LLMConfigurationError: If model re-initialization fails due to configuration issues.
            LLMClientError: If model re-initialization fails due to client-side issues.
        """
        new_settings = safety_settings if safety_settings is not None else {}

        if new_settings == self.safety_settings:
            self.logger.debug(
                f"Safety settings unchanged for Gemini. Current: {self.safety_settings}"
            )
            return

        # Store current state for potential rollback
        previous_safety_settings = self.safety_settings
        previous_generative_model = self.generative_model
        previous_chat = self.chat

        self.logger.info(
            f"Safety settings changing for Gemini from {previous_safety_settings} to {new_settings}. Re-initializing GenerativeModel."
        )

        try:
            # Step 1: Update the safety settings
            self.safety_settings = new_settings

            # Step 2: Create new GenerativeModel with updated safety settings
            new_generative_model = self.genai.GenerativeModel(
                model_name=self._model,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self._system_message if self._system_message else None,
            )

            # Step 3: Only update instance state after successful model creation
            self.generative_model = new_generative_model
            self.chat = None  # Reset chat session as context has changed

            self.logger.info(
                f"Gemini GenerativeModel successfully re-initialized with new safety settings: {self.safety_settings}"
            )

        except self.google_exceptions.PermissionDenied as e:
            self.logger.error(
                f"Permission denied during Gemini model re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Permission denied during Gemini model re-initialization. Check API key permissions. Error: {e}"
            ) from e

        except self.google_exceptions.NotFound as e:
            self.logger.error(
                f"Model not found during Gemini re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Gemini model '{self._model}' not found during re-initialization. Check model name. Error: {e}"
            ) from e

        except self.google_exceptions.InvalidArgument as e:
            self.logger.error(
                f"Invalid argument during Gemini model re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMConfigurationError(
                f"Invalid safety settings during Gemini model re-initialization. Check safety setting format or values. Error: {e}"
            ) from e

        except self.google_exceptions.ResourceExhausted as e:
            self.logger.error(
                f"Resource exhausted during Gemini model re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Resource exhausted during Gemini model re-initialization (rate limit?). Try again later. Error: {e}"
            ) from e

        except self.google_exceptions.GoogleAPIError as e:
            self.logger.error(
                f"Google API error during Gemini model re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Google API error during Gemini model re-initialization: {e}"
            ) from e

        except Exception as e:
            self.logger.exception(
                f"Unexpected error during Gemini model re-initialization (safety settings): {e}"
            )
            self._rollback_safety_settings_state(
                previous_safety_settings, previous_generative_model, previous_chat
            )
            raise LLMClientError(
                f"Unexpected error during Gemini model re-initialization: {e}"
            ) from e

    def _rollback_safety_settings_state(
        self,
        previous_safety_settings: Dict[Any, Any],
        previous_generative_model: Any,
        previous_chat: Optional[Any],
    ) -> None:
        """
        Rolls back the Gemini client state to the previous working configuration after safety settings failure.

        Args:
            previous_safety_settings: The previous safety settings to restore.
            previous_generative_model: The previous GenerativeModel instance to restore.
            previous_chat: The previous chat session to restore.
        """
        try:
            self.logger.warning(
                "Rolling back Gemini safety settings due to re-initialization failure."
            )

            # Restore the safety settings
            self.safety_settings = previous_safety_settings

            # Restore the GenerativeModel and chat session
            self.generative_model = previous_generative_model
            self.chat = previous_chat

            self.logger.info(
                "Successfully rolled back Gemini safety settings to previous working configuration."
            )

        except Exception as rollback_error:
            self.logger.critical(
                f"CRITICAL: Failed to rollback Gemini safety settings after re-initialization failure. "
                f"Client may be in an inconsistent state. Rollback error: {rollback_error}"
            )
            # Don't raise here as we're already in an error handling path

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens using the Gemini SDK (model.count_tokens).

        Falls back to approximation if the model is not properly initialized or
        if the API call fails.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        fallback_reason = None

        if not hasattr(self, "generative_model") or self.generative_model is None:
            self.logger.warning("Gemini model not initialized. Cannot count tokens using SDK.")
            fallback_reason = "Gemini model not initialized"
            # Fall through to approximation directly without trying API call
        else:
            try:
                # Use the model instance's count_tokens method
                count_response = self.generative_model.count_tokens(text)
                token_count = count_response.total_tokens
                self.logger.debug(
                    f"Counted {token_count} tokens for text (length {len(text)}) using Gemini SDK."
                )
                return token_count

            except (
                AttributeError
            ) as e:  # Should not happen if generative_model is set, but defensive
                self.logger.warning(
                    f"Gemini model object missing 'count_tokens' method: {e}. SDK/init issue? Falling back."
                )
                fallback_reason = f"Gemini SDK 'count_tokens' method missing: {e}"
            except self.google_exceptions.PermissionDenied as e:
                self.logger.error(
                    f"Gemini API permission denied during token count: {e}. Falling back."
                )
                fallback_reason = f"Gemini API permission denied: {e}"
            except self.google_exceptions.InvalidArgument as e:
                self.logger.error(
                    f"Gemini API invalid argument during token count: {e}. Falling back."
                )
                fallback_reason = f"Gemini API invalid argument: {e}"
            except self.google_exceptions.ResourceExhausted as e:  # E.g. Rate limit
                self.logger.warning(
                    f"Gemini API resource exhausted during token count: {e}. Falling back."
                )
                fallback_reason = f"Gemini API resource exhausted (rate limit?): {e}"
            except self.google_exceptions.NotFound as e:  # E.g. model not found for tokenization
                self.logger.error(
                    f"Gemini API resource not found during token count (model issue?): {e}. Falling back."
                )
                fallback_reason = f"Gemini API resource not found: {e}"
            except self.google_exceptions.InternalServerError as e:
                self.logger.error(
                    f"Gemini API internal server error during token count: {e}. Falling back."
                )
                fallback_reason = f"Gemini API internal server error: {e}"
            except (
                self.google_exceptions.GoogleAPIError
            ) as e:  # Catch-all for other Google API errors
                self.logger.error(f"Gemini API error during token count: {e}. Falling back.")
                fallback_reason = f"Google API error: {e}"
            except Exception as e:
                self.logger.error(
                    f"Unexpected error counting tokens with Gemini SDK: {e}. Falling back."
                )
                fallback_reason = f"Unexpected error with Gemini SDK: {e}"

        # Fallback approximation
        estimated_tokens = len(text) // 4
        self.logger.warning(
            f"Using approximate token count ({estimated_tokens}) for Gemini due to: {fallback_reason}. "
            f"Check SDK, API key, model name, and permissions."
        )
        return estimated_tokens

    def get_context_limit(self) -> int:
        """Returns the context window size (in tokens) for the configured Gemini model.

        Prioritizes dynamically fetched limit, then hardcoded dictionary, then default.
        """
        # 1. Prioritize dynamically fetched input_token_limit
        if self._input_token_limit is not None:
            self.logger.debug(
                f"Using dynamically fetched input token limit for {self._model}: {self._input_token_limit}"
            )
            return self._input_token_limit

        # 2. Fallback to hardcoded dictionary using the normalized self._model
        limit = GEMINI_CONTEXT_LIMITS.get(self._model)
        if (
            limit is not None
        ):  # Ensure limit is not None, as 0 could be a valid (though unlikely) limit
            self.logger.debug(f"Using hardcoded context limit for {self._model}: {limit}")
            return limit

        # 3. Fallback to default
        self.logger.warning(
            f"Context limit not found dynamically or in hardcoded list for Gemini model '{self._model}'. "
            f"Returning default: {DEFAULT_GEMINI_CONTEXT_LIMIT}"
        )
        return DEFAULT_GEMINI_CONTEXT_LIMIT

    # --- Gemini-Specific Provider Methods ---

    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Returns Gemini-specific capabilities."""
        base_capabilities = super().get_provider_capabilities()
        base_capabilities.update(
            {
                "supports_safety_settings": True,  # Gemini has configurable safety settings
                "supports_streaming": True,  # Gemini supports streaming
                "supports_chat_sessions": True,  # Gemini has stateful chat sessions
                "supports_system_instruction": True,  # Gemini uses system_instruction parameter
                "supported_models": self.SUPPORTED_MODELS,
                "context_limits": GEMINI_CONTEXT_LIMITS,
                "requires_alternating_roles": True,  # Gemini requires user/model alternation
                "supports_role_merging": True,  # We handle consecutive role merging
                "model_name_normalization": True,  # We normalize 'models/' prefix
            }
        )
        return base_capabilities

    def get_provider_settings(self) -> Dict[str, Any]:
        """Returns current Gemini-specific settings."""
        settings = super().get_provider_settings()
        settings.update(
            {
                "api_key_set": bool(self.api_key),
                "safety_settings": self.safety_settings,
                "generation_config": {
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                "chat_session_active": self.chat is not None,
                "model_normalized": self._model,
            }
        )
        return settings

    def set_provider_setting(self, setting_name: str, value: Any) -> bool:
        """Sets Gemini-specific settings."""
        if super().set_provider_setting(setting_name, value):
            return True

        # Gemini-specific settings
        if setting_name == "api_key":
            self.api_key = value
            # Re-configure the API key
            try:
                self.genai.configure(api_key=self.api_key)
                return True
            except Exception as e:
                self.logger.error(f"Failed to update Gemini API key: {e}")
                return False
        elif setting_name == "safety_settings":
            try:
                self.set_safety_settings(value)
                return True
            except Exception as e:
                self.logger.error(f"Failed to update Gemini safety settings: {e}")
                return False

        return False

    def supports_feature(self, feature_name: str) -> bool:
        """Checks Gemini-specific feature support."""
        if super().supports_feature(feature_name):
            return True

        gemini_features = {
            "safety_settings",
            "streaming",
            "chat_sessions",
            "system_instruction",
            "role_merging",
            "model_name_normalization",
            "generation_config",
            "native_token_counting",  # Gemini has model.count_tokens
        }

        return feature_name in gemini_features

    def get_safety_settings_info(self) -> Dict[str, Any]:
        """
        Returns information about current safety settings.
        Gemini-specific method.

        Returns:
            Dictionary with safety settings information.
        """
        return {
            "current_settings": self.safety_settings,
            "settings_active": bool(self.safety_settings),
            "can_modify": True,
            "requires_model_reinit": True,
        }

    def get_chat_session_info(self) -> Dict[str, Any]:
        """
        Returns information about the current chat session.
        Gemini-specific method.

        Returns:
            Dictionary with chat session information.
        """
        return {
            "session_active": self.chat is not None,
            "session_type": "stateful" if self.chat else "stateless",
            "history_length": len(self.chat.history) if self.chat else 0,
            "supports_history_translation": True,
        }

    def get_role_translation_info(self) -> Dict[str, Any]:
        """
        Returns information about role translation for Gemini.
        Gemini-specific method.

        Returns:
            Dictionary with role translation information.
        """
        return {
            "internal_roles": ["system", "user", "assistant"],
            "gemini_roles": ["user", "model"],
            "system_handling": "system_instruction",
            "consecutive_role_handling": "merge_with_separator",
            "separator": "\n\n",
        }
