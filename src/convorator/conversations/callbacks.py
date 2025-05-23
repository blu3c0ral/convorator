import os
from datetime import datetime
from typing import Optional, Dict, Any, Union

from .events import EventType, EnhancedMessageMetadata, OrchestrationStage


class DefaultFileResponseLogger:
    """
    A default messaging callback that logs LLM responses to a file.

    It records the timestamp, the source LLM agent, and the response content.
    It also prints a header whenever the OrchestrationStage changes.
    """

    def __init__(
        self, output_directory: str = "./convorator_logs", filename: str = "responses.log"
    ):
        """
        Initializes the DefaultFileResponseLogger.

        Args:
            output_directory (str): The directory where the log file will be saved.
                                    Defaults to "./convorator_logs".
            filename (str): The name of the log file. Defaults to "responses.log".
        """
        self.output_directory = output_directory
        self.filename = filename
        self.log_file_path = os.path.join(self.output_directory, self.filename)
        self.last_stage: Optional[OrchestrationStage] = None
        self._ensure_log_directory_exists()

    def _ensure_log_directory_exists(self) -> None:
        """Ensures that the log directory exists, creating it if necessary."""
        try:
            os.makedirs(self.output_directory, exist_ok=True)
        except OSError as e:
            # Consider logging this to a fallback logger or print if critical
            print(f"Error creating log directory {self.output_directory}: {e}")
            # Depending on desired robustness, could raise an error or disable logging

    def __call__(
        self,
        event_type: EventType,
        content: Union[str, Dict[str, Any]],
        metadata: EnhancedMessageMetadata,
    ) -> None:
        """
        The callback method invoked by the orchestrator.

        Args:
            event_type: The type of event that occurred.
            content: The content associated with the event (e.g., LLM response string).
            metadata: Rich metadata about the event.
        """
        if event_type == EventType.RESPONSE:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    # Check if stage has changed
                    current_stage = metadata.get("stage")
                    if current_stage and current_stage != self.last_stage:
                        f.write(f"\n--- STAGE: {current_stage.value} ---\n")
                        self.last_stage = current_stage

                    timestamp = metadata.get("timestamp", datetime.now().isoformat())
                    source_llm_name = metadata.get("source_entity_name", "UnknownLLM")

                    response_text = str(content)  # Ensure content is a string

                    log_entry = f"[{timestamp}] [{source_llm_name}]:\n{response_text}\n\n"
                    f.write(log_entry)

            except IOError as e:
                # Fallback logging if file write fails
                print(f"Error writing to log file {self.log_file_path}: {e}")
            except Exception as e:
                # Catch any other unexpected errors during callback execution
                print(f"Unexpected error in DefaultFileResponseLogger: {e}")
