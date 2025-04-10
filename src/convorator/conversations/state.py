# src/convorator/conversations/state.py

"""Manages conversation state for multi-agent interactions."""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Assuming standard project structure
try:
    from convorator.client.llm_client import Message
    from convorator.utils.logger import setup_logger
except ImportError:
    # Basic fallbacks if running standalone or structure differs
    import logging
    from dataclasses import dataclass

    setup_logger = lambda name: logging.getLogger(name)

    @dataclass
    class Message:
        role: str
        content: str

        def to_dict(self) -> Dict[str, str]:
            return {"role": self.role, "content": self.content}

    logging.warning("Using fallback logger and Message class for state.py")

logger = setup_logger(__name__)


class MultiAgentConversation:
    """
    Maintains the state of a conversation involving multiple agents/roles.
    Similar to the Conversation class in llm_client, but potentially used
    for logging the complete exchange from an observer's perspective.
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize a conversation.

        Args:
            system_message (Optional[str]): Initial system message. Defaults to None.
        """
        self.messages: List[Message] = []
        if system_message:
            # Use the proper method to ensure correct placement/replacement
            self.set_system_message(system_message)

    def set_system_message(self, content: str) -> None:
        """
        Set or update the system message for the conversation.

        If a system message exists, it's updated. Otherwise, it's inserted at the beginning.

        Args:
            content (str): The system message content.
        """
        # Check if a system message already exists and update it
        for i, message in enumerate(self.messages):
            if message.role == "system":
                if message.content != content:
                    logger.debug("Updating existing system message in MultiAgentConversation.")
                    self.messages[i] = Message("system", content)
                return

        # Otherwise insert a new system message at the beginning
        logger.debug("Inserting new system message in MultiAgentConversation.")
        self.messages.insert(0, Message("system", content))

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'Moderator', 'Primary').
            content (str): The message content.
        """
        # Simple append, assuming roles are managed externally
        self.messages.append(Message(role=role, content=content))
        logger.debug(
            f"Added message to MultiAgentConversation. Role: {role}, Content: {content[:100]}...",
        )

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in dictionary format.

        Returns:
            List[Dict[str, str]]: List of message dictionaries.
        """
        return [message.to_dict() for message in self.messages]

    def get_messages_by_role(self, role_name: str) -> List[Dict[str, str]]:
        """Filters messages by a specific role name."""
        return [msg.to_dict() for msg in self.messages if msg.role == role_name]

    def clear_conversation(self, clear_system_message: bool = False) -> None:
        """
        Clear all messages except optionally the system message.

        Args:
             clear_system_message (bool): If True, removes the system message too.
                                          If False, preserves the first system message if it exists.
        """
        system_msg_content = None
        if not clear_system_message:
            for message in self.messages:
                if message.role == "system":
                    system_msg_content = message.content
                    break

        self.messages = []
        if system_msg_content is not None:
            # Re-add the system message using the proper method
            self.set_system_message(system_msg_content)
        logger.info(
            f"MultiAgentConversation cleared (system message {'preserved' if system_msg_content else 'cleared'})."
        )

    def is_system_message_set(self) -> bool:
        """
        Check if a system message has been set in the conversation.

        Returns:
            bool: True if a system message exists, False otherwise.
        """
        return any(message.role == "system" for message in self.messages)

    # Removed switch_traditional_conversation_roles as it seems less relevant
    # for a multi-agent log with specific role names.
