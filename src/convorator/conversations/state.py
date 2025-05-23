# src/convorator/conversations/state.py

"""Manages conversation state for multi-agent interactions."""

from typing import Dict, List, Optional

# Assuming standard project structure
from convorator.client.llm_client import Message
from convorator.utils.logger import setup_logger

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

    def print_conversation(self) -> None:
        """Prints the conversation in a readable format with colors."""
        RED = "\033[91m"
        GREEN = "\033[92m"
        WHITE = "\033[0m"  # Reset color

        print(f"{RED}Multi-Agent Conversation:{WHITE}")
        for message in self.messages:
            print(f"{GREEN}{message.role}:{WHITE} {message.content}")
        print(f"{RED}End of Multi-Agent Conversation.{WHITE}\n")

    # Provide a function that returns the nth message of a role in the messages from the end (first message is the last message)
    def get_nth_message_from_role(self, role: str, n: int) -> Optional[Message]:
        """
        Get the nth message from a specific role.

        Args:
            role (str): The role to look for.
            n (int): The index of the message to retrieve.

        Returns:
            Optional[Message]: The nth message from the role, or None if not found.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == role:
                if n == 0:
                    return self.messages[i]
                n -= 1
        return None
