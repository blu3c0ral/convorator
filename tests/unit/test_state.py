import pytest
from convorator.conversations.state import MultiAgentConversation


def test_init_without_system_message():
    """Test initialization without a system message."""
    conversation = MultiAgentConversation()
    assert len(conversation.messages) == 0
    assert not conversation.is_system_message_set()


def test_init_with_system_message():
    """Test initialization with a system message."""
    system_message = "This is a system message."
    conversation = MultiAgentConversation(system_message=system_message)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == system_message
    assert conversation.is_system_message_set()


def test_set_system_message_when_not_exists():
    """Test setting a system message when none exists."""
    conversation = MultiAgentConversation()
    system_message = "This is a system message."

    conversation.set_system_message(system_message)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == system_message
    assert conversation.is_system_message_set()


def test_set_system_message_when_already_exists():
    """Test updating an existing system message."""
    original_message = "Original system message."
    conversation = MultiAgentConversation(system_message=original_message)

    new_message = "Updated system message."
    conversation.set_system_message(new_message)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == new_message
    assert conversation.is_system_message_set()


def test_add_message():
    """Test adding messages to the conversation."""
    conversation = MultiAgentConversation()

    conversation.add_message("user", "User message")
    conversation.add_message("Primary", "Primary agent message")
    conversation.add_message("Debater", "Debater agent message")

    assert len(conversation.messages) == 3
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "User message"
    assert conversation.messages[1].role == "Primary"
    assert conversation.messages[1].content == "Primary agent message"
    assert conversation.messages[2].role == "Debater"
    assert conversation.messages[2].content == "Debater agent message"


def test_get_messages():
    """Test retrieving messages as dictionaries."""
    conversation = MultiAgentConversation(system_message="System message")
    conversation.add_message("user", "User message")
    conversation.add_message("assistant", "Assistant message")

    messages = conversation.get_messages()

    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "System message"}
    assert messages[1] == {"role": "user", "content": "User message"}
    assert messages[2] == {"role": "assistant", "content": "Assistant message"}


def test_get_messages_by_role():
    """Test filtering messages by role."""
    conversation = MultiAgentConversation(system_message="System message")
    conversation.add_message("user", "User message 1")
    conversation.add_message("assistant", "Assistant message")
    conversation.add_message("user", "User message 2")

    user_messages = conversation.get_messages_by_role("user")
    assistant_messages = conversation.get_messages_by_role("assistant")
    system_messages = conversation.get_messages_by_role("system")
    nonexistent_messages = conversation.get_messages_by_role("nonexistent")

    assert len(user_messages) == 2
    assert user_messages[0] == {"role": "user", "content": "User message 1"}
    assert user_messages[1] == {"role": "user", "content": "User message 2"}

    assert len(assistant_messages) == 1
    assert assistant_messages[0] == {"role": "assistant", "content": "Assistant message"}

    assert len(system_messages) == 1
    assert system_messages[0] == {"role": "system", "content": "System message"}

    assert len(nonexistent_messages) == 0


def test_clear_conversation_preserve_system():
    """Test clearing conversation while preserving system message."""
    conversation = MultiAgentConversation(system_message="System message")
    conversation.add_message("user", "User message")
    conversation.add_message("assistant", "Assistant message")

    conversation.clear_conversation(clear_system_message=False)

    assert len(conversation.messages) == 1
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content == "System message"
    assert conversation.is_system_message_set()


def test_clear_conversation_clear_all():
    """Test clearing the entire conversation including system message."""
    conversation = MultiAgentConversation(system_message="System message")
    conversation.add_message("user", "User message")
    conversation.add_message("assistant", "Assistant message")

    conversation.clear_conversation(clear_system_message=True)

    assert len(conversation.messages) == 0
    assert not conversation.is_system_message_set()


def test_clear_conversation_when_no_system_message():
    """Test clearing a conversation that doesn't have a system message."""
    conversation = MultiAgentConversation()
    conversation.add_message("user", "User message")
    conversation.add_message("assistant", "Assistant message")

    conversation.clear_conversation(clear_system_message=False)

    assert len(conversation.messages) == 0
    assert not conversation.is_system_message_set()


def test_is_system_message_set():
    """Test checking if a system message exists."""
    # With system message
    conversation_with_system = MultiAgentConversation(system_message="System message")
    assert conversation_with_system.is_system_message_set()

    # Without system message
    conversation_without_system = MultiAgentConversation()
    assert not conversation_without_system.is_system_message_set()

    # Add then clear system message
    conversation_cleared = MultiAgentConversation(system_message="System message")
    conversation_cleared.clear_conversation(clear_system_message=True)
    assert not conversation_cleared.is_system_message_set()
