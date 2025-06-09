import pytest
from convorator.client.llm_client import Message, Conversation
import logging
import warnings


# --- Unit Tests for Message Class ---


def test_message_creation_and_attributes():
    """
    Tests basic creation of a Message object and attribute assignment.
    """
    role = "user"
    content = "Hello, world!"
    msg = Message(role=role, content=content)

    assert msg.role == role
    assert msg.content == content


def test_message_to_dict():
    """
    Tests the to_dict() method of the Message class.
    """
    role = "assistant"
    content = "This is a test message."
    msg = Message(role=role, content=content)
    expected_dict = {"role": role, "content": content}

    assert msg.to_dict() == expected_dict


def test_message_role_normalization_is_not_done_in_message_class():
    """
    Tests that the role is NOT normalized to lowercase upon Message creation.
    The current design specifies that role normalization is handled by the
    `Conversation.add_message` method, not the `Message` dataclass itself.
    This test verifies that the Message class stores the role as-is.
    """
    # Test with mixed case
    msg_mixed = Message(role="User", content="Mixed case role")
    assert msg_mixed.role == "User"

    # Test with uppercase
    msg_upper = Message(role="ASSISTANT", content="Upper case role")
    assert msg_upper.role == "ASSISTANT"

    # Test with lowercase
    msg_lower = Message(role="system", content="Lower case role")
    assert msg_lower.role == "system"


def test_message_with_empty_strings():
    """
    Tests Message creation with empty strings for role and content.
    """
    # Empty role
    msg_empty_role = Message(role="", content="Some content")
    assert msg_empty_role.role == ""
    assert msg_empty_role.content == "Some content"

    # Empty content
    msg_empty_content = Message(role="user", content="")
    assert msg_empty_content.role == "user"
    assert msg_empty_content.content == ""

    # Both empty
    msg_both_empty = Message(role="", content="")
    assert msg_both_empty.role == ""
    assert msg_both_empty.content == ""


def test_message_with_special_characters():
    """
    Tests Message creation with special characters, newlines, and unicode.
    """
    # Special characters
    msg_special = Message(role="user", content="Hello! @#$%^&*()_+-=[]{}|;':\",./<>?")
    assert msg_special.content == "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"

    # Newlines and tabs
    msg_newlines = Message(role="assistant", content="Line 1\nLine 2\tTabbed")
    assert msg_newlines.content == "Line 1\nLine 2\tTabbed"

    # Unicode characters
    msg_unicode = Message(role="system", content="Hello 世界! 🌟 café naïve résumé")
    assert msg_unicode.content == "Hello 世界! 🌟 café naïve résumé"


def test_message_with_long_strings():
    """
    Tests Message creation with very long strings.
    """
    long_role = "a" * 1000
    long_content = "This is a very long message. " * 100

    msg = Message(role=long_role, content=long_content)
    assert msg.role == long_role
    assert msg.content == long_content
    assert len(msg.role) == 1000
    assert len(msg.content) == 2900  # "This is a very long message. " is 29 chars * 100


def test_message_equality():
    """
    Tests equality comparison between Message objects.
    Since Message is a dataclass, it should support equality comparison.
    """
    msg1 = Message(role="user", content="Hello")
    msg2 = Message(role="user", content="Hello")
    msg3 = Message(role="user", content="Hi")
    msg4 = Message(role="assistant", content="Hello")

    # Same role and content should be equal
    assert msg1 == msg2

    # Different content should not be equal
    assert msg1 != msg3

    # Different role should not be equal
    assert msg1 != msg4


def test_message_string_representation():
    """
    Tests string representation of Message objects.
    """
    msg = Message(role="user", content="Hello, world!")

    # Should have a meaningful string representation
    str_repr = str(msg)
    assert "user" in str_repr
    assert "Hello, world!" in str_repr

    # repr should be reconstructable
    repr_str = repr(msg)
    assert "Message" in repr_str
    assert "role='user'" in repr_str
    assert "content='Hello, world!'" in repr_str


def test_message_to_dict_returns_new_dict():
    """
    Tests that to_dict() returns a new dictionary each time, not a reference.
    """
    msg = Message(role="user", content="Hello")

    dict1 = msg.to_dict()
    dict2 = msg.to_dict()

    # Should be equal in content
    assert dict1 == dict2

    # But should be different objects
    assert dict1 is not dict2

    # Modifying one shouldn't affect the other
    dict1["role"] = "modified"
    assert dict2["role"] == "user"


def test_message_to_dict_with_special_content():
    """
    Tests to_dict() method with special characters and edge cases.
    """
    # Test with special characters
    msg_special = Message(role="user", content="Content with\nnewlines\tand\ttabs")
    result = msg_special.to_dict()
    assert result == {"role": "user", "content": "Content with\nnewlines\tand\ttabs"}

    # Test with empty strings
    msg_empty = Message(role="", content="")
    result_empty = msg_empty.to_dict()
    assert result_empty == {"role": "", "content": ""}


def test_message_attribute_mutability():
    """
    Tests that Message attributes can be modified after creation.
    Note: This tests the current behavior - if immutability is desired,
    the Message class would need to be modified to use frozen=True.
    """
    msg = Message(role="user", content="Original content")

    # Modify attributes
    msg.role = "assistant"
    msg.content = "Modified content"

    assert msg.role == "assistant"
    assert msg.content == "Modified content"

    # Verify to_dict reflects the changes
    assert msg.to_dict() == {"role": "assistant", "content": "Modified content"}


def test_message_with_various_role_types():
    """
    Tests Message creation with different role values that might be encountered.
    """
    # Standard roles
    standard_roles = ["system", "user", "assistant", "model"]
    for role in standard_roles:
        msg = Message(role=role, content="Test content")
        assert msg.role == role

    # Non-standard but potentially valid roles
    other_roles = ["function", "tool", "human", "ai", "bot"]
    for role in other_roles:
        msg = Message(role=role, content="Test content")
        assert msg.role == role


def test_message_type_hints_are_suggestions():
    """
    Tests behavior when non-string types are passed to Message.
    Note: Python type hints are suggestions by default, not enforced at runtime.
    This test documents the current behavior - the class accepts any type.
    """
    # Numbers
    msg_number = Message(role=123, content=456)
    assert msg_number.role == 123
    assert msg_number.content == 456

    # None values
    msg_none = Message(role=None, content=None)
    assert msg_none.role is None
    assert msg_none.content is None

    # Boolean
    msg_bool = Message(role=True, content=False)
    assert msg_bool.role is True
    assert msg_bool.content is False

    # However, to_dict() expects string conversion to work
    # This might cause issues in practice
    dict_result = msg_number.to_dict()
    assert dict_result == {"role": 123, "content": 456}


# --- Unit Tests for Conversation Class ---


def test_conversation_empty_initialization():
    """
    Tests creating an empty Conversation with default parameters.
    """
    conv = Conversation()

    assert conv.messages == []
    assert conv.system_message is None
    assert conv.get_messages() == []


def test_conversation_initialization_with_system_message():
    """
    Tests creating a Conversation with a system message parameter.
    """
    system_msg = "You are a helpful assistant."
    conv = Conversation(system_message=system_msg)

    assert conv.system_message == system_msg
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == system_msg


def test_conversation_initialization_with_existing_messages():
    """
    Tests creating a Conversation with pre-existing messages.
    """
    initial_messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    conv = Conversation(messages=initial_messages)

    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[1].role == "assistant"
    assert conv.system_message is None


def test_conversation_post_init_system_message_insertion():
    """
    Tests __post_init__ behavior when system_message is provided but not in messages.
    """
    system_msg = "You are helpful."
    existing_messages = [Message(role="user", content="Hello")]

    conv = Conversation(messages=existing_messages, system_message=system_msg)

    # System message should be inserted at the beginning
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == system_msg
    assert conv.messages[1].role == "user"
    assert conv.messages[1].content == "Hello"


def test_conversation_post_init_existing_system_message_sync():
    """
    Tests __post_init__ behavior when system message exists in messages but system_message is None.
    """
    existing_messages = [
        Message(role="system", content="Existing system message"),
        Message(role="user", content="Hello"),
    ]

    conv = Conversation(messages=existing_messages, system_message=None)

    # system_message should be synced from the existing system message
    assert conv.system_message == "Existing system message"
    assert len(conv.messages) == 2


def test_conversation_add_message_role_normalization():
    """
    Tests that add_message() normalizes roles to lowercase.
    """
    conv = Conversation()

    conv.add_message("USER", "Hello")
    conv.add_message("ASSISTANT", "Hi")
    conv.add_message("System", "You are helpful")

    messages = conv.get_messages()
    # System message is always inserted at the beginning, so order is: system, user, assistant
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_conversation_add_message_different_roles():
    """
    Tests add_message() with various role types.
    """
    conv = Conversation()

    # Standard roles
    conv.add_message("system", "System message")
    conv.add_message("user", "User message")
    conv.add_message("assistant", "Assistant message")
    conv.add_message("model", "Model message")  # Gemini uses 'model'

    messages = conv.get_messages()
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "model"


def test_conversation_add_system_message_updates_existing():
    """
    Tests that adding a system message updates existing system message instead of duplicating.
    """
    conv = Conversation(system_message="Original system message")

    # Should have initial system message
    assert len(conv.messages) == 1
    assert conv.messages[0].content == "Original system message"

    # Add new system message - should update, not duplicate
    conv.add_message("system", "Updated system message")

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == "Updated system message"
    assert conv.system_message == "Updated system message"


def test_conversation_add_system_message_inserts_at_beginning():
    """
    Tests that system messages are inserted at the beginning when no system message exists.
    """
    conv = Conversation()
    conv.add_message("user", "Hello")
    conv.add_message("system", "You are helpful")

    messages = conv.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_conversation_add_user_message_convenience():
    """
    Tests the add_user_message() convenience method.
    """
    conv = Conversation()
    conv.add_user_message("Hello, how are you?")

    messages = conv.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, how are you?"


def test_conversation_add_assistant_message_convenience():
    """
    Tests the add_assistant_message() convenience method.
    """
    conv = Conversation()
    conv.add_assistant_message("I'm doing well, thank you!")

    messages = conv.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "I'm doing well, thank you!"


def test_conversation_get_messages_output_format():
    """
    Tests that get_messages() returns the correct dictionary format.
    """
    conv = Conversation()
    conv.add_message("system", "You are helpful")
    conv.add_user_message("What's 2+2?")
    conv.add_assistant_message("2+2 equals 4.")

    messages = conv.get_messages()

    # Should be a list of dictionaries
    assert isinstance(messages, list)
    assert len(messages) == 3

    # Each message should be a dictionary with role and content
    for msg in messages:
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg
        assert len(msg) == 2  # Only role and content keys

    # Verify specific content
    assert messages[0] == {"role": "system", "content": "You are helpful"}
    assert messages[1] == {"role": "user", "content": "What's 2+2?"}
    assert messages[2] == {"role": "assistant", "content": "2+2 equals 4."}


def test_conversation_clear_keep_system_true():
    """
    Tests clear() method with keep_system=True (default).
    """
    conv = Conversation(system_message="You are helpful")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi")

    # Should have 3 messages total
    assert len(conv.messages) == 3

    conv.clear(keep_system=True)

    # Should keep only system message
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == "You are helpful"
    assert conv.system_message == "You are helpful"


def test_conversation_clear_keep_system_false():
    """
    Tests clear() method with keep_system=False.
    """
    conv = Conversation(system_message="You are helpful")
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi")

    # Should have 3 messages total
    assert len(conv.messages) == 3

    conv.clear(keep_system=False)

    # Should clear everything
    assert len(conv.messages) == 0
    assert conv.system_message is None


def test_conversation_clear_default_behavior():
    """
    Tests clear() method with default parameters (should keep system message).
    """
    conv = Conversation(system_message="You are helpful")
    conv.add_user_message("Hello")

    conv.clear()  # Default should be keep_system=True

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.system_message == "You are helpful"


def test_conversation_clear_no_system_message():
    """
    Tests clear() behavior when there's no system message.
    """
    conv = Conversation()
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi")

    conv.clear(keep_system=True)

    # Should clear everything since there's no system message to keep
    assert len(conv.messages) == 0
    assert conv.system_message is None


def test_conversation_consecutive_same_role_warning(caplog):
    """
    Tests that adding consecutive messages with the same role generates a warning.
    """
    conv = Conversation()

    with caplog.at_level(logging.WARNING):
        conv.add_user_message("First user message")
        conv.add_user_message("Second user message")  # Should generate warning

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert "consecutive messages with the same role 'user'" in caplog.records[0].message.lower()

    # But messages should still be added
    messages = conv.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "user"


def test_conversation_consecutive_assistant_messages_warning(caplog):
    """
    Tests consecutive assistant messages also generate warnings.
    """
    conv = Conversation()

    with caplog.at_level(logging.WARNING):
        conv.add_assistant_message("First response")
        conv.add_assistant_message("Second response")  # Should generate warning

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert (
        "consecutive messages with the same role 'assistant'" in caplog.records[0].message.lower()
    )


def test_conversation_system_messages_dont_affect_consecutive_warning(caplog):
    """
    Tests that system messages don't interfere with consecutive message detection.
    """
    conv = Conversation()

    with caplog.at_level(logging.WARNING):
        conv.add_user_message("First user message")
        conv.add_message("system", "System intervention")
        conv.add_user_message(
            "Second user message"
        )  # Should still warn about consecutive user messages

    # Should have warning for consecutive user messages despite system message in between
    assert len(caplog.records) == 1
    assert "consecutive messages with the same role 'user'" in caplog.records[0].message.lower()


def test_conversation_non_standard_role_warning(caplog):
    """
    Tests that non-standard roles generate warnings.
    """
    conv = Conversation()

    with caplog.at_level(logging.WARNING):
        conv.add_message("function", "Function result")  # Non-standard role

    # Check that warning was logged
    assert len(caplog.records) == 1
    assert "non-standard role 'function'" in caplog.records[0].message.lower()


def test_conversation_standard_roles_no_warning(caplog):
    """
    Tests that standard roles don't generate warnings.
    """
    conv = Conversation()

    with caplog.at_level(logging.WARNING):
        conv.add_message("user", "User message")
        conv.add_message("assistant", "Assistant message")
        conv.add_message("model", "Model message")  # Gemini uses this

    # Should have no warnings for standard roles
    assert len(caplog.records) == 0


def test_conversation_role_alternation_typical_flow():
    """
    Tests typical conversation flow without warnings.
    """
    conv = Conversation(system_message="You are helpful")

    # Typical alternating flow should not generate warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there!")
        conv.add_user_message("How are you?")
        conv.add_assistant_message("I'm doing well!")

    # Should have no warnings
    assert len(warning_list) == 0

    # Verify message structure
    messages = conv.get_messages()
    assert len(messages) == 5  # system + 4 messages
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[4]["role"] == "assistant"


def test_conversation_complex_message_handling():
    """
    Tests complex scenarios with multiple operations.
    """
    conv = Conversation()

    # Start with user message
    conv.add_user_message("Initial question")

    # Add system message later (should insert at beginning)
    conv.add_message("system", "You are an expert")

    # Continue conversation
    conv.add_assistant_message("I can help with that")
    conv.add_user_message("Great!")

    # Update system message
    conv.add_message("system", "You are a specialized expert")

    messages = conv.get_messages()

    # Should have 4 messages total (system message updated, not duplicated)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a specialized expert"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Initial question"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "I can help with that"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Great!"


# --- Enhanced Tests for Edge Cases and Robustness ---


def test_conversation_init_with_conflicting_system_messages():
    """
    Tests initialization when a system_message is passed, but a different one
    is already in the message list. This documents the actual behavior: the
    constructor's system_message is ignored if the message list already
    starts with a system message.
    """
    initial_system_msg = Message(role="system", content="Initial system prompt")
    conflicting_system_msg = "Conflicting system prompt"

    conv = Conversation(
        messages=[initial_system_msg, Message(role="user", content="Hello")],
        system_message=conflicting_system_msg,
    )

    # The __post_init__ should NOT insert a new message, as one already exists.
    messages = conv.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Initial system prompt"
    # The system_message attribute from the constructor is kept, but not inserted.
    assert conv.system_message == conflicting_system_msg


def test_add_system_message_with_same_content_is_noop():
    """
    Tests that adding a system message with the same content as the existing one
    does not cause any changes (no-op).
    """
    conv = Conversation(system_message="You are helpful.")
    original_messages_obj = conv.messages  # Get a reference to the list object

    # Add the same system message again
    conv.add_message("system", "You are helpful.")

    # The list object itself should not have been replaced
    assert conv.messages is original_messages_obj
    assert len(conv.messages) == 1


def test_consecutive_check_after_only_system_messages(caplog):
    """
    Tests that adding a user message after only system messages does not
    trigger a consecutive role warning.
    """
    conv = Conversation()
    conv.add_message("system", "Setup prompt 1")
    conv.add_message("system", "Setup prompt 2")  # This will just update the first one

    with caplog.at_level(logging.WARNING):
        conv.add_user_message("This is the first user message.")

    # No WARNING-level logs should be present. DEBUG logs are ignored.
    warning_logs = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert not warning_logs
    assert len(conv.messages) == 2  # System + User


def test_clear_on_empty_conversation():
    """
    Tests that calling clear() on an already empty conversation works without error.
    """
    conv = Conversation()

    # Should not raise any exception
    try:
        conv.clear(keep_system=True)
        conv.clear(keep_system=False)
    except Exception as e:
        pytest.fail(f"clear() raised an unexpected exception on an empty conversation: {e}")

    assert conv.messages == []
    assert conv.system_message is None


def test_add_message_with_non_string_role_raises_error():
    """
    Tests that add_message raises an AttributeError for non-string roles
    because it calls .lower() on them. This documents a robustness constraint.
    """
    conv = Conversation()

    # Test with None
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'lower'"):
        conv.add_message(None, "Content with None role")

    # Test with an integer
    with pytest.raises(AttributeError, match="'int' object has no attribute 'lower'"):
        conv.add_message(123, "Content with int role")
