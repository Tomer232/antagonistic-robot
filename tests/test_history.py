"""Tests for conversation history management."""

import pytest

from roastcrowd.conversation.history import ConversationHistory


class TestConversationHistory:
    """Tests for the ConversationHistory class."""

    def test_add_user_message(self):
        """Adding a user message appends to history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_add_assistant_message(self):
        """Adding an assistant message appends to history."""
        history = ConversationHistory()
        history.add_assistant_message("Hi there")
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "assistant", "content": "Hi there"}

    def test_multi_turn_history(self):
        """Multiple turns build up correctly."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        history.add_user_message("How are you?")
        history.add_assistant_message("Fine")

        messages = history.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_clear(self):
        """clear() empties the history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        history.clear()
        assert history.get_messages() == []

    def test_get_messages_returns_copy(self):
        """get_messages returns a copy, not a reference."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        messages = history.get_messages()
        messages.append({"role": "user", "content": "Injected"})
        assert len(history.get_messages()) == 1  # original unchanged

    def test_truncation_drops_oldest(self):
        """When token limit exceeded, oldest turns are dropped."""
        # Use a very small token limit
        history = ConversationHistory(max_tokens=20)

        # Add enough messages to exceed the limit
        history.add_user_message("a " * 10)  # ~13 tokens
        history.add_assistant_message("b " * 10)  # ~13 tokens
        history.add_user_message("c " * 10)  # ~13 tokens
        history.add_assistant_message("d " * 10)  # ~13 tokens

        messages = history.get_messages()
        # Should have truncated some older messages
        assert len(messages) < 4

    def test_first_turn_preserved_during_truncation(self):
        """The first message is always preserved during truncation."""
        history = ConversationHistory(max_tokens=30)

        history.add_user_message("first message here")
        history.add_assistant_message("first response")
        history.add_user_message("second " * 20)
        history.add_assistant_message("second response " * 20)
        history.add_user_message("third " * 20)

        messages = history.get_messages()
        # First message should still be there
        assert messages[0]["content"] == "first message here"

    def test_token_estimation_reasonable(self):
        """Token estimation is in a reasonable range."""
        history = ConversationHistory()
        # "hello world" = 2 words * 1.3 = ~2.6 tokens
        history.add_user_message("hello world")
        tokens = history._estimate_tokens()
        assert 2 <= tokens <= 4
