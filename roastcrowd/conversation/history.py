"""Conversation history management with token-based truncation.

Maintains the list of message dicts (role and content) that get sent
to the LLM each turn. Supports truncation to prevent context window
overflow on long conversations.
"""

from typing import Dict, List


class ConversationHistory:
    """Manages the conversation message history for LLM context.

    Messages are stored as dicts with "role" and "content" keys.
    When the estimated token count exceeds max_tokens, the oldest
    turns are dropped (but the first turn is always preserved for
    context continuity).
    """

    TOKEN_MULTIPLIER = 1.3  # Estimate: 1 word ~ 1.3 tokens

    def __init__(self, max_tokens: int = 4000):
        self._messages: List[Dict[str, str]] = []
        self._max_tokens = max_tokens

    def add_user_message(self, text: str) -> None:
        """Add a user message to history."""
        self._messages.append({"role": "user", "content": text})
        self._truncate_if_needed()

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to history."""
        self._messages.append({"role": "assistant", "content": text})
        self._truncate_if_needed()

    def get_messages(self) -> List[Dict[str, str]]:
        """Return a copy of the full message history."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset history to empty."""
        self._messages.clear()

    def _estimate_tokens(self) -> int:
        """Estimate total tokens using word count * 1.3."""
        total_words = sum(
            len(msg["content"].split()) for msg in self._messages
        )
        return int(total_words * self.TOKEN_MULTIPLIER)

    def _truncate_if_needed(self) -> None:
        """If history exceeds token limit, drop oldest turns.

        Always keeps the first turn (first user + assistant pair)
        for context continuity. Drops messages in pairs to keep
        the conversation coherent.
        """
        while (
            self._estimate_tokens() > self._max_tokens
            and len(self._messages) > 2
        ):
            # Remove the second message (index 1), preserving index 0.
            # Remove in pairs (user + assistant) to keep conversation coherent.
            if len(self._messages) > 3:
                del self._messages[1]
                # Also remove its pair if it exists
                if (
                    len(self._messages) > 1
                    and self._messages[1]["role"] == "assistant"
                ):
                    del self._messages[1]
            else:
                break
