# audience_member.py
# Represents one AI audience member with fixed personality traits and cooldown.
# Interrupt decisions are now fully content-driven via Grok (no probability rules).

import time


class AudienceMember:
    """Holds personality configuration and cooldown state for one AI audience member."""

    COOLDOWN_SECONDS = 20.0  # seconds between interrupts

    def __init__(self, name, attitude, difficulty, aggressiveness, voice="onyx"):
        """
        Args:
            name: str — "MAYA", "JAKE", or "ALEX"
            attitude: str — "neutral" | "skeptical"
            difficulty: str — "medium" | "hard"
            aggressiveness: str — "medium" | "high"
            voice: str — TTS voice name (e.g. "sage", "onyx", "verse")
        """
        self.name = name
        self.attitude = attitude
        self.difficulty = difficulty
        self.aggressiveness = aggressiveness
        self.voice = voice

        # Cooldown state
        self.last_spoke_time = 0.0
        self.cooldown_seconds = self.COOLDOWN_SECONDS

    def is_on_cooldown(self):
        """Return True if still within post-speaking cooldown window."""
        return (time.time() - self.last_spoke_time) < self.cooldown_seconds

    def mark_spoke(self):
        """Call after this member speaks. Starts the cooldown timer."""
        self.last_spoke_time = time.time()

    def can_speak(self):
        """Return True if not on cooldown."""
        return not self.is_on_cooldown()
