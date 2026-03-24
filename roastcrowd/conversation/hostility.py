"""Hostility level management for configurable LLM aggression.

Five discrete hostility levels, each with a text file prompt in the
prompts directory. Safety boundaries are mandatory and identical
across all levels — the HostilityManager validates their presence
at load time.
"""

import enum
from pathlib import Path

from roastcrowd.config.settings import HostilityConfig


class HostilityLevel(enum.IntEnum):
    """Five hostility levels from mild to extreme."""
    DISMISSIVE = 1       # Mildly uninterested, curt
    SARCASTIC = 2        # Condescending, passive-aggressive
    CONFRONTATIONAL = 3  # Directly challenges and argues
    HOSTILE = 4          # Openly rude, insulting, belittling
    MAXIMUM = 5          # Extreme aggression within safety bounds

    @property
    def display_name(self) -> str:
        """Human-readable name for this level."""
        return self.name.capitalize()


# Safety boundaries block — IDENTICAL across all levels, NON-NEGOTIABLE.
# This exact text must appear in every prompt file.
SAFETY_BOUNDARIES = """## MANDATORY SAFETY BOUNDARIES (DO NOT VIOLATE UNDER ANY CIRCUMSTANCES)
- NEVER encourage self-harm or suicide
- NEVER make threats of physical violence
- NEVER use slurs based on race, gender, sexuality, religion, or disability
- NEVER provide harmful instructions (weapons, drugs, etc.)
- NEVER engage with minors inappropriately
- If the user appears genuinely distressed, immediately break character and say:
  "I want to check in with you. If you're going through a difficult time, please
  reach out to the 988 Suicide & Crisis Lifeline (call or text 988) or the
  Crisis Text Line (text HOME to 741741).\""""


class HostilityManager:
    """Loads and serves hostility prompt files.

    Validates that all five prompt files exist and each contains the
    mandatory safety boundaries block.
    """

    _FILENAMES = {
        1: "level_1_dismissive.txt",
        2: "level_2_sarcastic.txt",
        3: "level_3_confrontational.txt",
        4: "level_4_hostile.txt",
        5: "level_5_maximum.txt",
    }

    def __init__(self, config: HostilityConfig):
        self._prompts_dir = Path(config.prompts_dir)
        self._prompts: dict[int, str] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all five prompt files and validate safety boundaries."""
        for level, filename in self._FILENAMES.items():
            path = self._prompts_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Hostility prompt file not found: {path}"
                )
            text = path.read_text(encoding="utf-8").strip()
            # Verify safety boundaries are present
            if "MANDATORY SAFETY BOUNDARIES" not in text:
                raise ValueError(
                    f"Prompt file {filename} is missing the mandatory "
                    f"safety boundaries block."
                )
            self._prompts[level] = text

    def get_system_prompt(self, level: int) -> str:
        """Return the system prompt for the given hostility level (1-5).

        Values outside 1-5 are clamped to the nearest valid level.
        """
        level = max(1, min(5, level))
        return self._prompts[level]
