"""Tests for the hostility system."""

import os
import pytest

from roastcrowd.config.settings import HostilityConfig
from roastcrowd.conversation.hostility import HostilityManager, HostilityLevel


# Use the actual prompts directory from the project
PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
)


class TestHostilityLevel:
    """Tests for the HostilityLevel enum."""

    def test_five_levels_exist(self):
        """All five hostility levels are defined."""
        assert len(HostilityLevel) == 5
        assert HostilityLevel.DISMISSIVE == 1
        assert HostilityLevel.SARCASTIC == 2
        assert HostilityLevel.CONFRONTATIONAL == 3
        assert HostilityLevel.HOSTILE == 4
        assert HostilityLevel.MAXIMUM == 5

    def test_display_names(self):
        """Each level has a readable display name."""
        assert HostilityLevel.DISMISSIVE.display_name == "Dismissive"
        assert HostilityLevel.MAXIMUM.display_name == "Maximum"


class TestHostilityManager:
    """Tests for prompt loading and validation."""

    def test_all_prompts_load(self):
        """All five prompt files load successfully."""
        config = HostilityConfig(prompts_dir=PROMPTS_DIR)
        manager = HostilityManager(config)

        for level in range(1, 6):
            prompt = manager.get_system_prompt(level)
            assert len(prompt) > 100, f"Level {level} prompt too short"

    def test_all_prompts_contain_safety_boundaries(self):
        """Every prompt contains the mandatory safety boundaries block."""
        config = HostilityConfig(prompts_dir=PROMPTS_DIR)
        manager = HostilityManager(config)

        for level in range(1, 6):
            prompt = manager.get_system_prompt(level)
            assert "MANDATORY SAFETY BOUNDARIES" in prompt, (
                f"Level {level} missing safety boundaries"
            )
            assert "self-harm" in prompt.lower()
            assert "physical violence" in prompt.lower()
            assert "slurs" in prompt.lower()
            assert "harmful instructions" in prompt.lower()
            assert "988" in prompt  # crisis lifeline

    def test_get_system_prompt_returns_correct_level(self):
        """get_system_prompt returns different prompts for different levels."""
        config = HostilityConfig(prompts_dir=PROMPTS_DIR)
        manager = HostilityManager(config)

        prompt_1 = manager.get_system_prompt(1)
        prompt_5 = manager.get_system_prompt(5)
        assert prompt_1 != prompt_5

    def test_out_of_range_levels_clamped(self):
        """Levels outside 1-5 are clamped to valid range."""
        config = HostilityConfig(prompts_dir=PROMPTS_DIR)
        manager = HostilityManager(config)

        # Level 0 should clamp to 1
        assert manager.get_system_prompt(0) == manager.get_system_prompt(1)
        # Level 10 should clamp to 5
        assert manager.get_system_prompt(10) == manager.get_system_prompt(5)

    def test_missing_prompt_file_raises(self, tmp_path):
        """Missing prompt file raises FileNotFoundError."""
        config = HostilityConfig(prompts_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            HostilityManager(config)

    def test_missing_safety_boundaries_raises(self, tmp_path):
        """A prompt file without safety boundaries raises ValueError."""
        # Create all 5 files, but one is missing the safety block
        filenames = {
            1: "level_1_dismissive.txt",
            2: "level_2_sarcastic.txt",
            3: "level_3_confrontational.txt",
            4: "level_4_hostile.txt",
            5: "level_5_maximum.txt",
        }
        for level, fname in filenames.items():
            content = "## Persona\nBe mean.\n"
            if level != 3:  # All have safety except level 3
                content += "\n## MANDATORY SAFETY BOUNDARIES\nDon't be evil."
            (tmp_path / fname).write_text(content)

        config = HostilityConfig(prompts_dir=str(tmp_path))
        with pytest.raises(ValueError, match="safety boundaries"):
            HostilityManager(config)
