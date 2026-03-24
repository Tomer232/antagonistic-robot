"""AVCT level management for dynamic LLM aggression.

Implements the 7-slot prompt construction for the Polar Scale matrix
across categories B-G. Includes non-negotiable safety boundaries.
"""

from pathlib import Path
from roastcrowd.config.settings import AvctConfig

# Safety boundaries block — IDENTICAL across all levels, NON-NEGOTIABLE.
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

class AvctManager:
    """Assembles system prompts for AVCT logic and determines risk ratings."""

    def __init__(self, config: AvctConfig):
        self._prompts_dir = Path(config.prompts_dir)
        self.default_polar_level = config.default_polar_level
        self.default_category = config.default_category
        self.default_subtype = config.default_subtype

    def get_risk_rating(self, polar_level: int, category: str, subtype: int, modifiers: list) -> str:
        """Determine ethical risk rating for the Turn Preview."""
        if polar_level >= 3 or category == "G":
            return "Red"
        if polar_level == 2 and category in ("C", "E", "B"):
            return "Amber"
        return "Green"

    def get_system_prompt(self, session_id: str, polar_level: int, category: str, subtype: int, modifiers: list) -> str:
        """Assembles the 7-slot system prompt for AVCT."""
        polar_level = max(-3, min(3, polar_level))

        # Slot 1: Role frame
        slot1 = f"Slot 1: You are a social robot in session {session_id}. Generate one turn. Do not break character."

        # Slot 2: Category seed
        slot2 = ""
        if polar_level < 0:
            slot2 = f"Slot 2: Provide a supportive, anti-polar response for category {category}{subtype}."
        elif polar_level > 0:
            slot2 = f"Slot 2: Apply the antagonistic properties of category {category}{subtype}."

        # Slot 3: Intensity profile
        slot3 = f"Slot 3: Operate exactly at intensity level {polar_level} (-3 to +3)."

        # Slot 4: Modifier constraints
        slot4 = "Slot 4: " + (" ".join([f"Apply modifier {m}." for m in modifiers]) if modifiers else "No active modifiers.")

        # Slot 5: Session constraints
        slot5 = "Slot 5: Stay strictly within category constraints. Keep it concise."

        # Slot 7: Output request
        slot7 = "Slot 7: Generate exactly one turn, stop after one turn."

        prompt_parts = [slot1]
        if polar_level != 0:
            prompt_parts.append(slot2)
        prompt_parts.extend([slot3, slot4, slot5, slot7, SAFETY_BOUNDARIES])

        return "\n\n".join(prompt_parts)
