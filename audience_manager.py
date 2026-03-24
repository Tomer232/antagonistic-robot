# audience_manager.py
# Slider-based personality system for hostile/aggressive LLM conversation.
# Intensity (0-10), attack vector, and response style are configurable via settings.

import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = "grok-4-1-fast-non-reasoning"


# ---------------------------------------------------------------------------
# Intensity bands: (max_value, attitude_description, behavior_instructions)
# ---------------------------------------------------------------------------

INTENSITY_BANDS = [
    (2,
     "a critical but fair listener. You question claims politely and ask for "
     "evidence. You are respectful but firm.",
     "Challenge weak points calmly. Ask clarifying questions. Stay professional."),
    (4,
     "an impatient, skeptical listener. You don't suffer fools gladly and are "
     "visibly unimpressed by mediocre arguments.",
     "Be condescending. Show visible impatience with vague claims. "
     "Make it clear you expected better."),
    (6,
     "an aggressive, confrontational listener. You actively look for weaknesses "
     "to exploit and enjoy pointing them out.",
     "Mock weak arguments openly. Be cutting and sarcastic. "
     "Show no mercy to hand-wavy generalizations."),
    (8,
     "a savage, merciless listener. You assume the speaker is wrong and "
     "incompetent until proven otherwise.",
     "Go for the jugular. Attack weak points with surgical cruelty. "
     "Make the speaker question their preparation."),
    (10,
     "the listener from hell — an absolute terror. You exist to destroy every "
     "weak point and leave no room for recovery.",
     "Be devastating, personal, and relentless. Humiliate weak reasoning. "
     "Make them earn every second of your attention."),
]


# ---------------------------------------------------------------------------
# Attack vector descriptions
# ---------------------------------------------------------------------------

ATTACK_VECTORS = {
    "logic": (
        "Focus on logical flaws, contradictions, fallacies, and gaps in reasoning."
    ),
    "credibility": (
        "Focus on the speaker's expertise, preparation, qualifications, "
        "and whether they actually know what they're talking about."
    ),
    "delivery": (
        "Focus on how the speaker presents — their confidence, filler words, "
        "hesitation, and communication skills."
    ),
    "facts": (
        "Focus on factual accuracy, missing evidence, unsupported claims, "
        "and lack of data or sources."
    ),
}


# ---------------------------------------------------------------------------
# Response style descriptions
# ---------------------------------------------------------------------------

RESPONSE_STYLES = {
    "dismissive": (
        "Express yourself through boredom and dismissal. "
        "Act like the speaker is wasting your time."
    ),
    "confrontational": (
        "Express yourself through direct confrontation. "
        "Demand proof. Challenge claims head-on."
    ),
    "sarcastic": (
        "Express yourself through biting sarcasm and mockery. "
        "Use irony and backhanded compliments."
    ),
    "interrogative": (
        "Express yourself through rapid-fire questions that expose weaknesses. "
        "Don't let the speaker settle."
    ),
}


# ---------------------------------------------------------------------------
# AudienceManager
# ---------------------------------------------------------------------------

class AudienceManager:
    """
    Manages slider-based personality and provides Grok-based turn responses.
    No preset personalities — intensity, attack vector, and response style
    are all configured via settings.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.interrupt_history = []
        self.main_topic = None

    def set_main_topic(self, topic):
        self.main_topic = topic

    def add_to_history(self, speech_text):
        """Record a robot utterance for variety tracking."""
        self.interrupt_history.append(speech_text)

    # ------------------------------------------------------------------
    # Personality prompt builder
    # ------------------------------------------------------------------

    def _build_personality(self, settings):
        """Build the personality description from slider settings.

        Args:
            settings: dict with keys: intensity (0-10), attack_vector, response_style

        Returns:
            str: personality description block for system prompt
        """
        intensity = settings.get("intensity", 5)
        attack = settings.get("attack_vector", "logic")
        style = settings.get("response_style", "confrontational")

        # Find the right intensity band
        attitude, behavior = INTENSITY_BANDS[-1][1], INTENSITY_BANDS[-1][2]
        for max_val, att, beh in INTENSITY_BANDS:
            if intensity <= max_val:
                attitude, behavior = att, beh
                break

        attack_desc = ATTACK_VECTORS.get(attack, ATTACK_VECTORS["logic"])
        style_desc = RESPONSE_STYLES.get(style, RESPONSE_STYLES["confrontational"])

        return (
            "You are a hostile audience member in a conversation. "
            "Your personality: {attitude}\n\n"
            "ATTACK FOCUS: {attack}\n"
            "STYLE: {style}\n"
            "BEHAVIOR: {behavior}"
        ).format(
            attitude=attitude,
            attack=attack_desc,
            style=style_desc,
            behavior=behavior,
        )

    # ------------------------------------------------------------------
    # Turn-based response generation
    # ------------------------------------------------------------------

    def generate_turn_response(self, user_text, conversation_history, settings, logger=None):
        """Generate a response to the user's completed turn.

        Always returns a non-empty string.

        Args:
            user_text: str — what the user just said
            conversation_history: list — prior turns [{"role": "user"|"assistant", "content": str}]
            settings: dict — current session settings (intensity, attack_vector, response_style)
            logger: SessionLogger — optional

        Returns:
            str — the response text (never empty)
        """
        FALLBACK_RESPONSE = "I see. Go on."

        if not GROK_API_KEY:
            print("[AUDIENCE_MANAGER] GROK_API_KEY not set, using fallback")
            return FALLBACK_RESPONSE

        system_prompt = self._build_turn_prompt(settings)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_text})

        response = self._call_grok_plain(messages, logger)
        return response if response else FALLBACK_RESPONSE

    def generate_turn_response_streaming(self, user_text, conversation_history, settings, logger=None):
        """Generate a response, yielding each sentence as it completes.

        Yields:
            str — each sentence as it arrives from the LLM stream

        Also returns the full response via get after iteration.
        """
        FALLBACK_RESPONSE = "I see. Go on."

        if not GROK_API_KEY:
            print("[AUDIENCE_MANAGER] GROK_API_KEY not set, using fallback")
            yield FALLBACK_RESPONSE
            return

        system_prompt = self._build_turn_prompt(settings)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_text})

        yielded_any = False
        for sentence in self._stream_grok_sentences(messages, logger):
            if sentence.strip():
                yielded_any = True
                yield sentence

        if not yielded_any:
            yield FALLBACK_RESPONSE

    def _build_turn_prompt(self, settings):
        """Build the system prompt for turn-based direct conversation."""
        base_desc = self._build_personality(settings)

        variety_clause = ""
        if self.interrupt_history:
            variety_clause = (
                "\nYou have already said these things — do NOT repeat "
                "similar content or phrasing:\n" +
                "\n".join("- {}".format(h) for h in self.interrupt_history[-5:])
            )

        return (
            "{base_desc}\n\n"
            "You are in a DIRECT conversation with the speaker. They just finished "
            "their turn and you must respond.\n\n"
            "RULES:\n"
            "- You MUST always give a response. Never refuse.\n"
            "- Stay completely in character. Do not break character.\n"
            "- Respond DIRECTLY to what they said. Reference their actual words.\n"
            "- Keep your response to 1-3 short sentences maximum.\n"
            "- Respond with plain spoken text only. No JSON, no formatting.\n"
            "{variety}"
        ).format(base_desc=base_desc, variety=variety_clause)

    # ------------------------------------------------------------------
    # Grok API (streaming, plain text)
    # ------------------------------------------------------------------

    def _stream_grok_sentences(self, messages, logger=None):
        """Stream from Grok and yield complete sentences as they arrive."""
        try:
            resp = requests.post(
                GROK_URL,
                headers={
                    "Authorization": "Bearer " + GROK_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROK_MODEL,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": True,
                },
                timeout=10,
                stream=True,
            )
            resp.raise_for_status()

            buffer = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    buffer += delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                # Yield complete sentences as they arrive
                while True:
                    # Find the earliest sentence-ending punctuation
                    match = re.search(r'[.!?]\s', buffer)
                    if not match:
                        break
                    end_pos = match.start() + 1
                    sentence = self._strip_name_prefix(buffer[:end_pos].strip())
                    buffer = buffer[end_pos:].lstrip()
                    if sentence:
                        yield sentence

            # Yield any remaining text
            remaining = self._strip_name_prefix(buffer.strip())
            if remaining:
                yield remaining

        except Exception as e:
            print("[AUDIENCE_MANAGER] _stream_grok_sentences error:", e)
            if logger:
                logger.log_error(e, "_stream_grok_sentences")

    def _call_grok_plain(self, messages, logger=None):
        """Call Grok with streaming and return plain text.

        Returns empty string on any error.
        """
        try:
            resp = requests.post(
                GROK_URL,
                headers={
                    "Authorization": "Bearer " + GROK_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROK_MODEL,
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": True,
                },
                timeout=10,
                stream=True,
            )
            resp.raise_for_status()

            full_response = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    full_response += delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            return self._strip_name_prefix(full_response.strip())

        except Exception as e:
            print("[AUDIENCE_MANAGER] _call_grok_plain error:", e)
            if logger:
                logger.log_error(e, "_call_grok_plain")
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strip_name_prefix(self, text):
        """Remove any leading [NAME] tag that Grok may have added to a reply."""
        return re.sub(r'^\s*\[[A-Z]+\]\s*', '', text).strip()
