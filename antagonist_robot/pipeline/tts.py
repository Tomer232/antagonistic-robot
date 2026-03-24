"""TTS engine with pluggable backends. Primary implementation: OpenAI TTS.

Uses OpenAI's gpt-4o-mini-tts model which produces high-quality speech.
The base class allows swapping in alternative engines (e.g., edge-tts)
by changing config.
"""

import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from antagonist_robot.config.settings import TTSConfig
from antagonist_robot.pipeline.types import TTSResult


@dataclass
class VoiceInfo:
    """Metadata for an available TTS voice."""
    name: str
    gender: str
    locale: str


class TTSBase(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None) -> TTSResult:
        """Synthesize text to audio bytes."""
        ...

    @abstractmethod
    def list_voices(self) -> List[VoiceInfo]:
        """Return available voices."""
        ...


# Available OpenAI TTS voices with metadata
_OPENAI_VOICES = [
    VoiceInfo(name="alloy", gender="Female", locale="en-US"),
    VoiceInfo(name="echo", gender="Male", locale="en-US"),
    VoiceInfo(name="fable", gender="Male", locale="en-US"),
    VoiceInfo(name="onyx", gender="Male", locale="en-US"),
    VoiceInfo(name="nova", gender="Female", locale="en-US"),
    VoiceInfo(name="shimmer", gender="Female", locale="en-US"),
    VoiceInfo(name="coral", gender="Female", locale="en-US"),
    VoiceInfo(name="verse", gender="Male", locale="en-US"),
    VoiceInfo(name="ballad", gender="Male", locale="en-US"),
    VoiceInfo(name="ash", gender="Male", locale="en-US"),
    VoiceInfo(name="sage", gender="Female", locale="en-US"),
    VoiceInfo(name="marin", gender="Female", locale="en-US"),
    VoiceInfo(name="cedar", gender="Male", locale="en-US"),
]

_ALLOWED_VOICE_NAMES = {v.name for v in _OPENAI_VOICES}

# OpenAI TTS PCM format: 24 kHz, 16-bit, mono
PCM_SAMPLE_RATE = 24000


class OpenAITTSEngine(TTSBase):
    """TTS using OpenAI's gpt-4o-mini-tts model.

    Requests raw PCM audio (24kHz, 16-bit, mono) from the API and returns
    it as bytes in a TTSResult. The audio is NOT streamed to speakers here —
    that's the audio output module's job.
    """

    def __init__(self, config: TTSConfig):
        self._default_voice = config.default_voice
        self._model = config.model
        self._client = OpenAI(api_key=config.api_key)

    def synthesize(self, text: str, voice: Optional[str] = None) -> TTSResult:
        """Synthesize text to raw PCM audio bytes using OpenAI TTS.

        Args:
            text: The text to speak.
            voice: Optional voice name. Uses config default if not specified.

        Returns:
            TTSResult with PCM audio bytes at 24kHz 16-bit mono.
        """
        voice = voice or self._default_voice
        if voice not in _ALLOWED_VOICE_NAMES:
            voice = self._default_voice

        start = time.monotonic()

        response = self._client.audio.speech.create(
            model=self._model,
            voice=voice,
            input=text,
            response_format="pcm",
        )

        audio_bytes = response.content
        elapsed = time.monotonic() - start

        # Calculate duration from PCM byte count
        # PCM: 24kHz, 16-bit (2 bytes per sample), mono
        num_samples = len(audio_bytes) // 2
        duration = num_samples / PCM_SAMPLE_RATE

        return TTSResult(
            audio_bytes=audio_bytes,
            format="pcm",
            sample_rate=PCM_SAMPLE_RATE,
            duration_seconds=duration,
            synthesis_time_seconds=elapsed,
            voice=voice,
        )

    def list_voices(self) -> List[VoiceInfo]:
        """Return available OpenAI TTS voices."""
        return list(_OPENAI_VOICES)
