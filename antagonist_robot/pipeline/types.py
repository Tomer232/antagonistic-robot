"""Data types passed between pipeline components.

Every pipeline stage returns a dataclass that includes timing information
for latency logging. These are the only data structures that flow between
pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class AudioData:
    """Recorded audio from the microphone."""
    samples: np.ndarray          # float32 numpy array, normalized [-1, 1]
    sample_rate: int             # always 16000
    duration_seconds: float      # len(samples) / sample_rate
    recording_started: str       # ISO-format timestamp
    recording_ended: str         # ISO-format timestamp


@dataclass
class ASRResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str
    confidence: float            # average log probability from segments
    transcription_time_seconds: float


@dataclass
class LLMResult:
    """Result from LLM generation."""
    text: str
    model: str
    total_tokens: int
    generation_time_seconds: float


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio_bytes: bytes
    format: str                  # "pcm" or "wav"
    sample_rate: int             # sample rate of the audio
    duration_seconds: float
    synthesis_time_seconds: float
    voice: str


@dataclass
class TurnResult:
    """Complete result for one conversation turn."""
    turn_number: int
    user_audio: Optional[AudioData]
    transcript: str
    llm_response: str
    tts_result: Optional[TTSResult]
    hostility_level: int
    polar_level: int
    category: str
    subtype: int
    modifiers: list
    risk_rating: str
    latency: Dict[str, int]      # {"vad_ms": ..., "asr_ms": ..., "llm_ms": ..., "tts_ms": ..., "total_ms": ...}
    timestamp: str               # ISO-format
