# audio/tts.py
#
# Text-to-Speech using OpenAI TTS (gpt-4o-mini-tts).
# Voice is read from settings.json ("tts_voice"); falls back to "onyx".
# Audio streams directly to speakers via pyaudio as it downloads —
# no waiting for the full response before playback begins.

import os
import json

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables so OPENAI_API_KEY is available
load_dotenv()

# Single global OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

# Project root and settings file (same settings.json used everywhere else)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

ALLOWED_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
}
DEFAULT_VOICE = "onyx"  # default male, natural-sounding

# OpenAI TTS PCM format: 24 kHz, 16-bit, mono
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)

# Reuse a single PyAudio instance
_pa = pyaudio.PyAudio()


def _get_configured_voice() -> str:
    """
    Read the TTS voice from settings.json ("tts_voice").
    If missing/invalid, return DEFAULT_VOICE.
    """
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("tts_voice", DEFAULT_VOICE)
        if v in ALLOWED_VOICES:
            return v
    except Exception:
        pass
    return DEFAULT_VOICE


def synthesize_and_play(text: str, voice: str = None) -> None:
    """
    Synthesize the given text with OpenAI TTS and stream it to speakers.
    Audio starts playing as soon as the first bytes arrive from the API.
    """
    if not text:
        return

    if voice is None:
        voice = _get_configured_voice()
    elif voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    try:
        # Stream raw PCM from OpenAI TTS
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="pcm",
        ) as response:
            # Open pyaudio output stream
            stream = _pa.open(
                format=pyaudio.paInt16,
                channels=PCM_CHANNELS,
                rate=PCM_SAMPLE_RATE,
                output=True,
            )
            try:
                print(f"[TTS] Streaming voice={voice}: {text[:60]}...")
                for chunk in response.iter_bytes(chunk_size=4096):
                    stream.write(chunk)
            finally:
                stream.stop_stream()
                stream.close()
                print("[TTS] Playback finished")

    except Exception as e:
        # TTS failure should not crash the session
        print("[TTS] Error during synthesis:", e)
        print("[TTS] Fallback, text was:", text)
