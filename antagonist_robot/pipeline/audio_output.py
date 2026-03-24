"""Audio output routing with pluggable backends.

LaptopAudioOutput: plays through laptop speakers using pyaudio.
NAOAudioOutput: routes audio to NAO robot via TCP (nao_speaker_server.py).
"""

import socket
import time
from abc import ABC, abstractmethod

import numpy as np
import pyaudio

from antagonist_robot.pipeline.types import TTSResult


class AudioOutputBase(ABC):
    """Abstract base class for audio output."""

    @abstractmethod
    def play_audio(self, tts_result: TTSResult) -> None:
        """Play pre-synthesized audio. Blocks until done."""
        ...

    @abstractmethod
    def speak_text(self, text: str) -> None:
        """Send raw text to a device's built-in TTS. Blocks until done."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Immediately halt playback."""
        ...


class LaptopAudioOutput(AudioOutputBase):
    """Plays audio through laptop speakers using pyaudio.

    Handles raw PCM audio from OpenAI TTS (24kHz, 16-bit, mono).
    """

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = None

    def play_audio(self, tts_result: TTSResult) -> None:
        """Play PCM audio through laptop speakers. Blocks until done."""
        if tts_result.format == "pcm":
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=tts_result.sample_rate,
                output=True,
            )
            try:
                self._stream = stream
                # Write audio in chunks for responsive stop()
                chunk_size = 4096
                data = tts_result.audio_bytes
                for i in range(0, len(data), chunk_size):
                    if self._stream is None:
                        break  # stop() was called
                    stream.write(data[i:i + chunk_size])
            finally:
                stream.stop_stream()
                stream.close()
                self._stream = None
        else:
            raise ValueError(f"Unsupported audio format: {tts_result.format}")

    def speak_text(self, text: str) -> None:
        """Not supported for laptop output — always use play_audio."""
        raise NotImplementedError(
            "Laptop output requires pre-synthesized audio. Use play_audio()."
        )

    def stop(self) -> None:
        """Stop playback immediately."""
        self._stream = None


class NAOAudioOutput(AudioOutputBase):
    """Routes audio to NAO robot.

    Two modes:
    - use_builtin_tts=True: sends text directly to NAO's ALTextToSpeech
      via TCP to nao_speaker_server.py (skips local TTS for lower latency).
    - use_builtin_tts=False: would stream pre-synthesized audio to NAO's
      ALAudioPlayer (not yet implemented).
    """

    def __init__(self, ip: str, port: int, use_builtin_tts: bool):
        self._ip = ip
        self._port = port
        self._use_builtin_tts = use_builtin_tts

    @property
    def use_builtin_tts(self) -> bool:
        """Whether this output uses NAO's built-in TTS."""
        return self._use_builtin_tts

    def play_audio(self, tts_result: TTSResult) -> None:
        """Stream pre-synthesized audio to NAO's ALAudioPlayer.

        Not yet implemented — requires naoqi SDK.
        """
        raise NotImplementedError(
            "NAO audio streaming not yet implemented. "
            "Set use_builtin_tts=true to use NAO's built-in TTS instead."
        )

    def speak_text(self, text: str) -> None:
        """Send text to NAO's ALTextToSpeech via TCP.

        Connects to nao_speaker_server.py running on the robot.
        Protocol: send text + newline, wait for "ok" response.
        Blocks until the robot finishes speaking.
        """
        try:
            with socket.create_connection(
                (self._ip, self._port), timeout=30
            ) as s:
                s.sendall((text.strip() + "\n").encode("utf-8"))
                # Wait for "ok" acknowledgement from the robot
                response = b""
                while True:
                    chunk = s.recv(64)
                    if not chunk:
                        break
                    response += chunk
                    if b"ok" in response:
                        break
        except Exception as e:
            print(f"[NAO AUDIO] Socket error: {e}")

    def stop(self) -> None:
        """Cannot remotely stop NAO TTS currently."""
        pass
