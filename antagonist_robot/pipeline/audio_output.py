"""Audio output routing to the NAO robot via TCP.

NAOAudioOutput: routes audio to NAO robot via nao_speaker_server.py.
"""

import socket
import time
from abc import ABC, abstractmethod

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
