"""Tests for the audio output base class and implementations."""

import pytest

from antagonist_robot.pipeline.audio_output import (
    AudioOutputBase,
    LaptopAudioOutput,
    NAOAudioOutput,
)
from antagonist_robot.pipeline.types import TTSResult


class TestAudioOutputBase:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """AudioOutputBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioOutputBase()

    def test_has_required_methods(self):
        """AudioOutputBase defines all required abstract methods."""
        assert hasattr(AudioOutputBase, "play_audio")
        assert hasattr(AudioOutputBase, "speak_text")
        assert hasattr(AudioOutputBase, "stop")


class TestLaptopAudioOutput:
    """Tests for the laptop audio output."""

    def test_speak_text_raises(self):
        """LaptopAudioOutput.speak_text raises NotImplementedError."""
        output = LaptopAudioOutput()
        with pytest.raises(NotImplementedError):
            output.speak_text("Hello")

    def test_unsupported_format_raises(self):
        """Playing an unsupported audio format raises ValueError."""
        output = LaptopAudioOutput()
        result = TTSResult(
            audio_bytes=b"\x00",
            format="ogg",  # unsupported
            sample_rate=24000,
            duration_seconds=1.0,
            synthesis_time_seconds=0.1,
            voice="test",
        )
        with pytest.raises(ValueError, match="Unsupported"):
            output.play_audio(result)


class TestNAOAudioOutput:
    """Tests for the NAO audio output stub."""

    def test_play_audio_raises(self):
        """NAOAudioOutput.play_audio raises NotImplementedError."""
        output = NAOAudioOutput(ip="1.2.3.4", port=9600, use_builtin_tts=False)
        result = TTSResult(
            audio_bytes=b"\x00",
            format="pcm",
            sample_rate=24000,
            duration_seconds=1.0,
            synthesis_time_seconds=0.1,
            voice="test",
        )
        with pytest.raises(NotImplementedError):
            output.play_audio(result)

    def test_use_builtin_tts_property(self):
        """NAOAudioOutput exposes use_builtin_tts flag."""
        output_tts = NAOAudioOutput(ip="1.2.3.4", port=9600, use_builtin_tts=True)
        assert output_tts.use_builtin_tts is True

        output_no_tts = NAOAudioOutput(ip="1.2.3.4", port=9600, use_builtin_tts=False)
        assert output_no_tts.use_builtin_tts is False
