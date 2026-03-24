"""Tests for the SQLite session logger."""

import os
import tempfile

import numpy as np
import pytest

from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.pipeline.types import (
    ASRResult,
    AudioData,
    LLMResult,
    TTSResult,
    TurnResult,
)


@pytest.fixture
def logger(tmp_path):
    """Create a SessionLogger with a temp database."""
    db_path = str(tmp_path / "test.db")
    audio_dir = str(tmp_path / "audio")
    lg = SessionLogger(db_path=db_path, audio_dir=audio_dir, save_audio=True)
    yield lg
    lg.close()


def _make_turn(turn_number: int = 1) -> tuple:
    """Create test turn data."""
    audio = AudioData(
        samples=np.random.randn(16000).astype(np.float32),
        sample_rate=16000,
        duration_seconds=1.0,
        recording_started="2026-01-01T00:00:00Z",
        recording_ended="2026-01-01T00:00:01Z",
    )
    asr_result = ASRResult(
        text="Hello there",
        language="en",
        confidence=-0.5,
        transcription_time_seconds=0.1,
    )
    llm_result = LLMResult(
        text="Whatever.",
        model="test-model",
        total_tokens=10,
        generation_time_seconds=0.2,
    )
    tts_result = TTSResult(
        audio_bytes=b"\x00" * 48000,
        format="pcm",
        sample_rate=24000,
        duration_seconds=1.0,
        synthesis_time_seconds=0.3,
        voice="onyx",
    )
    turn = TurnResult(
        turn_number=turn_number,
        user_audio=audio,
        transcript="Hello there",
        llm_response="Whatever.",
        tts_result=tts_result,
        hostility_level=3,
        polar_level=3,
        category="D",
        subtype=2,
        modifiers=[],
        risk_rating="Amber",
        latency={"vad_ms": 100, "asr_ms": 50, "llm_ms": 200, "tts_ms": 300, "total_ms": 650},
        timestamp="2026-01-01T00:00:02Z",
    )
    return turn, asr_result, llm_result


class TestSessionLogger:
    """Tests for the SQLite session logger."""

    def test_create_session(self, logger):
        """create_session inserts a record."""
        logger.create_session("s1", "P001", 3, category="D", subtype=2, modifiers=[], config_snapshot=None)
        sessions = logger.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["participant_id"] == "P001"
        assert sessions[0]["polar_level"] == 3
        assert sessions[0]["end_time"] is None

    def test_log_turn(self, logger):
        """log_turn inserts a complete turn record."""
        logger.create_session("s1", "P001", 3, category="D", subtype=2, modifiers=[], config_snapshot=None)
        turn, asr_result, llm_result = _make_turn()
        logger.log_turn("s1", turn, asr_result, llm_result, "Be mean.")

        data = logger.export_session("s1")
        assert len(data["turns"]) == 1
        t = data["turns"][0]
        assert t["user_transcript"] == "Hello there"
        assert t["llm_output"] == "Whatever."
        assert t["latency_total_ms"] == 650

    def test_end_session(self, logger):
        """end_session sets the end_time."""
        logger.create_session("s1", "P001", 3, category="D", subtype=2, modifiers=[], config_snapshot=None)
        logger.end_session("s1")

        sessions = logger.get_sessions()
        assert sessions[0]["end_time"] is not None

    def test_get_sessions_ordered(self, logger):
        """get_sessions returns sessions in reverse chronological order."""
        logger.create_session("s1", "P001", 1, category="B", subtype=1, modifiers=[], config_snapshot=None)
        logger.create_session("s2", "P002", 2, category="C", subtype=1, modifiers=[], config_snapshot=None)
        sessions = logger.get_sessions()
        # Most recent first
        assert sessions[0]["session_id"] == "s2"

    def test_export_session(self, logger):
        """export_session returns session + all turns."""
        logger.create_session("s1", "P001", 3, category="D", subtype=2, modifiers=[], config_snapshot=None)

        for i in range(3):
            turn, asr_result, llm_result = _make_turn(turn_number=i + 1)
            logger.log_turn("s1", turn, asr_result, llm_result, "Prompt")

        data = logger.export_session("s1")
        assert data["session"]["session_id"] == "s1"
        assert len(data["turns"]) == 3

    def test_export_nonexistent_session(self, logger):
        """Exporting a non-existent session returns None/empty."""
        data = logger.export_session("nonexistent")
        assert data["session"] is None
        assert data["turns"] == []

    def test_audio_files_saved(self, logger, tmp_path):
        """Audio files are saved to disk when save_audio=True."""
        logger.create_session("s1", "P001", 3, category="D", subtype=2, modifiers=[], config_snapshot=None)
        turn, asr_result, llm_result = _make_turn()
        logger.log_turn("s1", turn, asr_result, llm_result, "Prompt")

        # Check user audio WAV exists
        user_wav = tmp_path / "audio" / "s1" / "turn_001_user.wav"
        assert user_wav.exists()

        # Check agent audio WAV exists
        agent_wav = tmp_path / "audio" / "s1" / "turn_001_agent.wav"
        assert agent_wav.exists()
