"""SQLite-based session and turn logger with WAV file saving.

Two tables: sessions (metadata) and turns (per-turn data with latency).
Audio files saved as WAV under data/audio/SESSION_ID/.
"""

import json
import sqlite3
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from roastcrowd.pipeline.types import ASRResult, LLMResult, TTSResult, TurnResult

class SessionLogger:
    """SQLite-based logger for research data collection."""

    def __init__(self, db_path: str, audio_dir: str, save_audio: bool = True):
        self._db_path = db_path
        self._audio_dir = audio_dir
        self._save_audio = save_audio

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create sessions and turns tables if they do not exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                polar_level INTEGER NOT NULL,
                category TEXT NOT NULL,
                subtype INTEGER NOT NULL,
                modifiers_json TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config_snapshot TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                user_audio_path TEXT,
                user_transcript TEXT,
                transcript_confidence REAL,
                llm_input TEXT,
                llm_output TEXT,
                llm_model TEXT,
                tokens_used INTEGER,
                tts_voice TEXT,
                tts_audio_path TEXT,
                
                hostility_level INTEGER,
                polar_level INTEGER,
                category TEXT,
                subtype INTEGER,
                modifiers_json TEXT,
                risk_rating TEXT,
                
                latency_vad_ms INTEGER,
                latency_asr_ms INTEGER,
                latency_llm_ms INTEGER,
                latency_tts_ms INTEGER,
                latency_total_ms INTEGER
            );
        """)
        
        # Super simple auto-migration for existing db schema
        migrations = [
            "ALTER TABLE sessions ADD COLUMN polar_level INTEGER DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN category TEXT DEFAULT 'D'",
            "ALTER TABLE sessions ADD COLUMN subtype INTEGER DEFAULT 1",
            "ALTER TABLE sessions ADD COLUMN modifiers_json TEXT DEFAULT '[]'",
            "ALTER TABLE sessions ADD COLUMN config_snapshot TEXT",
            "ALTER TABLE turns ADD COLUMN polar_level INTEGER DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN category TEXT DEFAULT 'D'",
            "ALTER TABLE turns ADD COLUMN subtype INTEGER DEFAULT 1",
            "ALTER TABLE turns ADD COLUMN modifiers_json TEXT DEFAULT '[]'",
            "ALTER TABLE turns ADD COLUMN risk_rating TEXT DEFAULT 'UNKNOWN'",
        ]
        for query in migrations:
            try:
                self._conn.execute(query)
            except sqlite3.OperationalError:
                pass  # column already exists
        
        self._conn.commit()

    def create_session(
        self,
        session_id: str,
        participant_id: str,
        polar_level: int,
        category: str,
        subtype: int,
        modifiers: list,
        config_snapshot: Optional[dict] = None,
    ) -> None:
        """Create a new session record in the database."""
        self._conn.execute(
            "INSERT INTO sessions (session_id, participant_id, hostility_level, polar_level, category, subtype, modifiers_json, "
            "start_time, config_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                participant_id,
                polar_level,
                polar_level,
                category,
                subtype,
                json.dumps(modifiers),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(config_snapshot) if config_snapshot else None,
            ),
        )
        self._conn.commit()

        if self._save_audio:
            session_audio_dir = Path(self._audio_dir) / session_id
            session_audio_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self,
        session_id: str,
        turn: TurnResult,
        asr_result: ASRResult,
        llm_result: LLMResult,
        system_prompt: str,
    ) -> None:
        """Log a complete turn to the database and save audio files."""
        user_audio_path = None
        tts_audio_path = None

        if self._save_audio:
            session_audio_dir = Path(self._audio_dir) / session_id

            if turn.user_audio is not None:
                user_audio_path = str(session_audio_dir / f"turn_{turn.turn_number:03d}_user.wav")
                self._save_wav(user_audio_path, turn.user_audio.samples, turn.user_audio.sample_rate)

            if turn.tts_result is not None:
                if turn.tts_result.format == "pcm":
                    tts_audio_path = str(session_audio_dir / f"turn_{turn.turn_number:03d}_agent.wav")
                    self._save_pcm_as_wav(tts_audio_path, turn.tts_result.audio_bytes, turn.tts_result.sample_rate)
                else:
                    ext = turn.tts_result.format
                    tts_audio_path = str(session_audio_dir / f"turn_{turn.turn_number:03d}_agent.{ext}")
                    with open(tts_audio_path, "wb") as f:
                        f.write(turn.tts_result.audio_bytes)

        llm_input_log = json.dumps({
            "system_prompt": system_prompt,
            "messages": turn.user_audio is not None and [{"role": "user", "content": turn.transcript}] or [],
        })

        self._conn.execute(
            "INSERT INTO turns (session_id, turn_number, timestamp, "
            "user_audio_path, user_transcript, transcript_confidence, "
            "llm_input, llm_output, llm_model, tokens_used, "
            "tts_voice, tts_audio_path, hostility_level, "
            "polar_level, category, subtype, modifiers_json, risk_rating, "
            "latency_vad_ms, latency_asr_ms, latency_llm_ms, "
            "latency_tts_ms, latency_total_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, turn.turn_number, turn.timestamp,
                user_audio_path, turn.transcript, asr_result.confidence,
                llm_input_log, turn.llm_response, llm_result.model, llm_result.total_tokens,
                turn.tts_result.voice if turn.tts_result else None, tts_audio_path,
                turn.hostility_level, turn.polar_level, turn.category, turn.subtype,
                json.dumps(turn.modifiers), turn.risk_rating,
                turn.latency.get("vad_ms"), turn.latency.get("asr_ms"), turn.latency.get("llm_ms"),
                turn.latency.get("tts_ms"), turn.latency.get("total_ms"),
            ),
        )
        self._conn.commit()

    def end_session(self, session_id: str) -> None:
        """Set the end time on a session record."""
        self._conn.execute(
            "UPDATE sessions SET end_time = ? WHERE session_id = ?",
            (datetime.now(timezone.utc).isoformat(), session_id),
        )
        self._conn.commit()

    def get_sessions(self) -> list:
        cursor = self._conn.execute("SELECT * FROM sessions ORDER BY start_time DESC")
        return [dict(row) for row in cursor.fetchall()]

    def export_session(self, session_id: str) -> dict:
        row = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            return {"session": None, "turns": []}

        session = dict(row)
        turns = [dict(r) for r in self._conn.execute("SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number", (session_id,)).fetchall()]
        return {"session": session, "turns": turns}

    def _save_wav(self, path: str, samples: np.ndarray, sample_rate: int) -> None:
        int16_samples = (samples * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_samples.tobytes())

    def _save_pcm_as_wav(self, path: str, pcm_bytes: bytes, sample_rate: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)

    def close(self) -> None:
        self._conn.close()
