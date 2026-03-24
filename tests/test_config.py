"""Tests for the configuration loader."""

import os
import tempfile
import pytest
import yaml

from antagonist_robot.config.settings import load_config, AppConfig


def _write_config(tmp_dir: str, config_dict: dict) -> str:
    """Write a config dict to a YAML file and return its path."""
    path = os.path.join(tmp_dir, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
    return path


def _minimal_config():
    """Return a minimal valid config dict."""
    return {
        "audio": {"sample_rate": 16000},
        "asr": {"model_size": "base.en"},
        "llm": {
            "base_url": "https://api.x.ai/v1",
            "model": "grok-4-fast",
            "api_key_env": "TEST_LLM_KEY",
        },
        "tts": {
            "engine": "openai",
            "api_key_env": "TEST_TTS_KEY",
        },
        "nao": {"mode": "simulated"},
        "hostility": {"default_level": 3, "prompts_dir": "prompts"},
        "logging": {"db_path": "data/test.db", "audio_dir": "data/audio"},
        "server": {"port": 8000},
    }


class TestLoadConfig:
    """Tests for load_config()."""

    def test_valid_config_loads(self, tmp_path, monkeypatch):
        """A valid YAML with required env vars produces an AppConfig."""
        monkeypatch.setenv("TEST_LLM_KEY", "fake-llm-key")
        monkeypatch.setenv("TEST_TTS_KEY", "fake-tts-key")

        config_dict = _minimal_config()
        path = _write_config(str(tmp_path), config_dict)

        config = load_config(path)
        assert isinstance(config, AppConfig)
        assert config.audio.sample_rate == 16000
        assert config.asr.model_size == "base.en"
        assert config.llm.model == "grok-4-fast"
        assert config.llm.api_key == "fake-llm-key"
        assert config.tts.api_key == "fake-tts-key"

    def test_missing_file_raises(self):
        """A non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_missing_llm_api_key_raises(self, tmp_path, monkeypatch):
        """Missing LLM API key env var raises ValueError."""
        monkeypatch.delenv("TEST_LLM_KEY", raising=False)
        monkeypatch.setenv("TEST_TTS_KEY", "fake-tts-key")

        path = _write_config(str(tmp_path), _minimal_config())

        with pytest.raises(ValueError, match="TEST_LLM_KEY"):
            load_config(path)

    def test_missing_tts_api_key_raises(self, tmp_path, monkeypatch):
        """Missing TTS API key env var raises ValueError."""
        monkeypatch.setenv("TEST_LLM_KEY", "fake-llm-key")
        monkeypatch.delenv("TEST_TTS_KEY", raising=False)

        path = _write_config(str(tmp_path), _minimal_config())

        with pytest.raises(ValueError, match="TEST_TTS_KEY"):
            load_config(path)

    def test_defaults_applied_for_missing_fields(self, tmp_path, monkeypatch):
        """Missing optional fields use their dataclass defaults."""
        monkeypatch.setenv("TEST_LLM_KEY", "fake-llm-key")
        monkeypatch.setenv("TEST_TTS_KEY", "fake-tts-key")

        # Minimal config without audio section
        config_dict = _minimal_config()
        del config_dict["audio"]
        path = _write_config(str(tmp_path), config_dict)

        config = load_config(path)
        assert config.audio.sample_rate == 16000  # default
        assert config.audio.silence_threshold_ms == 700  # default

    def test_extra_keys_ignored(self, tmp_path, monkeypatch):
        """Unknown keys in YAML are silently ignored."""
        monkeypatch.setenv("TEST_LLM_KEY", "fake-llm-key")
        monkeypatch.setenv("TEST_TTS_KEY", "fake-tts-key")

        config_dict = _minimal_config()
        config_dict["llm"]["unknown_field"] = "should be ignored"
        path = _write_config(str(tmp_path), config_dict)

        config = load_config(path)
        assert config.llm.model == "grok-4-fast"
