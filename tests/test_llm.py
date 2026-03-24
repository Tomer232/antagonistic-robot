"""Tests for the LLM engine with mocked API calls."""

from unittest.mock import MagicMock, patch

import pytest

from antagonist_robot.config.settings import LLMConfig
from antagonist_robot.pipeline.llm import LLMEngine
from antagonist_robot.pipeline.types import LLMResult


class TestLLMEngine:
    """Tests for the provider-agnostic LLM engine."""

    def _make_engine(self):
        """Create an LLMEngine with test config."""
        config = LLMConfig(
            base_url="https://api.test.com/v1",
            model="test-model",
            api_key="fake-key",
            max_tokens=100,
            temperature=0.7,
        )
        with patch("antagonist_robot.pipeline.llm.OpenAI") as mock_cls:
            engine = LLMEngine(config)
            return engine, mock_cls

    def test_generate_sends_correct_messages(self):
        """generate() sends system prompt + messages to the API."""
        config = LLMConfig(
            base_url="https://api.test.com/v1",
            model="test-model",
            api_key="fake-key",
        )

        with patch("antagonist_robot.pipeline.llm.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # Set up mock response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.model = "test-model"
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 42
            mock_client.chat.completions.create.return_value = mock_response

            engine = LLMEngine(config)
            result = engine.generate(
                "You are a test bot.",
                [{"role": "user", "content": "Hello"}],
            )

            # Verify the API was called with correct messages
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0] == {"role": "system", "content": "You are a test bot."}
            assert messages[1] == {"role": "user", "content": "Hello"}

    def test_generate_returns_llm_result(self):
        """generate() returns a properly populated LLMResult."""
        config = LLMConfig(
            base_url="https://api.test.com/v1",
            model="test-model",
            api_key="fake-key",
        )

        with patch("antagonist_robot.pipeline.llm.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "  Hello there!  "
            mock_response.model = "test-model-v2"
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 15
            mock_client.chat.completions.create.return_value = mock_response

            engine = LLMEngine(config)
            result = engine.generate("System", [{"role": "user", "content": "Hi"}])

            assert isinstance(result, LLMResult)
            assert result.text == "Hello there!"  # stripped
            assert result.model == "test-model-v2"
            assert result.total_tokens == 15
            assert result.generation_time_seconds >= 0

    def test_generate_uses_config_parameters(self):
        """generate() passes max_tokens and temperature from config."""
        config = LLMConfig(
            base_url="https://api.test.com/v1",
            model="test-model",
            api_key="fake-key",
            max_tokens=200,
            temperature=0.5,
        )

        with patch("antagonist_robot.pipeline.llm.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            mock_response.model = "test-model"
            mock_response.usage = MagicMock()
            mock_response.usage.total_tokens = 5
            mock_client.chat.completions.create.return_value = mock_response

            engine = LLMEngine(config)
            engine.generate("Sys", [{"role": "user", "content": "Hi"}])

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 200
            assert call_args.kwargs["temperature"] == 0.5
            assert call_args.kwargs["stream"] is False
