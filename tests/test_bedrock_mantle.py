"""Unit tests for BedrockMantleClient with mocked OpenAI SDK."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import openai

from src.config import IngestionConfig
from src.generation.models import GenerationConfig
from src.generation.providers.mantle import BedrockMantleClient


@pytest.fixture
def mantle_config():
    return IngestionConfig(
        llm_method="mantle",
        bedrock_mantle_api_key="mock_mantle_api_key",
        bedrock_mantle_base_url="https://bedrock-mantle.example.com/v1",
        bedrock_mantle_project_id="default_project",
        bedrock_mantle_model_id="openai.gpt-oss-120b",
    )


@pytest.fixture
def mock_openai_client():
    mock = MagicMock()

    # Create mock non-streaming completion response
    mock_choice = SimpleNamespace(
        message=SimpleNamespace(content="Dạ chào bạn, đây là câu trả lời từ Mantle."),
        finish_reason="stop",
    )
    mock_usage = SimpleNamespace(prompt_tokens=45, completion_tokens=20, total_tokens=65)
    mock_response = SimpleNamespace(choices=[mock_choice], usage=mock_usage)
    mock.chat.completions.create.return_value = mock_response

    return mock


def test_mantle_init_client_parameters(mantle_config):
    """Test client initialization passes base_url, api_key, and default_query."""
    with patch("src.generation.providers.mantle.OpenAI") as mock_openai_cls:
        client = BedrockMantleClient(app_config=mantle_config)
        mock_openai_cls.assert_called_once_with(
            api_key="mock_mantle_api_key",
            base_url="https://bedrock-mantle.example.com/v1",
            default_query={"project_id": "default_project"},
        )


def test_mantle_format_messages(mantle_config, mock_openai_client):
    """Test system prompts and user/assistant messages formatted into OpenAI schema."""
    client = BedrockMantleClient(app_config=mantle_config, openai_client=mock_openai_client)

    system_prompts = [{"text": "Bạn là trợ lý bán hàng chuyên nghiệp."}]
    messages = [
        {"role": "user", "content": [{"text": "Sản phẩm này có giá bao nhiêu?"}]},
        {"role": "assistant", "content": "Dạ giá là 500k ạ."},
        {"role": "user", "content": "Có được freeship không?"},
    ]

    formatted = client._format_messages(messages, system_prompts)
    assert len(formatted) == 4
    assert formatted[0] == {"role": "system", "content": "Bạn là trợ lý bán hàng chuyên nghiệp."}
    assert formatted[1] == {"role": "user", "content": "Sản phẩm này có giá bao nhiêu?"}
    assert formatted[2] == {"role": "assistant", "content": "Dạ giá là 500k ạ."}
    assert formatted[3] == {"role": "user", "content": "Có được freeship không?"}


def test_mantle_converse_sync_success(mantle_config, mock_openai_client):
    """Test synchronous converse call parses text, usage, and finish reason."""
    client = BedrockMantleClient(app_config=mantle_config, openai_client=mock_openai_client)

    gen_config = GenerationConfig(temperature=0.2, max_tokens=1000)
    res = client.converse(
        messages=[{"role": "user", "content": "Xin chào"}],
        system_prompts=[{"text": "System prompt"}],
        config=gen_config,
    )

    # Check payload sent to mock client
    mock_openai_client.chat.completions.create.assert_called_once()
    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "openai.gpt-oss-120b"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 1000
    assert kwargs["stream"] is False

    # Check returned data
    assert res["text"] == "Dạ chào bạn, đây là câu trả lời từ Mantle."
    assert res["token_usage"].input_tokens == 45
    assert res["token_usage"].output_tokens == 20
    assert res["token_usage"].total_tokens == 65
    assert res["stop_reason"] == "stop"


def test_mantle_converse_stream_success(mantle_config, mock_openai_client):
    """Test streaming converse_stream yields text chunks."""
    client = BedrockMantleClient(app_config=mantle_config, openai_client=mock_openai_client)

    # Mock streaming response
    chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Dạ "))])
    chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="chào "))])
    chunk3 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="bạn!"))])
    mock_openai_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    gen_config = GenerationConfig(temperature=0.1)
    tokens = list(
        client.converse_stream(
            messages=[{"role": "user", "content": "Xin chào"}],
            system_prompts=[],
            config=gen_config,
        )
    )

    assert tokens == ["Dạ ", "chào ", "bạn!"]
    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["stream"] is True


def test_mantle_retry_on_rate_limit(mantle_config, mock_openai_client):
    """Test retry with backoff on RateLimitError."""
    client = BedrockMantleClient(
        app_config=mantle_config,
        openai_client=mock_openai_client,
        max_retries=2,
        base_backoff_sec=0.01,
    )

    mock_choice = SimpleNamespace(
        message=SimpleNamespace(content="Success after retry"),
        finish_reason="stop",
    )
    mock_response = SimpleNamespace(choices=[mock_choice], usage=None)

    # First call raises RateLimitError, second succeeds
    mock_openai_client.chat.completions.create.side_effect = [
        openai.RateLimitError("Rate limit exceeded", response=MagicMock(), body={}),
        mock_response,
    ]

    res = client.converse(
        messages=[{"role": "user", "content": "Test"}],
        system_prompts=[],
        config=GenerationConfig(),
    )

    assert res["text"] == "Success after retry"
    assert mock_openai_client.chat.completions.create.call_count == 2
