"""Unit tests for LLMClientFactory and pluggable LLM providers."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List
import pytest

from src.config import IngestionConfig
from src.generation.models import GenerationConfig
from src.generation.providers import (
    BaseLLMClient,
    BedrockMantleClient,
    BedrockRuntimeClient,
    LLMClientFactory,
)


class DummyMockProvider(BaseLLMClient):
    """Mock provider for testing runtime dynamic registration."""

    def __init__(self, app_config=None, **kwargs):
        self.app_config = app_config
        self.kwargs = kwargs

    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        return {"text": "dummy response", "token_usage": None, "stop_reason": "stop"}

    def converse_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        yield "dummy "
        yield "response"


def test_factory_default_creates_runtime():
    """Default method without config creates BedrockRuntimeClient."""
    config = IngestionConfig(llm_method="runtime")
    client = LLMClientFactory.create(app_config=config)
    assert isinstance(client, BedrockRuntimeClient)


def test_factory_creates_mantle():
    """Setting llm_method to 'mantle' creates BedrockMantleClient."""
    config = IngestionConfig(
        llm_method="mantle",
        bedrock_mantle_api_key="test_key",
        bedrock_mantle_base_url="https://example.com/v1",
        bedrock_mantle_project_id="test_proj",
    )
    client = LLMClientFactory.create(app_config=config)
    assert isinstance(client, BedrockMantleClient)
    assert client.app_config.bedrock_mantle_project_id == "test_proj"


def test_factory_explicit_method_override():
    """Passing method argument explicitly overrides config."""
    config = IngestionConfig(llm_method="runtime")
    client = LLMClientFactory.create(method="mantle", app_config=config)
    assert isinstance(client, BedrockMantleClient)


def test_factory_case_insensitive():
    """Method key resolution is case-insensitive."""
    config = IngestionConfig(llm_method="MANTLE")
    client = LLMClientFactory.create(app_config=config)
    assert isinstance(client, BedrockMantleClient)

    client_runtime = LLMClientFactory.create(method="RUNTIME")
    assert isinstance(client_runtime, BedrockRuntimeClient)


def test_factory_dynamic_provider_registration():
    """Test dynamic registration of custom provider."""
    LLMClientFactory.register_provider("mock_custom", DummyMockProvider)
    providers = LLMClientFactory.get_registered_providers()
    assert "mock_custom" in providers

    client = LLMClientFactory.create(method="mock_custom")
    assert isinstance(client, DummyMockProvider)

    # Test converse and converse_stream
    result = client.converse([], [], GenerationConfig())
    assert result["text"] == "dummy response"

    stream_chunks = list(client.converse_stream([], [], GenerationConfig()))
    assert stream_chunks == ["dummy ", "response"]


def test_factory_raises_value_error_for_unknown():
    """Requesting an unregistered method raises ValueError with helpful message."""
    with pytest.raises(ValueError) as exc_info:
        LLMClientFactory.create(method="non_existent_provider_xyz")
    assert "Unknown LLM method: 'non_existent_provider_xyz'" in str(exc_info.value)
    assert "runtime" in str(exc_info.value)
    assert "mantle" in str(exc_info.value)
