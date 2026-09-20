"""AWS Bedrock Mantle client wrapper using OpenAI-compatible API gateway."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Iterator, List, Optional

import openai
from openai import OpenAI

from src.config import IngestionConfig, get_config
from src.generation.models import GenerationConfig, TokenUsage
from src.generation.providers.base import BaseLLMClient

logger = logging.getLogger(__name__)


class BedrockMantleClient(BaseLLMClient):
    """Bedrock Mantle client invoking OpenAI-compatible endpoint with telemetry and retry."""

    def __init__(
        self,
        app_config: Optional[IngestionConfig] = None,
        openai_client: Optional[Any] = None,
        max_retries: int = 3,
        base_backoff_sec: float = 0.5,
    ) -> None:
        self.app_config = app_config or get_config()
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._init_openai_client()

    def _init_openai_client(self) -> OpenAI:
        """Initializes the OpenAI client configured for Bedrock Mantle."""
        client_kwargs: Dict[str, Any] = {
            "api_key": self.app_config.bedrock_mantle_api_key or "default_key",
        }
        if self.app_config.bedrock_mantle_base_url:
            client_kwargs["base_url"] = self.app_config.bedrock_mantle_base_url

        if self.app_config.bedrock_mantle_project_id:
            client_kwargs["default_query"] = {
                "project_id": self.app_config.bedrock_mantle_project_id
            }

        return OpenAI(**client_kwargs)

    def _resolve_model_id(self, config: GenerationConfig) -> str:
        """Resolves active model ID for Bedrock Mantle."""
        return config.model_id or self.app_config.bedrock_mantle_model_id

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Formats system prompts and conversational messages into OpenAI messages schema."""
        formatted: List[Dict[str, str]] = []

        for sys_prompt in system_prompts:
            text = sys_prompt.get("text", "")
            if text:
                formatted.append({"role": "system", "content": text})

        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            if isinstance(raw_content, list):
                text_parts = [
                    part.get("text", "")
                    for part in raw_content
                    if isinstance(part, dict) and "text" in part
                ]
                content = "".join(text_parts)
            else:
                content = str(raw_content)

            formatted.append({"role": role, "content": content})

        return formatted

    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Executes synchronous Mantle completion with exponential backoff on rate limits.

        Returns:
            Dict containing:
                - "text": Synthesized markdown string
                - "token_usage": TokenUsage
                - "stop_reason": str
        """
        model_id = self._resolve_model_id(config)
        openai_messages = self._format_messages(messages, system_prompts)

        call_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": openai_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": False,
        }
        if config.stop_sequences:
            call_kwargs["stop"] = config.stop_sequences

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**call_kwargs)

                choice = response.choices[0] if response.choices else None
                full_text = choice.message.content if choice and choice.message else ""
                stop_reason = choice.finish_reason if choice else "stop"

                usage_data = getattr(response, "usage", None)
                usage = TokenUsage(
                    input_tokens=getattr(usage_data, "prompt_tokens", 0) if usage_data else 0,
                    output_tokens=getattr(usage_data, "completion_tokens", 0) if usage_data else 0,
                    total_tokens=getattr(usage_data, "total_tokens", 0) if usage_data else 0,
                )

                return {
                    "text": full_text or "",
                    "token_usage": usage,
                    "stop_reason": stop_reason or "stop",
                }

            except (openai.RateLimitError, openai.APIConnectionError) as e:
                last_err = e
                sleep_time = self.base_backoff_sec * (2**attempt) + random.uniform(0.1, 0.3)
                logger.warning(
                    "Bedrock Mantle transient error (%s). Attempt %d/%d, retrying in %.2fs: %s",
                    type(e).__name__,
                    attempt + 1,
                    self.max_retries,
                    sleep_time,
                    e,
                )
                time.sleep(sleep_time)
            except Exception as e:
                last_err = e
                logger.error("Bedrock Mantle unexpected API error: %s", e)
                raise

        raise last_err or RuntimeError("Bedrock Mantle failed after maximum retries")

    def converse_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Executes streaming Mantle completion, yielding incremental text tokens."""
        model_id = self._resolve_model_id(config)
        openai_messages = self._format_messages(messages, system_prompts)

        call_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": openai_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": True,
        }
        if config.stop_sequences:
            call_kwargs["stop"] = config.stop_sequences

        try:
            stream = self._client.chat.completions.create(**call_kwargs)

            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    text_delta = getattr(delta, "content", None) or ""
                    if text_delta:
                        yield text_delta

        except Exception as e:
            logger.error("Bedrock Mantle converse_stream error: %s", e)
            raise
