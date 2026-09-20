"""Factory and registry for pluggable LLM generation providers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from src.config import IngestionConfig, get_config
from src.generation.providers.base import BaseLLMClient
from src.generation.providers.mantle import BedrockMantleClient
from src.generation.providers.runtime import BedrockRuntimeClient

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Factory for creating and managing pluggable LLM inference client instances."""

    _registry: Dict[str, Type[BaseLLMClient]] = {
        "runtime": BedrockRuntimeClient,
        "mantle": BedrockMantleClient,
    }

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[BaseLLMClient]) -> None:
        """
        Dynamically registers a new LLM provider class into the factory.

        Args:
            name: Provider method key (case-insensitive, e.g. "ollama", "vllm", "openai").
            provider_cls: Subclass of BaseLLMClient.
        """
        normalized_name = name.strip().lower()
        cls._registry[normalized_name] = provider_cls
        logger.info("Registered LLM provider '%s' -> %s", normalized_name, provider_cls.__name__)

    @classmethod
    def get_registered_providers(cls) -> Dict[str, Type[BaseLLMClient]]:
        """Returns a copy of registered provider mappings."""
        return dict(cls._registry)

    @classmethod
    def create(
        cls,
        method: Optional[str] = None,
        app_config: Optional[IngestionConfig] = None,
        **kwargs: Any,
    ) -> BaseLLMClient:
        """
        Instantiates the requested or configured LLM client.

        Args:
            method: Explicit method key ("runtime", "mantle", or registered custom key).
                    If None, reads from app_config.llm_method (default: "runtime").
            app_config: IngestionConfig instance.
            **kwargs: Extra kwargs passed directly to provider constructor (e.g. mock clients).

        Returns:
            An instance of BaseLLMClient.

        Raises:
            ValueError: If the requested method is not registered.
        """
        config = app_config or get_config()
        selected_method = (method or config.llm_method or "runtime").strip().lower()

        provider_cls = cls._registry.get(selected_method)
        if provider_cls is None:
            available = sorted(list(cls._registry.keys()))
            raise ValueError(
                f"Unknown LLM method: '{selected_method}'. Registered providers: {available}"
            )

        return provider_cls(app_config=config, **kwargs)
