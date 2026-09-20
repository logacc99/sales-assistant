"""Abstract base class and contract for pluggable LLM inference providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

from src.generation.models import GenerationConfig


class BaseLLMClient(ABC):
    """Abstract interface for LLM conversational synthesis."""

    @abstractmethod
    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Executes non-streaming completion.

        Returns:
            Dict matching structure:
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "synthesized text"}],
                    }
                },
                "usage": {
                    "inputTokens": int,
                    "outputTokens": int,
                    "totalTokens": int,
                },
                "stopReason": str,
            }
        """
        pass

    @abstractmethod
    def converse_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Executes streaming completion, yielding string token deltas."""
        pass


# Backward compatibility alias
BaseBedrockClient = BaseLLMClient
