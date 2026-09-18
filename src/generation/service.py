"""Service facade providing high-level synchronous and streaming generation capabilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

from src.config import IngestionConfig, get_config
from src.generation.generator import GroundedResponseGenerator
from src.generation.models import (
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    ShopperContext,
    StreamChunk,
)
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class GenerationService:
    """Service facade coordinating generation requests for the REST API and assistant workflow."""

    def __init__(self, generator: Optional[GroundedResponseGenerator] = None) -> None:
        self.generator = generator or GroundedResponseGenerator()

    @classmethod
    def from_config(cls, app_config: Optional[IngestionConfig] = None) -> GenerationService:
        """Instantiates service with injected configuration."""
        cfg = app_config or get_config()
        generator = GroundedResponseGenerator(app_config=cfg)
        return cls(generator=generator)

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        shopper_context: Optional[ShopperContext] = None,
        config: Optional[GenerationConfig] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        require_citations: bool = True,
    ) -> GenerationResponse:
        """Executes full synchronous response generation."""
        req = GenerationRequest(
            query=query,
            chunks=chunks,
            shopper_context=shopper_context,
            config=config,
            conversation_history=conversation_history or [],
            require_citations=require_citations,
        )
        return self.generator.generate(req)

    def generate_stream(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        shopper_context: Optional[ShopperContext] = None,
        config: Optional[GenerationConfig] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        require_citations: bool = True,
    ) -> Iterator[StreamChunk]:
        """Executes streaming response generation yielding SSE-ready chunks."""
        req = GenerationRequest(
            query=query,
            chunks=chunks,
            shopper_context=shopper_context,
            config=config,
            conversation_history=conversation_history or [],
            require_citations=require_citations,
        )
        return self.generator.generate_stream(req)
