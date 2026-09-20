"""Pydantic schemas for end-to-end RAG chat assistant endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.api.v1.schemas.generation import (
    CitationModel,
    GenerationConfigModel,
    ShopperContextModel,
    TokenUsageModel,
)
from src.api.v1.schemas.retrieval import FilterCriteriaModel, RetrievedChunkModel


class ChatApiRequest(BaseModel):
    """Request payload for end-to-end RAG chat assistant."""

    query: str = Field(..., min_length=1, description="Shopper inquiry question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    filters: Optional[FilterCriteriaModel] = Field(
        default=None, description="Optional metadata filters for retrieval"
    )
    shopper_context: Optional[ShopperContextModel] = Field(
        default=None, description="Active cart and shopper context"
    )
    config: Optional[GenerationConfigModel] = Field(
        default=None, description="LLM generation parameters and method override ('runtime' or 'mantle')"
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list, description="Prior conversational turns: [{'role': 'user'|'assistant', 'content': '...'}]"
    )
    require_citations: bool = Field(
        default=True, description="Whether to extract and hydrate structured UI citations"
    )
    include_retrieved_chunks: bool = Field(
        default=False, description="Whether to include retrieved context chunks in response"
    )


class ChatApiResponse(BaseModel):
    """Response payload for end-to-end RAG chat assistant."""

    query: str = Field(..., description="Original shopper inquiry question")
    answer: str = Field(..., description="Synthesized conversational Markdown response")
    citations: List[CitationModel] = Field(
        default_factory=list, description="Structured UI citation cards"
    )
    retrieved_chunks: Optional[List[RetrievedChunkModel]] = Field(
        default=None, description="Raw context chunks used for grounding"
    )
    model_id: str = Field(..., description="Active foundation model utilized")
    method: str = Field(..., description="Active LLM invocation method ('runtime' or 'mantle')")
    latency_ms: float = Field(default=0.0, description="Total end-to-end latency in milliseconds")
    retrieval_latency_ms: float = Field(
        default=0.0, description="Upstream retrieval latency in milliseconds"
    )
    generation_latency_ms: float = Field(
        default=0.0, description="Downstream generation latency in milliseconds"
    )
    token_usage: TokenUsageModel = Field(
        default_factory=TokenUsageModel, description="Token consumption metrics"
    )
    refusal_triggered: bool = Field(
        default=False, description="Whether refusal / out-of-stock policy was triggered"
    )
    refusal_reason: Optional[str] = Field(
        default=None, description="Reason if refusal was triggered"
    )
    support_redirect_url: Optional[str] = Field(
        default=None, description="Customer support redirect URL if applicable"
    )
    degraded: bool = Field(
        default=False, description="Whether graceful fallback was triggered"
    )
    degradation_reason: Optional[str] = Field(
        default=None, description="Error explanation if degraded"
    )
