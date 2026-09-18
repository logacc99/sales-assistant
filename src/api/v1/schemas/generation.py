"""Pydantic schemas for LLM response generation endpoints matching SPEC-0004."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.api.v1.schemas.retrieval import RetrievedChunkModel


class CartItemModel(BaseModel):
    """Cart item schema for shopper context."""

    sku: str = Field(..., description="Product SKU identifier")
    product_id: str = Field(..., description="Unique product ID")
    name: str = Field(..., description="Product title")
    price: float = Field(..., ge=0.0, description="Product price")
    quantity: int = Field(default=1, ge=1, description="Quantity in cart")


class ShopperContextModel(BaseModel):
    """Shopper context schema including cart items and subtotal."""

    cart_items: List[CartItemModel] = Field(default_factory=list, description="Cart items")
    cart_subtotal: float = Field(default=0.0, ge=0.0, description="Cart total subtotal")
    current_time: Optional[datetime] = Field(default=None, description="Current UTC timestamp")
    locale: str = Field(default="vi_VN", description="Language/country locale code")
    user_id: Optional[str] = Field(default=None, description="Optional user ID")


class GenerationConfigModel(BaseModel):
    """Inference parameters for Bedrock Converse API."""

    model_id: Optional[str] = Field(
        default=None, description="Bedrock foundation model ID override"
    )
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Top-p probability cutoff"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=50, le=4096, description="Maximum completion tokens"
    )
    stop_sequences: List[str] = Field(default_factory=list, description="Optional stop tokens")


class GenerateApiRequest(BaseModel):
    """Request payload for synchronous and streaming generation."""

    query: str = Field(..., min_length=1, description="Shopper inquiry question")
    chunks: List[RetrievedChunkModel] = Field(
        default_factory=list, description="Pre-retrieved context chunks from SPEC-0003"
    )
    shopper_context: Optional[ShopperContextModel] = Field(
        default=None, description="Active cart and shopper context"
    )
    config: Optional[GenerationConfigModel] = Field(
        default=None, description="Bedrock model configuration overrides"
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list, description="Optional prior conversation turns: [{'role': 'user'|'assistant', 'content': '...'}]"
    )
    require_citations: bool = Field(
        default=True, description="Whether to extract and hydrate structured UI citations"
    )


class CitationModel(BaseModel):
    """Structured UI citation card object."""

    citation_type: str = Field(..., description="Citation type: product_cta, policy_link, promo_terms, support_redirect")
    title: str = Field(..., description="Display title for the UI card")
    url: str = Field(..., description="Verified destination link or support CTA")
    price_display: Optional[str] = Field(default=None, description="Formatted price display string")
    stock_status: Optional[str] = Field(default=None, description="Stock status (in_stock, out_of_stock, backorder)")
    promo_code: Optional[str] = Field(default=None, description="Promotion voucher code")
    discount_display: Optional[str] = Field(default=None, description="Formatted discount string")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Underlying chunk metadata attributes")


class TokenUsageModel(BaseModel):
    """Token consumption telemetry."""

    input_tokens: int = Field(default=0, description="Input prompt tokens consumed")
    output_tokens: int = Field(default=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens consumed")


class GenerateApiResponse(BaseModel):
    """Response payload for synchronous generation endpoint."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Synthesized Markdown text response")
    citations: List[CitationModel] = Field(
        default_factory=list, description="Structured UI citation cards"
    )
    model_id: str = Field(..., description="Bedrock model ID utilized")
    refusal_triggered: bool = Field(
        default=False, description="Whether refusal / out-of-stock policy was triggered"
    )
    refusal_reason: Optional[str] = Field(
        default=None, description="Reason if refusal was triggered"
    )
    support_redirect_url: Optional[str] = Field(
        default=None, description="Customer support redirect link if applicable"
    )
    token_usage: TokenUsageModel = Field(default_factory=TokenUsageModel, description="Token usage metrics")
    latency_ms: float = Field(default=0.0, description="End-to-end generation latency in milliseconds")
    degraded: bool = Field(default=False, description="True if fallback generation was triggered")
    degradation_reason: Optional[str] = Field(
        default=None, description="Error reason if degraded"
    )
