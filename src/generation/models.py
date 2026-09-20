"""Data models, request/response structures, and citation schemas for Bedrock generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from src.retrieval.models import RetrievedChunk


class CitationType(str, Enum):
    """Types of structured citations generated for frontend UI components."""

    PRODUCT_CTA = "product_cta"
    POLICY_LINK = "policy_link"
    PROMO_TERMS = "promo_terms"
    SUPPORT_REDIRECT = "support_redirect"


@dataclass
class CartItem:
    """Shopper's current cart item for promotion & stock evaluation."""

    sku: str
    product_id: str
    name: str
    price: float
    quantity: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert cart item to dictionary."""
        return {
            "sku": self.sku,
            "product_id": self.product_id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
        }


@dataclass
class ShopperContext:
    """Shopper session context passed to generator."""

    cart_items: List[CartItem] = field(default_factory=list)
    cart_subtotal: float = 0.0
    current_time: datetime = field(default_factory=datetime.utcnow)
    locale: str = "vi_VN"  # "vi_VN" or "en_US"
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert shopper context to dictionary."""
        return {
            "cart_items": [item.to_dict() for item in self.cart_items],
            "cart_subtotal": self.cart_subtotal,
            "current_time": self.current_time.isoformat(),
            "locale": self.locale,
            "user_id": self.user_id,
        }


@dataclass
class Citation:
    """Structured UI citation card object."""

    citation_type: CitationType
    title: str
    url: str
    price_display: Optional[str] = None
    stock_status: Optional[str] = None  # in_stock, out_of_stock, backorder
    promo_code: Optional[str] = None
    discount_display: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize citation to dict, preserving enums and dropping None values."""
        res: Dict[str, Any] = {
            "citation_type": self.citation_type.value if isinstance(self.citation_type, CitationType) else str(self.citation_type),
            "title": self.title,
            "url": self.url,
        }
        if self.price_display is not None:
            res["price_display"] = self.price_display
        if self.stock_status is not None:
            res["stock_status"] = self.stock_status
        if self.promo_code is not None:
            res["promo_code"] = self.promo_code
        if self.discount_display is not None:
            res["discount_display"] = self.discount_display
        if self.metadata:
            res["metadata"] = self.metadata
        return res


@dataclass
class GenerationConfig:
    """Inference parameter overrides for LLM generation."""

    model_id: Optional[str] = None  # None = resolve via configured provider default
    temperature: float = 0.1  # Low temperature for strict factual grounding
    top_p: float = 0.9
    max_tokens: int = 1500
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt_override: Optional[str] = None
    method: Optional[str] = None  # Optional provider override ("runtime" or "mantle")


@dataclass
class GenerationRequest:
    """Input payload to the generation subsystem."""

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    shopper_context: Optional[ShopperContext] = None
    config: Optional[GenerationConfig] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    require_citations: bool = True


@dataclass
class TokenUsage:
    """Token consumption telemetry."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GenerationResponse:
    """Final output response bundle with text, citations, and telemetry."""

    query: str
    answer: str  # Markdown text for conversational display
    citations: List[Citation] = field(default_factory=list)
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    refusal_triggered: bool = False
    refusal_reason: Optional[str] = None
    support_redirect_url: Optional[str] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    degraded: bool = False
    degradation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for REST API serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "model_id": self.model_id,
            "refusal_triggered": self.refusal_triggered,
            "refusal_reason": self.refusal_reason,
            "support_redirect_url": self.support_redirect_url,
            "token_usage": self.token_usage.__dict__,
            "latency_ms": round(self.latency_ms, 2),
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }


@dataclass
class StreamChunk:
    """Incremental event chunk for SSE streaming delivery."""

    event: str  # "delta", "citation", "done", "error"
    text: Optional[str] = None
    citation: Optional[Citation] = None
    data: Optional[Dict[str, Any]] = None

    def to_sse(self) -> str:
        """Format chunk as Server-Sent Event string."""
        import json

        payload: Dict[str, Any] = {"event": self.event}
        if self.text is not None:
            payload["text"] = self.text
        if self.citation is not None:
            payload["citation"] = self.citation.to_dict()
        if self.data is not None:
            payload["data"] = self.data
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
