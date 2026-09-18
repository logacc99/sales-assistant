"""Unit tests for generation data models and serialization."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.generation.models import (
    CartItem,
    Citation,
    CitationType,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    ShopperContext,
    StreamChunk,
    TokenUsage,
)
from src.retrieval.models import RetrievedChunk


def test_citation_serialization():
    cit = Citation(
        citation_type=CitationType.PRODUCT_CTA,
        title="Bếp Gas Weber",
        url="https://store.example.com/p/123",
        price_display="12.500.000 ₫",
        stock_status="in_stock",
        metadata={"sku": "BBQ-01"},
    )
    d = cit.to_dict()
    assert d["citation_type"] == "product_cta"
    assert d["title"] == "Bếp Gas Weber"
    assert d["url"] == "https://store.example.com/p/123"
    assert d["price_display"] == "12.500.000 ₫"
    assert d["stock_status"] == "in_stock"
    assert d["metadata"] == {"sku": "BBQ-01"}
    assert "promo_code" not in d  # None dropped


def test_shopper_context_serialization():
    now = datetime(2026, 9, 19, 10, 0, 0)
    item = CartItem(sku="SKU-1", product_id="P1", name="Product 1", price=100.0, quantity=2)
    ctx = ShopperContext(cart_items=[item], cart_subtotal=200.0, current_time=now, user_id="U123")

    d = ctx.to_dict()
    assert d["cart_subtotal"] == 200.0
    assert d["locale"] == "vi_VN"
    assert d["user_id"] == "U123"
    assert len(d["cart_items"]) == 1
    assert d["cart_items"][0]["sku"] == "SKU-1"


def test_generation_response_serialization():
    cit = Citation(
        citation_type=CitationType.SUPPORT_REDIRECT,
        title="Hỗ trợ",
        url="https://store.example.com/support",
    )
    usage = TokenUsage(input_tokens=150, output_tokens=50, total_tokens=200)
    resp = GenerationResponse(
        query="Bếp nướng giá bao nhiêu?",
        answer="Bếp có giá 12.500.000 ₫ [1]",
        citations=[cit],
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        refusal_triggered=False,
        token_usage=usage,
        latency_ms=123.456,
        degraded=False,
    )

    d = resp.to_dict()
    assert d["query"] == "Bếp nướng giá bao nhiêu?"
    assert d["latency_ms"] == 123.46
    assert len(d["citations"]) == 1
    assert d["token_usage"]["total_tokens"] == 200
    assert d["refusal_triggered"] is False


def test_stream_chunk_to_sse():
    cit = Citation(
        citation_type=CitationType.POLICY_LINK,
        title="Chính sách đổi trả",
        url="https://store.example.com/returns",
    )
    chunk = StreamChunk(event="citation", citation=cit)
    sse = chunk.to_sse()
    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")

    json_str = sse[len("data: ") : -2]
    parsed = json.loads(json_str)
    assert parsed["event"] == "citation"
    assert parsed["citation"]["title"] == "Chính sách đổi trả"
