"""Unit tests for prompt builder, XML formatting, and context budgeting."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.generation.budget import ContextBudgetManager
from src.generation.models import CartItem, GenerationRequest, ShopperContext
from src.generation.prompts import (
    NO_STACKING_DISCLAIMER_VI,
    OUT_OF_STOCK_NOTICE_VI,
    PromptBuilder,
)
from src.retrieval.models import RetrievedChunk


@pytest.fixture
def sample_chunks():
    c1 = RetrievedChunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        category="product",
        content="Bếp nướng Weber Spirit II E-210 với 2 đầu đốt thép không gỉ.",
        score=0.92,
        metadata={
            "name": "Bếp nướng Weber Spirit II",
            "sku": "BBQ-01",
            "price": 12500000.0,
            "stock_status": "in_stock",
            "product_url": "https://store.example.com/p/bbq-01",
        },
    )
    c2 = RetrievedChunk(
        chunk_id="chk_2",
        doc_id="doc_2",
        category="promotion",
        content="Giảm 10% cho đơn hàng từ 1.000.000đ khi nhập mã SALE10.",
        score=0.85,
        metadata={
            "promo_code": "SALE10",
            "discount_type": "percentage",
            "discount_value": 10.0,
            "min_spend": 1000000.0,
            "terms_url": "https://store.example.com/promo/sale10",
        },
    )
    c3 = RetrievedChunk(
        chunk_id="chk_3",
        doc_id="doc_3",
        category="policy",
        content="Chính sách đổi trả trong vòng 30 ngày kể từ ngày nhận hàng.",
        score=0.78,
        metadata={
            "section_title": "Chính sách đổi trả",
            "policy_type": "returns",
            "policy_url": "https://store.example.com/returns",
        },
    )
    return [c1, c2, c3]


def test_format_chunk_xml(sample_chunks):
    builder = PromptBuilder()

    xml1 = builder.format_chunk_xml(sample_chunks[0], 1)
    assert '<chunk index="[1]" category="product"' in xml1
    assert 'name="Bếp nướng Weber Spirit II"' in xml1
    assert 'price="12500000.0"' in xml1
    assert "Bếp nướng Weber Spirit II E-210" in xml1

    xml2 = builder.format_chunk_xml(sample_chunks[1], 2)
    assert '<chunk index="[2]" category="promotion"' in xml2
    assert 'promo_code="SALE10"' in xml2
    assert 'discount_type="percentage"' in xml2

    xml3 = builder.format_chunk_xml(sample_chunks[2], 3)
    assert '<chunk index="[3]" category="policy"' in xml3
    assert 'policy_type="returns"' in xml3


def test_build_messages_with_history(sample_chunks):
    builder = PromptBuilder()
    shopper_ctx = ShopperContext(
        cart_items=[CartItem(sku="BBQ-01", product_id="P1", name="Bếp Weber", price=12500000.0, quantity=1)],
        cart_subtotal=12500000.0,
    )
    history = [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Dạ chào bạn, tôi có thể giúp gì cho bạn?"},
    ]
    req = GenerationRequest(
        query="Bếp nướng có mã giảm giá nào không?",
        chunks=sample_chunks,
        shopper_context=shopper_ctx,
        conversation_history=history,
    )

    system_prompts, messages = builder.build_messages(req, sample_chunks)

    assert len(system_prompts) == 1
    assert "Trợ lý Bán hàng Trực tuyến" in system_prompts[0]["text"]
    assert NO_STACKING_DISCLAIMER_VI in system_prompts[0]["text"]
    assert OUT_OF_STOCK_NOTICE_VI in system_prompts[0]["text"]

    # History + current prompt
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert "<customer_query>" in messages[2]["content"][0]["text"]
    assert "<shopper_context>" in messages[2]["content"][0]["text"]
    assert '<chunk index="[1]"' in messages[2]["content"][0]["text"]


def test_context_budget_manager():
    # Setup manager with tight token limit
    manager = ContextBudgetManager(max_context_tokens=50, chars_per_token=3.0)

    c_short = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        category="product",
        content="Short content",
        score=0.9,
    )
    c_long = RetrievedChunk(
        chunk_id="c2",
        doc_id="d2",
        category="policy",
        content="Very long content " * 100,  # ~1800 chars -> 600 tokens
        score=0.8,
    )

    retained, report = manager.budget_chunks([c_short, c_long])
    assert len(retained) == 1
    assert retained[0].chunk_id == "c1"
    assert report.chunks_dropped == 1
    assert report.is_truncated is True
