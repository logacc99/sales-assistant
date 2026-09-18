"""Unit tests for citation hydrator and bracket parser."""

from __future__ import annotations

import pytest

from src.generation.citation_extractor import (
    CitationExtractor,
    SUPPORT_TITLE,
    SUPPORT_URL,
    format_currency_vnd,
)
from src.generation.models import CitationType
from src.retrieval.models import RetrievedChunk


@pytest.fixture
def sample_chunks():
    c1 = RetrievedChunk(
        chunk_id="chk_prod",
        doc_id="doc_prod",
        category="product",
        content="Bếp Weber Spirit II",
        score=0.9,
        metadata={
            "name": "Bếp Weber Spirit II",
            "price": 12500000.0,
            "product_url": "https://store.example.com/p/bbq",
            "stock_status": "in_stock",
            "sku": "SKU-BBQ",
        },
    )
    c2 = RetrievedChunk(
        chunk_id="chk_promo",
        doc_id="doc_promo",
        category="promotion",
        content="Mã SALE10",
        score=0.85,
        metadata={
            "promo_code": "SALE10",
            "discount_type": "percentage",
            "discount_value": 10.0,
            "terms_url": "https://store.example.com/promo",
        },
    )
    c3 = RetrievedChunk(
        chunk_id="chk_policy",
        doc_id="doc_policy",
        category="policy",
        content="Chính sách đổi trả",
        score=0.8,
        metadata={
            "section_title": "Đổi trả hàng",
            "policy_url": "https://store.example.com/policy/returns",
        },
    )
    return [c1, c2, c3]


def test_format_currency_vnd():
    assert format_currency_vnd(12500000.0) == "12.500.000 ₫"
    assert format_currency_vnd(50000) == "50.000 ₫"
    assert format_currency_vnd(None) is None


def test_parse_cited_indices():
    extractor = CitationExtractor()
    text = "Sản phẩm bếp nướng [1] rất tốt, áp dụng mã [2] và chính sách [3]."
    indices = extractor.parse_cited_indices(text, total_chunks=3)
    assert indices == [0, 1, 2]

    # Test combined brackets like [1, 2] and out-of-range index [99]
    text2 = "Xem chi tiết tại [1, 2] hoặc [99]."
    indices2 = extractor.parse_cited_indices(text2, total_chunks=3)
    assert indices2 == [0, 1]


def test_extract_citations_with_brackets(sample_chunks):
    extractor = CitationExtractor()
    answer = "Bạn có thể tham khảo Bếp Weber [1] và dùng mã SALE10 [2]."
    citations = extractor.extract_citations(answer, sample_chunks)

    assert len(citations) == 2
    assert citations[0].citation_type == CitationType.PRODUCT_CTA
    assert citations[0].title == "Bếp Weber Spirit II"
    assert citations[0].price_display == "12.500.000 ₫"

    assert citations[1].citation_type == CitationType.PROMO_TERMS
    assert citations[1].title == "Mã: SALE10"
    assert citations[1].discount_display == "Giảm 10.0%"


def test_extract_citations_fallback(sample_chunks):
    extractor = CitationExtractor(max_fallback_citations=2)
    answer = "Sản phẩm này rất tốt cho gia đình bạn."  # No brackets!
    citations = extractor.extract_citations(answer, sample_chunks)

    # Should fall back to top-2 chunks
    assert len(citations) == 2
    assert citations[0].citation_type == CitationType.PRODUCT_CTA
    assert citations[1].citation_type == CitationType.PROMO_TERMS


def test_extract_citations_with_out_of_stock_or_refusal(sample_chunks):
    extractor = CitationExtractor()
    answer = "Sản phẩm hiện đang tạm hết hàng [1]."
    citations = extractor.extract_citations(
        answer, sample_chunks, refusal_triggered=True, has_out_of_stock=True
    )

    # Should include chunk 1 and support redirect CTA
    types = [c.citation_type for c in citations]
    assert CitationType.PRODUCT_CTA in types
    assert CitationType.SUPPORT_REDIRECT in types

    support_cit = [c for c in citations if c.citation_type == CitationType.SUPPORT_REDIRECT][0]
    assert support_cit.url == SUPPORT_URL
    assert support_cit.title == SUPPORT_TITLE
