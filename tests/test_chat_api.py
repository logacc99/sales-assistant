"""Unit tests for the end-to-end RAG Chat API endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_generation_service, get_retrieval_service
from src.api.main import app
from src.generation.models import (
    Citation,
    CitationType,
    GenerationResponse,
    StreamChunk,
    TokenUsage,
)
from src.generation.service import GenerationService
from src.retrieval.models import RetrievedChunk
from src.retrieval.service import RetrievalService

client = TestClient(app)


@pytest.fixture
def mock_chat_services():
    mock_ret_service = MagicMock(spec=RetrievalService)
    mock_gen_service = MagicMock(spec=GenerationService)

    app.dependency_overrides[get_retrieval_service] = lambda: mock_ret_service
    app.dependency_overrides[get_generation_service] = lambda: mock_gen_service

    yield mock_ret_service, mock_gen_service

    app.dependency_overrides.clear()


def test_chat_endpoint_synchronous(mock_chat_services):
    """Test full synchronous chat flow with mock retrieval and generation."""
    mock_ret_service, mock_gen_service = mock_chat_services

    sample_chunk = RetrievedChunk(
        chunk_id="chk_weber",
        doc_id="doc_weber",
        category="product",
        content="Bếp nướng Weber Spirit II E-210 có giá 12.500.000 ₫.",
        score=0.92,
        metadata={
            "name": "Bếp Weber Spirit II",
            "price": 12500000.0,
            "stock_status": "in_stock",
            "product_url": "https://store.example.com/p/weber-spirit-2",
        },
    )
    mock_ret_service.search.return_value = SimpleNamespace(chunks=[sample_chunk])

    cit = Citation(
        citation_type=CitationType.PRODUCT_CTA,
        title="Bếp Weber Spirit II",
        url="https://store.example.com/p/weber-spirit-2",
        price_display="12.500.000 ₫",
        stock_status="in_stock",
    )
    usage = TokenUsage(input_tokens=150, output_tokens=50, total_tokens=200)

    mock_gen_service.generate.return_value = GenerationResponse(
        query="Bếp nướng Weber giá bao nhiêu?",
        answer="Dạ chào bạn, Bếp nướng Weber Spirit II [1] có giá là 12.500.000 ₫.",
        citations=[cit],
        model_id="openai.gpt-oss-120b",
        refusal_triggered=False,
        token_usage=usage,
        latency_ms=120.5,
        degraded=False,
    )

    payload = {
        "query": "Bếp nướng Weber giá bao nhiêu?",
        "top_k": 3,
        "config": {
            "method": "mantle",
            "temperature": 0.1,
        },
        "include_retrieved_chunks": True,
    }

    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "Bếp nướng Weber giá bao nhiêu?"
    assert "12.500.000" in data["answer"]
    assert data["method"] == "mantle"
    assert data["model_id"] == "openai.gpt-oss-120b"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_type"] == "product_cta"
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["chunk_id"] == "chk_weber"


def test_chat_endpoint_empty_query_fails():
    """Empty query string returns HTTP 400."""
    resp = client.post("/api/v1/chat", json={"query": "   "})
    assert resp.status_code == 400


def test_chat_stream_endpoint(mock_chat_services):
    """Test streaming chat endpoint returns SSE stream."""
    mock_ret_service, mock_gen_service = mock_chat_services

    sample_chunk = RetrievedChunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        category="product",
        content="Weber Spirit II",
        score=0.9,
    )
    mock_ret_service.search.return_value = SimpleNamespace(chunks=[sample_chunk])

    mock_gen_service.generate_stream.return_value = iter([
        StreamChunk(event="delta", text="Dạ chào bạn, "),
        StreamChunk(event="delta", text="Bếp Weber giá 12.5tr."),
        StreamChunk(event="done", data={"query": "Weber"}),
    ])

    payload = {"query": "Weber"}
    resp = client.post("/api/v1/chat/stream", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    content = resp.text
    assert '"event": "delta"' in content
    assert '"event": "done"' in content
