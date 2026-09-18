"""Unit tests for Generation REST API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_generation_service
from src.api.main import app
from src.generation.generator import GroundedResponseGenerator
from src.generation.models import (
    Citation,
    CitationType,
    GenerationResponse,
    StreamChunk,
    TokenUsage,
)
from src.generation.service import GenerationService

client = TestClient(app)


@pytest.fixture
def mock_generation_service():
    mock_service = MagicMock(spec=GenerationService)

    app.dependency_overrides[get_generation_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


def test_generate_endpoint_success(mock_generation_service):
    cit = Citation(
        citation_type=CitationType.PRODUCT_CTA,
        title="Bếp BBQ Gas",
        url="https://store.example.com/p/1",
        price_display="8.500.000 ₫",
        stock_status="in_stock",
    )
    usage = TokenUsage(input_tokens=120, output_tokens=40, total_tokens=160)
    mock_generation_service.generate.return_value = GenerationResponse(
        query="Bếp nướng gas",
        answer="Bếp nướng gas BBQ [1] có giá 8.500.000 ₫.",
        citations=[cit],
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        refusal_triggered=False,
        token_usage=usage,
        latency_ms=85.2,
        degraded=False,
    )

    payload = {
        "query": "Bếp nướng gas",
        "chunks": [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "category": "product",
                "content": "Bếp nướng gas ngoài trời",
                "score": 0.95,
                "metadata": {"name": "Bếp BBQ Gas", "price": 8500000.0, "product_url": "https://store.example.com/p/1"},
            }
        ],
        "shopper_context": {
            "cart_items": [{"sku": "SKU-1", "product_id": "P1", "name": "Item", "price": 1000.0, "quantity": 1}],
            "cart_subtotal": 1000.0,
        },
    }

    resp = client.post("/api/v1/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "Bếp nướng gas"
    assert "Bếp nướng gas BBQ [1]" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_type"] == "product_cta"
    assert data["citations"][0]["price_display"] == "8.500.000 ₫"
    assert data["token_usage"]["total_tokens"] == 160
    assert data["latency_ms"] == 85.2


def test_generate_endpoint_empty_chunks(mock_generation_service):
    mock_generation_service.generate.return_value = GenerationResponse(
        query="Câu hỏi không có trong catalog",
        answer="Dạ hiện tại cửa hàng chưa có thông tin...",
        citations=[
            Citation(
                citation_type=CitationType.SUPPORT_REDIRECT,
                title="Hỗ trợ",
                url="https://store.example.com/support",
            )
        ],
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        refusal_triggered=True,
        refusal_reason="empty_retrieved_context",
        support_redirect_url="https://store.example.com/support",
    )

    payload = {
        "query": "Câu hỏi không có trong catalog",
        "chunks": [],
    }

    resp = client.post("/api/v1/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["refusal_triggered"] is True
    assert data["refusal_reason"] == "empty_retrieved_context"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["citation_type"] == "support_redirect"


def test_generate_stream_endpoint(mock_generation_service):
    cit = Citation(
        citation_type=CitationType.PRODUCT_CTA,
        title="Bếp BBQ Gas",
        url="https://store.example.com/p/1",
    )

    events = [
        StreamChunk(event="delta", text="Dạ "),
        StreamChunk(event="delta", text="chào bạn!"),
        StreamChunk(event="citation", citation=cit),
        StreamChunk(event="done", data={"query": "Bếp gas", "latency_ms": 50.0}),
    ]
    mock_generation_service.generate_stream.return_value = iter(events)

    payload = {"query": "Bếp gas", "chunks": []}
    resp = client.post("/api/v1/generate/stream", json=payload)

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    body = resp.text
    assert 'data: {"event": "delta", "text": "Dạ "}' in body
    assert 'data: {"event": "delta", "text": "chào bạn!"}' in body
    assert '"event": "citation"' in body
    assert '"event": "done"' in body
