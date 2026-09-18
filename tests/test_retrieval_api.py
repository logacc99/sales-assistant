"""Unit tests for Retrieval REST API endpoints."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_hybrid_retriever,
    get_retrieval_service,
)
from src.api.main import app
from src.retrieval.models import (
    RetrievalResult,
    RetrievedChunk,
    SearchType,
)
from src.retrieval.service import RetrievalService

client = TestClient(app)


@pytest.fixture
def mock_retrieval_deps():
    mock_retriever = MagicMock()
    mock_service = RetrievalService(retriever=mock_retriever)

    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    yield {
        "retriever": mock_retriever,
        "service": mock_service,
    }

    app.dependency_overrides.clear()


def test_search_endpoint_success(mock_retrieval_deps):
    mock_retriever = mock_retrieval_deps["retriever"]

    sample_chunk = RetrievedChunk(
        chunk_id="chk_api_1",
        doc_id="doc_api_1",
        category="product",
        content="Bếp nướng gas ngoài trời BBQ 4 họng",
        score=0.952,
        lexical_score=14.5,
        vector_score=0.89,
        rrf_score=0.032,
        rerank_score=0.952,
        metadata={"product_name": "Bếp BBQ 4 họng", "price": 8500000.0},
    )

    mock_retriever.retrieve.return_value = RetrievalResult(
        query="bep nuong gas",
        search_type=SearchType.HYBRID,
        total_hits=1,
        chunks=[sample_chunk],
        latency_ms=35.4,
        retrieval_mode_used="hybrid_msearch_rrf_cohere_rerank",
        degraded=False,
    )

    resp = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "bep nuong gas",
            "search_type": "hybrid",
            "top_k": 5,
            "filters": {
                "category": "product",
                "stock_status": "in_stock",
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["query"] == "bep nuong gas"
    assert data["data"]["total_hits"] == 1
    assert data["data"]["chunks"][0]["chunk_id"] == "chk_api_1"
    assert data["data"]["chunks"][0]["score"] == 0.952


def test_search_endpoint_empty_query_fails():
    resp = client.post(
        "/api/v1/retrieval/search",
        json={"query": "   "},
    )
    assert resp.status_code == 400
    data = resp.json()
    error_msg = data.get("message") or data.get("detail", "")
    assert "cannot be empty" in error_msg


def test_explain_endpoint_success(mock_retrieval_deps):
    mock_retriever = mock_retrieval_deps["retriever"]

    sample_chunk = RetrievedChunk(
        chunk_id="chk_exp_1",
        doc_id="doc_exp_1",
        category="product",
        content="Bếp nướng dã ngoại",
        score=0.88,
        lexical_score=11.2,
        vector_score=0.85,
        rrf_score=0.031,
        rerank_score=0.88,
        lexical_rank=1,
        vector_rank=1,
        rerank_rank=1,
        metadata={"product_name": "Bếp dã ngoại"},
    )

    mock_retriever.retrieve.return_value = RetrievalResult(
        query="bếp dã ngoại",
        search_type=SearchType.HYBRID,
        total_hits=1,
        chunks=[sample_chunk],
        latency_ms=28.1,
        retrieval_mode_used="hybrid_msearch_rrf_cohere_rerank",
        degraded=False,
    )

    resp = client.post(
        "/api/v1/retrieval/explain",
        json={"query": "bếp dã ngoại", "top_k": 3},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    breakdown = data["data"]["scoring_breakdown"]
    assert len(breakdown) == 1
    assert breakdown[0]["chunk_id"] == "chk_exp_1"
    assert breakdown[0]["final_rank"] == 1
    assert breakdown[0]["lexical_rank"] == 1


def test_health_endpoint(mock_retrieval_deps):
    mock_retriever = mock_retrieval_deps["retriever"]
    mock_retriever.index_name = "sales-assistant-catalog"
    mock_retriever.client.ping.return_value = True
    mock_retriever.client.indices.exists.return_value = True
    mock_retriever.client.count.return_value = {"count": 42}
    mock_retriever.embedder = MagicMock()
    mock_retriever.reranker = MagicMock()

    resp = client.get("/api/v1/retrieval/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"
    assert data["data"]["opensearch_connected"] is True
    assert data["data"]["index_exists"] is True
    assert data["data"]["document_count"] == 42
    assert "is_serverless" in data["data"]
    assert "collection_type" in data["data"]


def test_health_endpoint_serverless_fallback(mock_retrieval_deps):
    mock_retriever = mock_retrieval_deps["retriever"]
    mock_retriever.index_name = "sales-assistant-index"
    # AOSS root HEAD / returns 404 or ping False
    mock_retriever.client.ping.return_value = False
    mock_retriever.client.indices.exists.return_value = True
    mock_retriever.client.count.return_value = {"count": 142}
    mock_retriever.embedder = MagicMock()
    mock_retriever.reranker = MagicMock()

    resp = client.get("/api/v1/retrieval/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"
    assert data["data"]["opensearch_connected"] is True
    assert data["data"]["index_exists"] is True
    assert data["data"]["document_count"] == 142
    assert data["data"]["is_serverless"] is True
    assert data["data"]["collection_type"] == "VECTORSEARCH"
