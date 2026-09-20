"""Unit and integration tests for OpenSearchHybridRetriever."""

from unittest.mock import MagicMock
import pytest

from src.retrieval.models import (
    FilterCriteria,
    RetrievalQuery,
    SearchType,
)
from src.retrieval.reranker import NoOpReranker
from src.retrieval.retriever import OpenSearchHybridRetriever


@pytest.fixture
def mock_clients():
    mock_os = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.05] * 1024
    reranker = NoOpReranker()
    return mock_os, mock_embedder, reranker


def test_empty_query_returns_empty_result(mock_clients):
    mock_os, mock_embedder, reranker = mock_clients
    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=reranker,
        index_name="sales-assistant-catalog",
    )
    res = retriever.retrieve(RetrievalQuery(query="   "))
    assert res.total_hits == 0
    assert len(res.chunks) == 0
    assert res.retrieval_mode_used == "empty_query"
    mock_os.search.assert_not_called()
    mock_os.msearch.assert_not_called()


def test_hybrid_msearch_retrieval(mock_clients):
    mock_os, mock_embedder, reranker = mock_clients

    # Mock msearch response
    bm25_hit = {
        "_id": "chk_bm25_1",
        "_score": 12.5,
        "_source": {
            "chunk_id": "chk_bm25_1",
            "doc_id": "doc_1",
            "category": "product",
            "content": "Bếp nướng gas 4 họng BBQ",
            "metadata": {"product_name": "Bếp BBQ 4 họng", "price": 8500000.0},
        },
    }
    knn_hit = {
        "_id": "chk_knn_1",
        "_score": 0.92,
        "_source": {
            "chunk_id": "chk_knn_1",
            "doc_id": "doc_2",
            "category": "product",
            "content": "Bếp nướng dã ngoại cao cấp",
            "metadata": {"product_name": "Bếp dã ngoại", "price": 1200000.0},
        },
    }

    mock_os.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [bm25_hit]}},
            {"hits": {"hits": [knn_hit]}},
        ]
    }

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=reranker,
        index_name="sales-assistant-catalog",
    )

    query = RetrievalQuery(
        query="bếp nướng gia đình",
        search_type=SearchType.HYBRID,
        top_k=5,
        filters=FilterCriteria(category="product", stock_status="in_stock"),
    )

    result = retriever.retrieve(query)

    assert result.total_hits == 2
    assert len(result.chunks) == 2
    assert "hybrid_msearch_rrf" in result.retrieval_mode_used
    assert result.degraded is False
    assert result.filters_applied == {"category": "product", "stock_status": "in_stock"}
    mock_os.msearch.assert_called_once()


def test_parent_child_context_expansion(mock_clients):
    mock_os, mock_embedder, reranker = mock_clients

    policy_hit = {
        "_id": "chk_pol_1",
        "_score": 15.0,
        "_source": {
            "chunk_id": "chk_pol_1",
            "doc_id": "doc_policy_1",
            "category": "policy",
            "content": "Thời gian đổi trả trong vòng 7 ngày.",
            "metadata": {
                "policy_type": "returns",
                "parent_section_content": "CHÍNH SÁCH ĐỔI TRẢ VÀ HOÀN TIỀN:\nThời gian đổi trả trong vòng 7 ngày. Hàng phải còn nguyên tem và hóa đơn mua hàng.",
            },
        },
    }

    mock_os.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [policy_hit]}},
            {"hits": {"hits": []}},
        ]
    }

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=reranker,
    )

    query = RetrievalQuery(
        query="đổi trả hàng lỗi",
        expand_parent_context=True,
        parent_max_chars=2000,
    )
    result = retriever.retrieve(query)

    assert result.total_hits == 1
    chunk = result.chunks[0]
    assert chunk.is_parent_expanded is True
    assert "CHÍNH SÁCH ĐỔI TRẢ VÀ HOÀN TIỀN:" in chunk.content


def test_point_lookups(mock_clients):
    mock_os, mock_embedder, reranker = mock_clients
    mock_os.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "prod_123",
                    "_score": 1.0,
                    "_source": {
                        "chunk_id": "prod_123",
                        "category": "product",
                        "content": "Bếp gas",
                        "metadata": {"product_id": "prod_123", "price": 500000.0},
                    },
                }
            ]
        }
    }

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=reranker,
    )

    found = retriever.search_by_product_id("prod_123")
    assert found is not None
    assert found.chunk_id == "prod_123"

    promos = retriever.search_promotions(min_spend=1000000.0)
    assert len(promos) == 1


def test_rerank_disabled_bypasses_reranking(mock_clients):
    mock_os, mock_embedder, _ = mock_clients
    mock_reranker = MagicMock()

    mock_os.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [{"_id": "h1", "_score": 10.0, "_source": {"chunk_id": "h1", "content": "Text 1"}}]}},
            {"hits": {"hits": [{"_id": "h2", "_score": 0.9, "_source": {"chunk_id": "h2", "content": "Text 2"}}]}},
        ]
    }

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=mock_reranker,
        rerank_enabled=False,
    )

    query = RetrievalQuery(
        query="bếp nướng gas",
        search_type=SearchType.HYBRID,
        top_k=2,
        rerank=True,
    )
    result = retriever.retrieve(query)

    assert result.total_hits == 2
    mock_reranker.rerank.assert_not_called()
    assert "_rerank" not in result.retrieval_mode_used
