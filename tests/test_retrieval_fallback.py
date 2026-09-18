"""Unit tests for graceful degradation fallback in hybrid retrieval."""

from unittest.mock import MagicMock
import pytest

from src.retrieval.models import RetrievalQuery, SearchType
from src.retrieval.reranker import BaseReranker
from src.retrieval.retriever import OpenSearchHybridRetriever


class FailingReranker(BaseReranker):
    def rerank(self, query, chunks, top_k):
        raise RuntimeError("Cohere Rerank API rate limit exceeded")


def test_fallback_when_bedrock_embedding_fails():
    mock_os = MagicMock()
    mock_embedder = MagicMock()
    # Bedrock embedder raises throttling exception
    mock_embedder.embed_text.side_effect = RuntimeError("AWS Bedrock ThrottlingException")

    # BM25 search fallback response
    bm25_hit = {
        "_id": "chk_fallback_1",
        "_score": 10.0,
        "_source": {
            "chunk_id": "chk_fallback_1",
            "category": "product",
            "content": "Bếp củi dã ngoại",
        },
    }
    mock_os.search.return_value = {"hits": {"hits": [bm25_hit]}}

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=None,
    )

    query = RetrievalQuery(query="bếp dã ngoại", search_type=SearchType.HYBRID)
    result = retriever.retrieve(query)

    assert result.degraded is True
    assert "Bedrock embedding failed" in (result.degradation_reason or "")
    assert "degraded_bm25" in result.retrieval_mode_used
    assert result.total_hits == 1
    assert result.chunks[0].chunk_id == "chk_fallback_1"
    mock_os.search.assert_called_once()


def test_fallback_when_cohere_rerank_fails():
    mock_os = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1] * 1024

    hit1 = {
        "_id": "chk_1",
        "_score": 12.0,
        "_source": {"chunk_id": "chk_1", "content": "Bếp 1"},
    }
    hit2 = {
        "_id": "chk_2",
        "_score": 0.85,
        "_source": {"chunk_id": "chk_2", "content": "Bếp 2"},
    }

    mock_os.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [hit1]}},
            {"hits": {"hits": [hit2]}},
        ]
    }

    retriever = OpenSearchHybridRetriever(
        client=mock_os,
        embedder=mock_embedder,
        reranker=FailingReranker(),
    )

    query = RetrievalQuery(
        query="bếp nướng",
        search_type=SearchType.HYBRID,
        rerank=True,
    )
    result = retriever.retrieve(query)

    assert result.degraded is True
    assert "Cohere rerank failed" in (result.degradation_reason or "")
    # Returns RRF order gracefully without crashing
    assert result.total_hits == 2
