"""Unit tests for retrieval models, default options, and candidate pool sizing."""

import pytest
from src.retrieval.models import (
    FilterCriteria,
    FusionAlgorithm,
    RerankerType,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    SearchType,
)


def test_retrieval_query_defaults():
    query = RetrievalQuery(query="bếp nướng gas ngoài trời")
    assert query.query == "bếp nướng gas ngoài trời"
    assert query.search_type == SearchType.HYBRID
    assert query.top_k == 5
    assert query.fusion_algorithm == FusionAlgorithm.RRF
    assert query.rrf_k == 60
    assert query.rerank is True
    assert query.rerank_model_id == "BAAI/bge-reranker-m3"
    assert query.reranker_type == RerankerType.LOCAL
    assert query.expand_parent_context is True
    assert query.parent_max_chars == 2000
    # candidate_pool_size = min(max(top_k * 3, 15), 30) -> min(max(15, 15), 30) == 15
    assert query.candidate_pool_size == 15


def test_candidate_pool_size_scaling():
    # top_k = 2 -> max(6, 15) -> 15
    q2 = RetrievalQuery(query="test", top_k=2)
    assert q2.candidate_pool_size == 15

    # top_k = 8 -> max(24, 15) -> 24
    q8 = RetrievalQuery(query="test", top_k=8)
    assert q8.candidate_pool_size == 24

    # top_k = 15 -> max(45, 15) -> 45 capped at 30
    q15 = RetrievalQuery(query="test", top_k=15)
    assert q15.candidate_pool_size == 30


def test_filter_criteria_to_dict():
    filters = FilterCriteria(
        category="product",
        stock_status="in_stock",
        min_price=1000000.0,
        max_price=5000000.0,
    )
    d = filters.to_dict()
    assert d["category"] == "product"
    assert d["stock_status"] == "in_stock"
    assert d["min_price"] == 1000000.0
    assert d["max_price"] == 5000000.0
    assert "policy_type" not in d


def test_retrieved_chunk_and_result_serialization():
    chunk = RetrievedChunk(
        chunk_id="chk_001",
        doc_id="doc_001",
        category="product",
        content="Bếp nướng gas cao cấp",
        score=0.9854,
        lexical_score=15.2,
        vector_score=0.88,
        rrf_score=0.0325,
        rerank_score=0.9854,
        lexical_rank=1,
        vector_rank=2,
        rerank_rank=1,
        metadata={"product_name": "Bếp nướng BBQ"},
    )
    c_dict = chunk.to_dict()
    assert c_dict["chunk_id"] == "chk_001"
    assert c_dict["score"] == 0.9854
    assert c_dict["metadata"]["product_name"] == "Bếp nướng BBQ"

    result = RetrievalResult(
        query="bếp nướng",
        search_type=SearchType.HYBRID,
        total_hits=1,
        chunks=[chunk],
        latency_ms=45.2,
        retrieval_mode_used="hybrid_msearch_rrf_cohere_rerank",
        degraded=False,
    )
    r_dict = result.to_dict()
    assert r_dict["query"] == "bếp nướng"
    assert r_dict["total_hits"] == 1
    assert len(r_dict["chunks"]) == 1
    assert r_dict["degraded"] is False
