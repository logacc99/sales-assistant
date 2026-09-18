"""Unit tests for Reciprocal Rank Fusion (RRF) and Linear Combination Fusion."""

import pytest
from src.retrieval.fusion import LinearCombinationFusion, ReciprocalRankFusion
from src.retrieval.models import RetrievedChunk


def _make_chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        doc_id=f"doc_{cid}",
        category="product",
        content=f"Content for {cid}",
        score=score,
    )


def test_rrf_scoring_and_ranking():
    # BM25 order: chunk_A (rank 1), chunk_B (rank 2)
    bm25_hits = [_make_chunk("chunk_A", 15.0), _make_chunk("chunk_B", 10.0)]
    # Vector order: chunk_B (rank 1), chunk_C (rank 2)
    vector_hits = [_make_chunk("chunk_B", 0.95), _make_chunk("chunk_C", 0.80)]

    fused = ReciprocalRankFusion.fuse(bm25_hits, vector_hits, rrf_k=60)

    assert len(fused) == 3
    # chunk_B appears in both lists: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~ 0.032522
    assert fused[0].chunk_id == "chunk_B"
    assert fused[0].lexical_rank == 2
    assert fused[0].vector_rank == 1
    assert fused[0].lexical_score == 10.0
    assert fused[0].vector_score == 0.95
    assert pytest.approx(fused[0].rrf_score, 1e-5) == (1.0 / 62.0 + 1.0 / 61.0)
    assert fused[0].score == fused[0].rrf_score

    # chunk_A is 1/(60+1) = 1/61 ~ 0.016393
    assert fused[1].chunk_id == "chunk_A"
    assert fused[1].lexical_rank == 1
    assert fused[1].vector_rank is None

    # chunk_C is 1/(60+2) = 1/62 ~ 0.016129
    assert fused[2].chunk_id == "chunk_C"
    assert fused[2].vector_rank == 2
    assert fused[2].lexical_rank is None


def test_linear_combination_fusion():
    bm25_hits = [_make_chunk("A", 20.0), _make_chunk("B", 10.0)]
    vector_hits = [_make_chunk("A", 0.5), _make_chunk("B", 1.0)]

    # Normalized BM25: A=1.0, B=0.0
    # Normalized Vector: A=0.0, B=1.0
    # Alpha = 0.5:
    # A = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    # B = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    fused_half = LinearCombinationFusion.fuse(bm25_hits, vector_hits, alpha=0.5)
    assert len(fused_half) == 2
    assert pytest.approx(fused_half[0].score, 1e-4) == 0.5
    assert pytest.approx(fused_half[1].score, 1e-4) == 0.5

    # Alpha = 0.8 (vector heavy):
    # A = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    # B = 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    # B must win
    fused_vec = LinearCombinationFusion.fuse(bm25_hits, vector_hits, alpha=0.8)
    assert fused_vec[0].chunk_id == "B"
    assert pytest.approx(fused_vec[0].score, 1e-4) == 0.8
    assert fused_vec[1].chunk_id == "A"
    assert pytest.approx(fused_vec[1].score, 1e-4) == 0.2
