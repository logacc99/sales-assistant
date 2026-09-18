"""Rank fusion algorithms for combining lexical and vector search results (RRF and Linear Combination)."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.retrieval.models import RetrievedChunk


class ReciprocalRankFusion:
    """Combines ranked candidate lists using Reciprocal Rank Fusion (RRF).
    
    Formula: RRF(d) = sum(1.0 / (k + rank_m(d))) for each search modality m.
    """

    @staticmethod
    def fuse(
        lexical_hits: List[RetrievedChunk],
        vector_hits: List[RetrievedChunk],
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """Fuses lexical and vector hits using RRF and assigns rank telemetry.
        
        Args:
            lexical_hits: Candidate chunks ordered by BM25 score descending.
            vector_hits: Candidate chunks ordered by vector similarity descending.
            rrf_k: Smoothing constant (default: 60).
            
        Returns:
            Deduplicated list of RetrievedChunk objects ordered by RRF score descending.
        """
        # Map chunk_id to merged RetrievedChunk
        merged_chunks: Dict[str, RetrievedChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # Process BM25 lexical candidates
        for rank, chunk in enumerate(lexical_hits, start=1):
            cid = chunk.chunk_id
            raw_lex = chunk.lexical_score if chunk.lexical_score is not None else chunk.score
            chunk.lexical_rank = rank
            chunk.lexical_score = raw_lex
            merged_chunks[cid] = chunk
            rrf_scores[cid] = 1.0 / (rrf_k + rank)

        # Process dense vector candidates
        for rank, chunk in enumerate(vector_hits, start=1):
            cid = chunk.chunk_id
            raw_vec = chunk.vector_score if chunk.vector_score is not None else chunk.score
            chunk.vector_rank = rank
            chunk.vector_score = raw_vec
            
            if cid in merged_chunks:
                existing = merged_chunks[cid]
                existing.vector_rank = rank
                existing.vector_score = raw_vec
                rrf_scores[cid] += 1.0 / (rrf_k + rank)
            else:
                merged_chunks[cid] = chunk
                rrf_scores[cid] = 1.0 / (rrf_k + rank)

        # Assign final rrf_score and score
        fused_list: List[RetrievedChunk] = []
        for cid, chunk in merged_chunks.items():
            final_rrf = rrf_scores[cid]
            chunk.rrf_score = final_rrf
            chunk.score = final_rrf
            fused_list.append(chunk)

        # Sort by RRF score descending, tie-breaking by lexical rank or vector rank
        fused_list.sort(
            key=lambda c: (
                c.rrf_score or 0.0,
                -(c.vector_rank if c.vector_rank is not None else 9999),
                -(c.lexical_rank if c.lexical_rank is not None else 9999),
            ),
            reverse=True,
        )

        return fused_list


class LinearCombinationFusion:
    """Combines lexical and vector candidate scores using min-max normalization and weighted sum."""

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        min_val = min(scores.values())
        max_val = max(scores.values())
        diff = max_val - min_val
        if diff <= 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / diff for k, v in scores.items()}

    @classmethod
    def fuse(
        cls,
        lexical_hits: List[RetrievedChunk],
        vector_hits: List[RetrievedChunk],
        alpha: float = 0.5,
    ) -> List[RetrievedChunk]:
        """Fuses candidate hits using normalized weighted sum: alpha * vector + (1 - alpha) * lexical."""
        merged_chunks: Dict[str, RetrievedChunk] = {}
        lexical_raw: Dict[str, float] = {}
        vector_raw: Dict[str, float] = {}

        for rank, chunk in enumerate(lexical_hits, start=1):
            cid = chunk.chunk_id
            raw_lex = chunk.lexical_score if chunk.lexical_score is not None else chunk.score
            chunk.lexical_rank = rank
            chunk.lexical_score = raw_lex
            merged_chunks[cid] = chunk
            lexical_raw[cid] = raw_lex

        for rank, chunk in enumerate(vector_hits, start=1):
            cid = chunk.chunk_id
            raw_vec = chunk.vector_score if chunk.vector_score is not None else chunk.score
            chunk.vector_rank = rank
            chunk.vector_score = raw_vec
            if cid not in merged_chunks:
                merged_chunks[cid] = chunk
            else:
                existing = merged_chunks[cid]
                existing.vector_rank = rank
                existing.vector_score = raw_vec
            vector_raw[cid] = raw_vec

        norm_lexical = cls._normalize(lexical_raw)
        norm_vector = cls._normalize(vector_raw)

        fused_list: List[RetrievedChunk] = []
        for cid, chunk in merged_chunks.items():
            s_lex = norm_lexical.get(cid, 0.0)
            s_vec = norm_vector.get(cid, 0.0)
            combined = alpha * s_vec + (1.0 - alpha) * s_lex
            chunk.score = combined
            fused_list.append(chunk)

        fused_list.sort(key=lambda c: c.score, reverse=True)
        return fused_list
