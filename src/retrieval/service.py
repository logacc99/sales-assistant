"""High-level retrieval service orchestrating query execution, explainability, and diagnostics."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.retrieval.models import (
    FilterCriteria,
    FusionAlgorithm,
    RetrievalQuery,
    RetrievalResult,
    SearchType,
)
from src.retrieval.retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service facade providing search and explain capabilities for assistant and REST API."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def search(
        self,
        query: str | RetrievalQuery,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 5,
        filters: Optional[FilterCriteria] = None,
        fusion_algorithm: FusionAlgorithm = FusionAlgorithm.RRF,
        score_threshold: float = 0.0,
        expand_parent_context: bool = True,
        rerank: bool = True,
    ) -> RetrievalResult:
        """Executes search with parameters wrapped into a RetrievalQuery."""
        if isinstance(query, RetrievalQuery):
            return self.retriever.retrieve(query)

        req = RetrievalQuery(
            query=query,
            search_type=search_type,
            top_k=top_k,
            filters=filters,
            fusion_algorithm=fusion_algorithm,
            score_threshold=score_threshold,
            expand_parent_context=expand_parent_context,
            rerank=rerank,
        )
        return self.retriever.retrieve(req)

    def explain(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[FilterCriteria] = None,
    ) -> Dict[str, Any]:
        """Provides detailed step-by-step scoring explanation for transparent inspection."""
        req = RetrievalQuery(
            query=query,
            search_type=SearchType.HYBRID,
            top_k=top_k,
            filters=filters,
            rerank=True,
            expand_parent_context=False,  # Keep original content for inspection
        )
        result = self.retriever.retrieve(req)

        breakdown = []
        for rank, chunk in enumerate(result.chunks, start=1):
            breakdown.append({
                "final_rank": rank,
                "chunk_id": chunk.chunk_id,
                "category": chunk.category,
                "final_score": chunk.score,
                "rerank_score": chunk.rerank_score,
                "rerank_rank": chunk.rerank_rank,
                "rrf_score": chunk.rrf_score,
                "lexical_score": chunk.lexical_score,
                "lexical_rank": chunk.lexical_rank,
                "vector_score": chunk.vector_score,
                "vector_rank": chunk.vector_rank,
                "content_snippet": chunk.content[:150] + ("..." if len(chunk.content) > 150 else ""),
                "product_name": chunk.metadata.get("product_name"),
            })

        return {
            "query": query,
            "latency_ms": result.latency_ms,
            "total_hits": result.total_hits,
            "retrieval_mode_used": result.retrieval_mode_used,
            "degraded": result.degraded,
            "degradation_reason": result.degradation_reason,
            "scoring_breakdown": breakdown,
        }
