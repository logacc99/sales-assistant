"""REST API endpoints for hybrid search, scoring explanation, and retrieval health verification."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_app_settings,
    get_hybrid_retriever,
    get_retrieval_service,
)
from src.api.v1.schemas.common import ApiResponse
from src.api.v1.schemas.retrieval import (
    ExplainApiResponse,
    RetrievalHealthResponse,
    SearchApiRequest,
    SearchApiResponse,
)
from src.config import Settings
from src.retrieval.models import (
    FilterCriteria,
    RetrievalQuery,
)
from src.retrieval.retriever import OpenSearchHybridRetriever
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=ApiResponse[SearchApiResponse],
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid knowledge retrieval",
    description="Retrieves top-K candidate chunks using BM25, Lucene k-NN, client-side RRF, and Cohere reranking.",
)
def execute_search(
    request: SearchApiRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> ApiResponse[SearchApiResponse]:
    """Primary hybrid retrieval endpoint with metadata filtering and parent-child expansion."""
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace only.",
        )

    filters = None
    if request.filters:
        filters = FilterCriteria(
            category=request.filters.category,
            stock_status=request.filters.stock_status,
            min_price=request.filters.min_price,
            max_price=request.filters.max_price,
            policy_type=request.filters.policy_type,
            promo_code=request.filters.promo_code,
            source_prefix=request.filters.source_prefix,
        )

    query_obj = RetrievalQuery(
        query=clean_query,
        search_type=request.search_type,
        top_k=request.top_k,
        filters=filters,
        fusion_algorithm=request.fusion_algorithm,
        rrf_k=request.rrf_k,
        hybrid_alpha=request.hybrid_alpha,
        score_threshold=request.score_threshold,
        expand_parent_context=request.expand_parent_context,
        parent_max_chars=request.parent_max_chars,
        rerank=request.rerank,
    )

    try:
        result = retrieval_service.retriever.retrieve(query_obj)
    except Exception as exc:
        logger.exception(f"Hybrid retrieval failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval query failed: {str(exc)}",
        )

    response_data = SearchApiResponse(
        query=result.query,
        search_type=result.search_type.value if hasattr(result.search_type, "value") else str(result.search_type),
        total_hits=result.total_hits,
        chunks=[
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "category": c.category,
                "content": c.content,
                "score": c.score,
                "lexical_score": c.lexical_score,
                "vector_score": c.vector_score,
                "rrf_score": c.rrf_score,
                "rerank_score": c.rerank_score,
                "lexical_rank": c.lexical_rank,
                "vector_rank": c.vector_rank,
                "rerank_rank": c.rerank_rank,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
                "is_parent_expanded": c.is_parent_expanded,
            }
            for c in result.chunks
        ],
        latency_ms=result.latency_ms,
        retrieval_mode_used=result.retrieval_mode_used,
        filters_applied=result.filters_applied,
        degraded=result.degraded,
        degradation_reason=result.degradation_reason,
    )

    return ApiResponse[SearchApiResponse](
        success=True,
        message=f"Retrieved {result.total_hits} chunks in {result.latency_ms:.2f}ms"
        + (f" (Degraded: {result.degradation_reason})" if result.degraded else ""),
        data=response_data,
    )


@router.post(
    "/explain",
    response_model=ApiResponse[ExplainApiResponse],
    status_code=status.HTTP_200_OK,
    summary="Explain retrieval scoring and rank fusion breakdown",
    description="Provides deep inspection of raw BM25, dense vector cosine similarity, RRF, and rerank scores.",
)
def explain_scoring(
    request: SearchApiRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> ApiResponse[ExplainApiResponse]:
    """Diagnostics endpoint detailing the ranking steps of hybrid search."""
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace only.",
        )

    filters = None
    if request.filters:
        filters = FilterCriteria(
            category=request.filters.category,
            stock_status=request.filters.stock_status,
            min_price=request.filters.min_price,
            max_price=request.filters.max_price,
            policy_type=request.filters.policy_type,
            promo_code=request.filters.promo_code,
            source_prefix=request.filters.source_prefix,
        )

    try:
        explanation = retrieval_service.explain(
            query=clean_query,
            top_k=request.top_k,
            filters=filters,
        )
    except Exception as exc:
        logger.exception(f"Scoring explanation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring explanation failed: {str(exc)}",
        )

    return ApiResponse[ExplainApiResponse](
        success=True,
        message=f"Explained ranking for {explanation['total_hits']} candidate chunks in {explanation['latency_ms']:.2f}ms",
        data=ExplainApiResponse(**explanation),
    )


@router.get(
    "/health",
    response_model=ApiResponse[RetrievalHealthResponse],
    status_code=status.HTTP_200_OK,
    summary="Verify retrieval subsystem health and dependencies",
    description="Checks OpenSearch cluster ping, target vector index presence, doc count, and Bedrock accessibility.",
)
def check_retrieval_health(
    retriever: OpenSearchHybridRetriever = Depends(get_hybrid_retriever),
    settings: Settings = Depends(get_app_settings),
) -> ApiResponse[RetrievalHealthResponse]:
    """Retrieval health verification probe."""
    target_index = retriever.index_name or settings.opensearch_index_name
    opensearch_ok = False
    index_exists = False
    doc_count = 0

    try:
        # Standard OpenSearch responds to ping (HEAD /). OpenSearch Serverless returns 404
        # or restricts cluster root, so fallback to checking collection index existence.
        reachable = False
        try:
            reachable = bool(retriever.client.ping())
        except Exception:
            reachable = False

        if not reachable:
            try:
                reachable = bool(retriever.client.indices.exists(index=target_index))
            except Exception:
                reachable = False

        if reachable:
            opensearch_ok = True
            index_exists = bool(retriever.client.indices.exists(index=target_index))
            if index_exists:
                stats = retriever.client.count(index=target_index)
                doc_count = stats.get("count", 0)
    except Exception as exc:
        logger.warning(f"Health check could not connect to OpenSearch: {exc}")

    bedrock_embedder_ready = retriever.embedder is not None
    bedrock_reranker_ready = retriever.reranker is not None

    overall_status = "healthy" if (opensearch_ok and index_exists) else "degraded"

    return ApiResponse[RetrievalHealthResponse](
        success=True,
        message=f"Retrieval subsystem status: {overall_status}",
        data=RetrievalHealthResponse(
            status=overall_status,
            opensearch_connected=opensearch_ok,
            index_exists=index_exists,
            index_name=target_index,
            document_count=doc_count,
            bedrock_embedder_ready=bedrock_embedder_ready,
            bedrock_reranker_ready=bedrock_reranker_ready,
            is_serverless=settings.opensearch_is_serverless,
            collection_type=settings.opensearch_collection_type,
        ),
    )
