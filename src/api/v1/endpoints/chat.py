"""FastAPI endpoints for end-to-end RAG chat assistant combining retrieval and generation."""

from __future__ import annotations

import logging
import time
from typing import Iterator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_app_settings,
    get_generation_service,
    get_retrieval_service,
)
from src.api.v1.schemas.chat import ChatApiRequest, ChatApiResponse
from src.api.v1.schemas.generation import CitationModel, TokenUsageModel
from src.api.v1.schemas.retrieval import RetrievedChunkModel
from src.config import Settings
from src.generation.models import (
    CartItem,
    GenerationConfig,
    ShopperContext,
)
from src.generation.service import GenerationService
from src.retrieval.models import (
    FilterCriteria,
    RetrievalQuery,
)
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_domain_shopper_context(payload: ChatApiRequest) -> ShopperContext | None:
    """Constructs ShopperContext domain object from payload."""
    if payload.shopper_context is None:
        return None

    cart_items = [
        CartItem(
            sku=item.sku,
            product_id=item.product_id,
            name=item.name,
            price=item.price,
            quantity=item.quantity,
        )
        for item in payload.shopper_context.cart_items
    ]
    ctx = ShopperContext(
        cart_items=cart_items,
        cart_subtotal=payload.shopper_context.cart_subtotal,
        locale=payload.shopper_context.locale,
        user_id=payload.shopper_context.user_id,
    )
    if payload.shopper_context.current_time is not None:
        ctx.current_time = payload.shopper_context.current_time
    return ctx


def _build_domain_generation_config(payload: ChatApiRequest) -> GenerationConfig | None:
    """Constructs GenerationConfig domain object from payload."""
    if payload.config is None:
        return None

    config = GenerationConfig(
        stop_sequences=payload.config.stop_sequences,
    )
    if payload.config.model_id:
        config.model_id = payload.config.model_id
    if payload.config.temperature is not None:
        config.temperature = payload.config.temperature
    if payload.config.top_p is not None:
        config.top_p = payload.config.top_p
    if payload.config.max_tokens is not None:
        config.max_tokens = payload.config.max_tokens
    if payload.config.method:
        config.method = payload.config.method

    return config


@router.post(
    "",
    response_model=ChatApiResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-End RAG Chat Assistant",
    description=(
        "Executes hybrid knowledge retrieval across OpenSearch and passes verified context "
        "chunks to the active LLM backend ('runtime' or 'mantle') for grounded synthesis."
    ),
)
def chat_response(
    payload: ChatApiRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    settings: Settings = Depends(get_app_settings),
) -> ChatApiResponse:
    """Synchronous end-to-end chat endpoint."""
    start_total = time.perf_counter()

    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    # 1. Hybrid Retrieval
    filters = None
    if payload.filters:
        filters = FilterCriteria(
            category=payload.filters.category,
            stock_status=payload.filters.stock_status,
            min_price=payload.filters.min_price,
            max_price=payload.filters.max_price,
            policy_type=payload.filters.policy_type,
            promo_code=payload.filters.promo_code,
            source_prefix=payload.filters.source_prefix,
        )

    retrieval_query = RetrievalQuery(
        query=clean_query,
        top_k=payload.top_k,
        filters=filters,
    )

    t_ret_start = time.perf_counter()
    try:
        search_result = retrieval_service.search(retrieval_query)
        chunks = search_result.chunks
    except Exception as e:
        logger.error("Retrieval failed during chat flow: %s", e)
        chunks = []
    retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0

    # 2. Grounded Generation
    shopper_ctx = _build_domain_shopper_context(payload)
    gen_config = _build_domain_generation_config(payload)

    t_gen_start = time.perf_counter()
    gen_result = generation_service.generate(
        query=clean_query,
        chunks=chunks,
        shopper_context=shopper_ctx,
        config=gen_config,
        conversation_history=payload.conversation_history,
        require_citations=payload.require_citations,
    )
    generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
    total_latency_ms = (time.perf_counter() - start_total) * 1000.0

    # 3. Format Response
    citations = [
        CitationModel(
            citation_type=c.citation_type.value if hasattr(c.citation_type, "value") else str(c.citation_type),
            title=c.title,
            url=c.url,
            price_display=c.price_display,
            stock_status=c.stock_status,
            promo_code=c.promo_code,
            discount_display=c.discount_display,
            metadata=c.metadata,
        )
        for c in gen_result.citations
    ]

    retrieved_chunk_models = None
    if payload.include_retrieved_chunks:
        retrieved_chunk_models = [
            RetrievedChunkModel(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                category=c.category,
                content=c.content,
                score=c.score,
                lexical_score=c.lexical_score,
                vector_score=c.vector_score,
                rrf_score=c.rrf_score,
                rerank_score=c.rerank_score,
                lexical_rank=c.lexical_rank,
                vector_rank=c.vector_rank,
                rerank_rank=c.rerank_rank,
                chunk_index=c.chunk_index,
                metadata=c.metadata,
                is_parent_expanded=c.is_parent_expanded,
            )
            for c in chunks
        ]

    active_method = (
        (gen_config.method if gen_config and gen_config.method else None)
        or settings.llm_method
        or "runtime"
    )

    return ChatApiResponse(
        query=clean_query,
        answer=gen_result.answer,
        citations=citations,
        retrieved_chunks=retrieved_chunk_models,
        model_id=gen_result.model_id,
        method=active_method,
        latency_ms=round(total_latency_ms, 2),
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
        token_usage=TokenUsageModel(
            input_tokens=gen_result.token_usage.input_tokens,
            output_tokens=gen_result.token_usage.output_tokens,
            total_tokens=gen_result.token_usage.total_tokens,
        ),
        refusal_triggered=gen_result.refusal_triggered,
        refusal_reason=gen_result.refusal_reason,
        support_redirect_url=gen_result.support_redirect_url,
        degraded=gen_result.degraded,
        degradation_reason=gen_result.degradation_reason,
    )


@router.post(
    "/stream",
    summary="End-to-End Streaming RAG Chat Assistant",
    description="Executes hybrid retrieval followed by streaming grounded LLM token deltas via SSE.",
)
def chat_response_stream(
    payload: ChatApiRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> StreamingResponse:
    """Streaming end-to-end chat endpoint."""
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    filters = None
    if payload.filters:
        filters = FilterCriteria(
            category=payload.filters.category,
            stock_status=payload.filters.stock_status,
            min_price=payload.filters.min_price,
            max_price=payload.filters.max_price,
            policy_type=payload.filters.policy_type,
            promo_code=payload.filters.promo_code,
            source_prefix=payload.filters.source_prefix,
        )

    retrieval_query = RetrievalQuery(
        query=clean_query,
        top_k=payload.top_k,
        filters=filters,
    )

    try:
        search_result = retrieval_service.search(retrieval_query)
        chunks = search_result.chunks
    except Exception as e:
        logger.error("Streaming chat retrieval failed: %s", e)
        chunks = []

    shopper_ctx = _build_domain_shopper_context(payload)
    gen_config = _build_domain_generation_config(payload)

    def event_generator() -> Iterator[str]:
        for event_chunk in generation_service.generate_stream(
            query=clean_query,
            chunks=chunks,
            shopper_context=shopper_ctx,
            config=gen_config,
            conversation_history=payload.conversation_history,
            require_citations=payload.require_citations,
        ):
            yield event_chunk.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
