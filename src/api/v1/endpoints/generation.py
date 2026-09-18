"""FastAPI endpoints for Bedrock LLM response generation matching SPEC-0004."""

from __future__ import annotations

import logging
from typing import Iterator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_generation_service
from src.api.v1.schemas.generation import (
    CitationModel,
    GenerateApiRequest,
    GenerateApiResponse,
    TokenUsageModel,
)
from src.generation.models import (
    CartItem,
    GenerationConfig,
    ShopperContext,
)
from src.generation.service import GenerationService
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter()


def _map_request_to_domain(payload: GenerateApiRequest):
    """Maps Pydantic API request schemas into generation domain dataclasses."""
    chunks = [
        RetrievedChunk(
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
        for c in payload.chunks
    ]

    shopper_ctx = None
    if payload.shopper_context is not None:
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
        shopper_ctx = ShopperContext(
            cart_items=cart_items,
            cart_subtotal=payload.shopper_context.cart_subtotal,
            locale=payload.shopper_context.locale,
            user_id=payload.shopper_context.user_id,
        )
        if payload.shopper_context.current_time is not None:
            shopper_ctx.current_time = payload.shopper_context.current_time

    config = None
    if payload.config is not None:
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

    return chunks, shopper_ctx, config


@router.post(
    "",
    response_model=GenerateApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronous LLM response generation",
    description="Synthesizes grounded conversational response from retrieved chunks and shopper context.",
)
def generate_response(
    payload: GenerateApiRequest,
    service: GenerationService = Depends(get_generation_service),
) -> GenerateApiResponse:
    """Executes full synchronous response generation."""
    try:
        chunks, shopper_ctx, config = _map_request_to_domain(payload)

        result = service.generate(
            query=payload.query,
            chunks=chunks,
            shopper_context=shopper_ctx,
            config=config,
            conversation_history=payload.conversation_history,
            require_citations=payload.require_citations,
        )

        citation_models = [
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
            for c in result.citations
        ]

        usage_model = TokenUsageModel(
            input_tokens=result.token_usage.input_tokens,
            output_tokens=result.token_usage.output_tokens,
            total_tokens=result.token_usage.total_tokens,
        )

        return GenerateApiResponse(
            query=result.query,
            answer=result.answer,
            citations=citation_models,
            model_id=result.model_id,
            refusal_triggered=result.refusal_triggered,
            refusal_reason=result.refusal_reason,
            support_redirect_url=result.support_redirect_url,
            token_usage=usage_model,
            latency_ms=round(result.latency_ms, 2),
            degraded=result.degraded,
            degradation_reason=result.degradation_reason,
        )
    except Exception as e:
        logger.error("Error generating response: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}",
        )


@router.post(
    "/stream",
    summary="Streaming LLM response generation via Server-Sent Events (SSE)",
    description="Streams real-time token deltas followed by structured citations event.",
)
def generate_response_stream(
    payload: GenerateApiRequest,
    service: GenerationService = Depends(get_generation_service),
) -> StreamingResponse:
    """Streams response tokens and trailing citations using Server-Sent Events (SSE)."""
    try:
        chunks, shopper_ctx, config = _map_request_to_domain(payload)

        def event_generator() -> Iterator[str]:
            for event_chunk in service.generate_stream(
                query=payload.query,
                chunks=chunks,
                shopper_context=shopper_ctx,
                config=config,
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
    except Exception as e:
        logger.error("Error in streaming generation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate generation stream: {str(e)}",
        )
