"""Grounded response generator orchestrating context budgeting, Bedrock synthesis, and citations."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from src.config import IngestionConfig, get_config
from src.generation.budget import ContextBudgetManager
from src.generation.citation_extractor import CitationExtractor, SUPPORT_TITLE, SUPPORT_URL
from src.generation.models import (
    Citation,
    GenerationConfig,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    TokenUsage,
)
from src.generation.prompts import (
    NO_STACKING_DISCLAIMER_VI,
    OUT_OF_STOCK_NOTICE_VI,
    PromptBuilder,
)
from src.generation.providers import BaseBedrockClient, BaseLLMClient, LLMClientFactory
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

EMPTY_CONTEXT_RESPONSE_VI = (
    "Dạ chào bạn, hiện tại cửa hàng chưa tìm thấy thông tin phù hợp với câu hỏi của bạn trong danh mục. "
    f"Bạn vui lòng kiểm tra lại hoặc liên hệ [{SUPPORT_TITLE}]({SUPPORT_URL}) để nhân viên tư vấn hỗ trợ chi tiết nhé!"
)

FALLBACK_ERROR_RESPONSE_VI = (
    "Dạ thành thật xin lỗi bạn, hệ thống tư vấn đang gặp sự cố kết nối tạm thời. "
    f"Bạn vui lòng thử lại sau giây lát hoặc liên hệ [{SUPPORT_TITLE}]({SUPPORT_URL}) để được hỗ trợ kịp thời nhé!"
)


class GroundedResponseGenerator:
    """Core generator coordinating context budgeting, prompt assembly, LLM call, and citation hydration."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        bedrock_client: Optional[BaseBedrockClient] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_extractor: Optional[CitationExtractor] = None,
        app_config: Optional[IngestionConfig] = None,
    ) -> None:
        self.app_config = app_config or get_config()
        active_client = llm_client or bedrock_client
        if active_client is None:
            self.llm_client = LLMClientFactory.create(app_config=self.app_config)
        else:
            self.llm_client = active_client
        self.bedrock_client = self.llm_client  # Backward-compatible attribute
        self.budget_manager = budget_manager or ContextBudgetManager()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_extractor = citation_extractor or CitationExtractor()

    def _resolve_config(self, req_config: Optional[GenerationConfig]) -> GenerationConfig:
        """Merges request-level generation config with application and provider defaults."""
        method = req_config.method if req_config and req_config.method else self.app_config.llm_method

        if method == "mantle":
            default_model = self.app_config.bedrock_mantle_model_id
        else:
            default_model = self.app_config.bedrock_generation_model_id

        if req_config is not None:
            model_id = req_config.model_id or default_model
            return GenerationConfig(
                model_id=model_id,
                temperature=req_config.temperature,
                top_p=req_config.top_p,
                max_tokens=req_config.max_tokens,
                stop_sequences=req_config.stop_sequences,
                system_prompt_override=req_config.system_prompt_override,
                method=method,
            )

        return GenerationConfig(
            model_id=default_model,
            temperature=self.app_config.bedrock_generation_temperature,
            max_tokens=self.app_config.bedrock_generation_max_tokens,
            method=method,
        )

    def _check_out_of_stock(self, chunks: List[RetrievedChunk], answer_text: str) -> bool:
        """Inspects if product chunks or generated answer indicate out-of-stock condition."""
        # Check product chunk metadata
        for chunk in chunks:
            if chunk.category == "product":
                status = (chunk.metadata or {}).get("stock_status", "").lower()
                if status in ("out_of_stock", "backorder"):
                    return True

        # Check generated answer keywords
        lower_ans = answer_text.lower()
        if "tạm hết hàng" in lower_ans or "hết hàng" in lower_ans or "out of stock" in lower_ans:
            return True

        return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Executes synchronous response generation with grounding and citation hydration.
        """
        start_time = time.perf_counter()
        config = self._resolve_config(request.config)

        # Handle empty context early
        if not request.chunks:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            support_cit = self.citation_extractor.create_support_citation()
            return GenerationResponse(
                query=request.query,
                answer=EMPTY_CONTEXT_RESPONSE_VI,
                citations=[support_cit] if request.require_citations else [],
                model_id=config.model_id,
                refusal_triggered=True,
                refusal_reason="empty_retrieved_context",
                support_redirect_url=SUPPORT_URL,
                latency_ms=latency_ms,
                degraded=False,
            )

        # 1. Budget and trim chunks
        retained_chunks, budget_report = self.budget_manager.budget_chunks(request.chunks)

        # 2. Build system and user messages
        system_prompts, messages = self.prompt_builder.build_messages(request, retained_chunks)

        # 3. Resolve active client and call LLM provider
        client = (
            self.llm_client
            if not config.method or config.method == self.app_config.llm_method
            else LLMClientFactory.create(method=config.method, app_config=self.app_config)
        )

        try:
            llm_result = client.converse(
                messages=messages,
                system_prompts=system_prompts,
                config=config,
            )
            answer_text = llm_result.get("text", "")
            usage = llm_result.get("token_usage", TokenUsage())
            degraded = False
            degradation_reason = None
        except Exception as e:
            logger.error("LLM generation failed: %s. Activating graceful degradation.", e)
            answer_text = FALLBACK_ERROR_RESPONSE_VI
            usage = TokenUsage()
            degraded = True
            degradation_reason = str(e)

        # 4. Out-of-stock and refusal detection
        has_out_of_stock = self._check_out_of_stock(retained_chunks, answer_text)
        refusal_triggered = has_out_of_stock or degraded

        # 5. Extract and hydrate citations
        citations: List[Citation] = []
        if request.require_citations:
            citations = self.citation_extractor.extract_citations(
                answer_text=answer_text,
                retained_chunks=retained_chunks,
                refusal_triggered=refusal_triggered,
                has_out_of_stock=has_out_of_stock,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return GenerationResponse(
            query=request.query,
            answer=answer_text,
            citations=citations,
            model_id=config.model_id,
            refusal_triggered=refusal_triggered,
            refusal_reason="product_out_of_stock" if has_out_of_stock else ("llm_error" if degraded else None),
            support_redirect_url=SUPPORT_URL if refusal_triggered else None,
            token_usage=usage,
            latency_ms=latency_ms,
            degraded=degraded,
            degradation_reason=degradation_reason,
        )

    def generate_stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """
        Executes streaming response generation yielding StreamChunk events.
        """
        start_time = time.perf_counter()
        config = self._resolve_config(request.config)

        if not request.chunks:
            support_cit = self.citation_extractor.create_support_citation()
            yield StreamChunk(event="delta", text=EMPTY_CONTEXT_RESPONSE_VI)
            if request.require_citations:
                yield StreamChunk(event="citation", citation=support_cit)
            yield StreamChunk(
                event="done",
                data={
                    "query": request.query,
                    "model_id": config.model_id,
                    "refusal_triggered": True,
                    "refusal_reason": "empty_retrieved_context",
                    "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                },
            )
            return

        retained_chunks, _ = self.budget_manager.budget_chunks(request.chunks)
        system_prompts, messages = self.prompt_builder.build_messages(request, retained_chunks)

        accumulated_text: List[str] = []
        degraded = False
        degradation_reason = None

        client = (
            self.llm_client
            if not config.method or config.method == self.app_config.llm_method
            else LLMClientFactory.create(method=config.method, app_config=self.app_config)
        )

        try:
            for text_delta in client.converse_stream(
                messages=messages,
                system_prompts=system_prompts,
                config=config,
            ):
                accumulated_text.append(text_delta)
                yield StreamChunk(event="delta", text=text_delta)
        except Exception as e:
            logger.error("Streaming LLM generation failed: %s", e)
            degraded = True
            degradation_reason = str(e)
            fallback = FALLBACK_ERROR_RESPONSE_VI
            accumulated_text.append(fallback)
            yield StreamChunk(event="error", text=fallback)

        full_answer = "".join(accumulated_text)
        has_out_of_stock = self._check_out_of_stock(retained_chunks, full_answer)
        refusal_triggered = has_out_of_stock or degraded

        if request.require_citations:
            citations = self.citation_extractor.extract_citations(
                answer_text=full_answer,
                retained_chunks=retained_chunks,
                refusal_triggered=refusal_triggered,
                has_out_of_stock=has_out_of_stock,
            )
            for cit in citations:
                yield StreamChunk(event="citation", citation=cit)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        yield StreamChunk(
            event="done",
            data={
                "query": request.query,
                "model_id": config.model_id,
                "refusal_triggered": refusal_triggered,
                "refusal_reason": "product_out_of_stock" if has_out_of_stock else None,
                "support_redirect_url": SUPPORT_URL if refusal_triggered else None,
                "latency_ms": round(latency_ms, 2),
                "degraded": degraded,
                "degradation_reason": degradation_reason,
            },
        )
