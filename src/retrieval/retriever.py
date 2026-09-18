"""Core OpenSearch hybrid retriever combining BM25, Lucene k-NN, client-side RRF, and Cohere reranking."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from src.config import get_config
from src.ingestion.embedder import BaseEmbedder, BedrockEmbedder
from src.ingestion.indexer import OpenSearchVectorIndexer
from src.retrieval.fusion import LinearCombinationFusion, ReciprocalRankFusion
from src.retrieval.models import (
    FilterCriteria,
    FusionAlgorithm,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    SearchType,
)
from src.retrieval.query_builder import OpenSearchQueryBuilder
from src.retrieval.reranker import BaseReranker, BedrockCohereReranker, NoOpReranker

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for information retrieval modules."""

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Executes retrieval according to the provided query specification."""
        pass

    @abstractmethod
    async def aretrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Asynchronous retrieval execution."""
        pass


class OpenSearchHybridRetriever(BaseRetriever):
    """Production hybrid retriever combining OpenSearch _msearch, Bedrock k-NN, and Bedrock Cohere Rerank."""

    def __init__(
        self,
        client: Optional[OpenSearch] = None,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        index_name: Optional[str] = None,
        default_top_k: int = 5,
    ) -> None:
        cfg = get_config()
        self.index_name = index_name or cfg.opensearch_index_name
        self.default_top_k = default_top_k

        if client is not None:
            self.client = client
        else:
            indexer = OpenSearchVectorIndexer()
            self.client = indexer.client

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = BedrockEmbedder(input_type="search_query")

        if reranker is not None:
            self.reranker = reranker
        else:
            try:
                self.reranker = BedrockCohereReranker()
            except Exception as exc:
                logger.warning(f"Could not initialize BedrockCohereReranker ({exc}). Falling back to NoOpReranker.")
                self.reranker = NoOpReranker()

    def _parse_hits(self, hits: List[Dict[str, Any]]) -> List[RetrievedChunk]:
        """Converts raw OpenSearch hit dictionaries into RetrievedChunk instances."""
        chunks: List[RetrievedChunk] = []
        for hit in hits:
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id") or hit.get("_id", "")
            doc_id = source.get("doc_id") or chunk_id
            category = source.get("category", "product")
            content = source.get("content", "")
            raw_score = float(hit.get("_score") or 0.0)
            chunk_index = source.get("chunk_index", 0)
            metadata = source.get("metadata", {})

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                category=category,
                content=content,
                score=raw_score,
                chunk_index=chunk_index,
                metadata=metadata,
                parent_content=metadata.get("parent_section_content"),
            )
            chunks.append(chunk)
        return chunks

    def _expand_parent_contexts(self, chunks: List[RetrievedChunk], max_chars: int = 2000) -> None:
        """Expands parent policy context in-place for policy chunks."""
        for chunk in chunks:
            if chunk.category == "policy" and chunk.metadata:
                parent = chunk.metadata.get("parent_section_content")
                if parent and isinstance(parent, str) and parent.strip():
                    expanded = parent[:max_chars] if len(parent) > max_chars else parent
                    chunk.content = expanded
                    chunk.parent_content = expanded
                    chunk.is_parent_expanded = True

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Executes synchronous retrieval with full graceful degradation telemetry."""
        start_time = time.perf_counter()

        clean_query = query.query.strip()
        if not clean_query:
            return RetrievalResult(
                query=query.query,
                search_type=query.search_type,
                total_hits=0,
                chunks=[],
                latency_ms=0.0,
                retrieval_mode_used="empty_query",
            )

        top_k = query.top_k or self.default_top_k
        degraded = False
        degradation_reason: Optional[str] = None
        retrieval_mode_used = ""
        candidate_pool_size = query.candidate_pool_size

        chunks: List[RetrievedChunk] = []

        # 1. Lexical BM25 Search
        if query.search_type == SearchType.LEXICAL:
            retrieval_mode_used = "lexical_bm25"
            bm25_q = OpenSearchQueryBuilder.build_bm25_query(
                clean_query, size=candidate_pool_size, filters=query.filters
            )
            resp = self.client.search(index=self.index_name, body=bm25_q)
            raw_hits = resp.get("hits", {}).get("hits", [])
            chunks = self._parse_hits(raw_hits)

            if query.rerank and self.reranker:
                try:
                    chunks = self.reranker.rerank(clean_query, chunks, top_k=top_k)
                    retrieval_mode_used += "_cohere_rerank"
                except Exception as exc:
                    degraded = True
                    degradation_reason = f"Cohere rerank failed: {exc}. Degraded to BM25 order."
                    chunks = chunks[:top_k]
            else:
                chunks = chunks[:top_k]

        # 2. Pure Dense Vector Search
        elif query.search_type == SearchType.VECTOR:
            query_vec: Optional[List[float]] = None
            try:
                query_vec = self.embedder.embed_text(clean_query, input_type="search_query")
            except Exception as emb_exc:
                logger.error(f"Bedrock vector embedding failed: {emb_exc}. Degrading to BM25.")
                degraded = True
                degradation_reason = f"Bedrock vector embedding failed: {emb_exc}. Degraded to BM25."

            if query_vec is not None:
                retrieval_mode_used = "vector_knn"
                knn_q = OpenSearchQueryBuilder.build_knn_query(
                    query_vec, size=candidate_pool_size, filters=query.filters
                )
                resp = self.client.search(index=self.index_name, body=knn_q)
                raw_hits = resp.get("hits", {}).get("hits", [])
                chunks = self._parse_hits(raw_hits)
            else:
                # Fallback to BM25
                retrieval_mode_used = "vector_degraded_bm25"
                bm25_q = OpenSearchQueryBuilder.build_bm25_query(
                    clean_query, size=candidate_pool_size, filters=query.filters
                )
                resp = self.client.search(index=self.index_name, body=bm25_q)
                raw_hits = resp.get("hits", {}).get("hits", [])
                chunks = self._parse_hits(raw_hits)

            if query.rerank and self.reranker:
                try:
                    chunks = self.reranker.rerank(clean_query, chunks, top_k=top_k)
                    retrieval_mode_used += "_cohere_rerank"
                except Exception as exc:
                    degraded = True
                    degradation_reason = (
                        f"{degradation_reason}; " if degradation_reason else ""
                    ) + f"Cohere rerank failed: {exc}. Degraded to vector order."
                    chunks = chunks[:top_k]
            else:
                chunks = chunks[:top_k]

        # 3. Hybrid Search (BM25 + k-NN via _msearch + client-side RRF)
        else:
            query_vec = None
            try:
                query_vec = self.embedder.embed_text(clean_query, input_type="search_query")
            except Exception as emb_exc:
                logger.error(f"Bedrock vector embedding failed during hybrid retrieval: {emb_exc}. Degrading to BM25.")
                degraded = True
                degradation_reason = f"Bedrock embedding failed: {emb_exc}. Degraded to BM25."

            if query_vec is not None:
                retrieval_mode_used = "hybrid_msearch"
                bm25_q = OpenSearchQueryBuilder.build_bm25_query(
                    clean_query, size=candidate_pool_size, filters=query.filters
                )
                knn_q = OpenSearchQueryBuilder.build_knn_query(
                    query_vec, size=candidate_pool_size, filters=query.filters
                )

                msearch_body = OpenSearchQueryBuilder.build_msearch_body(
                    self.index_name, bm25_q, knn_q
                )
                msearch_resp = self.client.msearch(body=msearch_body)

                responses = msearch_resp.get("responses", [])
                bm25_hits_raw = responses[0].get("hits", {}).get("hits", []) if len(responses) > 0 else []
                knn_hits_raw = responses[1].get("hits", {}).get("hits", []) if len(responses) > 1 else []

                bm25_candidates = self._parse_hits(bm25_hits_raw)
                knn_candidates = self._parse_hits(knn_hits_raw)

                if query.fusion_algorithm == FusionAlgorithm.LINEAR_COMBINATION:
                    retrieval_mode_used += "_linear"
                    fused_candidates = LinearCombinationFusion.fuse(
                        bm25_candidates, knn_candidates, alpha=query.hybrid_alpha
                    )
                else:
                    retrieval_mode_used += "_rrf"
                    fused_candidates = ReciprocalRankFusion.fuse(
                        bm25_candidates, knn_candidates, rrf_k=query.rrf_k
                    )

                # Filter by score threshold if specified
                if query.score_threshold > 0.0:
                    fused_candidates = [c for c in fused_candidates if c.score >= query.score_threshold]

                # Select candidates for re-ranking
                candidates_to_rerank = fused_candidates[:candidate_pool_size]

                if query.rerank and self.reranker:
                    try:
                        chunks = self.reranker.rerank(clean_query, candidates_to_rerank, top_k=top_k)
                        retrieval_mode_used += "_cohere_rerank"
                    except Exception as r_exc:
                        logger.warning(f"Cohere rerank failed: {r_exc}. Falling back to RRF order.")
                        degraded = True
                        degradation_reason = (
                            f"{degradation_reason}; " if degradation_reason else ""
                        ) + f"Cohere rerank failed: {r_exc}. Degraded to RRF order."
                        chunks = candidates_to_rerank[:top_k]
                else:
                    chunks = candidates_to_rerank[:top_k]

            else:
                # Degraded path: pure BM25
                retrieval_mode_used = "hybrid_degraded_bm25"
                bm25_q = OpenSearchQueryBuilder.build_bm25_query(
                    clean_query, size=candidate_pool_size, filters=query.filters
                )
                resp = self.client.search(index=self.index_name, body=bm25_q)
                raw_hits = resp.get("hits", {}).get("hits", [])
                chunks = self._parse_hits(raw_hits)[:top_k]

        # Post-ranking parent-child expansion on winning top_k chunks
        if query.expand_parent_context:
            self._expand_parent_contexts(chunks, max_chars=query.parent_max_chars)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=clean_query,
            search_type=query.search_type,
            total_hits=len(chunks),
            chunks=chunks,
            latency_ms=latency_ms,
            retrieval_mode_used=retrieval_mode_used,
            filters_applied=query.filters.to_dict() if query.filters else {},
            degraded=degraded,
            degradation_reason=degradation_reason,
        )

    async def aretrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Asynchronous execution delegating to retrieve (thread pool friendly)."""
        import asyncio
        return await asyncio.to_thread(self.retrieve, query)

    def search_by_product_id(self, product_id: str) -> Optional[RetrievedChunk]:
        """Direct point lookup for deterministic product verification."""
        q = {
            "size": 1,
            "query": {"term": {"metadata.product_id": product_id}},
        }
        resp = self.client.search(index=self.index_name, body=q)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return None
        return self._parse_hits(hits)[0]

    def search_promotions(self, min_spend: Optional[float] = None) -> List[RetrievedChunk]:
        """Retrieves currently active coupons matching spend criteria."""
        filter_clauses: List[Dict[str, Any]] = [{"term": {"category": "promotion"}}]
        if min_spend is not None:
            filter_clauses.append({"range": {"metadata.min_spend": {"lte": min_spend}}})

        q = {
            "size": 10,
            "query": {"bool": {"filter": filter_clauses}},
        }
        resp = self.client.search(index=self.index_name, body=q)
        hits = resp.get("hits", {}).get("hits", [])
        return self._parse_hits(hits)
