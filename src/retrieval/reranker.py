"""Re-ranking interfaces and AWS Bedrock Cohere Reranker implementation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from src.config import get_config
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for chunk re-ranking."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Re-scores candidate chunks given the search query and returns top_k hits."""
        pass


class NoOpReranker(BaseReranker):
    """Pass-through reranker preserving incoming rank order."""

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        selected = chunks[:top_k]
        for rank, chunk in enumerate(selected, start=1):
            chunk.rerank_rank = rank
            chunk.rerank_score = chunk.score
        return selected


class BedrockCohereReranker(BaseReranker):
    """AWS Bedrock Cohere Rerank client (cohere.rerank-v3-5:0)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        bedrock_client: Optional[Any] = None,
        region_name: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.model_id = model_id or cfg.bedrock_rerank_model_id
        self.region_name = region_name or cfg.aws_region

        if bedrock_client is not None:
            self.client = bedrock_client
        else:
            self.client = self._create_bedrock_client()

    def _create_bedrock_client(self) -> Any:
        """Initializes boto3 Bedrock Runtime client."""
        import boto3
        cfg = get_config()
        session_kwargs: dict[str, Any] = {"region_name": self.region_name}
        if cfg.aws_access_key_id and cfg.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = cfg.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
            if cfg.aws_session_token:
                session_kwargs["aws_session_token"] = cfg.aws_session_token

        session = boto3.Session(**session_kwargs)
        return session.client("bedrock-runtime", region_name=self.region_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Submits candidate chunks to Bedrock Cohere Rerank and re-orders hits."""
        if not chunks or top_k <= 0:
            return []

        # Prepare document texts for Cohere Rerank
        documents = [c.content for c in chunks]
        top_n = min(top_k, len(chunks))

        request_body = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "api_version": 2,
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
        except Exception as exc:
            logger.error(f"Bedrock Cohere reranking failed on model '{self.model_id}': {exc}")
            raise

        results = response_body.get("results", [])
        reranked_chunks: List[RetrievedChunk] = []

        for rank, res in enumerate(results, start=1):
            idx = res.get("index")
            rel_score = float(res.get("relevance_score", 0.0))
            if idx is not None and 0 <= idx < len(chunks):
                chunk = chunks[idx]
                chunk.rerank_score = rel_score
                chunk.rerank_rank = rank
                chunk.score = rel_score  # Update final score to reranker relevance score
                reranked_chunks.append(chunk)

        # Append any unranked chunks if top_n requested more than results returned
        if len(reranked_chunks) < top_n:
            seen_cids = {c.chunk_id for c in reranked_chunks}
            for chunk in chunks:
                if chunk.chunk_id not in seen_cids and len(reranked_chunks) < top_n:
                    reranked_chunks.append(chunk)

        return reranked_chunks
