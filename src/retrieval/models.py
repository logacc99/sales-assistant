"""Data models, request/response structures, and enums for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class SearchType(str, Enum):
    """Retrieval execution strategy."""
    HYBRID = "hybrid"
    VECTOR = "vector"
    LEXICAL = "lexical"


class FusionAlgorithm(str, Enum):
    """Rank fusion algorithm used to merge BM25 and dense vector results."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR_COMBINATION = "linear"  # Normalized weighted sum: alpha * dense + (1 - alpha) * bm25


class RerankerType(str, Enum):
    """Reranker implementation type."""
    BEDROCK_COHERE = "bedrock_cohere"
    NOOP = "noop"


@dataclass
class FilterCriteria:
    """Structured search filters for OpenSearch queries."""
    category: Optional[Literal["product", "promotion", "policy"]] = None
    stock_status: Optional[Literal["in_stock", "out_of_stock", "backorder"]] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    policy_type: Optional[Literal["shipping", "returns", "warranty", "privacy", "terms"]] = None
    promo_code: Optional[str] = None
    source_prefix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary omitting None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RetrievalQuery:
    """Input query specification for hybrid retrieval."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    top_k: int = 5
    filters: Optional[FilterCriteria] = None
    fusion_algorithm: FusionAlgorithm = FusionAlgorithm.RRF
    rrf_k: int = 60
    hybrid_alpha: float = 0.5  # Weight of dense vector in linear fusion (0.0 = pure BM25, 1.0 = pure vector)
    score_threshold: float = 0.0  # Minimum similarity/fusion score cutoff
    expand_parent_context: bool = True  # If True, replace content with parent_section_content on final top-K hits
    parent_max_chars: int = 2000  # Safety cap for injected parent content
    rerank: bool = True  # Always-on Bedrock Cohere Rerank by default
    rerank_model_id: str = "cohere.rerank-v3-5:0"
    reranker_type: RerankerType = RerankerType.BEDROCK_COHERE

    @property
    def candidate_pool_size(self) -> int:
        """Dynamic candidate pool size fetched from OpenSearch for reranking: min(max(top_k * 3, 15), 30)."""
        return min(max(self.top_k * 3, 15), 30)


@dataclass
class RetrievedChunk:
    """Individual chunk hit retrieved, scored, and re-ranked."""
    chunk_id: str
    doc_id: str
    category: str
    content: str
    score: float
    lexical_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_content: Optional[str] = None
    is_parent_expanded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary suitable for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "category": self.category,
            "content": self.content,
            "score": round(self.score, 4),
            "lexical_score": round(self.lexical_score, 4) if self.lexical_score is not None else None,
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
            "rrf_score": round(self.rrf_score, 6) if self.rrf_score is not None else None,
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "lexical_rank": self.lexical_rank,
            "vector_rank": self.vector_rank,
            "rerank_rank": self.rerank_rank,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "is_parent_expanded": self.is_parent_expanded,
        }


@dataclass
class RetrievalResult:
    """Full result bundle from the retrieval engine with degradation telemetry."""
    query: str
    search_type: SearchType
    total_hits: int
    chunks: List[RetrievedChunk]
    latency_ms: float
    retrieval_mode_used: str
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    degradation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary suitable for JSON serialization."""
        return {
            "query": self.query,
            "search_type": self.search_type.value if hasattr(self.search_type, "value") else str(self.search_type),
            "total_hits": self.total_hits,
            "chunks": [c.to_dict() for c in self.chunks],
            "latency_ms": round(self.latency_ms, 2),
            "retrieval_mode_used": self.retrieval_mode_used,
            "filters_applied": self.filters_applied,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }
