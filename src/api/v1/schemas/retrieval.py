"""Pydantic schemas for retrieval endpoints matching SPEC-0003."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.retrieval.models import FusionAlgorithm, SearchType


class FilterCriteriaModel(BaseModel):
    """Filter criteria model for search requests."""
    category: Optional[Literal["product", "promotion", "policy"]] = Field(
        default=None, description="Domain category filter"
    )
    stock_status: Optional[Literal["in_stock", "out_of_stock", "backorder"]] = Field(
        default=None, description="Inventory stock status filter"
    )
    min_price: Optional[float] = Field(default=None, ge=0.0, description="Minimum price boundary")
    max_price: Optional[float] = Field(default=None, ge=0.0, description="Maximum price boundary")
    policy_type: Optional[Literal["shipping", "returns", "warranty", "privacy", "terms"]] = Field(
        default=None, description="Policy subsection type filter"
    )
    promo_code: Optional[str] = Field(default=None, description="Promo coupon code filter")
    source_prefix: Optional[str] = Field(default=None, description="Filter documents by source URL prefix")


class SearchApiRequest(BaseModel):
    """Request payload for hybrid retrieval."""
    query: str = Field(..., min_length=1, description="Shopper inquiry text")
    search_type: SearchType = Field(
        default=SearchType.HYBRID, description="Retrieval strategy: hybrid, vector, or lexical"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filters: Optional[FilterCriteriaModel] = Field(default=None, description="Metadata filters")
    fusion_algorithm: FusionAlgorithm = Field(
        default=FusionAlgorithm.RRF, description="Fusion algorithm: rrf or linear"
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vector weight in linear fusion")
    score_threshold: float = Field(default=0.0, ge=0.0, description="Minimum score threshold cutoff")
    expand_parent_context: bool = Field(
        default=True, description="Expand parent section context for policy chunks"
    )
    parent_max_chars: int = Field(
        default=2000, ge=100, le=10000, description="Maximum character cap for parent policy context"
    )
    rerank: bool = Field(default=True, description="Enable Bedrock Cohere Rerank")


class RetrievedChunkModel(BaseModel):
    """Individual retrieved chunk hit."""
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_parent_expanded: bool = False


class SearchApiResponse(BaseModel):
    """Response payload for hybrid search."""
    query: str
    search_type: str
    total_hits: int
    chunks: List[RetrievedChunkModel]
    latency_ms: float
    retrieval_mode_used: str
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    degraded: bool = False
    degradation_reason: Optional[str] = None


class ExplainScoringItem(BaseModel):
    """Detailed score explanation for a single chunk."""
    final_rank: int
    chunk_id: str
    category: str
    final_score: float
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    lexical_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None
    content_snippet: str
    product_name: Optional[str] = None


class ExplainApiResponse(BaseModel):
    """Response payload for explain endpoint."""
    query: str
    latency_ms: float
    total_hits: int
    retrieval_mode_used: str
    degraded: bool
    degradation_reason: Optional[str] = None
    scoring_breakdown: List[ExplainScoringItem]


class RetrievalHealthResponse(BaseModel):
    """Retrieval service readiness health check."""
    status: str
    opensearch_connected: bool
    index_exists: bool
    index_name: str
    document_count: int
    bedrock_embedder_ready: bool
    bedrock_reranker_ready: bool
    is_serverless: bool = True
    collection_type: Optional[str] = "VECTORSEARCH"
