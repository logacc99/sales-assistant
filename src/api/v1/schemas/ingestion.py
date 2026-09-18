"""Pydantic schemas for Ingestion, Embedding, Indexing, and Search self-test endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ChunkPreviewRequest(BaseModel):
    """Request to preview chunk parsing on a file path or raw markdown string."""

    file_path: Optional[str] = Field(default=None, description="Path to a crawled markdown file")
    raw_markdown: Optional[str] = Field(default=None, description="Raw markdown string with optional frontmatter")
    source_url: Optional[str] = Field(default=None, description="Source URL if providing raw_markdown")


class ChunkPreviewItem(BaseModel):
    """Structured representation of a parsed chunk candidate."""

    id: str
    doc_id: str
    chunk_index: int
    content_hash: str
    category: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkPreviewResponse(BaseModel):
    """Response containing parsed chunks and count."""

    total_chunks: int
    chunks: list[ChunkPreviewItem]


class EmbedTextRequest(BaseModel):
    """Request payload to test Bedrock text embedding generation."""

    text: str = Field(..., description="Text string to generate embedding for")
    model_id: Optional[str] = Field(default=None, description="Optional Bedrock model override")


class EmbedTextResponse(BaseModel):
    """Result of vector embedding generation test."""

    model_id: str
    dimension: int
    norm: float
    embedding_preview: list[float]
    latency_ms: float


class IndexStatusResponse(BaseModel):
    """Status and health of the target OpenSearch index."""

    index_name: str
    exists: bool
    doc_count: int = 0
    dimension: int = 1024
    knn_engine: str = "lucene"
    vietnamese_analyzer_configured: bool = True


class IndexInitRequest(BaseModel):
    """Request to initialize or recreate the OpenSearch index."""

    index_name: Optional[str] = Field(default=None, description="Target index name")
    recreate: bool = Field(default=False, description="Whether to drop and recreate existing index")


class IndexInitResponse(BaseModel):
    """Index creation result."""

    index_name: str
    created: bool
    recreated: bool


class IngestFileRequest(BaseModel):
    """Request to run full ingestion pipeline on a single file."""

    file_path: str = Field(..., description="Path to target markdown file")
    enable_cdc: bool = Field(default=True, description="Enable Change Data Detection")
    index_name: Optional[str] = Field(default=None, description="OpenSearch index override")


class IngestFileResponse(BaseModel):
    """Summary of single-file ingestion."""

    file_path: str
    total_chunks: int
    skipped_cdc: int
    embedded_chunks: int
    indexed_chunks: int
    failed_chunks: int
    active_doc_ids: list[str] = Field(default_factory=list)


class IngestBatchRequest(BaseModel):
    """Request to run ingestion over a directory of files."""

    dir_path: str = Field(..., description="Directory containing crawled markdown files")
    pattern: str = Field(default="*.md", description="Glob pattern for matching files")
    enable_cdc: bool = Field(default=True, description="Enable Change Data Detection")
    enable_reconciliation: bool = Field(default=False, description="Purge stale orphaned docs")
    index_name: Optional[str] = Field(default=None, description="OpenSearch index override")


class IngestBatchResponse(BaseModel):
    """Summary of batch directory ingestion."""

    total_files: int
    total_chunks: int
    skipped_cdc: int
    embedded_chunks: int
    indexed_chunks: int
    failed_chunks: int
    purged_orphans: int
    file_summaries: list[dict[str, Any]] = Field(default_factory=list)


class CheckHashesRequest(BaseModel):
    """Request to query existing document content hashes via batch _mget."""

    doc_ids: list[str] = Field(..., description="List of document/chunk IDs to inspect")
    index_name: Optional[str] = Field(default=None, description="OpenSearch index override")


class CheckHashesResponse(BaseModel):
    """Document hash lookup results."""

    existing_hashes: dict[str, str]


class ReconcileRequest(BaseModel):
    """Request to reconcile and purge orphaned documents."""

    active_doc_ids: list[str] = Field(..., description="Active document IDs to preserve")
    source_prefix: Optional[str] = Field(default=None, description="Optional URL prefix filter")
    index_name: Optional[str] = Field(default=None, description="OpenSearch index override")


class ReconcileResponse(BaseModel):
    """Count of purged stale documents."""

    purged_count: int


class SearchTestRequest(BaseModel):
    """Self-test search request to verify indexed data retrieval."""

    query: str = Field(..., description="Search query string")
    search_type: Literal["hybrid", "lexical", "vector"] = Field(
        default="hybrid", description="Search algorithm to execute"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")
    category: Optional[str] = Field(default=None, description="Filter by category (product/policy/promotion)")
    index_name: Optional[str] = Field(default=None, description="Target index name")


class SearchResultItem(BaseModel):
    """Individual retrieved document from search self-test."""

    chunk_id: str
    score: float
    category: str
    content: str
    product_name: Optional[str] = None
    price: Optional[float] = None
    stock_status: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchTestResponse(BaseModel):
    """Search self-test response."""

    query: str
    total_hits: int
    search_type: str
    results: list[SearchResultItem]
