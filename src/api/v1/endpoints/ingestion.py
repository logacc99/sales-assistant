"""REST API endpoints for chunking, embedding, OpenSearch indexing, and search verification."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import (
    get_app_settings,
    get_chunker,
    get_embedder,
    get_indexer,
    get_pipeline,
)
from src.api.v1.schemas.common import ApiResponse
from src.api.v1.schemas.ingestion import (
    CheckHashesRequest,
    CheckHashesResponse,
    ChunkPreviewItem,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    EmbedTextRequest,
    EmbedTextResponse,
    IndexInitRequest,
    IndexInitResponse,
    IndexStatusResponse,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestFileRequest,
    IngestFileResponse,
    ReconcileRequest,
    ReconcileResponse,
    SearchResultItem,
    SearchTestRequest,
    SearchTestResponse,
)
from src.config import Settings
from src.ingestion.chunker import BaseChunker
from src.ingestion.embedder import BaseEmbedder
from src.ingestion.indexer import BaseVectorIndexer, OpenSearchVectorIndexer
from src.ingestion.models import EcomChunk
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# -------------------------------------------------------------------------
# 1. Chunking Preview Endpoint
# -------------------------------------------------------------------------
@router.post(
    "/chunk/preview",
    response_model=ApiResponse[ChunkPreviewResponse],
    status_code=status.HTTP_200_OK,
    summary="Preview chunking on file or raw markdown",
    description="Inspect boilerplate cleaning and structured chunk extraction without database operations.",
)
def preview_chunks(
    request: ChunkPreviewRequest,
    chunker: BaseChunker = Depends(get_chunker),
) -> ApiResponse[ChunkPreviewResponse]:
    """Inspects chunking candidate extraction from local file path or raw markdown string."""
    if not request.file_path and not request.raw_markdown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'raw_markdown' must be provided.",
        )

    chunks: list[EcomChunk] = []
    if request.file_path:
        p = Path(request.file_path)
        if not p.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}",
            )
        chunks = chunker.chunk(p)
    elif request.raw_markdown:
        chunks = chunker.chunk_markdown(
            request.raw_markdown,
            source_url=request.source_url or "",
        )

    items = []
    for c in chunks:
        meta_dict = {
            "source_url": c.metadata.source_url,
            "source_file": c.metadata.source_file,
            "page_title": c.metadata.page_title,
            "category": c.metadata.category,
        }
        if c.metadata.product:
            meta_dict.update(
                {
                    "product_id": c.metadata.product.product_id,
                    "name": c.metadata.product.name,
                    "price": c.metadata.product.price,
                    "currency": c.metadata.product.currency,
                    "stock_status": c.metadata.product.stock_status,
                }
            )
        elif c.metadata.policy:
            meta_dict.update(
                {
                    "policy_type": c.metadata.policy.policy_type,
                    "section_title": c.metadata.policy.section_title,
                }
            )

        items.append(
            ChunkPreviewItem(
                id=c.id,
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                content_hash=c.content_hash,
                category=c.metadata.category,
                content=c.content,
                metadata=meta_dict,
            )
        )

    return ApiResponse(
        success=True,
        message=f"Parsed {len(items)} chunks successfully",
        data=ChunkPreviewResponse(total_chunks=len(items), chunks=items),
    )


# -------------------------------------------------------------------------
# 2. Bedrock Embedding Test Endpoint
# -------------------------------------------------------------------------
@router.post(
    "/embed/text",
    response_model=ApiResponse[EmbedTextResponse],
    status_code=status.HTTP_200_OK,
    summary="Test Bedrock vector embedding generation",
    description="Validate AWS Bedrock credentials, latency, and vector dimensions on test text.",
)
def embed_text(
    request: EmbedTextRequest,
    embedder: BaseEmbedder = Depends(get_embedder),
) -> ApiResponse[EmbedTextResponse]:
    """Generates 1024-dim normalized vector via BedrockEmbedder."""
    try:
        t0 = time.perf_counter()
        vector = embedder.embed_text(request.text)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        norm = math.sqrt(sum(x * x for x in vector)) if vector else 0.0
        model_id = getattr(embedder, "model_id", "bedrock-embedding")

        return ApiResponse(
            success=True,
            message="Vector generated successfully",
            data=EmbedTextResponse(
                model_id=model_id,
                dimension=len(vector),
                norm=round(norm, 5),
                embedding_preview=[round(x, 5) for x in vector[:5]],
                latency_ms=latency_ms,
            ),
        )
    except Exception as exc:
        logger.exception(f"Embedding generation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bedrock embedding failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 3. OpenSearch Index Status
# -------------------------------------------------------------------------
@router.get(
    "/index/status",
    response_model=ApiResponse[IndexStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="Inspect OpenSearch index status",
    description="Check whether index exists, document count, and mapping health.",
)
def get_index_status(
    index_name: Optional[str] = Query(None, description="OpenSearch index name"),
    settings: Settings = Depends(get_app_settings),
    indexer: BaseVectorIndexer = Depends(get_indexer),
) -> ApiResponse[IndexStatusResponse]:
    """Returns OpenSearch index metadata and document count."""
    target_index = index_name or settings.opensearch_index_name
    client = getattr(indexer, "client", None)

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenSearch client is not configured.",
        )

    try:
        exists = client.indices.exists(index=target_index)
        doc_count = 0
        if exists:
            count_res = client.count(index=target_index)
            doc_count = count_res.get("count", 0)

        return ApiResponse(
            success=True,
            data=IndexStatusResponse(
                index_name=target_index,
                exists=exists,
                doc_count=doc_count,
                dimension=settings.bedrock_dimension,
                knn_engine="lucene",
                vietnamese_analyzer_configured=True,
            ),
        )
    except Exception as exc:
        logger.exception(f"Failed checking OpenSearch index status: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenSearch query failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 4. OpenSearch Index Initialization
# -------------------------------------------------------------------------
@router.post(
    "/index/init",
    response_model=ApiResponse[IndexInitResponse],
    status_code=status.HTTP_200_OK,
    summary="Initialize or recreate OpenSearch index",
    description="Creates index with dual k-NN and Vietnamese BM25 mappings.",
)
def init_index(
    request: IndexInitRequest,
    settings: Settings = Depends(get_app_settings),
    indexer: BaseVectorIndexer = Depends(get_indexer),
) -> ApiResponse[IndexInitResponse]:
    """Ensures index exists with proper mappings, optionally recreating it."""
    target_index = request.index_name or settings.opensearch_index_name
    try:
        created = indexer.ensure_index_exists(target_index, recreate=request.recreate)
        return ApiResponse(
            success=True,
            message=f"Index '{target_index}' processed successfully",
            data=IndexInitResponse(
                index_name=target_index,
                created=created,
                recreated=request.recreate,
            ),
        )
    except Exception as exc:
        logger.exception(f"Failed to initialize index '{target_index}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index initialization failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 5. Ingest Single File
# -------------------------------------------------------------------------
@router.post(
    "/ingest/file",
    response_model=ApiResponse[IngestFileResponse],
    status_code=status.HTTP_200_OK,
    summary="Ingest a single crawled markdown file",
    description="Runs chunking, CDC diffing, Bedrock embedding, and OpenSearch bulk indexing.",
)
def ingest_file(
    request: IngestFileRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> ApiResponse[IngestFileResponse]:
    """Processes a single file with Change Data Detection."""
    p = Path(request.file_path)
    if not p.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )

    pipeline.enable_cdc = request.enable_cdc
    if request.index_name:
        pipeline.index_name = request.index_name

    try:
        result = pipeline.ingest_file(p)
        return ApiResponse(
            success=True,
            message=f"Ingested file {p.name}: {result['indexed_chunks']} indexed, {result['skipped_cdc']} skipped (CDC)",
            data=IngestFileResponse(**result),
        )
    except Exception as exc:
        logger.exception(f"Failed ingesting file {request.file_path}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 6. Ingest Batch Directory
# -------------------------------------------------------------------------
@router.post(
    "/ingest/batch",
    response_model=ApiResponse[IngestBatchResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch ingest directory of markdown files",
    description="Processes directory with batch CDC filtering and optional orphan reconciliation.",
)
def ingest_batch(
    request: IngestBatchRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> ApiResponse[IngestBatchResponse]:
    """Iterates through crawled markdown directory with CDC and orphan cleanup."""
    p = Path(request.dir_path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {request.dir_path}",
        )

    pipeline.enable_cdc = request.enable_cdc
    pipeline.enable_reconciliation = request.enable_reconciliation
    if request.index_name:
        pipeline.index_name = request.index_name

    try:
        result = pipeline.ingest_directory(p, pattern=request.pattern)
        return ApiResponse(
            success=True,
            message=f"Batch ingestion complete: {result['indexed_chunks']} indexed across {result['total_files']} files",
            data=IngestBatchResponse(**result),
        )
    except Exception as exc:
        logger.exception(f"Batch ingestion failed for {request.dir_path}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ingestion failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 7. CDC Content Hash Check
# -------------------------------------------------------------------------
@router.post(
    "/cdc/check-hashes",
    response_model=ApiResponse[CheckHashesResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch query existing document content hashes",
    description="Uses OpenSearch _mget point lookups for fast CDC diff verification.",
)
def check_hashes(
    request: CheckHashesRequest,
    settings: Settings = Depends(get_app_settings),
    indexer: BaseVectorIndexer = Depends(get_indexer),
) -> ApiResponse[CheckHashesResponse]:
    """Returns existing {doc_id: content_hash} mappings from OpenSearch."""
    target_index = request.index_name or settings.opensearch_index_name
    try:
        hashes = indexer.get_existing_hashes(target_index, request.doc_ids)
        return ApiResponse(
            success=True,
            message=f"Found {len(hashes)} existing hashes for {len(request.doc_ids)} requested IDs",
            data=CheckHashesResponse(existing_hashes=hashes),
        )
    except Exception as exc:
        logger.exception(f"Failed checking hashes on {target_index}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hash check failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 8. Orphan Document Reconciliation
# -------------------------------------------------------------------------
@router.post(
    "/reconcile",
    response_model=ApiResponse[ReconcileResponse],
    status_code=status.HTTP_200_OK,
    summary="Purge orphaned / discontinued documents",
    description="Deletes indexed documents under source_prefix whose IDs are absent from active_doc_ids.",
)
def reconcile_orphans(
    request: ReconcileRequest,
    settings: Settings = Depends(get_app_settings),
    indexer: BaseVectorIndexer = Depends(get_indexer),
) -> ApiResponse[ReconcileResponse]:
    """Purges stale documents no longer in active dataset."""
    target_index = request.index_name or settings.opensearch_index_name
    try:
        deleted = indexer.purge_stale_documents(
            target_index,
            request.active_doc_ids,
            source_prefix=request.source_prefix,
        )
        return ApiResponse(
            success=True,
            message=f"Purged {deleted} stale documents from '{target_index}'",
            data=ReconcileResponse(purged_count=deleted),
        )
    except Exception as exc:
        logger.exception(f"Reconciliation failed on {target_index}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reconciliation failed: {str(exc)}",
        )


# -------------------------------------------------------------------------
# 9. Search Self-Test Endpoint
# -------------------------------------------------------------------------
@router.post(
    "/search/test",
    response_model=ApiResponse[SearchTestResponse],
    status_code=status.HTTP_200_OK,
    summary="Self-test search retrieval",
    description="Immediate verification of BM25 lexical, dense k-NN, or hybrid search against indexed documents.",
)
def search_test(
    request: SearchTestRequest,
    settings: Settings = Depends(get_app_settings),
    indexer: BaseVectorIndexer = Depends(get_indexer),
    embedder: BaseEmbedder = Depends(get_embedder),
) -> ApiResponse[SearchTestResponse]:
    """Executes verification search against OpenSearch index."""
    target_index = request.index_name or settings.opensearch_index_name
    client = getattr(indexer, "client", None)

    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenSearch client is not configured.",
        )

    try:
        if not client.indices.exists(index=target_index):
            return ApiResponse(
                success=True,
                message=f"Index '{target_index}' does not exist yet.",
                data=SearchTestResponse(
                    query=request.query,
                    total_hits=0,
                    search_type=request.search_type,
                    results=[],
                ),
            )

        # Build filter clauses
        filter_clauses: list[dict[str, Any]] = []
        if request.category:
            filter_clauses.append({"term": {"category": request.category}})

        # Build search query according to search_type
        if request.search_type == "lexical":
            query_body = {
                "size": request.top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": [
                                        "content^2",
                                        "content.folded^1.5",
                                        "metadata.product_name^3",
                                        "metadata.product_name.folded^2",
                                    ],
                                }
                            }
                        ],
                        "filter": filter_clauses,
                    }
                },
            }
        elif request.search_type == "vector":
            query_vec = embedder.embed_text(request.query)
            query_body = {
                "size": request.top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_vec,
                                        "k": request.top_k,
                                    }
                                }
                            }
                        ],
                        "filter": filter_clauses,
                    }
                },
            }
        else:  # hybrid
            query_vec = embedder.embed_text(request.query)
            query_body = {
                "size": request.top_k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": [
                                        "content^2",
                                        "content.folded^1.5",
                                        "metadata.product_name^3",
                                        "metadata.product_name.folded^2",
                                    ],
                                    "boost": 0.5,
                                }
                            },
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_vec,
                                        "k": request.top_k,
                                        "boost": 0.5,
                                    }
                                }
                            },
                        ],
                        "filter": filter_clauses,
                    }
                },
            }

        response = client.search(index=target_index, body=query_body)
        raw_hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {}).get("value", len(raw_hits))

        results: list[SearchResultItem] = []
        for h in raw_hits:
            source = h.get("_source", {})
            meta = source.get("metadata", {})
            results.append(
                SearchResultItem(
                    chunk_id=h.get("_id", ""),
                    score=float(h.get("_score", 0.0) or 0.0),
                    category=source.get("category", ""),
                    content=source.get("content", ""),
                    product_name=meta.get("product_name"),
                    price=meta.get("price"),
                    stock_status=meta.get("stock_status"),
                    metadata=meta,
                )
            )

        return ApiResponse(
            success=True,
            message=f"Search returned {len(results)} hits",
            data=SearchTestResponse(
                query=request.query,
                total_hits=total_hits,
                search_type=request.search_type,
                results=results,
            ),
        )
    except Exception as exc:
        logger.exception(f"Search self-test failed on {target_index}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(exc)}",
        )
