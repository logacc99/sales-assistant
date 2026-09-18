"""Unit tests for Ingestion and Indexing REST API endpoints."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_chunker,
    get_embedder,
    get_indexer,
    get_pipeline,
)
from src.api.main import app
from src.ingestion.models import (
    EcomChunk,
    EcomChunkMetadata,
    PolicyMetadata,
    ProductMetadata,
)

client = TestClient(app)


@pytest.fixture
def mock_deps():
    mock_c = MagicMock()
    mock_e = MagicMock()
    mock_i = MagicMock()
    mock_p = MagicMock()

    app.dependency_overrides[get_chunker] = lambda: mock_c
    app.dependency_overrides[get_embedder] = lambda: mock_e
    app.dependency_overrides[get_indexer] = lambda: mock_i
    app.dependency_overrides[get_pipeline] = lambda: mock_p

    yield {
        "chunker": mock_c,
        "embedder": mock_e,
        "indexer": mock_i,
        "pipeline": mock_p,
    }

    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# Chunk Preview Tests
# -----------------------------------------------------------------------------
def test_chunk_preview_raw_markdown(mock_deps):
    mock_chunker = mock_deps["chunker"]
    sample_chunk = EcomChunk.create(
        source_url="https://test.com/p1",
        doc_id="p1",
        chunk_index=0,
        content="Tên sản phẩm: Bếp nướng BBQ | Giá: 5000000 VND",
        metadata=EcomChunkMetadata(
            category="product",
            source_file="test.md",
            product=ProductMetadata(
                product_id="p1",
                name="Bếp nướng BBQ",
                price=5000000.0,
                stock_status="in_stock",
            ),
        ),
    )
    mock_chunker.chunk_markdown.return_value = [sample_chunk]

    response = client.post(
        "/api/v1/ingestion/chunk/preview",
        json={"raw_markdown": "# Bếp nướng\nGiá: 5.000.000đ"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_chunks"] == 1
    assert body["data"]["chunks"][0]["metadata"]["product_id"] == "p1"


def test_chunk_preview_file_path(mock_deps, tmp_path):
    mock_chunker = mock_deps["chunker"]
    test_file = tmp_path / "sample.md"
    test_file.write_text("# Test")

    sample_chunk = EcomChunk.create(
        source_url="https://test.com/policy",
        doc_id="pol1",
        chunk_index=0,
        content="Chính sách bảo hành 12 tháng",
        metadata=EcomChunkMetadata(
            category="policy",
            source_file="sample.md",
            policy=PolicyMetadata(
                policy_type="warranty",
                section_title="Bảo hành",
                effective_date="2026-01-01",
                policy_url="https://test.com/policy",
            ),
        ),
    )
    mock_chunker.chunk.return_value = [sample_chunk]

    response = client.post(
        "/api/v1/ingestion/chunk/preview",
        json={"file_path": str(test_file)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_chunks"] == 1
    assert body["data"]["chunks"][0]["metadata"]["policy_type"] == "warranty"


def test_chunk_preview_missing_args(mock_deps):
    response = client.post("/api/v1/ingestion/chunk/preview", json={})
    assert response.status_code == 400


def test_chunk_preview_file_not_found(mock_deps):
    response = client.post(
        "/api/v1/ingestion/chunk/preview",
        json={"file_path": "/nonexistent/path/doc.md"},
    )
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# Bedrock Embedder Tests
# -----------------------------------------------------------------------------
def test_embed_text_success(mock_deps):
    mock_embedder = mock_deps["embedder"]
    mock_embedder.embed_text.return_value = [0.1] * 1024
    mock_embedder.model_id = "cohere.embed-multilingual-v3.0"

    response = client.post(
        "/api/v1/ingestion/embed/text",
        json={"text": "Bếp than ngoài trời"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["dimension"] == 1024
    assert len(body["data"]["embedding_preview"]) == 5


def test_embed_text_failure(mock_deps):
    mock_embedder = mock_deps["embedder"]
    mock_embedder.embed_text.side_effect = RuntimeError("AWS Bedrock Throttling")

    response = client.post(
        "/api/v1/ingestion/embed/text",
        json={"text": "Bếp than ngoài trời"},
    )
    assert response.status_code == 500
    assert "AWS Bedrock Throttling" in response.json()["message"]


# -----------------------------------------------------------------------------
# OpenSearch Index Lifecycle Tests
# -----------------------------------------------------------------------------
def test_index_status(mock_deps):
    mock_indexer = mock_deps["indexer"]
    mock_indexer.client.indices.exists.return_value = True
    mock_indexer.client.count.return_value = {"count": 42}

    response = client.get("/api/v1/ingestion/index/status?index_name=test-catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["exists"] is True
    assert body["data"]["doc_count"] == 42


def test_index_init(mock_deps):
    mock_indexer = mock_deps["indexer"]
    mock_indexer.ensure_index_exists.return_value = True

    response = client.post(
        "/api/v1/ingestion/index/init",
        json={"index_name": "test-catalog", "recreate": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["created"] is True


# -----------------------------------------------------------------------------
# Ingestion Pipeline File & Batch Tests
# -----------------------------------------------------------------------------
def test_ingest_file(mock_deps, tmp_path):
    mock_pipeline = mock_deps["pipeline"]
    test_file = tmp_path / "page.md"
    test_file.write_text("dummy")

    mock_pipeline.ingest_file.return_value = {
        "file_path": str(test_file),
        "total_chunks": 5,
        "skipped_cdc": 3,
        "embedded_chunks": 2,
        "indexed_chunks": 2,
        "failed_chunks": 0,
        "active_doc_ids": ["d1", "d2"],
    }

    response = client.post(
        "/api/v1/ingestion/ingest/file",
        json={"file_path": str(test_file), "enable_cdc": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_chunks"] == 5
    assert body["data"]["skipped_cdc"] == 3


def test_ingest_batch(mock_deps, tmp_path):
    mock_pipeline = mock_deps["pipeline"]
    mock_pipeline.ingest_directory.return_value = {
        "total_files": 2,
        "total_chunks": 10,
        "skipped_cdc": 8,
        "embedded_chunks": 2,
        "indexed_chunks": 2,
        "failed_chunks": 0,
        "purged_orphans": 0,
        "file_summaries": [],
    }

    response = client.post(
        "/api/v1/ingestion/ingest/batch",
        json={"dir_path": str(tmp_path), "enable_cdc": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_files"] == 2


# -----------------------------------------------------------------------------
# CDC & Orphan Reconciliation Tests
# -----------------------------------------------------------------------------
def test_cdc_check_hashes(mock_deps):
    mock_indexer = mock_deps["indexer"]
    mock_indexer.get_existing_hashes.return_value = {"doc1": "hash123"}

    response = client.post(
        "/api/v1/ingestion/cdc/check-hashes",
        json={"doc_ids": ["doc1"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["existing_hashes"]["doc1"] == "hash123"


def test_reconcile_orphans(mock_deps):
    mock_indexer = mock_deps["indexer"]
    mock_indexer.purge_stale_documents.return_value = 4

    response = client.post(
        "/api/v1/ingestion/reconcile",
        json={"active_doc_ids": ["doc1"], "source_prefix": "https://test.com"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["purged_count"] == 4


# -----------------------------------------------------------------------------
# Search Self-Test Tests
# -----------------------------------------------------------------------------
def test_search_test_hybrid(mock_deps):
    mock_indexer = mock_deps["indexer"]
    mock_embedder = mock_deps["embedder"]

    mock_indexer.client.indices.exists.return_value = True
    mock_embedder.embed_text.return_value = [0.05] * 1024
    mock_indexer.client.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_id": "chunk_123",
                    "_score": 0.95,
                    "_source": {
                        "category": "product",
                        "content": "Bếp nướng BBQ than hoa",
                        "metadata": {
                            "product_name": "Bếp nướng BBQ",
                            "price": 3500000.0,
                            "stock_status": "in_stock",
                        },
                    },
                }
            ],
        }
    }

    response = client.post(
        "/api/v1/ingestion/search/test",
        json={
            "query": "bep nuong than",
            "search_type": "hybrid",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_hits"] == 1
    assert body["data"]["results"][0]["product_name"] == "Bếp nướng BBQ"
