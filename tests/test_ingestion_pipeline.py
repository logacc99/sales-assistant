"""End-to-end unit tests for IngestionPipeline orchestration and Change Data Detection (CDC)."""

from unittest.mock import MagicMock
from pathlib import Path
import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.models import EcomChunk, EcomChunkMetadata, ProductMetadata


@pytest.fixture
def mock_pipeline_components():
    mock_chunker = MagicMock()
    mock_embedder = MagicMock()
    mock_indexer = MagicMock()

    pipeline = IngestionPipeline(
        chunker=mock_chunker,
        embedder=mock_embedder,
        indexer=mock_indexer,
        index_name="test-catalog",
        enable_cdc=True,
        enable_reconciliation=False,
    )
    return pipeline, mock_chunker, mock_embedder, mock_indexer


def test_cdc_skips_unchanged_chunks(mock_pipeline_components, tmp_path):
    pipeline, mock_chunker, mock_embedder, mock_indexer = mock_pipeline_components

    test_file = tmp_path / "catalog.md"
    test_file.write_text("dummy content")

    chunk1 = EcomChunk.create(
        source_url="https://test.com/1",
        doc_id="doc1",
        chunk_index=0,
        content="Product 1 content",
        metadata=EcomChunkMetadata(category="product", source_file="catalog.md"),
    )
    chunk2 = EcomChunk.create(
        source_url="https://test.com/2",
        doc_id="doc1",
        chunk_index=1,
        content="Product 2 content",
        metadata=EcomChunkMetadata(category="product", source_file="catalog.md"),
    )

    mock_chunker.chunk.return_value = [chunk1, chunk2]

    # Scenario 1: Both chunks already exist in OpenSearch with identical hashes
    mock_indexer.get_existing_hashes.return_value = {
        chunk1.id: chunk1.content_hash,
        chunk2.id: chunk2.content_hash,
    }

    result = pipeline.ingest_file(test_file)

    assert result["total_chunks"] == 2
    assert result["skipped_cdc"] == 2
    assert result["embedded_chunks"] == 0
    assert result["indexed_chunks"] == 0
    # Crucial: 0 calls to Bedrock embedder!
    mock_embedder.embed_chunks.assert_not_called()
    mock_indexer.index_chunks.assert_not_called()


def test_cdc_embeds_only_modified_chunk(mock_pipeline_components, tmp_path):
    pipeline, mock_chunker, mock_embedder, mock_indexer = mock_pipeline_components

    test_file = tmp_path / "catalog.md"
    test_file.write_text("dummy content")

    chunk1 = EcomChunk.create(
        source_url="https://test.com/1",
        doc_id="doc1",
        chunk_index=0,
        content="Product 1 content",
        metadata=EcomChunkMetadata(category="product", source_file="catalog.md"),
    )
    chunk2 = EcomChunk.create(
        source_url="https://test.com/2",
        doc_id="doc1",
        chunk_index=1,
        content="Product 2 content modified price",
        metadata=EcomChunkMetadata(category="product", source_file="catalog.md"),
    )

    mock_chunker.chunk.return_value = [chunk1, chunk2]

    # Scenario 2: chunk1 unchanged, chunk2 has old/different hash in OpenSearch
    mock_indexer.get_existing_hashes.return_value = {
        chunk1.id: chunk1.content_hash,
        chunk2.id: "old_outdated_hash_xyz",
    }
    mock_embedder.embed_chunks.return_value = [chunk2]
    mock_indexer.index_chunks.return_value = (1, [])

    result = pipeline.ingest_file(test_file)

    assert result["total_chunks"] == 2
    assert result["skipped_cdc"] == 1
    assert result["embedded_chunks"] == 1
    assert result["indexed_chunks"] == 1
    # Only chunk2 passed to embed_chunks
    mock_embedder.embed_chunks.assert_called_once_with([chunk2])
    mock_indexer.index_chunks.assert_called_once_with("test-catalog", [chunk2])


def test_ingest_directory_with_orphan_reconciliation(mock_pipeline_components, tmp_path):
    pipeline, mock_chunker, mock_embedder, mock_indexer = mock_pipeline_components
    pipeline.enable_reconciliation = True

    f1 = tmp_path / "file1.md"
    f1.write_text("content 1")

    chunk = EcomChunk.create(
        source_url="https://test.com/item",
        doc_id="doc1",
        chunk_index=0,
        content="Active item",
        metadata=EcomChunkMetadata(category="product", source_file="file1.md"),
    )
    mock_chunker.chunk.return_value = [chunk]
    mock_indexer.get_existing_hashes.return_value = {}
    mock_embedder.embed_chunks.return_value = [chunk]
    mock_indexer.index_chunks.return_value = (1, [])
    mock_indexer.purge_stale_documents.return_value = 2

    summary = pipeline.ingest_directory(tmp_path)

    assert summary["total_files"] == 1
    assert summary["total_chunks"] == 1
    assert summary["indexed_chunks"] == 1
    assert summary["purged_orphans"] == 2
    mock_indexer.purge_stale_documents.assert_called_once_with(
        "test-catalog", [chunk.id], source_prefix=None
    )
