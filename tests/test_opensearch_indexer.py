"""Unit tests for OpenSearchVectorIndexer with mocked OpenSearch client."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.indexer import OpenSearchVectorIndexer
from src.ingestion.models import EcomChunk, EcomChunkMetadata, ProductMetadata


def test_ensure_index_exists_payload():
    """Verify OpenSearch index mapping and settings payload structure."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = False

    indexer = OpenSearchVectorIndexer(
        client=mock_client,
        dimension=1024,
    )
    indexer.ensure_index_exists("test-catalog")

    mock_client.indices.create.assert_called_once()
    call_args = mock_client.indices.create.call_args
    assert call_args[1]["index"] == "test-catalog"
    body = call_args[1]["body"]

    # Check k-NN setting
    assert body["settings"]["index"]["knn"] is True
    # Check Vietnamese analyzer
    assert "vietnamese_ascii_analyzer" in body["settings"]["analysis"]["analyzer"]
    # Check vector dimension
    assert body["mappings"]["properties"]["embedding"]["dimension"] == 1024
    assert body["mappings"]["properties"]["embedding"]["method"]["engine"] == "lucene"
    # Check folded multi-fields
    assert "folded" in body["mappings"]["properties"]["content"]["fields"]


def test_get_existing_hashes_mget():
    """Verify CDC batch _mget query and result extraction."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.mget.return_value = {
        "docs": [
            {
                "_id": "chunk-1",
                "found": True,
                "_source": {"content_hash": "hash_abc", "doc_id": "doc1"},
            },
            {
                "_id": "chunk-2",
                "found": False,
            },
        ]
    }

    indexer = OpenSearchVectorIndexer(client=mock_client)
    hashes = indexer.get_existing_hashes("test-catalog", ["chunk-1", "chunk-2"])

    mock_client.mget.assert_called_once()
    assert hashes == {"chunk-1": "hash_abc"}


def test_index_chunks_bulk():
    """Verify bulk indexing formatting and helper execution."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True

    indexer = OpenSearchVectorIndexer(client=mock_client)

    chunk = EcomChunk.create(
        source_url="https://bepbbq.com/san-pham/bep-gas/",
        doc_id="doc100",
        chunk_index=0,
        content="Bếp nướng gas",
        metadata=EcomChunkMetadata(
            category="product",
            source_file="gas.md",
            source_url="https://bepbbq.com/san-pham/bep-gas/",
            product=ProductMetadata(
                product_id="P1",
                name="Bếp gas",
                price=5000000.0,
            ),
        ),
        embedding=[0.1] * 1024,
    )

    with patch("src.ingestion.indexer.helpers.bulk") as mock_bulk:
        mock_bulk.return_value = (1, [])
        success, errors = indexer.index_chunks("test-catalog", [chunk])
        assert success == 1
        assert len(errors) == 0
        mock_bulk.assert_called_once()
        actions = mock_bulk.call_args[0][1]
        assert len(actions) == 1
        assert actions[0]["_id"] == chunk.id
        assert actions[0]["_source"]["content_hash"] == chunk.content_hash
        assert actions[0]["_source"]["metadata"]["price"] == 5000000.0


def test_purge_stale_documents():
    """Verify orphan reconciliation triggers delete_by_query with excluded active IDs."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.delete_by_query.return_value = {"deleted": 3}

    indexer = OpenSearchVectorIndexer(client=mock_client)
    deleted = indexer.purge_stale_documents(
        "test-catalog",
        active_doc_ids=["id1", "id2"],
        source_prefix="https://bepbbq.com/san-pham/",
    )

    assert deleted == 3
    mock_client.delete_by_query.assert_called_once()
    call_body = mock_client.delete_by_query.call_args[1]["body"]
    must_not = call_body["query"]["bool"]["must_not"]
    assert must_not[0]["terms"]["_id"] == ["id1", "id2"]
    must = call_body["query"]["bool"]["must"]
    assert must[0]["prefix"]["metadata.source_url"] == "https://bepbbq.com/san-pham/"
