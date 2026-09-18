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
    """Verify orphan reconciliation triggers search query and bulk delete for Serverless compatibility."""
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "stale-1"},
                {"_id": "stale-2"},
                {"_id": "stale-3"},
            ]
        }
    }

    indexer = OpenSearchVectorIndexer(client=mock_client)
    with patch("src.ingestion.indexer.helpers.bulk") as mock_bulk:
        mock_bulk.return_value = (3, [])
        deleted = indexer.purge_stale_documents(
            "test-catalog",
            active_doc_ids=["id1", "id2"],
            source_prefix="https://bepbbq.com/san-pham/",
        )

        assert deleted == 3
        mock_client.search.assert_called_once()
        call_body = mock_client.search.call_args[1]["body"]
        must_not = call_body["query"]["bool"]["must_not"]
        assert must_not[0]["terms"]["_id"] == ["id1", "id2"]
        must = call_body["query"]["bool"]["must"]
        assert must[0]["prefix"]["metadata.source_url"] == "https://bepbbq.com/san-pham/"

        mock_bulk.assert_called_once()
        actions = mock_bulk.call_args[0][1]
        assert len(actions) == 3
        assert actions[0]["_op_type"] == "delete"
        assert actions[0]["_id"] == "stale-1"


def test_serverless_auth_configuration():
    """Verify OpenSearch client configures SigV4 service='aoss' for serverless endpoints."""
    with patch("boto3.Session") as mock_session_cls, \
         patch("opensearchpy.AWSV4SignerAuth") as mock_auth_cls, \
         patch("src.ingestion.indexer.OpenSearch") as mock_os_cls:
        
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()
        mock_session_cls.return_value = mock_session

        indexer = OpenSearchVectorIndexer(
            host="https://abcd1234efgh5678.ap-southeast-1.aoss.amazonaws.com",
            port=443,
            use_aws_auth=True,
        )

        # Ensure AWSV4SignerAuth was initialized with service='aoss'
        mock_auth_cls.assert_called_once()
        call_args = mock_auth_cls.call_args
        assert call_args[0][2] == "aoss"


def test_as_langchain_vectorstore_serverless():
    """Verify as_langchain_vectorstore sets is_aoss=True."""
    mock_client = MagicMock()
    mock_client.transport.hosts = [{"http_auth": MagicMock()}]
    indexer = OpenSearchVectorIndexer(
        client=mock_client,
        host="https://test.aoss.amazonaws.com",
        port=443,
    )

    with patch("langchain_community.vectorstores.OpenSearchVectorSearch") as mock_vs:
        indexer.as_langchain_vectorstore("test-catalog")
        mock_vs.assert_called_once()
        kwargs = mock_vs.call_args[1]
        assert kwargs["is_aoss"] is True
        assert kwargs["use_ssl"] is True

