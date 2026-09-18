"""Unit tests for BedrockEmbedder with mocked Bedrock runtime client."""

import io
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.ingestion.embedder import BedrockEmbedder
from src.ingestion.models import EcomChunk, EcomChunkMetadata


def test_cohere_embedding_payload_and_dimension():
    """Verify Cohere Multilingual v3 request structure and 1024-dim vector response."""
    mock_client = MagicMock()
    fake_vector = [0.1] * 1024
    response_body = json.dumps({"embeddings": [fake_vector]}).encode("utf-8")
    mock_client.invoke_model.return_value = {"body": io.BytesIO(response_body)}

    embedder = BedrockEmbedder(
        model_id="cohere.embed-multilingual-v3.0",
        dimension=1024,
        client=mock_client,
    )

    vector = embedder.embed_text("Bếp nướng điện đa năng")
    assert len(vector) == 1024

    mock_client.invoke_model.assert_called_once()
    call_kwargs = mock_client.invoke_model.call_args[1]
    assert call_kwargs["modelId"] == "cohere.embed-multilingual-v3.0"
    payload = json.loads(call_kwargs["body"].decode("utf-8"))
    assert payload["texts"] == ["Bếp nướng điện đa năng"]
    assert payload["input_type"] == "search_document"


def test_titan_embedding_payload():
    """Verify Titan Text v2 request structure."""
    mock_client = MagicMock()
    fake_vector = [0.2] * 1024
    response_body = json.dumps({"embedding": fake_vector}).encode("utf-8")
    mock_client.invoke_model.return_value = {"body": io.BytesIO(response_body)}

    embedder = BedrockEmbedder(
        model_id="amazon.titan-embed-text-v2:0",
        dimension=1024,
        client=mock_client,
    )

    vector = embedder.embed_text("Chính sách bảo hành")
    assert len(vector) == 1024

    call_kwargs = mock_client.invoke_model.call_args[1]
    assert call_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    payload = json.loads(call_kwargs["body"].decode("utf-8"))
    assert payload["inputText"] == "Chính sách bảo hành"
    assert payload["dimensions"] == 1024
    assert payload["normalize"] is True


def test_embed_chunks_concurrency():
    """Verify concurrent batch chunk embedding with thread pool."""
    mock_client = MagicMock()
    fake_vector = [0.05] * 1024
    response_body = json.dumps({"embeddings": [fake_vector]}).encode("utf-8")
    mock_client.invoke_model.side_effect = lambda **kwargs: {
        "body": io.BytesIO(response_body)
    }

    embedder = BedrockEmbedder(
        model_id="cohere.embed-multilingual-v3.0",
        dimension=1024,
        max_workers=4,
        client=mock_client,
    )

    chunks = [
        EcomChunk.create(
            source_url=f"https://test.com/item-{i}",
            doc_id="doc1",
            chunk_index=i,
            content=f"Item content {i}",
            metadata=EcomChunkMetadata(category="product", source_file="test.md"),
        )
        for i in range(5)
    ]

    embedded = embedder.embed_chunks(chunks)
    assert len(embedded) == 5
    for c in embedded:
        assert c.embedding == fake_vector

    assert mock_client.invoke_model.call_count == 5


def test_retry_on_throttling():
    """Verify tenacity backoff retry logic on Bedrock ThrottlingException."""
    mock_client = MagicMock()
    fake_vector = [0.3] * 1024

    throttle_err = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeModel",
    )
    success_response = {"body": io.BytesIO(json.dumps({"embeddings": [fake_vector]}).encode("utf-8"))}

    # Fail twice with ThrottlingException, then succeed on 3rd try
    mock_client.invoke_model.side_effect = [throttle_err, throttle_err, success_response]

    embedder = BedrockEmbedder(
        model_id="cohere.embed-multilingual-v3.0",
        dimension=1024,
        client=mock_client,
    )

    vector = embedder.embed_text("Test retry text")
    assert len(vector) == 1024
    assert mock_client.invoke_model.call_count == 3


def test_get_langchain_embeddings():
    """Verify that get_langchain_embeddings returns a configured LangChain object."""
    mock_client = MagicMock()
    embedder = BedrockEmbedder(
        model_id="cohere.embed-multilingual-v3.0",
        dimension=1024,
        client=mock_client,
    )
    lc_embed = embedder.get_langchain_embeddings()
    assert lc_embed.model_id == "cohere.embed-multilingual-v3.0"
