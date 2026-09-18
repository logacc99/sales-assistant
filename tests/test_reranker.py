"""Unit tests for NoOpReranker and BedrockCohereReranker."""

import io
import json
from unittest.mock import MagicMock
import pytest

from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import BedrockCohereReranker, NoOpReranker


def _create_chunks():
    c1 = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        category="product",
        content="Bếp nướng củi mini",
        score=0.5,
    )
    c2 = RetrievedChunk(
        chunk_id="c2",
        doc_id="d2",
        category="product",
        content="Bếp nướng gas cao cấp 4 họng inox 304",
        score=0.4,
    )
    return [c1, c2]


def test_noop_reranker():
    chunks = _create_chunks()
    reranker = NoOpReranker()
    res = reranker.rerank(query="bếp nướng gas", chunks=chunks, top_k=1)
    assert len(res) == 1
    assert res[0].chunk_id == "c1"
    assert res[0].rerank_rank == 1
    assert res[0].rerank_score == 0.5


def test_bedrock_cohere_reranker_success():
    chunks = _create_chunks()
    mock_client = MagicMock()

    # Bedrock Cohere response flips the order: c2 (index 1) gets relevance 0.98, c1 (index 0) gets 0.45
    mock_payload = {
        "results": [
            {"index": 1, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.45},
        ]
    }
    mock_body = io.BytesIO(json.dumps(mock_payload).encode("utf-8"))
    mock_client.invoke_model.return_value = {"body": mock_body}

    reranker = BedrockCohereReranker(
        model_id="cohere.rerank-v3-5:0",
        bedrock_client=mock_client,
    )

    results = reranker.rerank(query="bếp nướng gas ngoài trời", chunks=chunks, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[0].rerank_rank == 1
    assert pytest.approx(results[0].score, 1e-4) == 0.98
    assert pytest.approx(results[0].rerank_score, 1e-4) == 0.98

    assert results[1].chunk_id == "c1"
    assert results[1].rerank_rank == 2
    assert pytest.approx(results[1].score, 1e-4) == 0.45

    mock_client.invoke_model.assert_called_once()
    call_args = mock_client.invoke_model.call_args[1]
    assert call_args["modelId"] == "cohere.rerank-v3-5:0"
    body_sent = json.loads(call_args["body"])
    assert body_sent["query"] == "bếp nướng gas ngoài trời"
    assert len(body_sent["documents"]) == 2


def test_bedrock_cohere_reranker_error():
    chunks = _create_chunks()
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = RuntimeError("AWS ThrottlingException")

    reranker = BedrockCohereReranker(
        model_id="cohere.rerank-v3-5:0",
        bedrock_client=mock_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        reranker.rerank(query="test", chunks=chunks, top_k=2)
    assert "ThrottlingException" in str(exc_info.value)
