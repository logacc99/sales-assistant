"""Unit tests for NoOpReranker, BedrockCohereReranker, LocalBGEReranker, and RerankerFactory."""

import io
import json
import os
from unittest.mock import MagicMock, patch
import pytest

from src.config import IngestionConfig
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import (
    BaseReranker,
    BedrockCohereReranker,
    LocalBGEReranker,
    NoOpReranker,
    RerankerFactory,
)


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


def test_local_bge_reranker_success():
    chunks = _create_chunks()
    mock_model = MagicMock()
    # Mock model predicts relevance: c1 -> 0.35, c2 -> 0.92
    mock_model.predict.return_value = [0.35, 0.92]

    reranker = LocalBGEReranker(
        model_id="BAAI/bge-reranker-m3",
        model_instance=mock_model,
        batch_size=16,
    )

    results = reranker.rerank(query="bếp nướng gas ngoài trời", chunks=chunks, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[0].rerank_rank == 1
    assert pytest.approx(results[0].score, 1e-4) == 0.92
    assert pytest.approx(results[0].rerank_score, 1e-4) == 0.92

    assert results[1].chunk_id == "c1"
    assert results[1].rerank_rank == 2
    assert pytest.approx(results[1].score, 1e-4) == 0.35

    mock_model.predict.assert_called_once()
    pairs_sent = mock_model.predict.call_args[0][0]
    assert len(pairs_sent) == 2
    assert pairs_sent[0] == ["bếp nướng gas ngoài trời", "Bếp nướng củi mini"]
    assert pairs_sent[1] == ["bếp nướng gas ngoài trời", "Bếp nướng gas cao cấp 4 họng inox 304"]


def test_local_bge_reranker_warmup():
    mock_model = MagicMock()
    reranker = LocalBGEReranker(model_instance=mock_model)
    reranker.initialize()
    mock_model.predict.assert_called_once()


def test_local_bge_reranker_empty_chunks():
    mock_model = MagicMock()
    reranker = LocalBGEReranker(model_instance=mock_model)
    assert reranker.rerank(query="test", chunks=[], top_k=5) == []
    mock_model.predict.assert_not_called()


def test_reranker_factory_creation():
    # 1. Local provider creation
    local_reranker = RerankerFactory.create("local", model_instance=MagicMock())
    assert isinstance(local_reranker, LocalBGEReranker)

    # 2. Bedrock provider creation
    bedrock_reranker = RerankerFactory.create("bedrock", bedrock_client=MagicMock())
    assert isinstance(bedrock_reranker, BedrockCohereReranker)

    # 3. NoOp provider creation
    noop_reranker = RerankerFactory.create("noop")
    assert isinstance(noop_reranker, NoOpReranker)

    # 4. Unknown provider raises ValueError
    with pytest.raises(ValueError) as exc:
        RerankerFactory.create("unknown_provider")
    assert "Unknown rerank method: 'unknown_provider'" in str(exc.value)


def test_reranker_factory_from_config():
    with patch.dict(os.environ, {"RERANK_METHOD": "noop"}, clear=False):
        cfg = IngestionConfig.from_env()
        reranker = RerankerFactory.create(app_config=cfg)
        assert isinstance(reranker, NoOpReranker)


def test_reranker_factory_custom_registration():
    class CustomReranker(BaseReranker):
        def rerank(self, query, chunks, top_k):
            return chunks[:top_k]

    RerankerFactory.register_reranker("custom_plugin", CustomReranker)
    instance = RerankerFactory.create("custom_plugin")
    assert isinstance(instance, CustomReranker)
    assert "custom_plugin" in RerankerFactory.get_registered_rerankers()
