"""Unit tests for Bedrock Converse client and GroundedResponseGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from botocore.exceptions import ClientError

from src.generation.bedrock_client import BedrockConverseClient
from src.generation.generator import GroundedResponseGenerator
from src.generation.models import (
    CitationType,
    GenerationConfig,
    GenerationRequest,
    TokenUsage,
)
from src.generation.service import GenerationService
from src.retrieval.models import RetrievedChunk


@pytest.fixture
def mock_boto_client():
    mock = MagicMock()
    # Mock converse response
    mock.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Dạ chào bạn, Bếp nướng Weber [1] có giá 12.500.000 ₫."}],
            }
        },
        "usage": {"inputTokens": 100, "outputTokens": 30, "totalTokens": 130},
        "stopReason": "end_turn",
    }
    # Mock converse_stream response
    mock.converse_stream.return_value = {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "Dạ chào bạn, "}}},
            {"contentBlockDelta": {"delta": {"text": "Bếp nướng Weber [1] "}}},
            {"contentBlockDelta": {"delta": {"text": "có giá 12.500.000 ₫."}}},
        ]
    }
    return mock


@pytest.fixture
def sample_chunk():
    return RetrievedChunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        category="product",
        content="Bếp nướng gas Weber Spirit II",
        score=0.95,
        metadata={
            "name": "Bếp Weber Spirit II",
            "price": 12500000.0,
            "product_url": "https://store.example.com/p/bbq",
            "stock_status": "in_stock",
        },
    )


def test_bedrock_client_converse(mock_boto_client):
    client = BedrockConverseClient(bedrock_client=mock_boto_client)
    res = client.converse(
        messages=[{"role": "user", "content": [{"text": "Test"}]}],
        system_prompts=[{"text": "System"}],
        config=GenerationConfig(),
    )

    assert "Bếp nướng Weber [1]" in res["text"]
    assert res["token_usage"].total_tokens == 130
    assert mock_boto_client.converse.call_count == 1


def test_bedrock_client_stream(mock_boto_client):
    client = BedrockConverseClient(bedrock_client=mock_boto_client)
    stream = client.converse_stream(
        messages=[{"role": "user", "content": [{"text": "Test"}]}],
        system_prompts=[{"text": "System"}],
        config=GenerationConfig(),
    )

    chunks = list(stream)
    assert len(chunks) == 3
    assert "".join(chunks) == "Dạ chào bạn, Bếp nướng Weber [1] có giá 12.500.000 ₫."


def test_bedrock_client_throttling_retry():
    mock_boto = MagicMock()
    throttle_err = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate limit"}}, "converse"
    )
    success_resp = {
        "output": {"message": {"content": [{"text": "Success after retry"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }
    # Fail once, then succeed
    mock_boto.converse.side_effect = [throttle_err, success_resp]

    client = BedrockConverseClient(bedrock_client=mock_boto, max_retries=2, base_backoff_sec=0.01)
    res = client.converse(
        messages=[{"role": "user", "content": [{"text": "Test"}]}],
        system_prompts=[{"text": "System"}],
        config=GenerationConfig(),
    )

    assert res["text"] == "Success after retry"
    assert mock_boto.converse.call_count == 2


def test_generator_normal_flow(mock_boto_client, sample_chunk):
    client = BedrockConverseClient(bedrock_client=mock_boto_client)
    generator = GroundedResponseGenerator(bedrock_client=client)

    req = GenerationRequest(query="Giá bếp nướng bao nhiêu?", chunks=[sample_chunk])
    resp = generator.generate(req)

    assert resp.refusal_triggered is False
    assert "Bếp nướng Weber [1]" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_type == CitationType.PRODUCT_CTA
    assert resp.citations[0].price_display == "12.500.000 ₫"
    assert resp.degraded is False


def test_generator_empty_context():
    generator = GroundedResponseGenerator()
    req = GenerationRequest(query="Câu hỏi không có dữ liệu", chunks=[])
    resp = generator.generate(req)

    assert resp.refusal_triggered is True
    assert resp.refusal_reason == "empty_retrieved_context"
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_type == CitationType.SUPPORT_REDIRECT


def test_generator_out_of_stock_trigger(sample_chunk):
    # Change chunk to out of stock
    sample_chunk.metadata["stock_status"] = "out_of_stock"

    mock_boto = MagicMock()
    mock_boto.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": "Sản phẩm hiện đang tạm hết hàng [1]. Nhân viên cửa hàng sẽ sớm liên hệ lại với bạn."
                    }
                ]
            }
        },
        "usage": {"inputTokens": 50, "outputTokens": 20, "totalTokens": 70},
    }
    client = BedrockConverseClient(bedrock_client=mock_boto)
    generator = GroundedResponseGenerator(bedrock_client=client)

    req = GenerationRequest(query="Bếp có còn hàng không?", chunks=[sample_chunk])
    resp = generator.generate(req)

    assert resp.refusal_triggered is True
    assert resp.refusal_reason == "product_out_of_stock"
    # Should contain product citation AND support redirect citation
    types = [c.citation_type for c in resp.citations]
    assert CitationType.PRODUCT_CTA in types
    assert CitationType.SUPPORT_REDIRECT in types


def test_generator_graceful_degradation_on_failure(sample_chunk):
    mock_boto = MagicMock()
    mock_boto.converse.side_effect = Exception("AWS Service Unavailable")

    client = BedrockConverseClient(bedrock_client=mock_boto, max_retries=1)
    generator = GroundedResponseGenerator(bedrock_client=client)

    req = GenerationRequest(query="Test", chunks=[sample_chunk])
    resp = generator.generate(req)

    assert resp.degraded is True
    assert "AWS Service Unavailable" in resp.degradation_reason
    assert resp.refusal_triggered is True


def test_generator_streaming(mock_boto_client, sample_chunk):
    client = BedrockConverseClient(bedrock_client=mock_boto_client)
    generator = GroundedResponseGenerator(bedrock_client=client)

    req = GenerationRequest(query="Giá bếp?", chunks=[sample_chunk])
    stream_events = list(generator.generate_stream(req))

    event_types = [e.event for e in stream_events]
    assert "delta" in event_types
    assert "citation" in event_types
    assert "done" in event_types

    # Find done event
    done_event = [e for e in stream_events if e.event == "done"][0]
    assert done_event.data["query"] == "Giá bếp?"


def test_generation_service(mock_boto_client, sample_chunk):
    client = BedrockConverseClient(bedrock_client=mock_boto_client)
    generator = GroundedResponseGenerator(bedrock_client=client)
    service = GenerationService(generator=generator)

    resp = service.generate(query="Test query", chunks=[sample_chunk])
    assert resp.query == "Test query"
    assert len(resp.citations) == 1
