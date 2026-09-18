"""AWS Bedrock vector embedding generator supporting Cohere and Titan models."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import boto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import get_config
from src.ingestion.models import EcomChunk

logger = logging.getLogger(__name__)


def _is_throttling_error(exception: BaseException) -> bool:
    """Checks if an exception is an AWS Bedrock throttling or rate limit error."""
    if isinstance(exception, ClientError):
        code = exception.response.get("Error", {}).get("Code", "")
        return code in (
            "ThrottlingException",
            "RequestLimitExceeded",
            "TooManyRequestsException",
            "ProvisionedThroughputExceededException",
        )
    return False


class BaseEmbedder(ABC):
    """Generates vector embeddings using AWS Bedrock."""

    model_id: str

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generates a normalized vector for a single text."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: list[EcomChunk]) -> list[EcomChunk]:
        """Embeds a list of chunks using ThreadPoolExecutor with exponential backoff."""
        pass

    @abstractmethod
    def get_langchain_embeddings(self) -> Any:
        """Returns the underlying langchain_aws.BedrockEmbeddings instance."""
        pass


class BedrockEmbedder(BaseEmbedder):
    """Bedrock dense vector embedder supporting Cohere Multilingual v3 and Titan Text v2."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        dimension: Optional[int] = None,
        max_workers: Optional[int] = None,
        client: Optional[Any] = None,
        region_name: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.model_id = model_id or cfg.bedrock_model_id
        self.dimension = dimension or cfg.bedrock_dimension
        self.max_workers = max_workers or cfg.bedrock_max_workers
        self.region_name = region_name or cfg.aws_region

        if client is not None:
            self.client = client
        else:
            session_kwargs: dict[str, Any] = {"region_name": self.region_name}
            if cfg.aws_access_key_id and cfg.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = cfg.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = cfg.aws_secret_access_key
                if cfg.aws_session_token:
                    session_kwargs["aws_session_token"] = cfg.aws_session_token

            session = boto3.Session(**session_kwargs)
            self.client = session.client("bedrock-runtime", region_name=self.region_name)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(_is_throttling_error),
        reraise=True,
    )
    def embed_text(self, text: str, input_type: str = "search_document") -> list[float]:
        """Generates embedding vector for a single text string with exponential backoff."""
        clean_text = text.strip()
        if not clean_text:
            clean_text = "empty"

        if "cohere" in self.model_id.lower():
            payload = {
                "texts": [clean_text],
                "input_type": input_type,
                "truncate": "END",
            }
            body_bytes = json.dumps(payload).encode("utf-8")
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body_bytes,
                accept="application/json",
                contentType="application/json",
            )
            resp_body = json.loads(response["body"].read())
            embeddings = resp_body.get("embeddings", [])
            if not embeddings:
                raise ValueError(f"Empty embeddings returned from Bedrock Cohere model: {resp_body}")
            vector = embeddings[0]
        else:
            # Titan v2 or similar
            payload = {
                "inputText": clean_text,
                "dimensions": self.dimension,
                "normalize": True,
            }
            body_bytes = json.dumps(payload).encode("utf-8")
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body_bytes,
                accept="application/json",
                contentType="application/json",
            )
            resp_body = json.loads(response["body"].read())
            vector = resp_body.get("embedding", [])
            if not vector:
                raise ValueError(f"Empty embedding returned from Bedrock Titan model: {resp_body}")

        if len(vector) != self.dimension:
            logger.warning(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        return vector

    def embed_chunks(self, chunks: list[EcomChunk]) -> list[EcomChunk]:
        """Embeds a list of EcomChunks concurrently using ThreadPoolExecutor."""
        if not chunks:
            return []

        def _embed_single(chunk: EcomChunk) -> EcomChunk:
            chunk.embedding = self.embed_text(chunk.content, input_type="search_document")
            return chunk

        embedded_chunks: list[EcomChunk] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(_embed_single, c): c for c in chunks}
            for future in as_completed(future_to_chunk):
                chunk = future.result()
                embedded_chunks.append(chunk)

        # Restore original order by chunk_index
        embedded_chunks.sort(key=lambda x: x.chunk_index)
        return embedded_chunks

    def get_langchain_embeddings(self) -> Any:
        """Returns langchain_aws BedrockEmbeddings wrapper."""
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(
            client=self.client,
            model_id=self.model_id,
            region_name=self.region_name,
        )
