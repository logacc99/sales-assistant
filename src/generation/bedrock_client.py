"""AWS Bedrock Converse API client wrapper with SigV4 auth, retry, and streaming."""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, EndpointConnectionError

from src.config import IngestionConfig, get_config
from src.generation.models import GenerationConfig, TokenUsage

logger = logging.getLogger(__name__)


class BaseBedrockClient(ABC):
    """Abstract interface for Bedrock conversational synthesis."""

    @abstractmethod
    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """Executes synchronous Converse API call."""
        pass

    @abstractmethod
    def converse_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        """Executes streaming Converse API call, yielding text chunks."""
        pass


class BedrockConverseClient(BaseBedrockClient):
    """AWS Bedrock client wrapping the Converse API with exponential backoff and telemetry."""

    def __init__(
        self,
        app_config: Optional[IngestionConfig] = None,
        bedrock_client: Optional[Any] = None,
        max_retries: int = 3,
        base_backoff_sec: float = 0.5,
    ) -> None:
        self.app_config = app_config or get_config()
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec

        if bedrock_client is not None:
            self._client = bedrock_client
        else:
            self._client = self._init_boto_client()

    def _init_boto_client(self) -> Any:
        """Initializes boto3 bedrock-runtime client with SigV4 credentials and timeout."""
        boto_config = BotoConfig(
            region_name=self.app_config.aws_region,
            read_timeout=self.app_config.bedrock_generation_timeout_seconds,
            connect_timeout=10,
            retries={"max_attempts": 1, "mode": "standard"},
        )
        kwargs: Dict[str, Any] = {"config": boto_config}
        if self.app_config.aws_access_key_id and self.app_config.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self.app_config.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.app_config.aws_secret_access_key
            if self.app_config.aws_session_token:
                kwargs["aws_session_token"] = self.app_config.aws_session_token

        return boto3.client("bedrock-runtime", **kwargs)

    def _build_inference_config(self, config: GenerationConfig) -> Dict[str, Any]:
        """Builds inference configuration dictionary for Bedrock Converse."""
        inf_config: Dict[str, Any] = {
            "temperature": config.temperature,
            "topP": config.top_p,
            "maxTokens": config.max_tokens,
        }
        if config.stop_sequences:
            inf_config["stopSequences"] = config.stop_sequences
        return inf_config

    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Executes synchronous Converse API request with exponential backoff on throttling.

        Returns:
            Dict containing:
                - "text": Synthesized markdown string
                - "token_usage": TokenUsage
                - "stop_reason": str
        """
        inf_config = self._build_inference_config(config)

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.converse(
                    modelId=config.model_id,
                    messages=messages,
                    system=system_prompts,
                    inferenceConfig=inf_config,
                )

                output_msg = response.get("output", {}).get("message", {})
                content_blocks = output_msg.get("content", [])
                text_chunks = [
                    block.get("text", "") for block in content_blocks if "text" in block
                ]
                full_text = "".join(text_chunks)

                usage_data = response.get("usage", {})
                usage = TokenUsage(
                    input_tokens=usage_data.get("inputTokens", 0),
                    output_tokens=usage_data.get("outputTokens", 0),
                    total_tokens=usage_data.get("totalTokens", 0),
                )

                return {
                    "text": full_text,
                    "token_usage": usage,
                    "stop_reason": response.get("stopReason", "end_turn"),
                }

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                last_err = e
                if code in ("ThrottlingException", "RequestLimitExceeded", "TooManyRequestsException"):
                    sleep_time = self.base_backoff_sec * (2**attempt) + random.uniform(0.1, 0.3)
                    logger.warning(
                        "Bedrock Converse throttled (%s). Attempt %d/%d, retrying in %.2fs",
                        code,
                        attempt + 1,
                        self.max_retries,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error("Bedrock Converse client error [%s]: %s", code, e)
                    raise
            except (EndpointConnectionError, Exception) as e:
                last_err = e
                logger.error("Bedrock Converse connection error: %s", e)
                raise

        raise last_err or RuntimeError("Bedrock Converse failed after maximum retries")

    def converse_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]],
        config: GenerationConfig,
    ) -> Iterator[str]:
        """
        Executes streaming Converse API request, yielding incremental text tokens.
        """
        inf_config = self._build_inference_config(config)

        try:
            response = self._client.converse_stream(
                modelId=config.model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inf_config,
            )

            stream = response.get("stream")
            if not stream:
                return

            for event in stream:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    text_delta = delta.get("text", "")
                    if text_delta:
                        yield text_delta

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            logger.error("Bedrock converse_stream ClientError [%s]: %s", code, e)
            raise
        except Exception as e:
            logger.error("Bedrock converse_stream unexpected exception: %s", e)
            raise
