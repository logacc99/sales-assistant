"""Unit tests for centralized configuration and environment variable loading."""

import os
from unittest.mock import patch

import pytest

from src.config import IngestionConfig, get_config


def test_default_config():
    """Verify default values when environment variables are not set."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = IngestionConfig.from_env()
        assert cfg.bedrock_model_id == "cohere.embed-multilingual-v3.0"
        assert cfg.bedrock_dimension == 1024
        assert cfg.bedrock_max_workers == 5
        assert cfg.aws_region == "us-east-1"
        assert cfg.opensearch_host == "localhost"
        assert cfg.opensearch_port == 9200
        assert cfg.opensearch_index_name == "sales-assistant-catalog"
        assert cfg.opensearch_use_aws_auth is True


def test_env_overrides():
    """Verify environment variable overrides."""
    env_vars = {
        "BEDROCK_EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
        "BEDROCK_EMBEDDING_DIMENSION": "512",
        "BEDROCK_MAX_WORKERS": "8",
        "AWS_REGION": "ap-southeast-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "OPENSEARCH_HOST": "opensearch.internal",
        "OPENSEARCH_PORT": "9201",
        "OPENSEARCH_INDEX_NAME": "custom-index",
        "OPENSEARCH_USE_AWS_AUTH": "false",
        "OPENSEARCH_USERNAME": "custom_user",
        "OPENSEARCH_PASSWORD": "custom_password",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        cfg = get_config()
        assert cfg.bedrock_model_id == "amazon.titan-embed-text-v2:0"
        assert cfg.bedrock_dimension == 512
        assert cfg.bedrock_max_workers == 8
        assert cfg.aws_region == "ap-southeast-1"
        assert cfg.aws_access_key_id == "test-key"
        assert cfg.aws_secret_access_key == "test-secret"
        assert cfg.opensearch_host == "opensearch.internal"
        assert cfg.opensearch_port == 9201
        assert cfg.opensearch_index_name == "custom-index"
        assert cfg.opensearch_use_aws_auth is False
        assert cfg.opensearch_username == "custom_user"
        assert cfg.opensearch_password == "custom_password"


def test_serverless_config_defaults():
    """Verify serverless defaults when an .aoss.amazonaws.com host is specified."""
    env_vars = {
        "OPENSEARCH_HOST": "https://xyz123.us-east-1.aoss.amazonaws.com",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        cfg = get_config()
        assert cfg.opensearch_is_serverless is True
        assert cfg.opensearch_port == 443
        assert cfg.opensearch_service_name == "aoss"
        assert cfg.opensearch_collection_type == "VECTORSEARCH"


def test_generation_and_rerank_config():
    """Verify generation provider, reranker region, and mantle configurations."""
    # Test defaults
    with patch.dict(os.environ, {}, clear=True):
        cfg = IngestionConfig.from_env()
        assert cfg.llm_method == "runtime"
        assert cfg.rerank_enabled is True
        assert cfg.rerank_method == "local"
        assert cfg.local_rerank_model_id == "BAAI/bge-reranker-m3"
        assert cfg.local_rerank_device == ""
        assert cfg.local_rerank_batch_size == 16
        assert cfg.bedrock_rerank_model_id == "cohere.rerank-v3-5:0"
        assert cfg.bedrock_rerank_region == "ap-northeast-1"
        assert cfg.bedrock_mantle_project_id == "default"
        assert cfg.bedrock_mantle_model_id == "openai.gpt-oss-120b"

    # Test overrides
    env_vars = {
        "LLM_METHOD": "mantle",
        "RERANK_ENABLED": "false",
        "RERANK_METHOD": "bedrock",
        "LOCAL_RERANK_MODEL_ID": "custom-bge-model",
        "LOCAL_RERANK_DEVICE": "cuda",
        "LOCAL_RERANK_BATCH_SIZE": "32",
        "BEDROCK_RERANK_REGION": "ap-northeast-1",
        "BEDROCK_MANTLE_API_KEY": "test-key-123",
        "BEDROCK_MANTLE_BASE_URL": "https://bedrock-mantle.ap-southeast-2.api.aws/v1",
        "BEDROCK_MANTLE_MODEL_ID": "custom-model",
        "BEDROCK_MANTLE_PROJECT_ID": "proj-abc",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        cfg = get_config()
        assert cfg.llm_method == "mantle"
        assert cfg.rerank_enabled is False
        assert cfg.rerank_method == "bedrock"
        assert cfg.local_rerank_model_id == "custom-bge-model"
        assert cfg.local_rerank_device == "cuda"
        assert cfg.local_rerank_batch_size == 32
        assert cfg.bedrock_rerank_region == "ap-northeast-1"
        assert cfg.bedrock_mantle_api_key == "test-key-123"
        assert cfg.bedrock_mantle_base_url == "https://bedrock-mantle.ap-southeast-2.api.aws/v1"
        assert cfg.bedrock_mantle_model_id == "custom-model"
        assert cfg.bedrock_mantle_project_id == "proj-abc"


