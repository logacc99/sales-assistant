"""Centralized configuration and environment variable loader for Sales Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load .env file from project root if it exists
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass


def _bool_from_env(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y", "t")


def _int_from_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


@dataclass
class IngestionConfig:
    """Dynamic configuration for ingestion, Bedrock embedding, and OpenSearch indexing."""

    # Bedrock embedding configuration
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv(
            "BEDROCK_EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3.0"
        )
    )
    bedrock_dimension: int = field(
        default_factory=lambda: _int_from_env("BEDROCK_EMBEDDING_DIMENSION", 1024)
    )
    bedrock_max_workers: int = field(
        default_factory=lambda: _int_from_env("BEDROCK_MAX_WORKERS", 5)
    )

    # AWS Credentials and Region
    aws_region: str = field(
        default_factory=lambda: os.getenv(
            "AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
    )
    aws_access_key_id: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID")
    )
    aws_secret_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    aws_session_token: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_SESSION_TOKEN")
    )

    # OpenSearch Connection and Authentication
    opensearch_host: str = field(
        default_factory=lambda: os.getenv("OPENSEARCH_HOST", "localhost")
    )
    opensearch_port: int = field(
        default_factory=lambda: _int_from_env("OPENSEARCH_PORT", 9200)
    )
    opensearch_index_name: str = field(
        default_factory=lambda: os.getenv("OPENSEARCH_INDEX_NAME", "sales-assistant-catalog")
    )
    opensearch_use_aws_auth: bool = field(
        default_factory=lambda: _bool_from_env("OPENSEARCH_USE_AWS_AUTH", True)
    )
    opensearch_username: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENSEARCH_USERNAME", "admin")
    )
    opensearch_password: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENSEARCH_PASSWORD", "admin")
    )

    @classmethod
    def from_env(cls) -> IngestionConfig:
        """Constructs IngestionConfig instance reading from current environment."""
        return cls()


def get_config() -> IngestionConfig:
    """Returns a fresh IngestionConfig instance loaded from the environment."""
    return IngestionConfig.from_env()


# Alias for standard application settings
Settings = IngestionConfig
