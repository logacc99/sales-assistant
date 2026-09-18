"""FastAPI dependency injection providers for service clients and pipelines."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from src.config import Settings, get_config
from src.ingestion.chunker import BaseChunker, CrawledMarkdownChunker
from src.ingestion.embedder import BaseEmbedder, BedrockEmbedder
from src.ingestion.indexer import BaseVectorIndexer, OpenSearchVectorIndexer
from src.ingestion.pipeline import IngestionPipeline


@lru_cache(maxsize=1)
def get_app_settings() -> Settings:
    """Provides application configuration settings."""
    return get_config()


@lru_cache(maxsize=1)
def get_chunker() -> BaseChunker:
    """Provides markdown chunker instance."""
    return CrawledMarkdownChunker()


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Provides Bedrock embedder instance."""
    return BedrockEmbedder()


@lru_cache(maxsize=1)
def get_indexer() -> BaseVectorIndexer:
    """Provides OpenSearch indexer instance."""
    return OpenSearchVectorIndexer()


def get_pipeline() -> IngestionPipeline:
    """Provides IngestionPipeline instance reusing standard singleton dependencies."""
    return IngestionPipeline(
        chunker=get_chunker(),
        embedder=get_embedder(),
        indexer=get_indexer(),
        auto_create_index=False,
    )
