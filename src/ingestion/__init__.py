"""Ingestion subsystem for multi-source document ingestion, web crawling, chunking, embedding, and indexing."""

from src.ingestion.chunker import BaseChunker, CrawledMarkdownChunker
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.crawler import crawl_website
from src.ingestion.embedder import BaseEmbedder, BedrockEmbedder
from src.ingestion.indexer import BaseVectorIndexer, OpenSearchVectorIndexer
from src.ingestion.models import (
    CategoryType,
    DiscountType,
    EcomChunk,
    EcomChunkMetadata,
    PolicyMetadata,
    PolicyType,
    ProductMetadata,
    PromotionMetadata,
    StockStatus,
)
from src.ingestion.pipeline import IngestionPipeline

__all__ = [
    "crawl_website",
    "CategoryType",
    "StockStatus",
    "DiscountType",
    "PolicyType",
    "ProductMetadata",
    "PromotionMetadata",
    "PolicyMetadata",
    "EcomChunkMetadata",
    "EcomChunk",
    "DocumentCleaner",
    "BaseChunker",
    "CrawledMarkdownChunker",
    "BaseEmbedder",
    "BedrockEmbedder",
    "BaseVectorIndexer",
    "OpenSearchVectorIndexer",
    "IngestionPipeline",
]
