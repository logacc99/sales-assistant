"""Orchestration pipeline for crawling ingestion, CDC diffing, embedding, and indexing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.config import get_config
from src.ingestion.chunker import BaseChunker, CrawledMarkdownChunker
from src.ingestion.embedder import BaseEmbedder, BedrockEmbedder
from src.ingestion.indexer import BaseVectorIndexer, OpenSearchVectorIndexer
from src.ingestion.models import EcomChunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document reading, chunking, Change Data Detection (CDC), embedding, and indexing."""

    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        embedder: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseVectorIndexer] = None,
        index_name: Optional[str] = None,
        enable_cdc: bool = True,
        enable_reconciliation: bool = False,
        auto_create_index: bool = True,
    ) -> None:
        cfg = get_config()
        self.chunker = chunker or CrawledMarkdownChunker()
        self.embedder = embedder or BedrockEmbedder()
        self.indexer = indexer or OpenSearchVectorIndexer()
        self.index_name = index_name or cfg.opensearch_index_name
        self.enable_cdc = enable_cdc
        self.enable_reconciliation = enable_reconciliation

        if auto_create_index and hasattr(self.indexer, "ensure_index_exists"):
            try:
                self.indexer.ensure_index_exists(self.index_name)
            except Exception as e:
                logger.warning(f"Could not ensure index '{self.index_name}' exists: {e}")

    def ingest_file(self, file_path: Path) -> dict[str, Any]:
        """Processes a single crawled markdown file: chunks, checks CDC, embeds new/modified, and indexes."""
        chunks: list[EcomChunk] = self.chunker.chunk(file_path)
        if not chunks:
            return {
                "file_path": str(file_path),
                "total_chunks": 0,
                "skipped_cdc": 0,
                "embedded_chunks": 0,
                "indexed_chunks": 0,
                "failed_chunks": 0,
                "active_doc_ids": [],
            }

        chunks_to_embed: list[EcomChunk] = chunks
        skipped_cdc = 0

        if self.enable_cdc:
            chunk_ids = [c.id for c in chunks]
            existing_hashes = self.indexer.get_existing_hashes(self.index_name, chunk_ids)
            chunks_to_embed = [
                c for c in chunks if existing_hashes.get(c.id) != c.content_hash
            ]
            skipped_cdc = len(chunks) - len(chunks_to_embed)

        indexed_count = 0
        failed_count = 0

        if chunks_to_embed:
            embedded_chunks = self.embedder.embed_chunks(chunks_to_embed)
            indexed_count, failed_items = self.indexer.index_chunks(
                self.index_name, embedded_chunks
            )
            failed_count = len(failed_items)

        return {
            "file_path": str(file_path),
            "total_chunks": len(chunks),
            "skipped_cdc": skipped_cdc,
            "embedded_chunks": len(chunks_to_embed),
            "indexed_chunks": indexed_count,
            "failed_chunks": failed_count,
            "active_doc_ids": [c.id for c in chunks],
        }

    def ingest_directory(self, dir_path: Path, pattern: str = "*.md") -> dict[str, Any]:
        """Iterates over markdown files in directory, performs batch CDC filtering, indexes, and reconciles orphans."""
        all_files = sorted(dir_path.glob(pattern))
        summary: dict[str, Any] = {
            "total_files": len(all_files),
            "total_chunks": 0,
            "skipped_cdc": 0,
            "embedded_chunks": 0,
            "indexed_chunks": 0,
            "failed_chunks": 0,
            "purged_orphans": 0,
            "file_summaries": [],
        }

        all_active_ids: list[str] = []

        for f in all_files:
            file_stat = self.ingest_file(f)
            summary["file_summaries"].append(file_stat)
            summary["total_chunks"] += file_stat["total_chunks"]
            summary["skipped_cdc"] += file_stat["skipped_cdc"]
            summary["embedded_chunks"] += file_stat["embedded_chunks"]
            summary["indexed_chunks"] += file_stat["indexed_chunks"]
            summary["failed_chunks"] += file_stat["failed_chunks"]
            all_active_ids.extend(file_stat["active_doc_ids"])

        if self.enable_reconciliation and all_active_ids:
            purged = self.reconcile_orphans(all_active_ids)
            summary["purged_orphans"] = purged

        return summary

    def reconcile_orphans(
        self, active_doc_ids: list[str], source_prefix: Optional[str] = None
    ) -> int:
        """Purges stale documents no longer present in active crawl batch."""
        return self.indexer.purge_stale_documents(
            self.index_name, active_doc_ids, source_prefix=source_prefix
        )
