"""Data contracts and schemas for e-commerce chunks and ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

CategoryType = Literal["product", "promotion", "policy"]
StockStatus = Literal["in_stock", "out_of_stock", "backorder"]
DiscountType = Literal["percentage", "fixed_amount", "free_shipping"]
PolicyType = Literal["shipping", "returns", "warranty", "privacy", "terms"]


@dataclass
class ProductMetadata:
    """Metadata schema for e-commerce product entities."""

    product_id: str
    name: str
    price: float
    currency: str = "VND"
    stock_status: StockStatus = "in_stock"
    product_url: str = ""
    sku: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionMetadata:
    """Metadata schema for promotional campaigns and discount vouchers."""

    promo_code: str
    discount_type: DiscountType
    discount_value: float
    start_date: datetime
    end_date: datetime
    min_spend: float = 0.0
    excluded_skus: list[str] = field(default_factory=list)
    applicable_categories: list[str] = field(default_factory=list)
    terms_url: Optional[str] = None
    stackable: bool = False


@dataclass
class PolicyMetadata:
    """Metadata schema for store policy and terms sections."""

    policy_type: PolicyType
    section_title: str
    effective_date: datetime
    policy_url: str
    parent_section_content: Optional[str] = None  # Parent-child small-to-big context expansion


@dataclass
class EcomChunkMetadata:
    """Top-level metadata container for chunk classification and lineage tracking."""

    category: CategoryType
    source_file: str
    source_type: Literal["crawler", "markdown", "csv", "json"] = "crawler"
    source_url: str = ""
    page_title: str = ""
    crawl_timestamp: Optional[datetime] = None  # Crawl time used for stale document reconciliation
    product: Optional[ProductMetadata] = None
    promotion: Optional[PromotionMetadata] = None
    policy: Optional[PolicyMetadata] = None


@dataclass
class EcomChunk:
    """Core chunk container representing a vector embedding and lexical search candidate."""

    id: str  # Deterministic hash: sha256(f"{source_url}#{chunk_index}")
    doc_id: str
    chunk_index: int
    content: str  # Formatted text used for vector embedding and BM25 lexical search
    content_hash: str  # sha256 of raw chunk content for Change Data Detection (CDC)
    metadata: EcomChunkMetadata
    embedding: Optional[list[float]] = None

    @classmethod
    def create(
        cls,
        source_url: str,
        doc_id: str,
        chunk_index: int,
        content: str,
        metadata: EcomChunkMetadata,
        embedding: Optional[list[float]] = None,
    ) -> EcomChunk:
        """Factory method computing deterministic id and content_hash."""
        chunk_id = hashlib.sha256(f"{source_url}#{chunk_index}".encode()).hexdigest()
        content_hash = hashlib.sha256(content.strip().encode()).hexdigest()
        return cls(
            id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            content=content.strip(),
            content_hash=content_hash,
            metadata=metadata,
            embedding=embedding,
        )

    def to_langchain_document(self) -> Any:
        """Converts EcomChunk to a standard langchain_core.documents.Document."""
        from langchain_core.documents import Document

        metadata_dict: dict[str, Any] = {
            "id": self.id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            "category": self.metadata.category,
            "source_file": self.metadata.source_file,
            "source_type": self.metadata.source_type,
            "source_url": self.metadata.source_url,
            "page_title": self.metadata.page_title,
            "crawl_timestamp": (
                self.metadata.crawl_timestamp.isoformat()
                if self.metadata.crawl_timestamp
                else None
            ),
            "product_id": self.metadata.product.product_id if self.metadata.product else None,
            "product_name": self.metadata.product.name if self.metadata.product else None,
            "price": self.metadata.product.price if self.metadata.product else None,
            "currency": self.metadata.product.currency if self.metadata.product else None,
            "stock_status": self.metadata.product.stock_status if self.metadata.product else None,
            "policy_type": self.metadata.policy.policy_type if self.metadata.policy else None,
            "parent_section_content": (
                self.metadata.policy.parent_section_content if self.metadata.policy else None
            ),
            "promo_code": self.metadata.promotion.promo_code if self.metadata.promotion else None,
        }

        return Document(
            page_content=self.content,
            metadata=metadata_dict,
        )
