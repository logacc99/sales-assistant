"""Hybrid citation extractor and hydrator for verified frontend UI cards."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.generation.models import Citation, CitationType
from src.retrieval.models import RetrievedChunk

SUPPORT_URL = "https://store.example.com/support"
SUPPORT_TITLE = "Trung tâm hỗ trợ khách hàng"


def format_currency_vnd(price: Optional[float]) -> Optional[str]:
    """Formats float price into standard Vietnamese currency display (e.g. 1.250.000 ₫)."""
    if price is None:
        return None
    try:
        formatted = f"{int(price):,}".replace(",", ".")
        return f"{formatted} ₫"
    except (ValueError, TypeError):
        return str(price)


class CitationExtractor:
    """Extracts and deterministically hydrates structured UI citations from chunk metadata."""

    def __init__(self, max_fallback_citations: int = 3) -> None:
        self.max_fallback_citations = max_fallback_citations
        # Matches citations like [1], [2], [1, 2], [1][2], etc.
        self._bracket_pattern = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    def parse_cited_indices(self, text: str, total_chunks: int) -> List[int]:
        """
        Parses 1-based index numbers cited inside brackets in the generated text.
        Returns ordered unique list of valid 0-based chunk indices.
        """
        if not text or total_chunks <= 0:
            return []

        cited_indices: List[int] = []
        seen: Set[int] = set()

        matches = self._bracket_pattern.findall(text)
        for match in matches:
            parts = match.split(",")
            for part in parts:
                clean_part = part.strip()
                if clean_part.isdigit():
                    num = int(clean_part)
                    # Convert from 1-based to 0-based index
                    idx = num - 1
                    if 0 <= idx < total_chunks and idx not in seen:
                        seen.add(idx)
                        cited_indices.append(idx)

        return cited_indices

    def hydrate_chunk(self, chunk: RetrievedChunk) -> Optional[Citation]:
        """Maps an individual chunk's metadata into a typed Citation object."""
        category = chunk.category
        meta = chunk.metadata or {}

        if category == "product":
            name = meta.get("name") or meta.get("product_name") or "Sản phẩm"
            url = meta.get("product_url") or meta.get("source_url") or ""
            price = meta.get("price")
            price_display = format_currency_vnd(price)
            stock = meta.get("stock_status") or "in_stock"
            return Citation(
                citation_type=CitationType.PRODUCT_CTA,
                title=name,
                url=url,
                price_display=price_display,
                stock_status=stock,
                metadata={
                    "sku": meta.get("sku"),
                    "product_id": meta.get("product_id"),
                    "chunk_id": chunk.chunk_id,
                },
            )

        elif category == "policy":
            title = meta.get("section_title") or "Chính sách cửa hàng"
            url = meta.get("policy_url") or meta.get("source_url") or SUPPORT_URL
            return Citation(
                citation_type=CitationType.POLICY_LINK,
                title=title,
                url=url,
                metadata={
                    "policy_type": meta.get("policy_type"),
                    "chunk_id": chunk.chunk_id,
                },
            )

        elif category == "promotion":
            promo_code = meta.get("promo_code") or "PROMO"
            discount_type = meta.get("discount_type")
            discount_val = meta.get("discount_value")
            discount_display = None
            if discount_val is not None:
                if discount_type == "percentage":
                    discount_display = f"Giảm {discount_val}%"
                elif discount_type == "fixed_amount":
                    discount_display = f"Giảm {format_currency_vnd(discount_val)}"
                elif discount_type == "free_shipping":
                    discount_display = "Miễn phí vận chuyển"
                else:
                    discount_display = str(discount_val)

            url = meta.get("terms_url") or meta.get("source_url") or "/promotions"
            return Citation(
                citation_type=CitationType.PROMO_TERMS,
                title=f"Mã: {promo_code}",
                url=url,
                promo_code=promo_code,
                discount_display=discount_display,
                metadata={
                    "min_spend": meta.get("min_spend"),
                    "chunk_id": chunk.chunk_id,
                },
            )

        return None

    def create_support_citation(self, title: str = SUPPORT_TITLE, url: str = SUPPORT_URL) -> Citation:
        """Creates a customer support redirect CTA card."""
        return Citation(
            citation_type=CitationType.SUPPORT_REDIRECT,
            title=title,
            url=url,
            metadata={"reason": "customer_support_redirect"},
        )

    def extract_citations(
        self,
        answer_text: str,
        retained_chunks: List[RetrievedChunk],
        refusal_triggered: bool = False,
        has_out_of_stock: bool = False,
    ) -> List[Citation]:
        """
        Executes hybrid citation extraction:
        1. Hydrates cited indices from bracket notation ([1], [2]).
        2. Falls back to top-K chunks if no brackets are present.
        3. Appends support redirect if refusal or out-of-stock is detected.
        """
        citations: List[Citation] = []
        seen_keys: Set[str] = set()

        # Step 1: Detect cited indices
        cited_indices = self.parse_cited_indices(answer_text, len(retained_chunks))

        # Step 2: Fallback if no citations found
        if not cited_indices and retained_chunks:
            cited_indices = list(range(min(len(retained_chunks), self.max_fallback_citations)))

        for idx in cited_indices:
            chunk = retained_chunks[idx]
            cit = self.hydrate_chunk(chunk)
            if cit:
                dedup_key = f"{cit.citation_type.value}:{cit.title}:{cit.url}"
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    citations.append(cit)

        # Step 3: Append support CTA if out of stock or refusal triggered
        if refusal_triggered or has_out_of_stock:
            support_cit = self.create_support_citation()
            dedup_key = f"{support_cit.citation_type.value}:{support_cit.title}:{support_cit.url}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                citations.append(support_cit)

        return citations
