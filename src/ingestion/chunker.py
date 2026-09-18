"""Markdown chunker for product listings, detail pages, and policy documents."""

from __future__ import annotations

import os
import re
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.models import (
    EcomChunk,
    EcomChunkMetadata,
    PolicyMetadata,
    ProductMetadata,
    PromotionMetadata,
)


class BaseChunker(ABC):
    """Splits cleaned markdown documents into semantically coherent chunk candidates."""

    @abstractmethod
    def chunk(self, file_path: Path) -> list[EcomChunk]:
        """Reads crawled markdown file, extracts YAML frontmatter, cleans boilerplate, returns chunks."""
        pass

    @abstractmethod
    def chunk_markdown(
        self, raw_markdown: str, source_file: str = "", source_url: str = ""
    ) -> list[EcomChunk]:
        """Directly parses raw markdown string into chunks without requiring disk I/O."""
        pass


class CrawledMarkdownChunker(BaseChunker):
    """Parses crawled markdown into typed EcomChunks (products, policies, promotions)."""

    def __init__(self, cleaner: Optional[DocumentCleaner] = None) -> None:
        self.cleaner = cleaner or DocumentCleaner()

    def chunk(self, file_path: Path) -> list[EcomChunk]:
        """Reads crawled markdown file, extracts frontmatter, cleans boilerplate, and chunks content."""
        if not file_path.exists():
            return []

        raw_text = file_path.read_text(encoding="utf-8")
        return self.chunk_markdown(raw_text, source_file=file_path.name)

    def chunk_markdown(
        self, raw_markdown: str, source_file: str = "", source_url: str = ""
    ) -> list[EcomChunk]:
        """Parses raw markdown string into typed EcomChunks."""
        frontmatter, raw_body = self.cleaner.extract_frontmatter(raw_markdown)

        url = source_url or frontmatter.get("url", "")
        page_title = frontmatter.get("title", "")
        crawl_timestamp = frontmatter.get("crawl_timestamp")
        filename = source_file or frontmatter.get("source_file", "raw_content.md")
        doc_id = frontmatter.get("content_hash", Path(filename).stem if filename else "raw_content")

        # Check for non-content pages
        if self.cleaner.is_non_content_page(url, page_title, raw_body):
            return []

        cleaned_body = self.cleaner.strip_boilerplate(raw_body)
        if not cleaned_body:
            return []

        # 1. Check if policy or informational page
        policy_type = self._detect_policy_type(url, page_title, cleaned_body)
        if policy_type:
            return self._chunk_policy_page(
                cleaned_body=cleaned_body,
                policy_type=policy_type,
                source_url=url,
                source_file=filename,
                page_title=page_title,
                doc_id=doc_id,
                crawl_timestamp=crawl_timestamp,
            )

        # 2. Check if dedicated single product detail page
        if self._is_product_detail_page(url, cleaned_body):
            detail_chunk = self._parse_product_detail(
                cleaned_body=cleaned_body,
                source_url=url,
                source_file=filename,
                page_title=page_title,
                doc_id=doc_id,
                crawl_timestamp=crawl_timestamp,
            )
            if detail_chunk:
                return [detail_chunk]

        # 3. Extract catalog product cards
        product_chunks = self._extract_catalog_products(
            cleaned_body=cleaned_body,
            source_url=url,
            source_file=filename,
            page_title=page_title,
            doc_id=doc_id,
            crawl_timestamp=crawl_timestamp,
        )

        if product_chunks:
            return product_chunks

        # 4. Fallback unstructured section chunking
        return self._chunk_generic_sections(
            cleaned_body=cleaned_body,
            source_url=url,
            source_file=filename,
            page_title=page_title,
            doc_id=doc_id,
            crawl_timestamp=crawl_timestamp,
        )

    def _detect_policy_type(self, url: str, title: str, body: str) -> Optional[str]:
        """Infers policy type from URL, title, or body keywords."""
        combined = f"{url} {title}".lower()
        if "doi-tra" in combined or "đổi trả" in combined:
            return "returns"
        if "bao-hanh" in combined or "bảo hành" in combined:
            return "warranty"
        if "van-chuyen" in combined or "vận chuyển" in combined or "giao-hang" in combined:
            return "shipping"
        if "bao-mat" in combined or "bảo mật" in combined:
            return "privacy"
        if "dieu-khoan" in combined or "thanh-toan" in combined or "quy-dinh" in combined:
            return "terms"
        return None

    def _is_product_detail_page(self, url: str, cleaned_body: str) -> bool:
        """Determines if the page is a dedicated single product detail page."""
        parsed = urllib.parse.urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) == 2 and path_parts[0] in ("san-pham", "product", "products", "item", "items") and not path_parts[1].startswith("page"):
            return True
        if "# " in cleaned_body and any(k in cleaned_body for k in ("Mã:", "SKU:", "Thêm vào giỏ hàng", "Add to cart", "Add to Cart")):
            return True
        return False

    def _extract_catalog_products(
        self,
        cleaned_body: str,
        source_url: str,
        source_file: str,
        page_title: str,
        doc_id: str,
        crawl_timestamp: Any,
    ) -> list[EcomChunk]:
        """Extracts discrete product listing cards from markdown."""
        chunks: list[EcomChunk] = []

        link_regex = re.compile(
            r"^\[\s*([^!\]\n]+?)\s*\]\((https?://[^)\s]+/(?:san-pham|product|products|item|items)/([^/\s)]+)/?)\)$"
        )
        price_regex = re.compile(r"([0-9]{1,3}(?:[.,][0-9]{3})*\s*(?:₫|VND|VNĐ|\$|USD)?)")

        lines = cleaned_body.splitlines()
        seen_urls: set[str] = set()
        chunk_index = 0

        for idx, line in enumerate(lines):
            line_str = line.strip()
            m = link_regex.match(line_str)
            if not m:
                continue

            name = m.group(1).strip()
            prod_url = m.group(2).strip()
            slug = m.group(3).strip()

            # Ignore image links, empty names, and button symbols like [**+**]
            if name.startswith("!") or name.startswith("[") or not re.search(r"[\w\d]{3,}", name):
                continue

            if prod_url in seen_urls:
                continue

            # Look in the next 5 lines for price
            price = 0.0
            found_price = False
            for lookahead in range(idx + 1, min(len(lines), idx + 6)):
                price_m = price_regex.search(lines[lookahead])
                if price_m:
                    price_digits = re.sub(r"[^\d]", "", price_m.group(1))
                    if price_digits:
                        price = float(price_digits)
                        found_price = True
                        break

            # In catalog listings, valid product cards always feature a price
            if not found_price:
                continue

            seen_urls.add(prod_url)

            # Check lines preceding the product name (up to 4 lines back) for stock and add-to-cart
            card_prefix_lines = lines[max(0, idx - 4) : idx]
            card_prefix_text = " ".join(card_prefix_lines)

            stock_status = "in_stock"
            if "hết hàng" in card_prefix_text.lower() or "out of stock" in card_prefix_text.lower():
                stock_status = "out_of_stock"

            add_to_cart_match = re.search(r"add-to-cart=(\d+)", card_prefix_text)
            product_id = add_to_cart_match.group(1) if add_to_cart_match else slug

            breadcrumb = page_title.split("-")[0].strip() if "-" in page_title else page_title
            semantic_content = (
                f"Tên sản phẩm: {name} | "
                f"Danh mục: {breadcrumb} | "
                f"Giá: {price:,.0f} VND | "
                f"Tình trạng: {stock_status} | "
                f"Link: {prod_url}"
            )

            product_meta = ProductMetadata(
                product_id=product_id,
                name=name,
                price=price,
                currency="VND",
                stock_status=stock_status,
                product_url=prod_url,
                sku=slug,
                categories=[breadcrumb] if breadcrumb else [],
            )

            chunk_meta = EcomChunkMetadata(
                category="product",
                source_file=source_file,
                source_type="crawler",
                source_url=prod_url,
                page_title=page_title,
                crawl_timestamp=crawl_timestamp,
                product=product_meta,
            )

            chunk = EcomChunk.create(
                source_url=prod_url,
                doc_id=doc_id,
                chunk_index=chunk_index,
                content=semantic_content,
                metadata=chunk_meta,
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def _parse_product_detail(
        self,
        cleaned_body: str,
        source_url: str,
        source_file: str,
        page_title: str,
        doc_id: str,
        crawl_timestamp: Any,
    ) -> Optional[EcomChunk]:
        """Parses a dedicated product detail page into a single high-fidelity chunk."""
        # 1. Product Title: extract from H1 (# Title) or fallback to page_title
        h1_match = re.search(r"^#\s+([^\n]+)", cleaned_body, re.MULTILINE)
        if h1_match:
            name = h1_match.group(1).strip()
        else:
            name = page_title.split("-")[0].strip() if "-" in page_title else page_title

        name = name.strip()

        # 2. Extract price (must have thousand separators or currency symbol to avoid matching model numbers like 28")
        price_pattern = re.compile(
            r"(?:[0-9]{1,3}(?:[.,][0-9]{3})+\s*(?:₫|VND|VNĐ|\$|USD)?|[0-9]{1,3}(?:[.,][0-9]{3})*\s*(?:₫|VND|VNĐ|\$|USD))"
        )
        price_match = price_pattern.search(cleaned_body)
        price = 0.0
        if price_match:
            price_digits = re.sub(r"[^\d]", "", price_match.group(0))
            if price_digits:
                price = float(price_digits)

        # 3. Stock status
        stock_status = "in_stock"
        if "hết hàng" in cleaned_body.lower() or "out of stock" in cleaned_body.lower():
            stock_status = "out_of_stock"

        # 4. Slug & SKU
        slug_match = re.search(r"/(?:san-pham|product|products|item|items)/([^/]+)/?", source_url)
        slug = slug_match.group(1) if slug_match else "unknown"

        sku_match = re.search(r"(?:Mã|SKU):\s*([^\s\n,]+)", cleaned_body, re.IGNORECASE)
        product_id = sku_match.group(1).strip() if sku_match else slug
        sku = product_id

        # 5. Categories & Tags
        categories: list[str] = []
        cat_match = re.search(r"(?:Danh mục|Categories|Category):\s*([^\n]+)", cleaned_body, re.IGNORECASE)
        if cat_match:
            categories = re.findall(r"\[([^\]]+)\]", cat_match.group(1))

        tags: list[str] = []
        tag_match = re.search(r"(?:Thẻ|Tags|Tag):\s*([^\n]+)", cleaned_body, re.IGNORECASE)
        if tag_match:
            tags = re.findall(r"\[([^\]]+)\]", tag_match.group(1))

        # 6. Attributes & Specs
        attributes: dict[str, Any] = {
            "sku": sku,
            "tags": tags,
        }

        # Clean body to extract meaningful description paragraphs
        lines = cleaned_body.splitlines()
        desc_lines: list[str] = []
        for line in lines:
            line_s = line.strip()
            # Skip image links, share URLs, buttons, and short navigational snippets
            if (
                not line_s
                or line_s.startswith("![")
                or line_s.startswith("[ ![")
                or line_s.startswith("![]")
                or line_s.startswith("[![]")
                or line_s.startswith("[](whatsapp:")
                or line_s.startswith("[](https://")
                or line_s.startswith("[ ](")
                or line_s.startswith("* [ ](")
                or line_s.startswith("* [](")
                or line_s.startswith("# ")
                or re.match(r"^(?:Mã|SKU):", line_s, re.IGNORECASE)
                or re.match(r"^(?:Danh mục|Categories|Category):", line_s, re.IGNORECASE)
                or re.match(r"^(?:Thẻ|Tags|Tag):", line_s, re.IGNORECASE)
                or re.match(r"^\[(?:Trang chủ|Home)\]", line_s, re.IGNORECASE)
                or "Thêm vào giỏ hàng" in line_s
                or "Add to cart" in line_s
                or "số lượng" in line_s.lower()
                or "quantity" in line_s.lower()
                or re.match(r"^\*\s*\[\s*(?:Mô tả|Description|Đánh giá|Reviews)\b", line_s, re.IGNORECASE)
                or re.match(r"^#{1,4}\s*(?:Đánh giá|Reviews)\b", line_s, re.IGNORECASE)
                or "Chưa có đánh giá nào" in line_s
                or "No reviews yet" in line_s
                or "Chỉ những khách hàng đã đăng nhập" in line_s
                or "Only logged in customers" in line_s
            ):
                continue
            if re.match(r"^[0-9]{1,3}(?:[.,][0-9]{3})*\s*(?:₫|VND|VNĐ|\$|USD)?$", line_s):
                continue
            desc_lines.append(line_s)

        description_text = "\n".join(desc_lines).strip()

        # Semantic content template
        categories_str = ", ".join(categories) if categories else "Sản phẩm"
        semantic_content = (
            f"Tên sản phẩm: {name} | "
            f"Mã SKU: {sku} | "
            f"Danh mục: {categories_str} | "
            f"Giá: {price:,.0f} VND | "
            f"Tình trạng: {stock_status} | "
            f"Link: {source_url}\n"
            f"Mô tả sản phẩm:\n{description_text}"
        )

        product_meta = ProductMetadata(
            product_id=product_id,
            name=name,
            price=price,
            currency="VND",
            stock_status=stock_status,
            product_url=source_url,
            sku=sku,
            categories=categories,
            attributes=attributes,
        )

        chunk_meta = EcomChunkMetadata(
            category="product",
            source_file=source_file,
            source_type="crawler",
            source_url=source_url,
            page_title=page_title,
            crawl_timestamp=crawl_timestamp,
            product=product_meta,
        )

        return EcomChunk.create(
            source_url=source_url,
            doc_id=doc_id,
            chunk_index=0,
            content=semantic_content,
            metadata=chunk_meta,
        )

    def _chunk_policy_page(
        self,
        cleaned_body: str,
        policy_type: str,
        source_url: str,
        source_file: str,
        page_title: str,
        doc_id: str,
        crawl_timestamp: Any,
    ) -> list[EcomChunk]:
        """Splits policy markdown by headers with small-to-big parent context expansion."""
        chunks: list[EcomChunk] = []

        # Split on markdown headers (#, ##, ###)
        sections = re.split(r"\n(?=#{1,3}\s+)", cleaned_body)
        chunk_index = 0

        for sec in sections:
            sec_text = sec.strip()
            if not sec_text:
                continue

            lines = sec_text.splitlines()
            section_title = lines[0].lstrip("# ").strip() if lines else page_title
            section_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else sec_text

            if not section_body:
                section_body = section_title

            # Small chunk for embedding
            embedding_text = (
                f"Chính sách: {policy_type} | "
                f"Tiêu đề: {section_title} | "
                f"Nội dung: {section_body[:500]}"
            )

            policy_meta = PolicyMetadata(
                policy_type=policy_type,  # type: ignore
                section_title=section_title,
                effective_date=crawl_timestamp or None,
                policy_url=source_url,
                parent_section_content=sec_text,  # Full parent section
            )

            chunk_meta = EcomChunkMetadata(
                category="policy",
                source_file=source_file,
                source_type="crawler",
                source_url=source_url,
                page_title=page_title,
                crawl_timestamp=crawl_timestamp,
                policy=policy_meta,
            )

            chunk = EcomChunk.create(
                source_url=f"{source_url}#{chunk_index}",
                doc_id=doc_id,
                chunk_index=chunk_index,
                content=embedding_text,
                metadata=chunk_meta,
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def _chunk_generic_sections(
        self,
        cleaned_body: str,
        source_url: str,
        source_file: str,
        page_title: str,
        doc_id: str,
        crawl_timestamp: Any,
    ) -> list[EcomChunk]:
        """Fallback chunker splitting generic markdown into paragraphs."""
        chunks: list[EcomChunk] = []
        paragraphs = [p.strip() for p in cleaned_body.split("\n\n") if len(p.strip()) > 30]

        for idx, para in enumerate(paragraphs):
            chunk_meta = EcomChunkMetadata(
                category="policy",
                source_file=source_file,
                source_type="crawler",
                source_url=source_url,
                page_title=page_title,
                crawl_timestamp=crawl_timestamp,
            )
            chunk = EcomChunk.create(
                source_url=f"{source_url}#{idx}",
                doc_id=doc_id,
                chunk_index=idx,
                content=para[:600],
                metadata=chunk_meta,
            )
            chunks.append(chunk)

        return chunks
