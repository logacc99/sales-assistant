"""Document cleaner for parsing YAML frontmatter and stripping web boilerplate."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional
import yaml


class DocumentCleaner:
    """Cleans crawled markdown documents, extracts frontmatter, and strips boilerplate."""

    NON_CONTENT_URL_PATTERNS = [
        re.compile(r"/gio-hang/?", re.IGNORECASE),
        re.compile(r"/cart/?", re.IGNORECASE),
        re.compile(r"/tai-khoan/?", re.IGNORECASE),
        re.compile(r"/my-account/?", re.IGNORECASE),
        re.compile(r"/lost-password/?", re.IGNORECASE),
        re.compile(r"/thanh-toan/?", re.IGNORECASE),
        re.compile(r"/checkout/?", re.IGNORECASE),
    ]

    FOOTER_PATTERNS = [
        # Related / cross-sell product sections
        re.compile(r"\n#{1,4}\s+(?:Sản phẩm tương tự|Sản phẩm liên quan|Related products|You may also like)", re.IGNORECASE),
        # Standard corporate & informational footer headings (Vietnamese & English)
        re.compile(
            r"\n#{1,4}\s+(?:VỀ CHÚNG TÔI|THÔNG TIN|LIÊN HỆ|NHẬN TIN|CHÍNH SÁCH|HỖ TRỢ|ABOUT US|CONTACT|INFORMATION|CUSTOMER SERVICE)",
            re.IGNORECASE,
        ),
        # Business registration & legal entity markers
        re.compile(r"\n(?:CTY|CÔNG TY)\s+(?:TNHH|CP|CỔ PHẦN)", re.IGNORECASE),
        re.compile(r"\n(?:Giấy\s+)?(?:CNĐKKD|ĐKKD|MSDN|GPKD):\s*\d+", re.IGNORECASE),
        # Standard copyright line (any brand)
        re.compile(r"\n(?:©|\(c\)|Copyright)\s*(?:\d{4})?", re.IGNORECASE),
    ]

    HEADER_END_PATTERNS = [
        re.compile(r"FILTER BY PRICE.*?\n", re.IGNORECASE),
        re.compile(r"Showing all \d+ results.*?\n", re.IGNORECASE),
        re.compile(r"Thứ tự theo mức độ phổ biến.*?\n", re.IGNORECASE),
        re.compile(r"\[Trang chủ\]\(https?://[^)]+\)\s*/\s*\[Sản phẩm\]\(https?://[^)]+\)[^\n]*\n", re.IGNORECASE),
    ]

    def extract_frontmatter(self, raw_text: str) -> tuple[dict[str, Any], str]:
        """Extracts YAML frontmatter and returns (metadata_dict, body_markdown)."""
        raw_text = raw_text.strip()
        if not raw_text.startswith("---"):
            return {}, raw_text

        parts = raw_text.split("---", 2)
        if len(parts) < 3:
            return {}, raw_text

        yaml_content = parts[1]
        body = parts[2].strip()

        try:
            metadata = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            metadata = {}

        if "crawl_timestamp" in metadata and isinstance(metadata["crawl_timestamp"], str):
            try:
                metadata["crawl_timestamp"] = datetime.fromisoformat(metadata["crawl_timestamp"])
            except ValueError:
                pass

        return metadata, body

    def is_non_content_page(self, url: str, title: str, body: str) -> bool:
        """Determines if a page is a non-content utility page (cart, account, login)."""
        for pattern in self.NON_CONTENT_URL_PATTERNS:
            if pattern.search(url):
                return True

        # If page body is empty or practically empty
        if len(body.strip()) < 20:
            return True

        return False

    def strip_boilerplate(self, body: str) -> str:
        """Strips header navigation, sidebar trees, and footers from markdown body."""
        cleaned = body

        # 1. Cut footer
        for footer_pat in self.FOOTER_PATTERNS:
            match = footer_pat.search(cleaned)
            if match:
                cleaned = cleaned[: match.start()].strip()

        # 2. Strip header navigation if a known separator exists
        # In catalog listings, products typically start after breadcrumbs or filter widgets
        header_cutoff = -1
        for header_pat in self.HEADER_END_PATTERNS:
            matches = list(header_pat.finditer(cleaned))
            if matches:
                last_match = matches[-1]
                if last_match.end() > header_cutoff:
                    header_cutoff = last_match.end()

        if header_cutoff > 0:
            cleaned = cleaned[header_cutoff:].strip()
        else:
            # Fallback: remove skip link and initial nav lists if present
            cleaned = re.sub(r"^\[Skip to content\]\([^\)]+\)\s*", "", cleaned, flags=re.IGNORECASE)

        # 3. Clean sidebar category listing if still present in body
        cat_match = re.search(
            r"(?:Danh mục sản phẩm|Product Categories|Categories)\s*\n(?:\s*\*\s*\[[^\]]+\]\([^\)]+\)\s*\n)+",
            cleaned,
            re.IGNORECASE,
        )
        if cat_match:
            cleaned = cleaned[: cat_match.start()] + cleaned[cat_match.end() :]

        return cleaned.strip()
