"""Unit tests for the Crawl4AI-based web crawler module in src/ingestion/crawler.py."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Optional
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ingestion.crawler import (
    _extract_markdown_text,
    _generate_frontmatter,
    async_crawl_website,
    crawl_website,
)


@dataclass
class MockCrawlResult:
    """Mock result imitating Crawl4AI CrawlResult."""
    url: str
    markdown: Any
    success: bool = True
    metadata: Optional[dict[str, Any]] = None
    title: Optional[str] = None
    error_message: Optional[str] = None


class TestCrawl4AIParserAndHelpers(unittest.TestCase):
    """Verifies frontmatter generation, markdown extraction, and hash calculations."""

    def test_generate_frontmatter(self) -> None:
        frontmatter = _generate_frontmatter(
            url="https://store.example.com/product/123",
            title='Sample "Widget" Pro\nNew Edition',
            timestamp_iso="2026-09-15T12:00:00Z",
            content_hash="abcdef123456",
        )
        self.assertTrue(frontmatter.startswith("---\n"))
        self.assertIn('url: "https://store.example.com/product/123"', frontmatter)
        self.assertIn('title: "Sample \\"Widget\\" Pro New Edition"', frontmatter)
        self.assertIn('crawl_timestamp: "2026-09-15T12:00:00Z"', frontmatter)
        self.assertIn('content_hash: "abcdef123456"', frontmatter)
        self.assertIn('category: "crawled"', frontmatter)
        self.assertTrue(frontmatter.endswith("---\n\n"))

    def test_extract_markdown_text(self) -> None:
        # String markdown
        res_str = MockCrawlResult(url="https://ex.com", markdown="# Heading\nParagraph")
        self.assertEqual(_extract_markdown_text(res_str), "# Heading\nParagraph")

        # Object markdown with raw_markdown attribute
        mock_obj = MagicMock()
        mock_obj.raw_markdown = "## Object Markdown"
        res_obj = MockCrawlResult(url="https://ex.com", markdown=mock_obj)
        self.assertEqual(_extract_markdown_text(res_obj), "## Object Markdown")

        # Missing markdown
        res_empty = MagicMock(spec=[])
        self.assertEqual(_extract_markdown_text(res_empty), "")


class TestCrawl4AIExecution(unittest.IsolatedAsyncioTestCase):
    """Verifies async crawl workflow, file creation, hashing, and sync wrapper."""

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", False)
    async def test_raises_import_error_when_crawl4ai_not_installed(self) -> None:
        with self.assertRaises(ImportError) as ctx:
            await async_crawl_website("https://example.com")
        self.assertIn("crawl4ai is not installed", str(ctx.exception))

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_async_crawl_website_success(self, mock_crawler_cls) -> None:
        mock_crawler_instance = AsyncMock()
        mock_crawler_cls.return_value.__aenter__.return_value = mock_crawler_instance

        # Return mock results for two pages
        mock_crawler_instance.arun.return_value = [
            MockCrawlResult(
                url="https://store.example.com/products/headphones",
                markdown="# Wireless Headphones\nPrice: $99.99\nIn stock.",
                metadata={"title": "Wireless Headphones - Store"},
            ),
            MockCrawlResult(
                url="https://store.example.com/policies/shipping",
                markdown="# Shipping Policy\nFree shipping on orders over $50.",
                metadata={"title": "Shipping Policy - Store"},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                max_depth=2,
                max_pages=5,
                output_dir=out_dir,
                delay_seconds=0,
            )

            self.assertEqual(len(files), 2)
            # Files now live under website/store.example.com/
            expected_subdir = out_dir / "website" / "store.example.com"
            for f in files:
                self.assertTrue(f.exists())
                self.assertTrue(str(f).startswith(str(expected_subdir)))
                content = f.read_text(encoding="utf-8")
                self.assertTrue(content.startswith("---\n"))
                self.assertIn('category: "crawled"', content)
                self.assertIn("content_hash:", content)

            # Check specific product file content
            prod_file = next(f for f in files if "headphones" in f.name)
            prod_content = prod_file.read_text(encoding="utf-8")
            self.assertIn("# Wireless Headphones", prod_content)
            self.assertIn("Price: $99.99", prod_content)

            # Validate hash integrity
            body = prod_content.split("---\n\n", 1)[1].rstrip("\n")
            expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            self.assertIn(f'content_hash: "{expected_hash}"', prod_content)

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    def test_sync_crawl_website_facade(self, mock_crawler_cls) -> None:
        mock_crawler_instance = AsyncMock()
        mock_crawler_cls.return_value.__aenter__.return_value = mock_crawler_instance
        mock_crawler_instance.arun.return_value = [
            MockCrawlResult(
                url="https://store.example.com/item",
                markdown="# Item Details",
                metadata={"title": "Item Page"},
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            files = crawl_website(
                start_url="https://store.example.com/",
                output_dir=out_dir,
            )
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].exists())
            # File should be nested under website/store.example.com/
            self.assertIn("website", str(files[0]))
            self.assertIn("store.example.com", str(files[0]))


class TestCrawl4AIEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Covers domain filtering, skip logic, max_pages cap, and filename slug generation."""

    def _make_crawler_mock(self, mock_cls, results):
        """Wire up an AsyncWebCrawler mock with the given list of MockCrawlResult."""
        instance = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = instance
        instance.arun.return_value = results
        return instance

    # ------------------------------------------------------------------
    # Domain filtering
    # ------------------------------------------------------------------

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_cross_domain_results_are_skipped(self, mock_crawler_cls) -> None:
        """Results whose netloc does not match the start_url domain must not be saved."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://evil.com/steal",
                    markdown="# Stolen Page",
                    metadata={"title": "Bad"},
                ),
                MockCrawlResult(
                    url="https://store.example.com/home",
                    markdown="# Home Page",
                    metadata={"title": "Home"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        # Only the store.example.com page should be saved
        self.assertEqual(len(files), 1)
        self.assertIn("home", files[0].name)

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_allowed_domain_overrides_start_url_netloc(self, mock_crawler_cls) -> None:
        """allowed_domain should restrict pages even when start_url has a different netloc."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://cdn.example.com/assets",
                    markdown="# CDN Page",
                    metadata={"title": "CDN"},
                ),
                MockCrawlResult(
                    url="https://store.example.com/product",
                    markdown="# Product",
                    metadata={"title": "Product"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                allowed_domain="cdn.example.com",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        # Only the cdn.example.com page passes the domain filter
        self.assertEqual(len(files), 1)
        self.assertIn("assets", files[0].name)

    # ------------------------------------------------------------------
    # Failed / empty results
    # ------------------------------------------------------------------

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_failed_result_is_skipped(self, mock_crawler_cls) -> None:
        """Results with success=False must be skipped; no file should be written."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://store.example.com/broken",
                    markdown="",
                    success=False,
                    error_message="503 Service Unavailable",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(files, [])

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_empty_markdown_result_is_skipped(self, mock_crawler_cls) -> None:
        """Results that yield no markdown text must not produce a file."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://store.example.com/blank",
                    markdown="   ",          # whitespace-only → stripped to ""
                    metadata={"title": "Blank"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(files, [])

    # ------------------------------------------------------------------
    # URL deduplication
    # ------------------------------------------------------------------

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_duplicate_url_is_saved_once(self, mock_crawler_cls) -> None:
        """BFSDeepCrawlStrategy re-fetches the start URL — only one file must be written."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://store.example.com/",
                    markdown="# Homepage v1",
                    metadata={"title": "Home"},
                ),
                # Same URL fetched again (BFS depth-0 re-fetch)
                MockCrawlResult(
                    url="https://store.example.com/",
                    markdown="# Homepage v2 (duplicate fetch)",
                    metadata={"title": "Home"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(len(files), 1)

    # ------------------------------------------------------------------
    # max_pages cap
    # ------------------------------------------------------------------

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_max_pages_cap_is_respected(self, mock_crawler_cls) -> None:
        """No more than max_pages files should be written regardless of result count."""
        results = [
            MockCrawlResult(
                url=f"https://store.example.com/page{i}",
                markdown=f"# Page {i}",
                metadata={"title": f"Page {i}"},
            )
            for i in range(10)
        ]
        self._make_crawler_mock(mock_crawler_cls, results)
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                max_pages=3,
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(len(files), 3)

    # ------------------------------------------------------------------
    # Filename slug generation
    # ------------------------------------------------------------------

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_root_path_produces_index_slug(self, mock_crawler_cls) -> None:
        """A URL with no meaningful path (root '/') must yield slug 'index'."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://store.example.com/",
                    markdown="# Homepage content",
                    metadata={"title": "Home"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].name.startswith("crawled_index_"))

    @patch("src.ingestion.crawler.CRAWL4AI_AVAILABLE", True)
    @patch("src.ingestion.crawler.AsyncWebCrawler")
    async def test_special_chars_in_path_are_sanitized(self, mock_crawler_cls) -> None:
        """Special characters in the URL path must be replaced with underscores in the filename."""
        self._make_crawler_mock(
            mock_crawler_cls,
            [
                MockCrawlResult(
                    url="https://store.example.com/category/shoes & bags?sort=asc",
                    markdown="# Shoes & Bags",
                    metadata={"title": "Shoes and Bags"},
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            files = await async_crawl_website(
                start_url="https://store.example.com/",
                output_dir=Path(tmpdir),
                delay_seconds=0,
            )
        self.assertEqual(len(files), 1)
        # No spaces, &, or ? in the filename
        self.assertNotRegex(files[0].name, r"[ &?]")
        self.assertTrue(files[0].name.startswith("crawled_"))

    # ------------------------------------------------------------------
    # Frontmatter helpers — additional edge cases
    # ------------------------------------------------------------------

    def test_frontmatter_title_newline_is_stripped(self) -> None:
        """Newlines in the title must be collapsed to a space in frontmatter."""
        from src.ingestion.crawler import _generate_frontmatter
        fm = _generate_frontmatter(
            url="https://example.com",
            title="Line One\nLine Two",
            timestamp_iso="2026-01-01T00:00:00Z",
            content_hash="deadbeef",
        )
        self.assertNotIn("\n", fm.split("---\n")[1].split("title:")[1].split("\n")[0])
        self.assertIn("Line One Line Two", fm)

    def test_extract_markdown_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only markdown strings should be treated as empty after strip."""
        from src.ingestion.crawler import _extract_markdown_text
        result = MockCrawlResult(url="https://ex.com", markdown="   \t\n  ")
        self.assertEqual(_extract_markdown_text(result), "")


if __name__ == "__main__":
    unittest.main()
