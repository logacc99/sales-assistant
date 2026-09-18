"""Modern Web Crawler using Crawl4AI for dynamic JavaScript rendering and RAG Markdown ingestion."""

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Optional, Union
import urllib.parse

logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None
    BFSDeepCrawlStrategy = None


def _generate_frontmatter(url: str, title: str, timestamp_iso: str, content_hash: str) -> str:
    """Builds standardized YAML frontmatter for ingested documents."""
    safe_title = title.replace('"', '\\"').replace("\n", " ").strip()
    return (
        f"---\n"
        f'url: "{url}"\n'
        f'title: "{safe_title}"\n'
        f'crawl_timestamp: "{timestamp_iso}"\n'
        f'content_hash: "{content_hash}"\n'
        f'category: "crawled"\n'
        f"---\n\n"
    )


def _extract_markdown_text(result: Any) -> str:
    """Extracts raw markdown string from CrawlResult."""
    if not hasattr(result, "markdown"):
        return ""
    md = result.markdown
    if hasattr(md, "raw_markdown"):
        return str(md.raw_markdown).strip()
    return str(md).strip()


async def async_crawl_website(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 15,
    allowed_domain: Optional[str] = None,
    output_dir: Union[str, Path] = Path("data/raw"),
    headless: bool = True,
    magic: bool = True,
    delay_seconds: float = 0.5,
) -> list[Path]:
    """
    Asynchronously crawls web pages using Crawl4AI with Playwright headless browser.

    Args:
        start_url: Entry point URL for crawl.
        max_depth: Maximum link hop depth from start_url.
        max_pages: Maximum number of pages to save.
        allowed_domain: Domain restriction; defaults to netloc of start_url.
        output_dir: Directory where resulting Markdown files are saved.
        headless: Whether to run Chromium in headless mode.
        magic: Enables anti-bot stealth heuristics and human-like delays.
        delay_seconds: Pause between page requests.

    Returns:
        List of Path objects for all newly written files in `output_dir`.
    """
    if not CRAWL4AI_AVAILABLE:
        raise ImportError(
            "crawl4ai is not installed. Please install project dependencies using:\n"
            "  pip install -r requirements.txt\n"
            "  crawl4ai-setup"
        )

    parsed_start = urllib.parse.urlparse(start_url)
    target_domain = (allowed_domain or parsed_start.netloc).lower()

    # Auto-nest under {output_dir}/website/{domain}/
    out_path = Path(output_dir) / "website" / target_domain
    out_path.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(
        headless=headless,
        verbose=False,
    )

    # Configure BFS deep crawling with domain boundary guard
    deep_crawl = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        include_external=False,
        max_pages=max_pages,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=magic,
        deep_crawl_strategy=deep_crawl,
        delay_before_return_html=delay_seconds,
    )

    logger.info(
        "Starting Crawl4AI crawl at %s (domain: %s, max_depth: %d, max_pages: %d)",
        start_url,
        target_domain,
        max_depth,
        max_pages,
    )

    saved_files: list[Path] = []
    seen_urls: set[str] = set()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        raw_results = await crawler.arun(start_url, config=run_config)

        # Normalize single vs multi result
        results_list = raw_results if isinstance(raw_results, (list, tuple)) else [raw_results]

        for result in results_list:
            if not getattr(result, "success", True):
                logger.warning("Failed crawl on %s: %s", getattr(result, "url", ""), getattr(result, "error_message", "Unknown error"))
                continue

            page_url = getattr(result, "url", start_url)

            # Deduplicate: BFSDeepCrawlStrategy re-fetches the start URL as depth-0
            normalized_url = page_url.rstrip("/").lower()
            if normalized_url in seen_urls:
                logger.debug("Skipping duplicate URL: %s", page_url)
                continue
            seen_urls.add(normalized_url)

            parsed_url = urllib.parse.urlparse(page_url)
            if target_domain not in parsed_url.netloc.lower():
                continue

            markdown_content = _extract_markdown_text(result)
            if not markdown_content:
                logger.debug("Skipping empty markdown output for %s", page_url)
                continue

            # Extract title from metadata or fallback to URL
            metadata = getattr(result, "metadata", {}) or {}
            page_title = metadata.get("title") or getattr(result, "title", None) or page_url

            content_hash = hashlib.sha256(markdown_content.encode("utf-8")).hexdigest()
            timestamp_iso = datetime.now(timezone.utc).isoformat()

            frontmatter = _generate_frontmatter(
                url=page_url,
                title=page_title,
                timestamp_iso=timestamp_iso,
                content_hash=content_hash,
            )
            full_document = frontmatter + markdown_content + "\n"

            # Create sanitized filename
            url_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", parsed_url.path.strip("/")).strip("_")
            if not url_slug:
                url_slug = "index"
            file_name = f"crawled_{url_slug}_{content_hash[:8]}.md"
            file_path = out_path / file_name

            file_path.write_text(full_document, encoding="utf-8")
            saved_files.append(file_path)
            logger.info("Saved %s (%d chars)", file_path.name, len(full_document))

            if len(saved_files) >= max_pages:
                break

    logger.info("Crawl4AI completed. Ingested %d pages into %s", len(saved_files), out_path)
    return saved_files


def crawl_website(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 15,
    allowed_domain: Optional[str] = None,
    output_dir: Union[str, Path] = Path("data/raw"),
    headless: bool = True,
    magic: bool = True,
    delay_seconds: float = 0.5,
    **kwargs: Any,
) -> list[Path]:
    """
    Synchronous entry point for crawling websites using Crawl4AI.
    """
    return asyncio.run(
        async_crawl_website(
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            allowed_domain=allowed_domain,
            output_dir=output_dir,
            headless=headless,
            magic=magic,
            delay_seconds=delay_seconds,
        )
    )

if __name__ == "__main__":
    files = crawl_website(
        start_url="https://bepbbq.com/",
        max_depth=1,
        max_pages=10,
        output_dir="data/raw",
    )
    print(f"Crawled {len(files)} pages.")