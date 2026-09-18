"""Pydantic schemas for web crawler endpoints."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    """Request payload for triggering a web crawl."""

    start_url: str = Field(..., description="Root URL to start crawling from", examples=["https://bepbbq.com"])
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum number of pages to crawl")
    max_depth: int = Field(default=2, ge=1, le=5, description="BFS crawl depth limit")
    allowed_domain: Optional[str] = Field(default=None, description="Optional domain restriction")
    output_dir: Optional[str] = Field(default=None, description="Optional custom directory to save markdown files")
    delay_seconds: float = Field(default=0.5, ge=0.0, description="Polite crawl delay between requests")


class CrawlPageResult(BaseModel):
    """Result of an individual crawled page."""

    file_path: str
    url: Optional[str] = None
    title: Optional[str] = None


class CrawlResponse(BaseModel):
    """Summary of crawl execution."""

    total_crawled: int = Field(..., description="Number of successfully saved pages")
    saved_files: list[str] = Field(default_factory=list, description="Saved file paths on disk")
    errors: list[str] = Field(default_factory=list, description="Non-fatal crawl error messages")
