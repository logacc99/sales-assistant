"""REST API endpoints for Crawl4AI web crawling."""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status

from src.api.v1.schemas.common import ApiResponse
from src.api.v1.schemas.crawler import CrawlRequest, CrawlResponse
from src.ingestion.crawler import async_crawl_website

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/crawl",
    response_model=ApiResponse[CrawlResponse],
    status_code=status.HTTP_200_OK,
    summary="Trigger web crawler",
    description="Crawl a target website using Crawl4AI and save markdown files with frontmatter.",
)
async def trigger_crawl(request: CrawlRequest) -> ApiResponse[CrawlResponse]:
    """Triggers asynchronous website crawler with polite rate-limiting and BFS depth limits."""
    try:
        kwargs = {
            "start_url": request.start_url,
            "max_depth": request.max_depth,
            "max_pages": request.max_pages,
            "allowed_domain": request.allowed_domain,
            "delay_seconds": request.delay_seconds,
        }
        if request.output_dir:
            kwargs["output_dir"] = Path(request.output_dir)

        saved_paths = await async_crawl_website(**kwargs)
        saved_str_paths = [str(p) for p in saved_paths]

        return ApiResponse(
            success=True,
            message=f"Successfully crawled {len(saved_str_paths)} pages",
            data=CrawlResponse(
                total_crawled=len(saved_str_paths),
                saved_files=saved_str_paths,
                errors=[],
            ),
        )
    except Exception as exc:
        logger.exception(f"Web crawl failed for {request.start_url}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crawling failed: {str(exc)}",
        )
