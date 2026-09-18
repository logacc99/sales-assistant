"""Unit tests for Crawler API endpoints."""

from unittest.mock import AsyncMock, patch
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns successful status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"


def test_root_endpoint():
    """Verify GET / returns documentation and info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Sales Assistant API"
    assert data["docs"] == "/docs"


@patch("src.api.v1.endpoints.crawler.async_crawl_website", new_callable=AsyncMock)
def test_crawl_endpoint_success(mock_crawl):
    """Verify POST /api/v1/crawler/crawl triggers crawler and returns file list."""
    mock_crawl.return_value = [
        Path("data/raw/website/test.com/page1.md"),
        Path("data/raw/website/test.com/page2.md"),
    ]

    payload = {
        "start_url": "https://test.com",
        "max_pages": 5,
        "max_depth": 2,
    }

    response = client.post("/api/v1/crawler/crawl", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total_crawled"] == 2
    assert len(body["data"]["saved_files"]) == 2
    mock_crawl.assert_awaited_once()


@patch("src.api.v1.endpoints.crawler.async_crawl_website", new_callable=AsyncMock)
def test_crawl_endpoint_failure(mock_crawl):
    """Verify POST /api/v1/crawler/crawl handles errors gracefully."""
    mock_crawl.side_effect = RuntimeError("Playwright browser launch failed")

    payload = {
        "start_url": "https://test.com",
        "max_pages": 2,
    }

    response = client.post("/api/v1/crawler/crawl", json=payload)
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert "Playwright browser launch failed" in body["message"]
