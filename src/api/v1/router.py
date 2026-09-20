"""Aggregates all API v1 endpoints under a unified router."""

from fastapi import APIRouter

from src.api.v1.endpoints import chat, crawler, generation, ingestion, retrieval

api_v1_router = APIRouter()

api_v1_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["RAG Chat Assistant"],
)

api_v1_router.include_router(
    crawler.router,
    prefix="/crawler",
    tags=["Web Crawler"],
)

api_v1_router.include_router(
    ingestion.router,
    prefix="/ingestion",
    tags=["Ingestion & Indexing"],
)

api_v1_router.include_router(
    retrieval.router,
    prefix="/retrieval",
    tags=["Hybrid Retrieval"],
)

api_v1_router.include_router(
    generation.router,
    prefix="/generate",
    tags=["LLM Generation"],
)


