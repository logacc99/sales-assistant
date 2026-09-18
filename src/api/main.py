"""FastAPI application entrypoint, middleware configuration, and global error handling."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.router import api_v1_router
from src.api.v1.schemas.common import ApiResponse, ErrorResponse, HealthResponse
from src.config import get_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup and shutdown procedures."""
    cfg = get_config()
    logger.info(f"Starting Sales Assistant API [Region: {cfg.aws_region}]...")
    yield
    logger.info("Shutting down Sales Assistant API...")


def create_app() -> FastAPI:
    """Application factory for Sales Assistant FastAPI server."""
    app = FastAPI(
        title="Sales Assistant RAG & Ingestion API",
        description=(
            "Modular REST API for website crawling, semantic markdown chunking, "
            "AWS Bedrock embeddings, OpenSearch vector indexing, CDC diffing, and retrieval verification."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                success=False,
                message=str(exc.detail),
                error_code=f"HTTP_{exc.status_code}",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                success=False,
                message="Request validation error",
                error_code="VALIDATION_ERROR",
                details={"errors": exc.errors()},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                success=False,
                message="Internal server error",
                error_code="INTERNAL_SERVER_ERROR",
                details={"error": str(exc)},
            ).model_dump(mode="json"),
        )

    # Base Health Endpoints
    @app.get("/", tags=["System Health"])
    def root() -> dict[str, str]:
        return {
            "name": "Sales Assistant API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    @app.get(
        "/health",
        response_model=ApiResponse[HealthResponse],
        tags=["System Health"],
        summary="Service Health Status",
    )
    def health() -> ApiResponse[HealthResponse]:
        return ApiResponse(
            success=True,
            data=HealthResponse(
                status="healthy",
                version="0.1.0",
                opensearch_configured=True,
                bedrock_configured=True,
            ),
        )

    # Mount API Version 1
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


app = create_app()
