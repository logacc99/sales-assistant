"""Standardized response envelope schemas for API v1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


def current_utc_time() -> datetime:
    return datetime.now(timezone.utc)


class ApiResponse(BaseModel, Generic[T]):
    """Unified API response envelope."""

    success: bool = Field(default=True, description="Indicates if request succeeded")
    message: str = Field(default="Success", description="Human-readable status message")
    data: Optional[T] = Field(default=None, description="Typed payload")
    timestamp: datetime = Field(default_factory=current_utc_time, description="UTC response timestamp")


class ErrorResponse(BaseModel):
    """Unified error response schema."""

    success: bool = Field(default=False, description="Always false for error responses")
    message: str = Field(..., description="Error summary message")
    error_code: str = Field(default="INTERNAL_ERROR", description="Machine-readable error classification")
    details: Optional[dict[str, Any]] = Field(default=None, description="Contextual error metadata")
    timestamp: datetime = Field(default_factory=current_utc_time, description="UTC response timestamp")


class HealthResponse(BaseModel):
    """Application health status."""

    status: str = "ok"
    version: str = "0.1.0"
    opensearch_configured: bool = True
    bedrock_configured: bool = True
