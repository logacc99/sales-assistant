"""Sales Assistant Unified API Gateway & Server Entrypoint.

Provides CLI command execution, environment validation, pre-flight health diagnostics,
and launches the Uvicorn ASGI server hosting the FastAPI application.
Conforms to SPEC-0000: Section 6.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Optional, Sequence

import uvicorn

from src.config import IngestionConfig, get_config

logger = logging.getLogger("sales_assistant.entrypoint")


def _bool_from_env(key: str, default: bool) -> bool:
    """Helper to parse boolean values from environment variables."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y", "t")


def build_parser() -> argparse.ArgumentParser:
    """Constructs command-line argument parser for the API entrypoint."""
    default_host = os.getenv("APP_HOST", os.getenv("HOST", "127.0.0.1"))
    default_port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
    default_reload = _bool_from_env("APP_RELOAD", True)
    default_workers = int(os.getenv("APP_WORKERS", "1"))
    default_log_level = os.getenv("APP_LOG_LEVEL", "info")

    parser = argparse.ArgumentParser(
        prog="sales-assistant",
        description="Sales Assistant RAG & Ingestion API Server Entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help="Network interface to bind server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="TCP port to listen on.",
    )
    reload_group = parser.add_mutually_exclusive_group()
    reload_group.add_argument(
        "--reload",
        dest="reload",
        action="store_true",
        default=default_reload,
        help="Enable auto-reload on code changes (Development mode).",
    )
    reload_group.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        help="Disable auto-reload on code changes (Production mode).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Number of worker processes (enforced to 1 if reload is enabled).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default=default_log_level,
        help="Logging level for Uvicorn and API components.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="Execute pre-flight connectivity & environment diagnostics without starting HTTP server.",
    )

    return parser


def run_preflight_checks(cfg: Optional[IngestionConfig] = None) -> dict[str, Any]:
    """Evaluates environment configuration, AWS Bedrock credentials, and OpenSearch connectivity.

    Returns a structured diagnostics dictionary.
    """
    if cfg is None:
        cfg = get_config()

    report: dict[str, Any] = {
        "status": "healthy",
        "aws_region": cfg.aws_region,
        "bedrock_model": cfg.bedrock_model_id,
        "bedrock_dimension": cfg.bedrock_dimension,
        "opensearch_target": f"{cfg.opensearch_host}:{cfg.opensearch_port}",
        "opensearch_index": cfg.opensearch_index_name,
        "opensearch_reachable": False,
        "bedrock_configured": bool(cfg.aws_region),
        "errors": [],
        "warnings": [],
    }

    # Verify OpenSearch connectivity
    try:
        from src.ingestion.indexer import OpenSearchVectorIndexer

        indexer = OpenSearchVectorIndexer()
        if hasattr(indexer, "client") and indexer.client is not None:
            if indexer.client.ping():
                report["opensearch_reachable"] = True
            else:
                report["warnings"].append(
                    f"OpenSearch cluster at {cfg.opensearch_host}:{cfg.opensearch_port} did not respond to ping."
                )
    except Exception as exc:
        report["warnings"].append(
            f"OpenSearch connectivity check failed ({cfg.opensearch_host}:{cfg.opensearch_port}): {exc}"
        )

    # Check AWS credentials presence
    if not (cfg.aws_access_key_id and cfg.aws_secret_access_key):
        report["warnings"].append(
            "AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) not found in environment; "
            "relying on default IAM role / SSO chain."
        )

    if report["errors"]:
        report["status"] = "unhealthy"
    elif report["warnings"]:
        report["status"] = "degraded"

    return report


def print_startup_banner(host: str, port: int, cfg: IngestionConfig) -> None:
    """Prints formatted ASCII banner displaying runtime settings, docs links, and routes."""
    banner = f"""
================================================================================
  🛍️  SALES ASSISTANT — API GATEWAY & RAG RUNTIME
================================================================================
  Status:            Online (Ready for requests)
  API Version:       v0.1.0
  Listening Address: http://{host}:{port}
  Swagger UI:        http://{host}:{port}/docs
  ReDoc:             http://{host}:{port}/redoc
  OpenSearch Host:   {cfg.opensearch_host}:{cfg.opensearch_port} (Index: {cfg.opensearch_index_name})
  Bedrock Model:     {cfg.bedrock_model_id} ({cfg.bedrock_dimension}-d)
  Active Routes:
    - [GET]  /health
    - [POST] /api/v1/crawler/crawl
    - [POST] /api/v1/ingestion/run
    - [POST] /api/v1/ingestion/index-raw
    - [POST] /api/v1/ingestion/search
================================================================================
"""
    print(banner, flush=True)


def print_diagnostics_report(report: dict[str, Any]) -> None:
    """Prints structured pre-flight diagnostics results to stdout."""
    status_icon = "✅" if report["status"] == "healthy" else ("⚠️" if report["status"] == "degraded" else "❌")
    os_icon = "✅" if report["opensearch_reachable"] else "⚠️"
    bedrock_icon = "✅" if report["bedrock_configured"] else "❌"

    output = f"""
================================================================================
  🩺  SALES ASSISTANT — PRE-FLIGHT SYSTEM DIAGNOSTICS
================================================================================
  Overall Health:     {status_icon} {report['status'].upper()}
  AWS Region:         {report['aws_region']}
  Bedrock Model:      {bedrock_icon} {report['bedrock_model']} ({report['bedrock_dimension']}-d)
  OpenSearch Target:  {os_icon} {report['opensearch_target']} (Index: {report['opensearch_index']})
  OpenSearch Online:  {report['opensearch_reachable']}
"""
    if report["warnings"]:
        output += "\n  Warnings:\n"
        for w in report["warnings"]:
            output += f"    - {w}\n"

    if report["errors"]:
        output += "\n  Errors:\n"
        for e in report["errors"]:
            output += f"    - {e}\n"

    output += "================================================================================\n"
    print(output, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main application launcher and process supervisor."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = get_config()

    # Pre-flight health diagnostics
    diagnostics = run_preflight_checks(cfg)

    if args.check:
        print_diagnostics_report(diagnostics)
        return 1 if diagnostics["errors"] else 0

    # Validate workers vs reload
    workers = args.workers
    if args.reload and workers > 1:
        logger.warning(
            "Uvicorn auto-reload is enabled; forcing worker count from %d to 1.", workers
        )
        workers = 1

    # Render startup banner
    print_startup_banner(args.host, args.port, cfg)

    # Launch Uvicorn ASGI server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
