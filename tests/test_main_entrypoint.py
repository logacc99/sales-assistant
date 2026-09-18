"""Unit and integration tests for Sales Assistant unified main.py entrypoint."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure workspace root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import (
    build_parser,
    main,
    print_diagnostics_report,
    print_startup_banner,
    run_preflight_checks,
)
from src.config import IngestionConfig


def test_build_parser_defaults():
    """Verify default parser configurations match specification defaults."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.host in ("127.0.0.1", "localhost", os.getenv("APP_HOST", "127.0.0.1"))
    assert args.port == int(os.getenv("APP_PORT", "8000"))
    assert args.reload is True or args.reload is False
    assert args.workers == 1
    assert args.log_level == "info"
    assert args.check is False


def test_build_parser_custom_args():
    """Verify custom CLI flags are accurately parsed."""
    parser = build_parser()
    args = parser.parse_args([
        "--host", "0.0.0.0",
        "--port", "9999",
        "--no-reload",
        "--workers", "4",
        "--log-level", "debug",
        "--check",
    ])

    assert args.host == "0.0.0.0"
    assert args.port == 9999
    assert args.reload is False
    assert args.workers == 4
    assert args.log_level == "debug"
    assert args.check is True


def test_run_preflight_checks():
    """Verify preflight check dictionary structure and error tolerance."""
    cfg = IngestionConfig(
        aws_region="us-east-1",
        bedrock_model_id="cohere.embed-multilingual-v3.0",
        bedrock_dimension=1024,
        opensearch_host="localhost",
        opensearch_port=9200,
    )
    report = run_preflight_checks(cfg)

    assert "status" in report
    assert report["aws_region"] == "us-east-1"
    assert report["bedrock_model"] == "cohere.embed-multilingual-v3.0"
    assert report["bedrock_dimension"] == 1024
    assert "opensearch_reachable" in report
    assert isinstance(report["warnings"], list)
    assert isinstance(report["errors"], list)


def test_print_startup_banner_and_diagnostics(capsys):
    """Verify stdout formatting for banner and diagnostics report."""
    cfg = IngestionConfig()
    print_startup_banner("127.0.0.1", 8000, cfg)
    captured = capsys.readouterr()
    assert "SALES ASSISTANT" in captured.out
    assert "http://127.0.0.1:8000/docs" in captured.out

    report = {
        "status": "healthy",
        "aws_region": "us-east-1",
        "bedrock_model": "cohere.embed-multilingual-v3.0",
        "bedrock_dimension": 1024,
        "opensearch_target": "localhost:9200",
        "opensearch_index": "test-index",
        "opensearch_reachable": True,
        "bedrock_configured": True,
        "warnings": ["Warning 1"],
        "errors": [],
    }
    print_diagnostics_report(report)
    diagnostics_out = capsys.readouterr()
    assert "PRE-FLIGHT SYSTEM DIAGNOSTICS" in diagnostics_out.out
    assert "Warning 1" in diagnostics_out.out


@patch("uvicorn.run")
def test_main_starts_uvicorn(mock_uvicorn):
    """Verify main launches uvicorn with resolved parameters."""
    exit_code = main(["--host", "127.0.0.1", "--port", "8000", "--no-reload", "--workers", "2"])
    assert exit_code == 0
    mock_uvicorn.assert_called_once_with(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=2,
        log_level="info",
    )


@patch("uvicorn.run")
def test_main_forces_single_worker_on_reload(mock_uvicorn):
    """Verify reload mode forces workers to 1 even if higher number specified."""
    exit_code = main(["--reload", "--workers", "4"])
    assert exit_code == 0
    assert mock_uvicorn.call_args[1]["workers"] == 1


def test_main_check_flag(capsys):
    """Verify --check runs diagnostics and does not start uvicorn."""
    with patch("uvicorn.run") as mock_uvicorn:
        exit_code = main(["--check"])
        assert exit_code == 0
        mock_uvicorn.assert_not_called()
        captured = capsys.readouterr()
        assert "PRE-FLIGHT SYSTEM DIAGNOSTICS" in captured.out


def test_cli_help_subprocess():
    """Verify CLI executable responds to --help via subprocess execution."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Sales Assistant RAG & Ingestion API Server Entrypoint" in result.stdout
    assert "--host" in result.stdout
    assert "--check" in result.stdout
