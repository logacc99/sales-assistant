"""Pytest configuration and root sys.path setup."""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for test discovery and execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
