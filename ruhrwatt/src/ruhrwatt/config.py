"""Paths and runtime settings, read from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Root of the data lake (raw / bronze / silver / gold)."""
    return Path(os.environ.get("RUHRWATT_DATA_DIR", "data")).resolve()


def raw_dir(source: str) -> Path:
    """Raw, untouched downloads for one source, e.g. raw/smard."""
    return data_dir() / "raw" / source


def request_delay() -> float:
    """Seconds to sleep between HTTP requests."""
    return float(os.environ.get("RUHRWATT_REQUEST_DELAY", "0.2"))
