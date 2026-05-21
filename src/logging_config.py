"""Shared loguru configuration helpers for trading entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config import (
    get_default_log_app_name,
    get_default_log_error_name,
    get_default_log_error_retention,
    get_default_log_retention,
    get_default_log_rotation,
)


def setup_logging(
    log_level: str = "INFO",
    *,
    app_log_name: str | None = None,
    error_log_name: str | None = None,
    log_dir: str | Path | None = None,
) -> None:
    """Configure loguru handlers for a trading entry point."""
    log_dir = Path(log_dir) if log_dir is not None else Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    app_log_name = app_log_name or get_default_log_app_name()
    error_log_name = error_log_name or get_default_log_error_name()

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        log_dir / f"{app_log_name}_{{time:YYYY-MM-DD}}.log",
        rotation=get_default_log_rotation(),
        retention=get_default_log_retention(),
        level=log_level,
        enqueue=True,
    )

    logger.add(
        log_dir / f"{error_log_name}_{{time:YYYY-MM-DD}}.log",
        rotation=get_default_log_rotation(),
        retention=get_default_log_error_retention(),
        level="ERROR",
        enqueue=True,
    )