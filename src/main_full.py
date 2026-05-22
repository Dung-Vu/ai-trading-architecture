#!/usr/bin/env python3
"""Compatibility entry point for the full unified trading bot.

This module now delegates to the canonical `src.main` runtime.
"""

from __future__ import annotations

from typing import Any

import src.config as app_config

from src.bot.full_trading_bot import FullTradingBot
from src.main import main as _main_impl
from src.main import parse_args as _parse_args_impl
from src.main import run_backtest as _run_backtest_impl
from src.logging_config import setup_logging


parse_args = _parse_args_impl


def run_backtest(config: Any, args: Any | None = None, *, days: int | None = None) -> None:
    """Compatibility adapter preserving the legacy `(config, args)` signature."""
    resolved_days = days
    if resolved_days is None and args is not None:
        resolved_days = getattr(args, "backtest_days", None)
    if resolved_days is None:
        resolved_days = getattr(config, "backtest_days", 90)
    return _run_backtest_impl(config, days=resolved_days)


def main() -> None:
    """Main entry point."""
    return _main_impl(
        parse_args_fn=parse_args,
        run_backtest_fn=lambda config, *, days=None: run_backtest(config, days=days),
        setup_logging_fn=setup_logging,
        setup_logging_kwargs={
            "app_log_name": "full_trading",
            "error_log_name": "full_error",
        },
        config_loader=app_config.load_config,
        bot_class=FullTradingBot,
    )


__all__ = [
    "FullTradingBot",
    "main",
    "parse_args",
    "run_backtest",
    "setup_logging",
]


if __name__ == "__main__":
    main()