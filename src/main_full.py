#!/usr/bin/env python3
"""Compatibility entry point for the full unified trading bot."""

from __future__ import annotations

import src.config as app_config

from src.bot.full_trading_bot import FullTradingBot
from src.cli import main as _main_impl
from src.cli import parse_args as _parse_args_impl
from src.cli import run_backtest as _run_backtest_impl
from src.debate.runtime import build_debate_engine
from src.logging_config import setup_logging


parse_args = _parse_args_impl
run_backtest = _run_backtest_impl


def main() -> None:
    """Main entry point."""
    return _main_impl(
        parse_args_fn=parse_args,
        run_backtest_fn=run_backtest,
        setup_logging_fn=setup_logging,
        config_loader=app_config.load_config,
        debate_engine_builder=build_debate_engine,
        bot_class=FullTradingBot,
    )


__all__ = [
    "FullTradingBot",
    "build_debate_engine",
    "main",
    "parse_args",
    "run_backtest",
    "setup_logging",
]


if __name__ == "__main__":
    main()