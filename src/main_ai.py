#!/usr/bin/env python3
"""Compatibility entry point for the lean AI trading bot."""

from __future__ import annotations

import src.config as app_config

from src.ai_cli import main as _main_impl
from src.ai_cli import parse_args as _parse_args_impl
from src.ai_cli import run_backtest as _run_backtest_impl
from src.ai_cli import run_debate_only as _run_debate_only_impl
from src.bot.ai_trading_bot import AITradingBot
from src.debate.runtime import build_debate_engine
from src.logging_config import setup_logging


parse_args = _parse_args_impl
run_backtest = _run_backtest_impl


async def run_debate_only(config, symbol: str = "BTC/USDT") -> None:
    """Run a single debate without trade execution."""
    await _run_debate_only_impl(
        config,
        symbol=symbol,
        debate_engine_builder=build_debate_engine,
    )


def main() -> None:
    """Entry point for AI trading bot."""
    return _main_impl(
        parse_args_fn=parse_args,
        run_debate_only_fn=run_debate_only,
        run_backtest_fn=run_backtest,
        setup_logging_fn=setup_logging,
        config_loader=app_config.load_config,
        debate_engine_builder=build_debate_engine,
        bot_class=AITradingBot,
    )


__all__ = [
    "AITradingBot",
    "build_debate_engine",
    "main",
    "parse_args",
    "run_backtest",
    "run_debate_only",
    "setup_logging",
]


if __name__ == "__main__":
    main()
