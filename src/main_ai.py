#!/usr/bin/env python3
"""Compatibility entry point for the lean AI trading bot.

This module now delegates to the canonical `src.main` runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import src.config as app_config
from loguru import logger

from src.bot.ai_trading_bot import AITradingBot
from src.main import main as _main_impl
from src.main import parse_args as _parse_args_impl
from src.main import run_debate_only as _run_debate_only_impl
from src.logging_config import setup_logging
from src.runtime_helpers import run_backtest_from_config


parse_args = _parse_args_impl


def run_backtest(config: Any, days: int = 90) -> None:
    """Run the legacy AI backtest variant with the SMA proxy wiring."""
    from src.strategy.sma_cross import SMACrossStrategy

    symbol = config.trading.symbols[0] if config.trading.symbols else "BTC/USDT"
    params = {
        "symbol": symbol.replace("/", ""),
        "quote_asset": "USDT",
        "initial_capital": config.trading.initial_capital,
        "sma_fast": config.strategy.sma_fast,
        "sma_slow": config.strategy.sma_slow,
        "rsi_period": config.strategy.rsi_period,
    }

    try:
        run_backtest_from_config(
            config,
            days=days,
            heading=f"📈 Starting AI Backtest ({days} days)...",
            strategy_class=SMACrossStrategy,
            strategy_name="ai_debate (sma_cross proxy)",
            symbol=symbol,
            parameters=params,
            output_path=Path(__file__).parent.parent / "logs" / "ai_backtest_results.html",
            use_utc=True,
        )
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")


async def run_debate_only(config, symbol: str = "BTC/USDT") -> None:
    """Run a single debate without trade execution."""
    await _run_debate_only_impl(config, symbol=symbol)


def main() -> None:
    """Entry point for AI trading bot."""
    return _main_impl(
        parse_args_fn=parse_args,
        run_debate_only_fn=run_debate_only,
        run_backtest_fn=lambda config, *, days=None: run_backtest(
            config,
            days=90 if days is None else days,
        ),
        setup_logging_fn=setup_logging,
        setup_logging_kwargs={
            "app_log_name": "ai_trading",
            "error_log_name": "ai_error",
        },
        config_loader=app_config.load_config,
        bot_class=AITradingBot,
    )


__all__ = [
    "AITradingBot",
    "main",
    "parse_args",
    "run_backtest",
    "run_debate_only",
    "setup_logging",
]


if __name__ == "__main__":
    main()
