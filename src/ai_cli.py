"""CLI entry points for the lean AI trading bot."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable

import src.config as app_config
from loguru import logger

from src.bot.ai_trading_bot import AITradingBot
from src.debate.runtime import build_debate_engine
from src.logging_config import setup_logging
from src.runtime_helpers import (
    add_common_runtime_args,
    load_runtime_config,
    run_backtest_from_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Trading with Multi-Agent Debate Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main_ai --mode dryrun --strategy ai_debate
      Run AI debate trading in dry-run mode

  python -m src.main_ai --mode dryrun --strategy sma_cross
      Run SMA Cross strategy with AI debate confirmation

  python -m src.main_ai --debate-only --symbol BTC/USDT
      Run a single debate without executing trades

  python -m src.main_ai --backtest --backtest-days 90
      Backtest the AI strategy over 90 days

  python -m src.main_ai --mode dryrun --strategy ai_debate --optimize
      Run trading with DSPy prompt optimization enabled
        """,
    )

    add_common_runtime_args(
        parser,
        strategy_choices=["sma_cross", "bbands", "ai_debate"],
        strategy_help="Trading strategy (default: ai_debate)",
        default_strategy="ai_debate",
        include_interval=True,
        initial_capital_help="Initial capital for dry-run/backtest (default: 10000)",
    )

    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run debate engine only, no trade execution",
    )
    parser.add_argument(
        "--debate-symbol",
        type=str,
        default=None,
        help="Single symbol for debate-only mode",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of live trading",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=90,
        help="Number of days for backtest (default: 90)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable DSPy prompt optimization",
    )

    return parser.parse_args()


async def run_debate_only(
    config: Any,
    symbol: str = "BTC/USDT",
    debate_engine_builder: Callable[..., tuple[Any, Any]] = build_debate_engine,
) -> None:
    """Run a single debate without trade execution."""
    logger.info(f"🧠 Running debate-only for {symbol}...")

    try:
        engine, _llm_model = debate_engine_builder(config, [symbol])
        market_data = {
            "price": 67500.0,
            "rsi": 45,
            "macd": 0.0,
            "volume": 1000,
            "bb_upper": 68000,
            "bb_lower": 67000,
        }

        result = engine.run_debate(market_data=market_data, symbol=symbol)

        logger.info("=" * 60)
        logger.info("🧠 DEBATE RESULT")
        logger.info("=" * 60)
        logger.info(f"  Action:     {result.action}")
        logger.info(f"  Confidence: {result.confidence:.1f}%")
        logger.info(f"  Reason:     {result.reason[:200]}...")
        logger.info(f"  Stop Loss:  ${result.stop_loss:,.2f}")
        logger.info(f"  Take Profit: ${result.take_profit:,.2f}")
        logger.info(f"  Risk:       {result.risk_decision}")
        logger.info("=" * 60)
    except ImportError as exc:
        logger.error(f"Debate engine dependencies missing: {exc}")
    except Exception as exc:
        logger.error(f"Debate failed: {exc}")


def run_backtest(config: Any, days: int = 90) -> None:
    """Run backtest with AI strategy."""
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
            output_path=Path(__file__).parent.parent
            / "logs"
            / "ai_backtest_results.html",
            use_utc=True,
        )
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")


def main(
    *,
    parse_args_fn: Callable[[], argparse.Namespace] | None = None,
    run_debate_only_fn: Callable[..., Any] | None = None,
    run_backtest_fn: Callable[[Any, int], None] | None = None,
    setup_logging_fn: Callable[..., None] | None = None,
    config_loader: Callable[..., Any] | None = None,
    debate_engine_builder: Callable[..., tuple[Any, Any]] | None = None,
    bot_class: type[AITradingBot] = AITradingBot,
) -> None:
    """Entry point for AI trading bot."""
    parse_args_fn = parse_args if parse_args_fn is None else parse_args_fn
    run_debate_only_fn = (
        run_debate_only if run_debate_only_fn is None else run_debate_only_fn
    )
    run_backtest_fn = run_backtest if run_backtest_fn is None else run_backtest_fn
    setup_logging_fn = setup_logging if setup_logging_fn is None else setup_logging_fn
    config_loader = app_config.load_config if config_loader is None else config_loader
    debate_engine_builder = (
        build_debate_engine
        if debate_engine_builder is None
        else debate_engine_builder
    )

    args = parse_args_fn()
    config = load_runtime_config(args, config_loader=config_loader)

    setup_logging_fn(
        log_level=config.monitoring.log_level,
        app_log_name="ai_trading",
        error_log_name="ai_error",
    )

    logger.info("=" * 60)
    logger.info("🤖 AI-Powered Trading Architecture")
    logger.info("=" * 60)
    logger.info(f"  Mode:     {config.trading.mode}")
    logger.info(f"  Strategy: {config.strategy.name}")
    logger.info(f"  Symbols:  {config.trading.symbols}")
    logger.info(f"  Capital:  ${config.trading.initial_capital:,.2f}")
    if args.optimize:
        logger.info("  DSPy:     ✅ Optimization enabled")
    logger.info("=" * 60)

    if args.debate_only:
        symbol = args.debate_symbol or config.trading.symbols[0]
        asyncio.run(
            run_debate_only_fn(
                config,
                symbol=symbol,
                debate_engine_builder=debate_engine_builder,
            )
        )
    elif args.backtest:
        run_backtest_fn(config, days=args.backtest_days)
    else:
        bot = bot_class(
            config=config,
            mode=args.mode,
            strategy=args.strategy,
            symbols=args.symbols,
            interval=args.interval,
        )

        async def _run() -> None:
            await bot.setup()
            await bot.run()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as exc:
            logger.error(f"Fatal error: {exc}")
            sys.exit(1)
