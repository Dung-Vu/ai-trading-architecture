"""CLI entry points for the full unified trading bot."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Callable

import src.config as app_config
from loguru import logger

from src.bot.full_trading_bot import FullTradingBot
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
        description="Full Unified AI Trading Bot (Phase 1 + 2 + 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main_full --mode dryrun --strategy sma_cross
      Dry-run with SMA Cross strategy (classic)

  python -m src.main_full --mode dryrun --strategy ai_debate
      Dry-run with AI debate engine (Phase 2)

  python -m src.main_full --mode testnet --strategy ai_debate
      Testnet trading with full AI stack

  python -m src.main_full --mode dryrun --strategy bbands
      Dry-run with Bollinger Bands strategy

  python -m src.main_full --mode dryrun --strategy ai_debate \\
      --symbols BTC/USDT ETH/USDT --interval 30
      Fast-cycle AI trading on BTC + ETH
        """,
    )

    add_common_runtime_args(
        parser,
        strategy_choices=["sma_cross", "bbands", "ai_debate"],
        strategy_help="Trading strategy (default: ai_debate for trading, config value for backtest)",
        default_strategy=None,
        include_interval=True,
        symbols_help="Trading symbols (default: BTC/USDT ETH/USDT SOL/USDT)",
    )

    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable Mem0 self-learning memory",
    )

    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Disable news/sentiment pipeline",
    )

    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Disable auto-tuner",
    )

    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run debate engine only, no trade execution",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest instead of live trading",
    )

    parser.add_argument(
        "--backtest-days",
        type=int,
        default=10,
        help="Number of days for backtest (default: 10)",
    )

    return parser.parse_args()



def run_backtest(config, args):
    """Run backtest with the selected strategy."""
    run_backtest_from_config(
        config,
        days=args.backtest_days,
        heading="📈 Starting Backtest...",
        output_path=Path(__file__).parent.parent / "logs" / "backtest_results.html",
    )


# ─── Main Entry Point ──────────────────────────────────────────────────

def main(
    *,
    parse_args_fn: Callable[[], argparse.Namespace] | None = None,
    run_backtest_fn: Callable[[Any, argparse.Namespace], None] | None = None,
    setup_logging_fn: Callable[..., None] | None = None,
    config_loader: Callable[..., Any] | None = None,
    debate_engine_builder: Callable[..., tuple[Any, Any]] | None = None,
    bot_class: type[FullTradingBot] = FullTradingBot,
) -> None:
    """Main entry point for the full unified trading runtime."""
    parse_args_fn = parse_args if parse_args_fn is None else parse_args_fn
    run_backtest_fn = run_backtest if run_backtest_fn is None else run_backtest_fn
    setup_logging_fn = setup_logging if setup_logging_fn is None else setup_logging_fn
    config_loader = app_config.load_config if config_loader is None else config_loader
    debate_engine_builder = (
        build_debate_engine
        if debate_engine_builder is None
        else debate_engine_builder
    )

    args = parse_args_fn()
    config = load_runtime_config(
        args,
        default_strategy="ai_debate",
        keep_config_strategy_for_backtest=True,
        config_loader=config_loader,
    )

    setup_logging_fn(
        log_level=config.monitoring.log_level,
        app_log_name="full_trading",
        error_log_name="full_error",
    )

    logger.info("=" * 60)
    logger.info("🚀 Full Unified AI Trading Architecture (Phase 1+2+4)")
    logger.info("=" * 60)
    logger.info(f"  Mode:        {config.trading.mode}")
    logger.info(f"  Strategy:    {config.strategy.name}")
    logger.info(f"  Symbols:     {config.trading.symbols}")
    logger.info(f"  Memory:      {'Enabled' if not args.no_memory else 'Disabled'}")
    logger.info(f"  News:        {'Enabled' if not args.no_news else 'Disabled'}")
    logger.info(f"  Auto-tune:   {'Enabled' if not args.no_autotune else 'Disabled'}")
    logger.info("=" * 60)

    if args.backtest:
        run_backtest_fn(config, args)
    elif args.debate_only:
        logger.info("🧠 Debate-only mode — running single debate...")
        engine, _ = debate_engine_builder(config, config.trading.symbols)

        sample_data = {
            "price": 67500,
            "rsi": 28,
            "sma_fast": 67200,
            "sma_slow": 68000,
            "volume": 2.5,
        }
        result = engine.run_debate(sample_data, symbol=config.trading.symbols[0])
        logger.info(
            f"Debate result: {result.action} "
            f"(confidence={result.confidence:.0f}%)"
        )
        logger.info(f"Reasoning: {result.reason[:200]}")
    else:
        bot = bot_class(
            config=config,
            mode=config.trading.mode,
            strategy=config.strategy.name,
            symbols=config.trading.symbols,
            interval=args.interval,
            enable_memory=not args.no_memory,
            enable_news=not args.no_news,
            enable_autotune=not args.no_autotune,
        )

        async def async_main() -> None:
            await bot.setup()
            await bot.run()

        asyncio.run(async_main())