#!/usr/bin/env python3
"""
Canonical application entry point for the trading architecture.

Read this module to understand the top-level runtime flow:
1. Parse CLI args.
2. Load config.
3. Route to data pipeline, debate-only, backtest, monitoring, or trading.
4. Start the canonical TradingBot runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable

import src.config as app_config
from loguru import logger
from src.config import get_default_loop_interval_seconds
from src.logging_config import setup_logging
from src.runtime_helpers import (
    add_common_runtime_args,
    load_runtime_config,
    run_backtest_from_config,
)
from src.shared_utils import normalize_market_symbol


def parse_args() -> argparse.Namespace:
    """Parse the canonical CLI surface for the trading runtime."""
    parser = argparse.ArgumentParser(
        description="AI Autonomous Trading Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode dryrun --strategy ai_debate
      Run the canonical AI trading bot in dry-run mode

  python -m src.main --mode testnet --strategy sma_cross
      Run testnet trading with the classic strategy flow

  python -m src.main --debate-only --debate-symbol BTC/USDT
      Run one debate round without trade execution

  python -m src.main --backtest --backtest-days 30
      Run a backtest over the last 30 days

  python -m src.main --data-pipeline
      Collect market data only
        """,
    )

    add_common_runtime_args(
        parser,
        strategy_choices=["sma_cross", "bbands", "ai_debate"],
        strategy_help="Trading strategy (default: ai_debate for trading, config value for backtest)",
        default_strategy=None,
        include_interval=True,
        initial_capital_help="Initial capital for dry-run/backtest (default: 10000)",
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
        "--data-pipeline",
        action="store_true",
        help="Run data collection pipeline only",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start monitoring bot only",
    )
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run a single debate without trade execution",
    )
    parser.add_argument(
        "--debate-symbol",
        type=str,
        default=None,
        help="Single symbol for debate-only mode",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory integrations in the trading bot",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Disable news and sentiment enrichment",
    )
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="Disable autotune and weekly optimization checks",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable prompt optimization hooks when supported",
    )

    return parser.parse_args()


def run_data_pipeline(config: Any) -> None:
    """Run the data collection pipeline."""
    from src.data.config import DataConfig as PipelineConfig
    from src.data.quality_gates import QualityGates
    from src.data.questdb_writer import QuestDBWriter
    from src.data.redis_cache import RedisCache
    from src.data.binance_connector import BinanceConnector

    logger.info("🚀 Starting Data Pipeline...")

    # Load pipeline config
    pipeline_cfg = PipelineConfig.load_from_env()

    # Initialize components
    quality_gates = QualityGates(
        max_latency_ms=pipeline_cfg.max_latency_ms,
        z_score_threshold=pipeline_cfg.z_score_threshold,
        max_spread_pct=pipeline_cfg.max_spread_pct,
    )

    questdb = QuestDBWriter(addr=pipeline_cfg.questdb_addr)
    redis_cache = RedisCache(url=pipeline_cfg.redis_url)

    # Convert symbols from trading format (BTC/USDT) to cryptofeed format (BTC-USDT)
    cf_symbols = [normalize_market_symbol(s) for s in config.trading.symbols]

    connector = BinanceConnector(
        symbols=cf_symbols,
        channels=pipeline_cfg.channels,
        candle_interval=pipeline_cfg.candle_interval,
        questdb_writer=questdb,
        redis_cache=redis_cache,
        quality_gates=quality_gates,
    )

    logger.info(f"📊 Collecting data for symbols: {cf_symbols}")
    logger.info("Press Ctrl+C to stop...")

    try:
        connector.start()
    except KeyboardInterrupt:
        logger.info("🛑 Stopping data pipeline...")
        connector.stop()
    finally:
        questdb.close()


def run_backtest(config: Any, *, days: int | None = None) -> None:
    """Run backtest with the selected strategy."""
    run_backtest_from_config(
        config,
        days=days if days is not None else getattr(config, "backtest_days", 90),
        heading="📈 Starting Backtest...",
        output_path=Path(__file__).parent.parent / "logs" / "backtest_results.html",
        chart_saved_message="📊 Results chart saved to {path}",
    )


async def run_debate_only(config: Any, *, symbol: str = "BTC/USDT") -> None:
    """Run a single debate round without entering the trading loop."""
    from src.debate import run_debate as run_debate_facade

    logger.info(f"🧠 Running debate-only for {symbol}...")

    market_data = {
        "price": 67500.0,
        "rsi": 45,
        "macd": 0.0,
        "volume": 1000,
        "bb_upper": 68000,
        "bb_lower": 67000,
    }
    result = run_debate_facade(config, market_data, symbol=symbol)

    logger.info("=" * 60)
    logger.info("🧠 DEBATE RESULT")
    logger.info("=" * 60)
    logger.info(f"  Action:     {result.get('action', 'HOLD')}")
    logger.info(f"  Confidence: {result.get('confidence', 0.0):.1f}%")
    logger.info(f"  Reason:     {result.get('reason', result.get('reasoning', ''))[:200]}...")
    logger.info(f"  Stop Loss:  ${result.get('stop_loss', 0):,.2f}")
    logger.info(f"  Take Profit: ${result.get('take_profit', 0):,.2f}")
    logger.info(f"  Risk:       {result.get('risk_decision', 'APPROVE')}")
    logger.info("=" * 60)


def run_dry_run(config: Any, **kwargs: Any) -> None:
    """Run the real dry-run trading bot loop."""
    config.trading.mode = "dryrun"
    kwargs.pop("mode", None)
    if not kwargs:
        return run_trading_bot(config)
    return run_trading_bot(config, mode="dryrun", **kwargs)


def run_trading_bot(
    config: Any,
    *,
    mode: str | None = None,
    strategy: str | None = None,
    symbols: list[str] | None = None,
    interval: int | None = None,
    enable_memory: bool = True,
    enable_news: bool = True,
    enable_autotune: bool = True,
    bot_class: type[Any] | None = None,
) -> None:
    """Run the real trading bot loop for dryrun/testnet/live modes."""
    if bot_class is None:
        from src.bot import TradingBot as bot_class

    bot = bot_class(
        config=config,
        mode=mode or config.trading.mode,
        strategy=strategy or config.strategy.name,
        symbols=symbols or config.trading.symbols,
        interval=interval if interval is not None else getattr(
            config.trading,
            "interval",
            get_default_loop_interval_seconds(),
        ),
        enable_memory=enable_memory,
        enable_news=enable_news,
        enable_autotune=enable_autotune,
    )

    async def _run() -> None:
        await bot.setup()
        await bot.run()

    asyncio.run(_run())


def run_monitor(config: Any) -> None:
    """Start the Telegram monitoring bot."""
    from src.monitoring import build_telegram_bot

    logger.info("📱 Starting Telegram Monitoring Bot...")

    if not config.monitoring.telegram_bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set. Cannot start monitoring bot.")
        sys.exit(1)

    bot = build_telegram_bot(config)
    if bot is None:
        logger.error("❌ Telegram monitoring config incomplete.")
        sys.exit(1)

    logger.info("✅ Monitoring bot started. Press Ctrl+C to stop...")
    bot.start_polling()


def main(
    *,
    parse_args_fn: Callable[[], argparse.Namespace] | None = None,
    setup_logging_fn: Callable[..., None] | None = None,
    setup_logging_kwargs: dict[str, Any] | None = None,
    config_loader: Callable[..., Any] | None = None,
    run_data_pipeline_fn: Callable[[Any], None] | None = None,
    run_backtest_fn: Callable[..., None] | None = None,
    run_monitor_fn: Callable[[Any], None] | None = None,
    run_dry_run_fn: Callable[..., None] | None = None,
    run_trading_bot_fn: Callable[..., None] | None = None,
    run_debate_only_fn: Callable[..., Any] | None = None,
    bot_class: type[Any] | None = None,
) -> None:
    """Canonical main entry point with injectable seams for compatibility wrappers."""
    parse_args_fn = parse_args if parse_args_fn is None else parse_args_fn
    setup_logging_fn = setup_logging if setup_logging_fn is None else setup_logging_fn
    setup_logging_kwargs = {} if setup_logging_kwargs is None else setup_logging_kwargs
    config_loader = app_config.load_config if config_loader is None else config_loader
    run_data_pipeline_fn = (
        run_data_pipeline if run_data_pipeline_fn is None else run_data_pipeline_fn
    )
    run_backtest_fn = run_backtest if run_backtest_fn is None else run_backtest_fn
    run_monitor_fn = run_monitor if run_monitor_fn is None else run_monitor_fn
    run_dry_run_fn = run_dry_run if run_dry_run_fn is None else run_dry_run_fn
    run_trading_bot_fn = (
        run_trading_bot if run_trading_bot_fn is None else run_trading_bot_fn
    )
    run_debate_only_fn = (
        run_debate_only if run_debate_only_fn is None else run_debate_only_fn
    )

    args = parse_args_fn()
    config = load_runtime_config(
        args,
        default_strategy="ai_debate",
        keep_config_strategy_for_backtest=True,
        config_loader=config_loader,
    )

    # Setup logging
    setup_logging_fn(
        log_level=config.monitoring.log_level,
        **setup_logging_kwargs,
    )

    logger.info("=" * 60)
    logger.info("🚀 AI Autonomous Trading Architecture")
    logger.info("=" * 60)
    logger.info(f"  Mode:     {config.trading.mode}")
    logger.info(f"  Strategy: {config.strategy.name}")
    logger.info(f"  Symbols:  {config.trading.symbols}")
    logger.info(f"  Memory:   {'Disabled' if args.no_memory else 'Enabled'}")
    logger.info(f"  News:     {'Disabled' if args.no_news else 'Enabled'}")
    logger.info(f"  Autotune: {'Disabled' if args.no_autotune else 'Enabled'}")
    if args.optimize:
        logger.info("  Optimize: Enabled")
    logger.info("=" * 60)

    trading_kwargs = {
        "mode": config.trading.mode,
        "strategy": config.strategy.name,
        "symbols": config.trading.symbols,
        "interval": args.interval,
        "enable_memory": not args.no_memory,
        "enable_news": not args.no_news,
        "enable_autotune": not args.no_autotune,
        "bot_class": bot_class,
    }

    if args.data_pipeline:
        run_data_pipeline_fn(config)
    elif args.backtest:
        config.backtest_days = args.backtest_days
        run_backtest_fn(config, days=args.backtest_days)
    elif args.monitor:
        run_monitor_fn(config)
    elif args.debate_only:
        symbol = args.debate_symbol or config.trading.symbols[0]
        asyncio.run(run_debate_only_fn(config, symbol=symbol))
    elif config.trading.mode == "dryrun":
        run_dry_run_fn(config, **trading_kwargs)
    elif config.trading.mode == "testnet":
        logger.info("🔗 Connecting to Binance Testnet...")
        run_trading_bot_fn(config, **trading_kwargs)
    elif config.trading.mode == "live":
        logger.warning("⚠️ LIVE trading mode — use at your own risk!")
        logger.warning("⚠️ Ensure Risk Engine and Kill Switch are configured!")
        run_trading_bot_fn(config, **trading_kwargs)


__all__ = [
    "main",
    "parse_args",
    "run_backtest",
    "run_data_pipeline",
    "run_debate_only",
    "run_dry_run",
    "run_monitor",
    "run_trading_bot",
]


if __name__ == "__main__":
    main()
