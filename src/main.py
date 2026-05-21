#!/usr/bin/env python3
"""
AI Autonomous Trading Architecture — Main Entry Point

Usage:
    python -m src.main --mode dryrun          # Run dry-run trading
    python -m src.main --mode testnet         # Run on Binance testnet
    python -m src.main --backtest             # Run backtest
    python -m src.main --data-pipeline        # Run data collection only
    python -m src.main --monitor              # Start monitoring bot only
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Autonomous Trading Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --mode dryrun              # Dry-run with SMA Cross strategy
  python -m src.main --mode testnet             # Live testnet trading
  python -m src.main --backtest                 # Run backtest (last 3 months)
  python -m src.main --data-pipeline            # Collect market data only
  python -m src.main --monitor                  # Start Telegram monitoring bot
  python -m src.main --mode dryrun --strategy bbands  # Use Bollinger Bands strategy
        """,
    )

    add_common_runtime_args(
        parser,
        strategy_choices=["sma_cross", "bbands"],
        strategy_help="Trading strategy (default: sma_cross)",
        default_strategy="sma_cross",
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

    return parser.parse_args()


def run_data_pipeline(config):
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


def run_backtest(config):
    """Run backtest with the selected strategy."""
    run_backtest_from_config(
        config,
        days=getattr(config, "backtest_days", 90),
        heading="📈 Starting Backtest...",
        output_path=Path(__file__).parent.parent / "logs" / "backtest_results.html",
        chart_saved_message="📊 Results chart saved to {path}",
    )


def run_dry_run(config):
    """Run the real dry-run trading bot loop."""
    config.trading.mode = "dryrun"
    return run_trading_bot(config)


def run_trading_bot(config):
    """Run the real trading bot loop for dryrun/testnet/live modes."""
    from src.main_full import FullTradingBot

    bot = FullTradingBot(
        config=config,
        mode=config.trading.mode,
        strategy=config.strategy.name,
        symbols=config.trading.symbols,
        interval=getattr(
            config.trading,
            "interval",
            get_default_loop_interval_seconds(),
        ),
        enable_memory=False,
        enable_news=False,
        enable_autotune=False,
    )

    async def _run() -> None:
        await bot.setup()
        await bot.run()

    asyncio.run(_run())


def run_monitor(config):
    """Start the Telegram monitoring bot."""
    from src.monitoring.telegram_bot import TelegramBot

    logger.info("📱 Starting Telegram Monitoring Bot...")

    if not config.monitoring.telegram_bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set. Cannot start monitoring bot.")
        sys.exit(1)

    bot = TelegramBot(
        bot_token=config.monitoring.telegram_bot_token,
        chat_id=config.monitoring.telegram_chat_id,
    )

    logger.info("✅ Monitoring bot started. Press Ctrl+C to stop...")
    bot.start_polling()


def main():
    args = parse_args()
    config = load_runtime_config(args, config_loader=app_config.load_config)

    # Setup logging
    setup_logging(log_level=config.monitoring.log_level)

    logger.info("=" * 60)
    logger.info("🚀 AI Autonomous Trading Architecture")
    logger.info("=" * 60)
    logger.info(f"  Mode:     {config.trading.mode}")
    logger.info(f"  Strategy: {config.strategy.name}")
    logger.info(f"  Symbols:  {config.trading.symbols}")
    logger.info("=" * 60)

    # Route to appropriate mode
    if args.data_pipeline:
        run_data_pipeline(config)
    elif args.backtest:
        config.backtest_days = args.backtest_days
        run_backtest(config)
    elif args.monitor:
        run_monitor(config)
    elif config.trading.mode == "dryrun":
        run_dry_run(config)
    elif config.trading.mode == "testnet":
        logger.info("🔗 Connecting to Binance Testnet...")
        run_trading_bot(config)
    elif config.trading.mode == "live":
        logger.warning("⚠️ LIVE trading mode — use at your own risk!")
        logger.warning("⚠️ Ensure Risk Engine and Kill Switch are configured!")
        run_trading_bot(config)


if __name__ == "__main__":
    main()
