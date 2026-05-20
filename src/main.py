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

from loguru import logger


def setup_logging(log_level: str = "INFO"):
    """Configure loguru logging."""
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console output
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # File output (daily rotation, 30 days retention)
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        enqueue=True,
    )

    # Error-only file
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="90 days",
        level="ERROR",
        enqueue=True,
    )


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

    parser.add_argument(
        "--mode",
        choices=["dryrun", "testnet", "live"],
        default="dryrun",
        help="Trading mode (default: dryrun)",
    )
    parser.add_argument(
        "--strategy",
        choices=["sma_cross", "bbands"],
        default="sma_cross",
        help="Trading strategy (default: sma_cross)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trading symbols (default: BTC/USDT ETH/USDT)",
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
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml config file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000,
        help="Initial capital for dry-run/backtest (default: 10000)",
    )

    return parser.parse_args()


def run_data_pipeline(config):
    """Run the data collection pipeline."""
    from src.data.binance_connector import BinanceConnector
    from src.data.config import DataConfig as PipelineConfig
    from src.data.quality_gates import QualityGates
    from src.data.questdb_writer import QuestDBWriter
    from src.data.redis_cache import RedisCache

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
    cf_symbols = [s.replace("/", "-") for s in config.trading.symbols]
    pipeline_cfg.symbols = cf_symbols

    connector = BinanceConnector(
        config=pipeline_cfg,
        questdb_writer=questdb,
        redis_cache=redis_cache,
        quality_gates=quality_gates,
    )

    logger.info(f"📊 Collecting data for symbols: {cf_symbols}")
    logger.info("Press Ctrl+C to stop...")

    try:
        questdb.connect()
        asyncio.run(redis_cache.connect())
        connector.start()
    except KeyboardInterrupt:
        logger.info("🛑 Stopping data pipeline...")
        connector.stop()
    finally:
        asyncio.run(redis_cache.close())
        questdb.close()


def run_backtest(config):
    """Run backtest with the selected strategy."""
    from src.strategy.backtest import BacktestRunner
    from src.strategy.bbands import BBandsStrategy
    from src.strategy.metrics import MetricsCalculator
    from src.strategy.sma_cross import SMACrossStrategy

    logger.info("📈 Starting Backtest...")

    # Select strategy
    if config.strategy.name == "sma_cross":
        strategy_class = SMACrossStrategy
    elif config.strategy.name == "bbands":
        strategy_class = BBandsStrategy
    else:
        raise ValueError(f"Unknown strategy: {config.strategy.name}")

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=getattr(config, "backtest_days", 90))

    logger.info(f"📅 Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"📊 Symbol: {config.trading.symbols[0]}")
    logger.info(f"💰 Initial capital: ${config.trading.initial_capital:,.2f}")

    # Parameters
    params = {
        "symbol": config.trading.symbols[0],
        "sma_fast": config.strategy.sma_fast,
        "sma_slow": config.strategy.sma_slow,
        "rsi_period": config.strategy.rsi_period,
    }

    runner = BacktestRunner(
        strategy_class=strategy_class,
        symbol=config.trading.symbols[0],
        start_date=start_date,
        end_date=end_date,
        parameters=params,
        initial_capital=config.trading.initial_capital,
    )

    results = runner.run()

    if results:
        metrics = MetricsCalculator.summarize(
            results.get("trades", []),
            results.get("equity_curve", None),
        )

        logger.info("=" * 60)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"  Win Rate:         {metrics.get('win_rate_pct', 0):.1f}%")
        logger.info(f"  Total Trades:     {metrics.get('total_trades', 0)}")
        logger.info(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        logger.info("=" * 60)

        # Save plot
        output_path = Path(__file__).parent.parent / "logs" / "backtest_results.html"
        runner.plot_results(output_path=str(output_path))
        logger.info(f"📊 Results chart saved to {output_path}")
    else:
        logger.warning("⚠️ No backtest results returned")


def run_dry_run(config):
    """Run dry-run trading."""
    from src.execution.dry_run import DryRunExecutor
    from src.monitoring.trading_logger import TradingLogger
    from src.risk.kill_switch import KillSwitch
    from src.risk.risk_engine import RiskEngine

    logger.info("🤖 Starting Dry-Run Trading...")
    logger.info(f"💰 Initial capital: ${config.trading.initial_capital:,.2f}")
    logger.info(f"📊 Symbols: {config.trading.symbols}")
    logger.info(f"📋 Strategy: {config.strategy.name}")

    # Initialize components
    trading_logger = TradingLogger()
    trading_logger.setup()

    risk_engine = RiskEngine(
        max_daily_loss_pct=config.risk.max_daily_loss_pct / 100,
        max_drawdown_pct=config.risk.max_drawdown_pct / 100,
        max_position_pct=config.risk.max_position_pct / 100,
        max_leverage=config.risk.max_leverage,
    )
    risk_engine.update_peak_equity(config.trading.initial_capital)
    logger.debug(
        f"RiskEngine initialized with drawdown={risk_engine.get_status().current_drawdown_pct:.2%}"
    )

    kill_switch = KillSwitch()
    dry_run_executor = DryRunExecutor(initial_balance=config.trading.initial_capital)

    logger.info("✅ Dry-run ready. Press Ctrl+C to stop...")
    logger.info("ℹ️  This simulates trading without real orders.")

    # In a full implementation, you'd loop here and call strategy logic
    # For now, this sets up the infrastructure
    try:
        while not kill_switch.is_active():
            # Trading loop would go here
            import time
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("🛑 Stopping dry-run trading...")

    # Final report
    portfolio = dry_run_executor.get_portfolio()
    trades = dry_run_executor.get_trade_log()

    logger.info("=" * 60)
    logger.info("📊 DRY-RUN SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total Value:    ${portfolio['total_value']:,.2f}")
    logger.info(f"  Cash:           ${portfolio['cash']:,.2f}")
    logger.info(f"  Total Trades:   {len(trades)}")
    logger.info("=" * 60)


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


def run_unified_bot(config, args) -> None:
    """Run the production unified bot for exchange-backed modes."""
    from src.main_full import FullTradingBot

    bot = FullTradingBot(
        config=config,
        mode=config.trading.mode,
        strategy=config.strategy.name,
        symbols=config.trading.symbols,
        interval=getattr(args, "interval", 60),
    )

    async def _run() -> None:
        await bot.setup()
        await bot.run()

    asyncio.run(_run())


def main():
    args = parse_args()

    # Load config
    from src.config import load_config
    config = load_config(config_path=args.config)

    # Override with CLI args
    config.trading.mode = args.mode
    config.trading.initial_capital = args.initial_capital
    config.monitoring.log_level = args.log_level

    if args.symbols:
        config.trading.symbols = args.symbols
    if args.strategy:
        config.strategy.name = args.strategy

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
        run_unified_bot(config, args)
    elif config.trading.mode == "live":
        logger.warning("⚠️ LIVE trading mode — use at your own risk!")
        logger.warning("⚠️ Ensure Risk Engine and Kill Switch are configured!")
        run_unified_bot(config, args)


if __name__ == "__main__":
    main()
