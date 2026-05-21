"""Shared helpers for CLI parsing, config overrides, and backtest orchestration."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.reports.utils import extract_equity, extract_trades
from src.config import (
    get_default_initial_capital,
    env_str,
    get_default_loop_interval_seconds,
    get_trading_symbols,
    load_config,
)


def add_common_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    strategy_choices: list[str],
    strategy_help: str,
    default_strategy: str | None,
    include_interval: bool = False,
    symbols_help: str = "Trading symbols (default: BTC/USDT ETH/USDT)",
    initial_capital_help: str = "Initial capital (default: 10000)",
) -> None:
    """Add the common trading-runtime CLI flags shared by entry points."""
    parser.add_argument(
        "--mode",
        choices=["dryrun", "testnet", "live"],
        default=env_str("TRADING_MODE", "dryrun"),
        help="Trading mode (default: dryrun)",
    )
    parser.add_argument(
        "--strategy",
        choices=strategy_choices,
        default=default_strategy,
        help=strategy_help,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=get_trading_symbols(),
        help=symbols_help,
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
        default=env_str("LOG_LEVEL", "INFO"),
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=get_default_initial_capital(),
        help=initial_capital_help,
    )

    if include_interval:
        parser.add_argument(
            "--interval",
            type=int,
            default=get_default_loop_interval_seconds(),
            help="Trading loop interval in seconds (default: 60)",
        )


def load_runtime_config(
    args: argparse.Namespace,
    *,
    default_strategy: str | None = None,
    keep_config_strategy_for_backtest: bool = False,
    config_loader: Callable[..., Any] = load_config,
) -> Any:
    """Load the shared app config and apply standard CLI overrides."""
    config = config_loader(config_path=getattr(args, "config", None))

    if hasattr(args, "mode"):
        config.trading.mode = args.mode
    if hasattr(args, "initial_capital"):
        config.trading.initial_capital = args.initial_capital
    if hasattr(args, "log_level"):
        config.monitoring.log_level = args.log_level

    symbols = getattr(args, "symbols", None)
    if symbols:
        config.trading.symbols = symbols

    strategy = getattr(args, "strategy", None)
    if strategy:
        config.strategy.name = strategy
    elif default_strategy and not (
        keep_config_strategy_for_backtest and getattr(args, "backtest", False)
    ):
        config.strategy.name = default_strategy

    return config


def build_backtest_dates(
    days: int,
    *,
    use_utc: bool = False,
) -> tuple[datetime, datetime]:
    """Return a backtest date window ending at now."""
    end_date = datetime.now(timezone.utc) if use_utc else datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def resolve_standard_backtest_spec(
    config: Any,
    *,
    symbol: str | None = None,
) -> tuple[type[Any], str, str, dict[str, Any]]:
    """Return the classic backtest strategy class, symbol, and parameters."""
    from src.strategy.bbands import BBandsStrategy
    from src.strategy.sma_cross import SMACrossStrategy

    active_symbol = symbol or (
        config.trading.symbols[0] if config.trading.symbols else "BTC/USDT"
    )
    strategy_name = config.strategy.name

    if strategy_name == "sma_cross":
        return (
            SMACrossStrategy,
            strategy_name,
            active_symbol,
            {
                "symbol": active_symbol,
                "sma_fast": config.strategy.sma_fast,
                "sma_slow": config.strategy.sma_slow,
                "rsi_period": config.strategy.rsi_period,
            },
        )

    if strategy_name == "bbands":
        return (
            BBandsStrategy,
            strategy_name,
            active_symbol,
            {"symbol": active_symbol},
        )

    raise ValueError(f"Unknown strategy for backtest: {strategy_name}")


def build_backtest_runner(
    *,
    strategy_class: type[Any],
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    parameters: dict[str, Any],
    initial_capital: float,
) -> Any:
    """Create a BacktestRunner with the shared parameter wiring."""
    from src.strategy.backtest import BacktestRunner

    return BacktestRunner(
        strategy_class=strategy_class,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        initial_capital=initial_capital,
    )


def summarize_backtest_results(results: dict[str, Any]) -> dict[str, Any]:
    """Compute the shared metric summary for a backtest result payload."""
    from src.strategy.metrics import MetricsCalculator

    equity_curve = extract_equity(results)
    if equity_curve is None:
        equity_curve = pd.Series(dtype=float)

    return MetricsCalculator.summarize(
        extract_trades(results),
        equity_curve,
    )


def run_backtest_from_config(
    config: Any,
    *,
    days: int,
    heading: str,
    output_path: Path | None,
    use_utc: bool = False,
    chart_saved_message: str = "📊 Chart saved to {path}",
    strategy_class: type[Any] | None = None,
    strategy_name: str | None = None,
    symbol: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run a backtest from config using either default or explicit strategy wiring."""
    if strategy_class is None or strategy_name is None or parameters is None:
        (
            resolved_strategy_class,
            resolved_strategy_name,
            resolved_symbol,
            resolved_parameters,
        ) = resolve_standard_backtest_spec(config, symbol=symbol)
        strategy_class = strategy_class or resolved_strategy_class
        strategy_name = strategy_name or resolved_strategy_name
        symbol = symbol or resolved_symbol
        parameters = parameters or resolved_parameters

    start_date, end_date = build_backtest_dates(days, use_utc=use_utc)
    return run_backtest_workflow(
        heading=heading,
        strategy_class=strategy_class,
        strategy_name=strategy_name,
        symbol=symbol or (config.trading.symbols[0] if config.trading.symbols else "BTC/USDT"),
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        initial_capital=config.trading.initial_capital,
        output_path=output_path,
        chart_saved_message=chart_saved_message,
    )


def log_backtest_metrics(metrics: dict[str, Any]) -> None:
    """Emit the standard backtest summary block."""
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


def run_backtest_workflow(
    *,
    heading: str,
    strategy_class: type[Any],
    strategy_name: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    parameters: dict[str, Any],
    initial_capital: float,
    output_path: Path | None = None,
    empty_results_message: str = "⚠️ No backtest results returned",
    chart_saved_message: str = "📊 Chart saved to {path}",
) -> dict[str, Any] | None:
    """Run the shared backtest flow and emit the common result logging."""
    logger.info(heading)
    logger.info(
        f"📅 Backtest: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
    )
    logger.info(f"📊 Symbol: {symbol}")
    logger.info(f"📋 Strategy: {strategy_name}")
    logger.info(f"💰 Capital: ${initial_capital:,.2f}")

    runner = build_backtest_runner(
        strategy_class=strategy_class,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        initial_capital=initial_capital,
    )
    results = runner.run()
    if not results:
        logger.warning(empty_results_message)
        return None

    metrics = summarize_backtest_results(results)
    log_backtest_metrics(metrics)

    if output_path is not None:
        runner.plot_results(output_path=str(output_path))
        logger.info(chart_saved_message.format(path=output_path))

    return metrics
