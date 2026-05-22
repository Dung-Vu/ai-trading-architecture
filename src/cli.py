"""Compatibility shim for the legacy full-bot CLI module."""

from __future__ import annotations

from src.main_full import FullTradingBot, main, parse_args, run_backtest


__all__ = ["FullTradingBot", "main", "parse_args", "run_backtest"]