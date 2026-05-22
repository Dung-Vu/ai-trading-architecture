"""Compatibility shim for the legacy lean AI CLI module."""

from __future__ import annotations

from src.main_ai import AITradingBot, main, parse_args, run_backtest, run_debate_only


__all__ = ["AITradingBot", "main", "parse_args", "run_backtest", "run_debate_only"]
