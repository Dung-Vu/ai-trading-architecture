"""Bot implementations."""

from __future__ import annotations

from typing import Any


__all__ = ["AITradingBot", "FullTradingBot"]


def __getattr__(name: str) -> Any:
	if name == "AITradingBot":
		from src.bot.ai_trading_bot import AITradingBot

		return AITradingBot
	if name == "FullTradingBot":
		from src.bot.full_trading_bot import FullTradingBot

		return FullTradingBot
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
