"""Public bot facade.

Read this package root first.

Flow owner:
1. `TradingBot.run()` owns the main loop.
2. `TradingBot._process_symbol()` owns the per-symbol pipeline.
3. Legacy bot variants remain available as compatibility exports.
"""

from __future__ import annotations

from typing import Any


__all__ = [
	"TradingBot",
	"AITradingBot",
	"FullTradingBot",
	"create_trading_bot",
]


def create_trading_bot(*args: Any, **kwargs: Any) -> Any:
	"""Create the canonical trading bot implementation."""
	from src.bot.full_trading_bot import FullTradingBot

	return FullTradingBot(*args, **kwargs)


def __getattr__(name: str) -> Any:
	if name == "TradingBot":
		from src.bot.full_trading_bot import FullTradingBot

		return FullTradingBot
	if name == "AITradingBot":
		from src.bot.ai_trading_bot import AITradingBot

		return AITradingBot
	if name == "FullTradingBot":
		from src.bot.full_trading_bot import FullTradingBot

		return FullTradingBot
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
