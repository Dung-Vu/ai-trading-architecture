"""Public monitoring facade.

Read this package root for alert delivery and human-facing notifications.
`build_telegram_bot(config)` is the canonical constructor for runtime alerts.
"""

from __future__ import annotations

from typing import Any


def build_telegram_bot(config: Any):
    """Build a TelegramBot when monitoring config is complete, else return None."""
    monitoring = getattr(config, "monitoring", config)
    token = getattr(monitoring, "telegram_bot_token", "")
    chat_id = getattr(monitoring, "telegram_chat_id", "")
    if not token or not chat_id:
        return None

    from .telegram_bot import TelegramBot

    return TelegramBot(bot_token=token, chat_id=chat_id)


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "TelegramBot":
        from .telegram_bot import TelegramBot
        return TelegramBot
    if name == "TradingLogger":
        from .trading_logger import TradingLogger
        return TradingLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TelegramBot", "TradingLogger", "build_telegram_bot"]
