"""Monitoring module — Telegram notifications, logging, and alert formatting."""


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "TelegramBot":
        from .telegram_bot import TelegramBot
        return TelegramBot
    if name == "TradingLogger":
        from .trading_logger import TradingLogger
        return TradingLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TelegramBot", "TradingLogger"]
