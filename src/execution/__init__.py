"""Execution module for crypto trading bot."""


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "ExchangeClient":
        from .exchange_client import ExchangeClient
        return ExchangeClient
    if name == "OrderManager":
        from .order_manager import OrderManager
        return OrderManager
    if name == "DryRunExecutor":
        from .dry_run import DryRunExecutor
        return DryRunExecutor
    if name == "PositionSizer":
        from .position_sizer import PositionSizer
        return PositionSizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ExchangeClient",
    "OrderManager",
    "DryRunExecutor",
    "PositionSizer",
]
