"""Public execution facade.

Read this package root for the order-execution boundary.
`build_dry_run_executor(config)` is the smallest constructor most callers need.
"""

from __future__ import annotations

from typing import Any


def build_dry_run_executor(config: Any):
    """Create the standard dry-run executor from the app config surface."""
    from .dry_run import DryRunExecutor

    trading = getattr(config, "trading", None)
    initial_balance = getattr(trading, "initial_capital", None)
    if initial_balance is None:
        initial_balance = getattr(config, "initial_capital", 10_000.0)
    return DryRunExecutor(initial_balance=float(initial_balance or 0.0))


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
    "build_dry_run_executor",
]
