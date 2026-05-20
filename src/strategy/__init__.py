"""Strategy module for crypto trading bot.

Provides base strategy classes, concrete implementations,
backtest runner, and performance metrics calculator.
"""


def __getattr__(name):
    """Lazy imports to avoid requiring all dependencies at package level."""
    if name == "BaseStrategy":
        from .base import BaseStrategy
        return BaseStrategy
    if name == "SMACrossStrategy":
        from .sma_cross import SMACrossStrategy
        return SMACrossStrategy
    if name == "BBandsStrategy":
        from .bbands import BBandsStrategy
        return BBandsStrategy
    if name == "BacktestRunner":
        from .backtest import BacktestRunner
        return BacktestRunner
    if name == "MetricsCalculator":
        from .metrics import MetricsCalculator
        return MetricsCalculator
    if name == "ParameterOptimizer":
        from .optimizer import ParameterOptimizer
        return ParameterOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseStrategy",
    "SMACrossStrategy",
    "BBandsStrategy",
    "BacktestRunner",
    "MetricsCalculator",
    "ParameterOptimizer",
]
