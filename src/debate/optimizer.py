"""
Compatibility facade for DSPy prompt optimization.

The implementation is split across ``dspy_infra`` and ``prompt_optimizer``.
"""

from __future__ import annotations

from src.debate.dspy_infra import (
    DebateResultOutput,
    MarketInput,
    TradeDemonstration,
    dspy,
    pnl_weighted_metric,
    sharpe_metric,
)
from src.debate.prompt_optimizer import DSPyOptimizer


__all__ = [
    "DSPyOptimizer",
    "DebateResultOutput",
    "MarketInput",
    "TradeDemonstration",
    "dspy",
    "pnl_weighted_metric",
    "sharpe_metric",
]
