"""DSPy signatures, output models, and optimization metrics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class MarketInput(dspy.Signature):
        """
        Given market data with technical indicators, predict trading action.

        The input contains current price, technical indicators, recent price
        action, and market context. The output is a structured trading decision.
        """

        market_data = dspy.InputField(
            desc="Market data including price, technical indicators, volume, and context"
        )
        action = dspy.OutputField(desc="Trading action: BUY, SELL, or HOLD")
        confidence = dspy.OutputField(desc="Confidence in the action (0-100)")
        reason = dspy.OutputField(desc="Detailed reasoning for the decision")
        stop_loss = dspy.OutputField(desc="Recommended stop-loss price")
        take_profit = dspy.OutputField(desc="Recommended take-profit price")

else:

    class MarketInput:  # type: ignore[no-redef]
        """Placeholder used only so importing this module gives a clean error later."""


class DebateResultOutput(BaseModel):
    """Structured output from the DSPy debate program."""

    action: str = Field(..., description="BUY, SELL, or HOLD")
    confidence: float = Field(..., ge=0, le=100, description="Confidence level")
    reason: str = Field(..., description="Detailed reasoning")
    stop_loss: float = Field(..., gt=0, description="Stop-loss price")
    take_profit: float = Field(..., gt=0, description="Take-profit price")


class TradeDemonstration(BaseModel):
    """A single trade used as a demonstration for DSPy optimization."""

    market_data_text: str = Field(
        ..., description="Text description of market conditions"
    )
    action: str = Field(..., description="The action that was taken")
    pnl: float = Field(..., description="Realized P&L from this trade")
    was_profitable: bool = Field(
        ..., description="Whether the trade was profitable"
    )


def sharpe_metric(demo: Any, pred: Any, trace: Any = None) -> float:
    """
    DSPy metric that rewards correct action prediction and confidence calibration.
    """
    del trace
    if not hasattr(pred, "action") or not hasattr(demo, "action"):
        return 0.0

    action_match = pred.action.upper() == demo.action.upper()
    if not action_match:
        return 0.0

    confidence = float(getattr(pred, "confidence", 50))
    if 60 <= confidence <= 90:
        return 1.0
    if confidence > 90:
        return 0.8
    return 0.6


def pnl_weighted_metric(demo: Any, pred: Any, trace: Any = None) -> float:
    """DSPy metric weighted by the absolute PnL of the demonstration trade."""
    base = sharpe_metric(demo, pred, trace)
    pnl = float(getattr(demo, "pnl", 0))
    pnl_weight = min(abs(pnl) / 100.0, 2.0)
    return base * (1 + pnl_weight)


__all__ = [
    "DebateResultOutput",
    "MarketInput",
    "TradeDemonstration",
    "dspy",
    "pnl_weighted_metric",
    "sharpe_metric",
]
