"""
AI Debate Engine — LangGraph-based multi-agent debate for trading decisions.

Architecture:
    Market Data → Bull Agent → Bear Agent → Devil's Advocate
              → (repeat N rounds)
              → Judge/Synthesis → Risk Manager → Final Decision

Anti-Sycophancy Rules (enforced in every agent prompt):
    1. Each round MUST provide NEW evidence.
    2. MUST rebut the strongest opposing point.
    3. Stance changes require explicit justification.
    4. All claims MUST have specific numbers.
    5. FORBIDDEN to agree unless genuinely convinced.

Usage:
    >>> from debate import DebateEngine, DebateConfig
    >>> from debate.llm_client import LLMClient
    >>>
    >>> llm = LLMClient(model="bailian/qwen3.6-plus")
    >>> config = DebateConfig(max_rounds=3, symbols=["BTC/USDT"])
    >>> engine = DebateEngine(config, llm)
    >>> result = engine.run_debate(
    ...     market_data={"price": 67500, "rsi": 45, "volume": ...},
    ...     symbol="BTC/USDT",
    ... )
    >>> print(result.action, result.confidence)

Public facade:
    `run_debate(config, market_data, symbol=...) -> dict`
    lets callers use the debate package without first learning its builders.
"""

from __future__ import annotations

from typing import Any

from .agents import (
    BaseAgent,
    BearAgent,
    BullAgent,
    DevilsAdvocate,
    JudgeAgent,
    RiskManagerAgent,
)
from .llm_client import LLMClient
from .models import (
    AgentOutput,
    DebateConfig,
    DebateResult,
    DebateRound,
)

__all__ = [
    # Engine
    "DebateEngine",
    "DebateState",
    "run_debate",
    # Models
    "DebateConfig",
    "DebateResult",
    "DebateRound",
    "AgentOutput",
    # Agents
    "BaseAgent",
    "BullAgent",
    "BearAgent",
    "DevilsAdvocate",
    "JudgeAgent",
    "RiskManagerAgent",
    # LLM
    "LLMClient",
    # Optimizer
    "DSPyOptimizer",
]


def run_debate(
    config: Any,
    market_data: dict[str, Any],
    *,
    symbol: str = "BTC/USDT",
    symbols: list[str] | None = None,
    current_positions: dict[str, Any] | None = None,
    portfolio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a debate through the public package facade and return a normalized dict."""
    from .runtime import build_debate_engine, normalize_debate_result

    engine, _ = build_debate_engine(config, symbols or [symbol])
    result = engine.run_debate(
        market_data=market_data,
        current_positions=current_positions or {},
        portfolio=portfolio or {},
        symbol=symbol,
    )
    return normalize_debate_result(result, include_reason_alias=True)


def __getattr__(name: str):
    """Lazy imports for optional debate submodules."""
    if name in {"DebateEngine", "DebateState"}:
        from .debate_engine import DebateEngine, DebateState

        return {"DebateEngine": DebateEngine, "DebateState": DebateState}[name]
    if name == "DSPyOptimizer":
        from .optimizer import DSPyOptimizer
        return DSPyOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
