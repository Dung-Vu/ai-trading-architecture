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
    >>> llm = LLMClient(model="anthropic/claude-sonnet-4")
    >>> config = DebateConfig(max_rounds=3, symbols=["BTC/USDT"])
    >>> engine = DebateEngine(config, llm)
    >>> result = engine.run_debate(
    ...     market_data={"price": 67500, "rsi": 45, "volume": ...},
    ...     symbol="BTC/USDT",
    ... )
    >>> print(result.action, result.confidence)
"""

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


def __getattr__(name: str):
    """Lazy imports for optional debate submodules."""
    if name in {"DebateEngine", "DebateState"}:
        from .debate_engine import DebateEngine, DebateState

        return {"DebateEngine": DebateEngine, "DebateState": DebateState}[name]
    if name == "DSPyOptimizer":
        from .optimizer import DSPyOptimizer
        return DSPyOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
