"""Public runtime contracts for the trading architecture.

Read this module first to understand the high-level bot flow:
market data -> strategy -> debate -> risk -> execution -> memory -> monitor.
Each protocol here describes the smallest public surface another module needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class PortfolioSnapshot:
    """Minimal portfolio state passed between trading modules."""

    cash: float
    total_value: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass(slots=True)
class TradingDecision:
    """Normalized trading decision emitted by debate or strategy layers."""

    symbol: str
    action: str
    confidence: float = 0.0
    reasoning: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradeExecutionResult:
    """Normalized execution result emitted by execution layers."""

    symbol: str
    side: str
    quantity: float
    price: float
    status: str = "executed"
    pnl: float | None = None
    order_info: dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Provides the latest market snapshot for one symbol."""

    async def get(self, symbol: str) -> dict[str, Any]: ...


class StrategyEngine(Protocol):
    """Generates a raw action proposal from market data."""

    def generate(self, symbol: str, market_data: dict[str, Any]) -> str: ...


class DebateService(Protocol):
    """Turns market context into a final trading decision."""

    async def run(
        self,
        market_data: dict[str, Any],
        *,
        symbol: str,
        current_positions: dict[str, Any] | None = None,
        portfolio: PortfolioSnapshot | dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> TradingDecision | dict[str, Any]: ...


class RiskService(Protocol):
    """Approves or rejects a proposed trade against portfolio limits."""

    def check(
        self,
        decision: TradingDecision | dict[str, Any],
        portfolio: PortfolioSnapshot | dict[str, Any],
    ) -> tuple[bool, str]: ...


class ExecutionService(Protocol):
    """Executes an approved trade decision and returns an execution record."""

    async def execute(
        self,
        decision: TradingDecision | dict[str, Any],
        *,
        symbol: str,
        price: float,
    ) -> TradeExecutionResult | dict[str, Any] | None: ...


class MemoryService(Protocol):
    """Persists and recalls trading context across loops."""

    def query(self, symbol: str, market_data: dict[str, Any]) -> str: ...

    async def record(
        self,
        result: TradeExecutionResult | dict[str, Any],
        decision: TradingDecision | dict[str, Any],
        market_data: dict[str, Any],
    ) -> None: ...


class MonitorService(Protocol):
    """Emits alerts and human-facing runtime notifications."""

    async def alert(
        self,
        result: TradeExecutionResult | dict[str, Any],
        decision: TradingDecision | dict[str, Any],
    ) -> None: ...