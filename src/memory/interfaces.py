"""Abstract contracts for the persistent trade-memory layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trade_memory import PerformanceSummary


class TradeMemoryInterface(ABC):
    """Abstract contract for trade logging and analytics backends."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the backing stores."""

    @abstractmethod
    async def close(self) -> None:
        """Close any open backing-store resources."""

    @abstractmethod
    async def log_trade(self, trade: dict[str, Any]) -> int:
        """Persist a trade record and return its identifier."""

    @abstractmethod
    async def log_debate(self, debate_result: dict[str, Any]) -> int:
        """Persist a debate record and return its identifier."""

    @abstractmethod
    async def get_trade_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        strategy: str | None = None,
        limit: int = 100,
        before_cursor: tuple[datetime, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical trade records matching the provided filters."""

    @abstractmethod
    async def get_performance_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> "PerformanceSummary":
        """Return aggregate trading performance over a time window."""

    @abstractmethod
    async def get_strategy_performance(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, "PerformanceSummary"]:
        """Return aggregate performance grouped by strategy name."""

    @abstractmethod
    async def get_trade_patterns(self) -> dict[str, Any]:
        """Return mined trade patterns for review or optimization."""