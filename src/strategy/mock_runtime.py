"""Minimal runtime shims for constructing Lumibot strategies in tests and tools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class MockFilledPositions(list):
    """Minimal filled-position container matching Lumibot expectations."""

    def get_list(self) -> "MockFilledPositions":
        return self


class MockDataSource:
    """Minimal data source used when a strategy is instantiated off-broker."""

    SOURCE = "MOCK"

    def __init__(self) -> None:
        self.datetime_start = None
        self.datetime_end = None
        self._data_store: dict[str, Any] = {}

    def get_datetime(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)


class MockBroker:
    """Minimal broker shim that satisfies Lumibot strategy construction."""

    IS_BACKTESTING_BROKER = True

    def __init__(self) -> None:
        self.name = "mock"
        self.data_source = MockDataSource()
        self._filled_positions = MockFilledPositions()
        self.quote_assets: set[str] = set()
        self.market = "24/7"

    def _add_subscriber(self, subscriber: Any) -> None:
        del subscriber


def ensure_strategy_runtime_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject mock runtime objects when a strategy is built off the Lumibot loop."""
    runtime_kwargs = dict(kwargs)
    broker = runtime_kwargs.get("broker")
    if broker is None:
        broker = MockBroker()
        runtime_kwargs["broker"] = broker

    if runtime_kwargs.get("data_source") is None:
        runtime_kwargs["data_source"] = broker.data_source

    return runtime_kwargs