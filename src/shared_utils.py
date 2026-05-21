"""Shared utility helpers for small cross-cutting runtime patterns."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime
from typing import TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


def normalize_market_symbol(symbol: str, separator: str = "-") -> str:
    """Convert market symbols like BTC/USDT into exchange/feed-safe forms."""
    return symbol.replace("/", separator)


def trim_mapping_size(
    mapping: MutableMapping[KeyT, ValueT],
    max_entries: int,
) -> None:
    """Evict oldest inserted items until the mapping size is within bounds."""
    if max_entries < 0:
        raise ValueError("max_entries must be non-negative")

    while len(mapping) > max_entries:
        oldest_key = next(iter(mapping))
        mapping.pop(oldest_key, None)


def parse_iso_timestamp(ts: str | None) -> float:
    """Parse ISO-8601 timestamps into unix seconds, tolerating trailing Z."""
    if not ts:
        return 0.0

    try:
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0