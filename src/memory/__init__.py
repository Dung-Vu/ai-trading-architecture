"""
Memory & Learning Layer — Trade history, weekly review, self-learning memory,
and knowledge graph.

Architecture:
    TradeMemory      → PostgreSQL (trades, debates) + Redis (hot cache)
    WeeklyReviewer   → Generates weekly performance reports & insights
    Mem0Memory       → Semantic vector memory (Qdrant) for trade recall
    KnowledgeGraph   → Pattern graph for condition→action→outcome lookup

Usage:
    >>> from src.memory import TradeMemory, Mem0Memory, KnowledgeGraph
    >>> memory = TradeMemory()
    >>> await memory.connect()
    >>> await memory.log_trade({...})
    >>> mem0 = Mem0Memory(qdrant_url="http://localhost:6333")
    >>> mem0.add_trade_memory(trade_data, debate_data)
    >>> kg = KnowledgeGraph()
    >>> kg.add_pattern("RSI < 30", "BUY", "win", 0.7)

Public facade:
    `build_trade_memory(config)` and `build_knowledge_graph()` expose the main
    runtime constructors without requiring callers to read internal modules.
"""

from __future__ import annotations

from typing import Any

from src.config import get_default_database_url, get_default_redis_url

from .interfaces import TradeMemoryInterface
from .trade_memory import TradeMemory
from .weekly_review import WeeklyReviewer
from .mem0_memory import Mem0Memory
from .knowledge_graph import KnowledgeGraph


def build_trade_memory(config: Any) -> TradeMemory:
    """Create the standard trade-memory adapter from the app config surface."""
    return TradeMemory(
        db_url=getattr(config, "database_url", get_default_database_url()),
        redis_url=getattr(config, "redis_url", get_default_redis_url()),
    )


def build_knowledge_graph() -> KnowledgeGraph:
    """Create an in-process knowledge graph for pattern recall."""
    return KnowledgeGraph()

__all__ = [
    "TradeMemoryInterface",
    "TradeMemory",
    "WeeklyReviewer",
    "Mem0Memory",
    "KnowledgeGraph",
    "build_trade_memory",
    "build_knowledge_graph",
]
