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
"""

from .interfaces import TradeMemoryInterface
from .trade_memory import TradeMemory
from .weekly_review import WeeklyReviewer
from .mem0_memory import Mem0Memory
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "TradeMemoryInterface",
    "TradeMemory",
    "WeeklyReviewer",
    "Mem0Memory",
    "KnowledgeGraph",
]
