from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.knowledge_graph import KnowledgeGraph
from src.memory.mem0_memory import _InMemoryStore
from src.memory.trade_memory import TradeMemory


class _PoolAcquireContext:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, connection):
        self._connection = connection

    def acquire(self):
        return _PoolAcquireContext(self._connection)


def test_knowledge_graph_returns_stable_pattern_ids():
    graph = KnowledgeGraph()
    pattern_id = graph.add_pattern(
        condition="RSI below 30 and volume spike",
        action="BUY",
        outcome="win",
        confidence=0.8,
    )

    first = graph.query_pattern("BTC has RSI below 30")
    second = graph.query_pattern("BTC has RSI below 30")

    assert first[0]["pattern_id"] == pattern_id
    assert second[0]["pattern_id"] == pattern_id


def test_in_memory_store_reindexes_search_after_update():
    store = _InMemoryStore()
    memory_id = store.add({
        "symbol": "BTC/USDT",
        "decision_reasoning": "oversold bounce with strong volume",
    })
    store.add({
        "symbol": "ETH/USDT",
        "decision_reasoning": "trend breakout continuation",
    })

    initial = store.search("oversold BTC", limit=5)
    assert initial[0]["_id"] == memory_id

    assert store.update(memory_id, {"outcome_notes": "momentum failure after entry"})

    updated = store.search("failure", limit=5)
    assert updated[0]["_id"] == memory_id


@pytest.mark.asyncio
async def test_trade_memory_history_supports_strategy_and_cursor_filters():
    connection = MagicMock()
    connection.fetch = AsyncMock(return_value=[{
        "id": 5,
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "symbol": "BTC/USDT",
        "side": "BUY",
        "quantity": 0.1,
        "price": 100.0,
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "strategy": "ai_debate",
        "mode": "dryrun",
        "ai_confidence": 75.0,
        "debate_result": '{"decision": "BUY"}',
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "order_id": "order-1",
    }])

    memory = TradeMemory(enable_redis=False)
    memory._connected = True
    memory._pg_pool = _FakePool(connection)

    cursor_timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    rows = await memory.get_trade_history(
        symbol="BTC/USDT",
        strategy="ai_debate",
        before_cursor=(cursor_timestamp, 42),
        limit=25,
    )

    query, *params = connection.fetch.call_args.args

    assert "symbol = $1" in query
    assert "strategy = $2" in query
    assert "(timestamp, id) < ($3, $4)" in query
    assert "ORDER BY timestamp DESC, id DESC" in query
    assert params == ["BTC/USDT", "ai_debate", cursor_timestamp, 42, 25]
    assert rows[0]["debate_result"]["decision"] == "BUY"
