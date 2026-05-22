from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.mem0_memory import Mem0Memory
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
        self.close = AsyncMock()

    def acquire(self):
        return _PoolAcquireContext(self._connection)


class _FakeMem0Client:
    def __init__(self, config):
        self.config = config
        self._metadata = {}

    def add(self, _memory_text, user_id, metadata):
        memory_id = f"{user_id}-1"
        self._metadata[memory_id] = metadata
        return {"memory_id": memory_id}

    def search(self, query, limit):
        del query, limit
        memory_id, metadata = next(iter(self._metadata.items()))
        return [{"memory_id": memory_id, "score": 0.91, "metadata": metadata}]


class _FailingMem0Client(_FakeMem0Client):
    def add(self, _memory_text, user_id, metadata):
        del _memory_text, user_id, metadata
        raise RuntimeError("mem0 add failed")


@pytest.mark.asyncio
async def test_trade_memory_connects_logs_trade_and_caches_latest_trade():
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value=123)
    pool = _FakePool(connection)

    redis_client = MagicMock()
    redis_client.ping = AsyncMock()
    redis_client.hset = AsyncMock()
    redis_client.expire = AsyncMock()
    redis_client.close = AsyncMock()

    memory = TradeMemory(
        db_url="postgresql://postgres:postgres@localhost:5432/trading_db",
        redis_url="redis://localhost:6379",
        enable_redis=True,
    )
    memory._ensure_schema = AsyncMock()
    fake_asyncpg = SimpleNamespace(create_pool=AsyncMock(return_value=pool))
    fake_aioredis = SimpleNamespace(from_url=MagicMock(return_value=redis_client))

    with patch("src.memory.trade_memory.asyncpg", fake_asyncpg), patch(
        "src.memory.trade_memory.aioredis",
        fake_aioredis,
    ):
        await memory.connect()
        row_id = await memory.log_trade({
            "timestamp": datetime(2026, 5, 20, tzinfo=timezone.utc).isoformat(),
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 65000.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "strategy": "ai_debate",
            "mode": "testnet",
            "ai_confidence": 82.0,
            "debate_result": {"action": "BUY", "confidence": 82},
            "stop_loss": 62000.0,
            "take_profit": 70000.0,
            "order_id": "order-123",
        })
        await memory.close()

    assert row_id == 123
    memory._ensure_schema.assert_awaited_once()
    connection.fetchval.assert_awaited_once()
    redis_client.hset.assert_awaited_once()
    redis_client.expire.assert_awaited_once_with("trade:latest:BTC/USDT", 86400)
    pool.close.assert_awaited_once()
    redis_client.close.assert_awaited_once()


def test_mem0_memory_uses_vector_store_client_for_add_and_search():
    with patch("src.memory.mem0_memory.MEM0_AVAILABLE", True), patch(
        "src.memory.mem0_memory.QDRANT_AVAILABLE",
        True,
    ), patch(
        "src.memory.mem0_memory.Mem0Client",
        _FakeMem0Client,
    ):
        memory = Mem0Memory(qdrant_url="http://qdrant:6333")

        memory_id = memory.add_trade_memory(
            trade={
                "timestamp": datetime(2026, 5, 20, tzinfo=timezone.utc).isoformat(),
                "symbol": "BTC/USDT",
                "side": "BUY",
                "price": 65000.0,
                "quantity": 0.1,
                "indicators": {"rsi": 28, "sma_fast": 64000, "sma_slow": 64500},
                "market_conditions": {"trend": "uptrend", "volume_high": True},
                "strategy": "ai_debate",
                "stop_loss": 62000.0,
                "take_profit": 70000.0,
            },
            debate={
                "action": "BUY",
                "confidence": 84,
                "reasoning": "Oversold bounce with strong volume.",
                "risk_decision": "APPROVE",
            },
        )

        results = memory.query_similar_trades("BTC oversold bounce", limit=1)

    assert memory_id == "trader_BTC/USDT-1"
    assert memory.get_stats()["store_type"] == "mem0_qdrant"
    assert results[0]["memory_id"] == memory_id
    assert results[0]["trade_data"]["symbol"] == "BTC/USDT"
    assert results[0]["debate_action"] == "BUY"
    assert results[0]["similarity_score"] == 0.91


def test_mem0_memory_falls_back_after_runtime_add_failure():
    with patch("src.memory.mem0_memory.MEM0_AVAILABLE", True), patch(
        "src.memory.mem0_memory.QDRANT_AVAILABLE",
        True,
    ), patch(
        "src.memory.mem0_memory.Mem0Client",
        _FailingMem0Client,
    ):
        memory = Mem0Memory(qdrant_url="http://qdrant:6333")

        memory_id = memory.add_trade_memory(
            trade={
                "timestamp": datetime(2026, 5, 20, tzinfo=timezone.utc).isoformat(),
                "symbol": "BTC/USDT",
                "side": "BUY",
                "price": 65000.0,
                "quantity": 0.1,
                "indicators": {"rsi": 28},
                "market_conditions": {"trend": "uptrend", "volume_high": True},
            },
            debate={
                "action": "BUY",
                "confidence": 84,
                "reasoning": "Oversold bounce with strong volume.",
            },
        )
        results = memory.query_similar_trades("BTC oversold bounce", limit=1)

    assert memory_id
    assert memory.get_stats()["store_type"] == "in_memory_fallback"
    assert results[0]["trade_data"]["symbol"] == "BTC/USDT"