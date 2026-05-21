"""
TradeMemory — Persistent trade logging and performance analytics.

Stores trades and debate results in PostgreSQL, with optional Redis
caching for hot data. Provides query methods for trade history,
performance summaries, and pattern detection.

Tables created (if not exists):
    - trades: all executed trades with full metadata
    - debates: debate engine results for audit trail

Usage:
    >>> memory = TradeMemory()
    >>> await memory.connect()
    >>> await memory.log_trade({
    ...     "symbol": "BTC/USDT", "side": "BUY", "quantity": 0.01,
    ...     "price": 67500.0, "strategy": "ai_debate", "mode": "dryrun",
    ... })
    >>> history = await memory.get_trade_history(symbol="BTC/USDT", limit=10)
    >>> summary = await memory.get_performance_summary()
    >>> await memory.close()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.config import (
    get_default_database_url,
    get_default_pg_pool_command_timeout,
    get_default_pg_pool_max_size,
    get_default_pg_pool_min_size,
    get_default_redis_cache_ttl_seconds,
    get_default_redis_url,
)
from .analytics import TradeMemoryAnalyticsMixin
from .interfaces import TradeMemoryInterface
from .schema import (
    DebateRecord,
    PerformanceSummary,
    TradeRecord,
    ensure_trade_memory_schema,
)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]



# ─── TradeMemory ───────────────────────────────────────────────────────
class TradeMemory(TradeMemoryAnalyticsMixin, TradeMemoryInterface):
    """
    Persistent trade memory backed by PostgreSQL and Redis.

    Handles:
    - Logging trades and debate results
    - Querying trade history with filters
    - Computing performance metrics
    - Detecting trading patterns for AI optimization
    """

    def __init__(
        self,
        db_url: str | None = None,
        redis_url: str | None = None,
        enable_redis: bool = True,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        pool_command_timeout: int | None = None,
        redis_cache_ttl_seconds: int | None = None,
    ) -> None:
        """
        Initialize TradeMemory.

        Args:
            db_url: PostgreSQL connection URL.
            redis_url: Redis connection URL.
            enable_redis: Whether to use Redis for caching.
        """
        self._db_url = db_url or get_default_database_url()
        self._redis_url = redis_url or get_default_redis_url()
        self._enable_redis = enable_redis
        self._pg_pool_min_size = (
            pool_min_size if pool_min_size is not None else get_default_pg_pool_min_size()
        )
        self._pg_pool_max_size = (
            pool_max_size if pool_max_size is not None else get_default_pg_pool_max_size()
        )
        self._pg_pool_command_timeout = (
            pool_command_timeout
            if pool_command_timeout is not None
            else get_default_pg_pool_command_timeout()
        )
        self._redis_cache_ttl_seconds = (
            redis_cache_ttl_seconds
            if redis_cache_ttl_seconds is not None
            else get_default_redis_cache_ttl_seconds()
        )

        self._pg_pool: asyncpg.Pool | None = None
        self._redis: aioredis.Redis | None = None

        self._connected = False

    # ─── Connection Lifecycle ──────────────────────────────────────────

    async def connect(self) -> None:
        """Establish database and Redis connections."""
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install: pip install asyncpg")

        if self._connected:
            return

        try:
            self._pg_pool = await asyncpg.create_pool(
                dsn=self._db_url,
                min_size=self._pg_pool_min_size,
                max_size=self._pg_pool_max_size,
                command_timeout=self._pg_pool_command_timeout,
            )
            logger.info("PostgreSQL connection pool created")

            async with self._pg_pool.acquire() as conn:
                await self._ensure_schema(conn)

            # Redis connection (optional)
            if self._enable_redis:
                if aioredis is None:
                    logger.warning("redis package not installed, Redis caching disabled")
                else:
                    try:
                        self._redis = aioredis.from_url(
                            self._redis_url,
                            decode_responses=True,
                            encoding="utf-8",
                        )
                        await self._redis.ping()
                        logger.info(f"Redis connected to {self._redis_url}")
                    except Exception:
                        logger.warning("Redis connection failed, continuing without cache")
                        self._redis = None

            self._connected = True

        except Exception:
            logger.exception("Failed to connect to database")
            raise

    async def _ensure_schema(self, conn: Any) -> None:
        """Initialize or verify the TradeMemory schema version."""
        await ensure_trade_memory_schema(conn)

    async def close(self) -> None:
        """Close all connections."""
        if self._pg_pool:
            await self._pg_pool.close()
            logger.info("PostgreSQL pool closed")
            self._pg_pool = None

        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed")
            self._redis = None

        self._connected = False

    def _require_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected:
            raise RuntimeError("TradeMemory not connected — call connect() first")

    # ─── Trade Logging ─────────────────────────────────────────────────

    async def log_trade(self, trade: dict[str, Any]) -> int:
        """
        Log a trade to PostgreSQL.

        Args:
            trade: Trade data dict with keys matching TradeRecord fields.

        Returns:
            Database row ID of the inserted trade.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        record = TradeRecord(**trade)
        ts = datetime.fromisoformat(record.timestamp)

        async with self._pg_pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO trades
                    (timestamp, symbol, side, quantity, price, pnl, pnl_pct,
                     strategy, mode, ai_confidence, debate_result,
                     stop_loss, take_profit, order_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                RETURNING id
                """,
                ts,
                record.symbol,
                record.side,
                record.quantity,
                record.price,
                record.pnl,
                record.pnl_pct,
                record.strategy,
                record.mode,
                record.ai_confidence,
                json.dumps(record.debate_result) if record.debate_result else None,
                record.stop_loss,
                record.take_profit,
                record.order_id,
            )

        logger.info(
            f"[TradeMemory] Logged trade #{row_id}: {record.side} {record.quantity} "
            f"{record.symbol} @ {record.price} (PnL={record.pnl:+.2f})"
        )

        # Cache in Redis
        await self._cache_trade(record)

        return row_id

    async def log_debate(self, debate_result: dict[str, Any]) -> int:
        """
        Log a debate result to PostgreSQL.

        Args:
            debate_result: Debate data dict with keys matching DebateRecord fields.

        Returns:
            Database row ID of the inserted debate.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        record = DebateRecord(**debate_result)
        ts = datetime.fromisoformat(record.timestamp)

        async with self._pg_pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO debates
                    (timestamp, symbol, bull_arg, bear_arg, devil_arg,
                     judge_action, judge_confidence, risk_action,
                     risk_reasoning, rounds, latency_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                ts,
                record.symbol,
                record.bull_arg,
                record.bear_arg,
                record.devil_arg,
                record.judge_action,
                record.judge_confidence,
                record.risk_action,
                record.risk_reasoning,
                record.rounds,
                record.latency_seconds,
            )

        logger.info(
            f"[TradeMemory] Logged debate #{row_id}: {record.symbol} "
            f"→ {record.judge_action} (confidence={record.judge_confidence:.1f})"
        )

        return row_id

    # ─── Redis Caching ─────────────────────────────────────────────────

    async def _cache_trade(self, record: TradeRecord) -> None:
        """Cache the latest trade in Redis for fast access."""
        if not self._redis:
            return

        try:
            key = f"trade:latest:{record.symbol}"
            await self._redis.hset(
                key,
                mapping={
                    "timestamp": record.timestamp,
                    "side": record.side,
                    "quantity": str(record.quantity),
                    "price": str(record.price),
                    "pnl": str(record.pnl),
                    "strategy": record.strategy,
                },
            )
            # Expire after 24 hours
            await self._redis.expire(key, self._redis_cache_ttl_seconds)
        except Exception:
            logger.warning(f"Failed to cache trade in Redis for {record.symbol}")

    # ─── Query Methods ─────────────────────────────────────────────────

    async def get_trade_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        strategy: str | None = None,
        limit: int = 100,
        before_cursor: tuple[datetime, int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query trade history with optional filters.

        Args:
            symbol: Filter by trading pair.
            start_date: Filter trades from this date onward.
            end_date: Filter trades up to this date.
            strategy: Optional strategy filter.
            limit: Maximum number of trades to return.
            before_cursor: Optional pagination cursor as (timestamp, id).

        Returns:
            List of trade dicts, newest first.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        if strategy:
            conditions.append(f"strategy = ${param_idx}")
            params.append(strategy)
            param_idx += 1

        if before_cursor:
            cursor_timestamp, cursor_id = before_cursor
            conditions.append(f"(timestamp, id) < (${param_idx}, ${param_idx + 1})")
            params.extend([cursor_timestamp, cursor_id])
            param_idx += 2

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT id, timestamp, symbol, side, quantity, price,
                   pnl, pnl_pct, strategy, mode, ai_confidence,
                   debate_result, stop_loss, take_profit, order_id
            FROM trades{where}
            ORDER BY timestamp DESC, id DESC
            LIMIT ${param_idx}
        """

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results: list[dict[str, Any]] = []
        for row in rows:
            trade_dict = dict(row)
            # Parse JSONB debate_result
            if trade_dict.get("debate_result") and isinstance(trade_dict["debate_result"], str):
                try:
                    trade_dict["debate_result"] = json.loads(trade_dict["debate_result"])
                except json.JSONDecodeError:
                    trade_dict["debate_result"] = None
            results.append(trade_dict)

        logger.debug(f"[TradeMemory] Retrieved {len(results)} trades from history")
        return results

    async def get_recent_trades(
        self, symbol: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get the most recent trades for a specific symbol.

        Args:
            symbol: Trading pair.
            limit: Number of recent trades.

        Returns:
            List of recent trade dicts, newest first.
        """
        return await self.get_trade_history(symbol=symbol, limit=limit)

    async def get_debate_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Query debate history.

        Args:
            symbol: Filter by trading pair.
            start_date: Filter debates from this date.
            end_date: Filter debates up to this date.
            limit: Maximum number of debates.

        Returns:
            List of debate dicts, newest first.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        query = f"""
            SELECT id, timestamp, symbol, bull_arg, bear_arg, devil_arg,
                   judge_action, judge_confidence, risk_action,
                   risk_reasoning, rounds, latency_seconds
            FROM debates WHERE {where}
            ORDER BY timestamp DESC, id DESC
            LIMIT ${param_idx}
        """

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    # ─── Context Manager ───────────────────────────────────────────────

    async def __aenter__(self) -> "TradeMemory":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        await self.close()
        return None
