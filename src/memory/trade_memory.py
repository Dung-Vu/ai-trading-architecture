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
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel, Field

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]


# ─── Trade Record ──────────────────────────────────────────────────────
class TradeRecord(BaseModel):
    """Structured trade record for logging."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    symbol: str = Field(..., description="Trading pair, e.g. BTC/USDT")
    side: str = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    price: float = Field(..., gt=0, description="Execution price")
    pnl: float = Field(default=0.0, description="Realized P&L (0 for opening trades)")
    pnl_pct: float = Field(default=0.0, description="P&L as percentage")
    strategy: str = Field(default="unknown", description="Strategy name")
    mode: str = Field(default="dryrun", description="dryrun, testnet, live")
    ai_confidence: float | None = Field(default=None, ge=0, le=100, description="AI confidence score")
    debate_result: dict[str, Any] | None = Field(default=None, description="Full debate result if AI-driven")
    stop_loss: float | None = Field(default=None, description="Stop-loss price")
    take_profit: float | None = Field(default=None, description="Take-profit price")
    order_id: str | None = Field(default=None, description="Exchange order ID")


# ─── Debate Record ─────────────────────────────────────────────────────
class DebateRecord(BaseModel):
    """Structured debate result for logging."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    symbol: str = Field(..., description="Trading pair")
    bull_arg: str = Field(default="", description="Bull agent's argument summary")
    bear_arg: str = Field(default="", description="Bear agent's argument summary")
    devil_arg: str = Field(default="", description="Devil's advocate argument summary")
    judge_action: str = Field(default="HOLD", description="Judge's final action: BUY, SELL, HOLD")
    judge_confidence: float = Field(default=50.0, ge=0, le=100, description="Judge confidence")
    risk_action: str = Field(default="APPROVE", description="Risk manager action: APPROVE, REJECT, REDUCE")
    risk_reasoning: str = Field(default="", description="Risk manager reasoning")
    rounds: int = Field(default=0, description="Number of debate rounds")
    latency_seconds: float = Field(default=0.0, description="Total debate latency")


# ─── Performance Summary ───────────────────────────────────────────────
class PerformanceSummary(BaseModel):
    """Aggregated performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0


# ─── TradeMemory ───────────────────────────────────────────────────────
class TradeMemory:
    """
    Persistent trade memory backed by PostgreSQL and Redis.

    Handles:
    - Logging trades and debate results
    - Querying trade history with filters
    - Computing performance metrics
    - Detecting trading patterns for AI optimization
    """

    # SQL DDL statements
    _TRADES_DDL = """
    CREATE TABLE IF NOT EXISTS trades (
        id              SERIAL PRIMARY KEY,
        timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol          VARCHAR(32) NOT NULL,
        side            VARCHAR(8) NOT NULL CHECK (side IN ('BUY', 'SELL')),
        quantity        DOUBLE PRECISION NOT NULL CHECK (quantity > 0),
        price           DOUBLE PRECISION NOT NULL CHECK (price > 0),
        pnl             DOUBLE PRECISION DEFAULT 0.0,
        pnl_pct         DOUBLE PRECISION DEFAULT 0.0,
        strategy        VARCHAR(64) DEFAULT 'unknown',
        mode            VARCHAR(16) DEFAULT 'dryrun',
        ai_confidence   DOUBLE PRECISION,
        debate_result   JSONB,
        stop_loss       DOUBLE PRECISION,
        take_profit     DOUBLE PRECISION,
        order_id        VARCHAR(128),
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
    CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
    """

    _DEBATES_DDL = """
    CREATE TABLE IF NOT EXISTS debates (
        id              SERIAL PRIMARY KEY,
        timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol          VARCHAR(32) NOT NULL,
        bull_arg        TEXT DEFAULT '',
        bear_arg        TEXT DEFAULT '',
        devil_arg       TEXT DEFAULT '',
        judge_action    VARCHAR(8) DEFAULT 'HOLD',
        judge_confidence DOUBLE PRECISION DEFAULT 50.0,
        risk_action     VARCHAR(16) DEFAULT 'APPROVE',
        risk_reasoning  TEXT DEFAULT '',
        rounds          INTEGER DEFAULT 0,
        latency_seconds DOUBLE PRECISION DEFAULT 0.0,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_debates_symbol ON debates(symbol);
    CREATE INDEX IF NOT EXISTS idx_debates_timestamp ON debates(timestamp);
    CREATE INDEX IF NOT EXISTS idx_debates_judge_action ON debates(judge_action);
    """

    def __init__(
        self,
        db_url: str = "postgresql://postgres:postgres@localhost:5432/trading_db",
        redis_url: str = "redis://localhost:6379",
        enable_redis: bool = True,
    ) -> None:
        """
        Initialize TradeMemory.

        Args:
            db_url: PostgreSQL connection URL.
            redis_url: Redis connection URL.
            enable_redis: Whether to use Redis for caching.
        """
        self._db_url = db_url
        self._redis_url = redis_url
        self._enable_redis = enable_redis

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
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info("PostgreSQL connection pool created")

            # Run DDL to ensure tables exist
            async with self._pg_pool.acquire() as conn:
                await conn.execute(self._TRADES_DDL)
                await conn.execute(self._DEBATES_DDL)
            logger.info("Database tables verified/created")

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
                        await cast(Awaitable[Any], self._redis.ping())
                        logger.info(f"Redis connected to {self._redis_url}")
                    except Exception:
                        logger.warning("Redis connection failed, continuing without cache")
                        self._redis = None

            self._connected = True

        except Exception:
            logger.exception("Failed to connect to database")
            raise

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

        return int(row_id)

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

        return int(row_id)

    # ─── Redis Caching ─────────────────────────────────────────────────

    async def _cache_trade(self, record: TradeRecord) -> None:
        """Cache the latest trade in Redis for fast access."""
        if not self._redis:
            return

        try:
            key = f"trade:latest:{record.symbol}"
            await cast(Awaitable[Any], self._redis.hset(
                key,
                mapping={
                    "timestamp": record.timestamp,
                    "side": record.side,
                    "quantity": str(record.quantity),
                    "price": str(record.price),
                    "pnl": str(record.pnl),
                    "strategy": record.strategy,
                },
            ))
            # Expire after 24 hours
            await self._redis.expire(key, 86400)
        except Exception:
            logger.warning(f"Failed to cache trade in Redis for {record.symbol}")

    # ─── Query Methods ─────────────────────────────────────────────────

    async def get_trade_history(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query trade history with optional filters.

        Args:
            symbol: Filter by trading pair.
            start_date: Filter trades from this date onward.
            end_date: Filter trades up to this date.
            limit: Maximum number of trades to return.

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

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT id, timestamp, symbol, side, quantity, price,
                   pnl, pnl_pct, strategy, mode, ai_confidence,
                   debate_result, stop_loss, take_profit, order_id
            FROM trades{where}
            ORDER BY timestamp DESC
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
            ORDER BY timestamp DESC
            LIMIT ${param_idx}
        """

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    # ─── Performance Analytics ─────────────────────────────────────────

    async def get_performance_summary(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        strategy: str | None = None,
    ) -> PerformanceSummary:
        """
        Compute aggregated performance metrics.

        Args:
            start_date: Start of period.
            end_date: End of period.
            strategy: Filter by strategy name.

        Returns:
            PerformanceSummary with all computed metrics.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        # Only analyze closing trades (SELL side with realized PnL)
        conditions = ["side = 'SELL'", "pnl != 0"]
        params: list[Any] = []
        param_idx = 1

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

        where = " AND ".join(conditions)

        async with self._pg_pool.acquire() as conn:
            # Get aggregate stats
            stats = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE pnl > 0) as winning_trades,
                    COUNT(*) FILTER (WHERE pnl <= 0) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as best_trade,
                    COALESCE(MIN(pnl), 0) as worst_trade,
                    COALESCE(SUM(pnl) FILTER (WHERE pnl > 0), 0) as gross_profit,
                    COALESCE(ABS(SUM(pnl) FILTER (WHERE pnl < 0)), 0) as gross_loss
                FROM trades WHERE {where}
                """,
                *params,
            )

            if stats is None or stats["total_trades"] == 0:
                return PerformanceSummary()

            # Get all PnLs for Sharpe calculation
            pnl_rows = await conn.fetch(
                f"SELECT pnl FROM trades WHERE {where} ORDER BY timestamp", *params
            )
            pnls = [float(row["pnl"]) for row in pnl_rows]

            # Compute max drawdown
            max_dd = self._compute_max_drawdown(pnls)

            # Compute Sharpe ratio
            sharpe = self._compute_sharpe(pnls)

            gross_profit = float(stats["gross_profit"])
            gross_loss = float(stats["gross_loss"])
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            total = int(stats["total_trades"])
            wins = int(stats["winning_trades"])

            return PerformanceSummary(
                total_trades=total,
                winning_trades=wins,
                losing_trades=int(stats["losing_trades"]),
                win_rate=(wins / total * 100) if total > 0 else 0.0,
                total_pnl=float(stats["total_pnl"]),
                avg_pnl=float(stats["avg_pnl"]),
                best_trade=float(stats["best_trade"]),
                worst_trade=float(stats["worst_trade"]),
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                profit_factor=profit_factor,
            )

    async def get_strategy_performance(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, PerformanceSummary]:
        """
        Get performance broken down by strategy.

        Args:
            start_date: Start of period.
            end_date: End of period.

        Returns:
            Dict mapping strategy name → PerformanceSummary.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        conditions = ["side = 'SELL'", "pnl != 0"]
        params: list[Any] = []
        param_idx = 1

        if start_date:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        where = " AND ".join(conditions)

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT strategy FROM trades WHERE {where}
                """,
                *params,
            )

        strategies = [row["strategy"] for row in rows]
        result: dict[str, PerformanceSummary] = {}

        for strat in strategies:
            result[strat] = await self.get_performance_summary(
                start_date=start_date, end_date=end_date, strategy=strat
            )

        return result

    # ─── Pattern Detection ─────────────────────────────────────────────

    async def get_trade_patterns(
        self, min_samples: int = 5
    ) -> dict[str, Any]:
        """
        Identify common trading patterns from historical data.

        Detects patterns like:
        - Symbol + side combinations with high/low win rates
        - Strategy performance by symbol
        - Time-of-day performance
        - Confidence vs. outcome correlation

        Args:
            min_samples: Minimum trades required for a pattern to be reported.

        Returns:
            Dict with pattern categories and their insights.
        """
        self._require_connected()
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool is None")

        patterns: dict[str, Any] = {
            "symbol_side_patterns": [],
            "strategy_symbol_patterns": [],
            "time_of_day_patterns": [],
            "confidence_correlation": {},
            "risk_action_outcomes": [],
        }

        async with self._pg_pool.acquire() as conn:
            # Pattern 1: Symbol + Side win rates
            rows = await conn.fetch(
                """
                SELECT symbol, side,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(SUM(pnl), 0) as total_pnl,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY symbol, side
                HAVING COUNT(*) >= $1
                ORDER BY total DESC
                """,
                min_samples,
            )

            for row in rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["symbol_side_patterns"].append({
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(float(row["total_pnl"]), 2),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 2: Strategy performance by symbol
            strat_rows = await conn.fetch(
                """
                SELECT strategy, symbol,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY strategy, symbol
                HAVING COUNT(*) >= $1
                ORDER BY total DESC
                """,
                min_samples,
            )

            for row in strat_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["strategy_symbol_patterns"].append({
                    "strategy": row["strategy"],
                    "symbol": row["symbol"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 3: Time of day performance (UTC hour)
            time_rows = await conn.fetch(
                """
                SELECT EXTRACT(HOUR FROM timestamp) as hour,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades
                WHERE side = 'SELL' AND pnl != 0
                GROUP BY hour
                HAVING COUNT(*) >= $1
                ORDER BY avg_pnl DESC
                """,
                max(min_samples, 3),
            )

            for row in time_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["time_of_day_patterns"].append({
                    "hour_utc": int(row["hour"]),
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

            # Pattern 4: AI confidence correlation
            conf_rows = await conn.fetch(
                """
                SELECT ai_confidence, pnl
                FROM trades
                WHERE ai_confidence IS NOT NULL AND side = 'SELL' AND pnl != 0
                ORDER BY timestamp
                """
            )

            if conf_rows:
                high_conf = [float(r["pnl"]) for r in conf_rows if float(r["ai_confidence"] or 0) >= 70]
                low_conf = [float(r["pnl"]) for r in conf_rows if float(r["ai_confidence"] or 0) < 70]

                patterns["confidence_correlation"] = {
                    "high_confidence_avg_pnl": round(
                        sum(high_conf) / len(high_conf) if high_conf else 0, 2
                    ),
                    "low_confidence_avg_pnl": round(
                        sum(low_conf) / len(low_conf) if low_conf else 0, 2
                    ),
                    "high_confidence_count": len(high_conf),
                    "low_confidence_count": len(low_conf),
                }

            # Pattern 5: Risk action outcomes
            debate_rows = await conn.fetch(
                """
                SELECT risk_action,
                       COUNT(*) as total,
                       COUNT(*) FILTER (WHERE pnl > 0) as wins,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM debates d
                JOIN trades t ON d.symbol = t.symbol
                    AND t.timestamp BETWEEN d.timestamp
                    AND d.timestamp + INTERVAL '1 hour'
                WHERE t.side = 'SELL' AND t.pnl != 0
                GROUP BY d.risk_action
                """
            )

            for row in debate_rows:
                win_rate = (int(row["wins"]) / int(row["total"]) * 100) if row["total"] > 0 else 0
                patterns["risk_action_outcomes"].append({
                    "risk_action": row["risk_action"],
                    "total_trades": int(row["total"]),
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(float(row["avg_pnl"]), 2),
                })

        logger.info(f"[TradeMemory] Pattern analysis complete: {sum(len(v) if isinstance(v, list) else 1 for v in patterns.values())} patterns found")
        return patterns

    # ─── Static Helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """
        Compute maximum drawdown from a sequence of PnL values.

        Args:
            pnls: List of trade PnLs in chronological order.

        Returns:
            Maximum drawdown as percentage.
        """
        if not pnls:
            return 0.0

        equity = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        return round(max_dd * 100, 2)

    @staticmethod
    def _compute_sharpe(pnls: list[float], risk_free_rate: float = 0.0) -> float:
        """
        Compute Sharpe ratio from a sequence of PnL values.

        Args:
            pnls: List of trade PnLs.
            risk_free_rate: Annual risk-free rate.

        Returns:
            Sharpe ratio.
        """
        if not pnls or len(pnls) < 2:
            return 0.0

        import statistics

        avg = statistics.mean(pnls)
        std = statistics.stdev(pnls)

        if std == 0:
            return 0.0

        # Annualize (assuming ~250 trading periods)
        sharpe = (avg - risk_free_rate / 250) / std * (250 ** 0.5)
        return float(round(sharpe, 3))

    # ─── Context Manager ───────────────────────────────────────────────

    async def __aenter__(self) -> TradeMemory:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        await self.close()
        return None
