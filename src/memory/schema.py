"""Trade memory record models and database schema management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    """Structured trade record for logging."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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


class DebateRecord(BaseModel):
    """Structured debate result for logging."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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


TRADES_DDL = """
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

DEBATES_DDL = """
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

SCHEMA_META_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    component       VARCHAR(64) PRIMARY KEY,
    version         INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SCHEMA_COMPONENT = "trade_memory"
SCHEMA_VERSION = 1


async def ensure_trade_memory_schema(conn: Any) -> None:
    """Initialize or verify the TradeMemory schema version."""
    await conn.execute(SCHEMA_META_DDL)
    row = await conn.fetchrow(
        "SELECT version FROM schema_migrations WHERE component = $1",
        SCHEMA_COMPONENT,
    )

    current_version = int(row["version"]) if row else 0

    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            "Database schema version is newer than this application supports"
        )

    if current_version == 0:
        await conn.execute(TRADES_DDL)
        await conn.execute(DEBATES_DDL)
        await conn.execute(
            """
            INSERT INTO schema_migrations(component, version)
            VALUES ($1, $2)
            ON CONFLICT (component)
            DO UPDATE SET version = EXCLUDED.version, updated_at = NOW()
            """,
            SCHEMA_COMPONENT,
            SCHEMA_VERSION,
        )
        logger.info(f"Database schema initialized at version {SCHEMA_VERSION}")
        return

    if current_version < SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported TradeMemory schema migration path: {current_version} -> {SCHEMA_VERSION}"
        )

    logger.info(f"Database schema version {current_version} verified")