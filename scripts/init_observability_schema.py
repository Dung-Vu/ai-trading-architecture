"""Initialize local observability tables used by the Grafana dashboard."""

from __future__ import annotations

import asyncio
import os
import urllib.parse
import urllib.request
from pathlib import Path

import asyncpg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


QUESTDB_DDL = [
    """
    CREATE TABLE IF NOT EXISTS trades (
        timestamp TIMESTAMP,
        symbol SYMBOL,
        side SYMBOL,
        price DOUBLE,
        amount DOUBLE,
        quantity DOUBLE,
        trade_id SYMBOL,
        exchange SYMBOL,
        pnl DOUBLE,
        strategy_name SYMBOL,
        notes STRING
    ) TIMESTAMP(timestamp) PARTITION BY DAY
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        timestamp TIMESTAMP,
        portfolio_value DOUBLE,
        initial_capital DOUBLE,
        drawdown_pct DOUBLE
    ) TIMESTAMP(timestamp) PARTITION BY DAY
    """,
]


POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS open_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(8) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
    entry_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    current_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    unrealized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    pnl_pct DOUBLE PRECISION NOT NULL DEFAULT 0,
    open_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(16) NOT NULL DEFAULT 'open'
);

CREATE INDEX IF NOT EXISTS idx_open_positions_status
    ON open_positions(status);
CREATE INDEX IF NOT EXISTS idx_open_positions_symbol
    ON open_positions(symbol);

CREATE TABLE IF NOT EXISTS bot_status (
    id SERIAL PRIMARY KEY,
    is_running INTEGER NOT NULL DEFAULT 0,
    mode VARCHAR(16) NOT NULL DEFAULT 'dryrun',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bot_decisions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(32) NOT NULL DEFAULT '',
    strategy_name VARCHAR(64) NOT NULL DEFAULT 'unknown',
    action VARCHAR(8) NOT NULL DEFAULT 'HOLD',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    reasoning TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_bot_decisions_strategy
    ON bot_decisions(strategy_name);
CREATE INDEX IF NOT EXISTS idx_bot_decisions_timestamp
    ON bot_decisions(timestamp);

INSERT INTO bot_status(is_running, mode)
SELECT 0, 'dryrun'
WHERE NOT EXISTS (SELECT 1 FROM bot_status);
"""


def load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def questdb_base_url() -> str:
    raw = os.environ.get("QUESTDB_HTTP_URL") or os.environ.get("QUESTDB_HTTP_ADDR")
    if raw:
        return raw if raw.startswith(("http://", "https://")) else f"http://{raw}"
    return "http://localhost:9000"


def postgres_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("PG_DSN")
    if dsn:
        return dsn
    password = os.environ.get("POSTGRES_PASSWORD")
    user = os.environ.get("POSTGRES_USER", "trading_user")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "trading_db")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD or PG_DSN or DATABASE_URL must be set."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def exec_questdb(sql: str) -> None:
    params = urllib.parse.urlencode({"query": " ".join(sql.split())})
    url = f"{questdb_base_url().rstrip('/')}/exec?{params}"
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status >= 400:
            raise RuntimeError(f"QuestDB DDL failed with HTTP {response.status}")


async def init_postgres() -> None:
    conn = await asyncpg.connect(postgres_dsn())
    try:
        await conn.execute(POSTGRES_DDL)
    finally:
        await conn.close()


async def main() -> None:
    load_dotenv()
    for ddl in QUESTDB_DDL:
        exec_questdb(ddl)
    await init_postgres()
    print("Observability schema initialized.")


if __name__ == "__main__":
    asyncio.run(main())
