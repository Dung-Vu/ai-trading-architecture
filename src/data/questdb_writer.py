"""QuestDB writer for persisting market data via InfluxDB line protocol."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime

from loguru import logger
from questdb.ingress import Sender

from src.config import get_default_questdb_http_addr


class QuestDBWriter:
    """Async-compatible writer for QuestDB using the InfluxDB line protocol.

    Creates tables via the QuestDB REST API and writes data through the
    line protocol sender with auto-flush configuration.

    Parameters
    ----------
    addr : str
        QuestDB line protocol address (e.g. "localhost:9000").
    auto_flush_rows : int
        Number of rows before auto-flush triggers.
    """

    # DDL statements to ensure tables exist
    _DDL_TRADES = """\
CREATE TABLE IF NOT EXISTS trades (
    symbol SYMBOL,
    side SYMBOL,
    price DOUBLE,
    amount DOUBLE,
    trade_id SYMBOL,
    exchange SYMBOL
) TIMESTAMP(ts) PARTITION BY HOUR;
"""

    _DDL_OHLCV = """\
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol SYMBOL,
    interval SYMBOL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    trades_count LONG
) TIMESTAMP(ts) PARTITION BY DAY;
"""

    _DDL_TICKER = """\
CREATE TABLE IF NOT EXISTS ticker_latest (
    symbol SYMBOL,
    bid DOUBLE,
    ask DOUBLE,
    exchange SYMBOL
) TIMESTAMP(ts) PARTITION BY DAY DEDUP UPSERT KEYS(symbol, exchange);
"""

    _DDL_NEWS = """\
CREATE TABLE IF NOT EXISTS news (
    symbol SYMBOL,
    source SYMBOL,
    title STRING,
    url STRING,
    content_snippet STRING,
    sentiment_score DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY;
"""

    def __init__(
        self,
        addr: str | None = None,
        auto_flush_rows: int = 5000,
    ) -> None:
        self._addr = addr or get_default_questdb_http_addr()
        self._auto_flush_rows = auto_flush_rows
        self._sender: Sender | None = None
        self._http_base = f"http://{self._addr}"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create the QuestDB line-protocol sender.

        Uses Sender.from_conf with http transport and auto-flush settings.
        Also ensures all tables exist via DDL over the REST API.
        """
        try:
            conf = (
                f"http::addr={self._addr};"
                f"auto_flush_rows={self._auto_flush_rows};"
                f"auto_flush_interval=1000;"
            )
            self._sender = Sender.from_conf(conf)
            logger.info(f"QuestDB sender connected to {self._addr}")
        except Exception:
            logger.exception("Failed to connect to QuestDB")
            raise

        # Ensure tables exist
        try:
            self._ensure_tables()
        except Exception:
            logger.warning("Failed to ensure tables exist (they may already exist)")

    def _ensure_tables(self) -> None:
        """Execute DDL statements to create tables if they don't exist."""
        for ddl in (
            self._DDL_TRADES,
            self._DDL_OHLCV,
            self._DDL_TICKER,
            self._DDL_NEWS,
        ):
            try:
                self._exec_ddl(ddl)
            except Exception:
                logger.debug(f"DDL already applied or failed: {ddl.strip()[:60]}...")

    def _exec_ddl(self, ddl: str) -> None:
        """Execute a DDL statement via QuestDB REST API."""
        url = f"{self._http_base}/exec"
        params = {"query": ddl}

        req = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError("Failed to execute QuestDB DDL request") from exc

        if result.get("ddl") != "OK":
            raise RuntimeError(f"DDL failed: {result}")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def _coerce_timestamp(self, value: object) -> int | datetime | None:
        """Convert incoming timestamps into QuestDB sender compatible values."""
        if value is None or value == "":
            return None

        if isinstance(value, (int, datetime)):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

        return None

    def write_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        trade_id: str,
        exchange: str,
        ts_ns: int,
    ) -> None:
        """Write a trade record to the 'trades' table.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTC-USDT").
        side : str
            Trade side ("buy" or "sell").
        price : float
            Execution price.
        amount : float
            Trade quantity.
        trade_id : str
            Unique trade identifier.
        exchange : str
            Exchange name (e.g. "Binance").
        ts_ns : int
            Timestamp in nanoseconds.
        """
        if self._sender is None:
            raise RuntimeError("QuestDBWriter not connected — call connect() first")

        try:
            self._sender.row(
                "trades",
                symbols={
                    "symbol": symbol,
                    "side": side,
                    "trade_id": trade_id,
                    "exchange": exchange,
                },
                columns={
                    "price": price,
                    "amount": amount,
                },
                at=ts_ns,
            )
            return True
        except Exception:
            logger.exception(f"Failed to write trade: {symbol} {side} {price}")
            return False

    def write_ohlcv(
        self,
        symbol: str,
        interval: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        trades_count: int,
        ts_ns: int,
    ) -> None:
        """Write an OHLCV candle to the 'ohlcv' table.

        Parameters
        ----------
        symbol : str
            Trading pair.
        interval : str
            Candle interval (e.g. "1m").
        open_, high, low, close : float
            OHLC prices.
        volume : float
            Total volume for the candle.
        trades_count : int
            Number of trades in the candle.
        ts_ns : int
            Timestamp in nanoseconds (candle close time).
        """
        if self._sender is None:
            raise RuntimeError("QuestDBWriter not connected — call connect() first")

        try:
            self._sender.row(
                "ohlcv",
                symbols={
                    "symbol": symbol,
                    "interval": interval,
                },
                columns={
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "trades_count": trades_count,
                },
                at=ts_ns,
            )
            return True
        except Exception:
            logger.exception(f"Failed to write OHLCV: {symbol} {interval}")
            return False

    def write_ticker(
        self,
        symbol: str,
        bid: float,
        ask: float,
        exchange: str,
        ts_ns: int,
    ) -> None:
        """Write a ticker snapshot to 'ticker_latest' with DEDUP UPSERT.

        The table is defined with DEDUP UPSERT KEYS(symbol, exchange) so
        each write replaces the previous row for the same symbol+exchange.

        Parameters
        ----------
        symbol : str
            Trading pair.
        bid : float
            Best bid price.
        ask : float
            Best ask price.
        exchange : str
            Exchange name.
        ts_ns : int
            Timestamp in nanoseconds.
        """
        if self._sender is None:
            raise RuntimeError("QuestDBWriter not connected — call connect() first")

        try:
            self._sender.row(
                "ticker_latest",
                symbols={
                    "symbol": symbol,
                    "exchange": exchange,
                },
                columns={
                    "bid": bid,
                    "ask": ask,
                },
                at=ts_ns,
            )
            return True
        except Exception:
            logger.exception(f"Failed to write ticker: {symbol}")
            return False

    def write_news(self, record: dict[str, object]) -> bool:
        """Write a news record to the 'news' table."""
        if self._sender is None:
            raise RuntimeError("QuestDBWriter not connected — call connect() first")

        try:
            at = self._coerce_timestamp(record.get("timestamp"))
            self._sender.row(
                "news",
                symbols={
                    "symbol": str(record.get("symbol", "")),
                    "source": str(record.get("source", "")),
                },
                columns={
                    "title": str(record.get("title", "")),
                    "url": str(record.get("url", "")),
                    "content_snippet": str(record.get("content_snippet", "")),
                    "sentiment_score": float(record.get("sentiment_score", 0.0)),
                },
                at=at,
            )
            return True
        except Exception:
            logger.exception(
                f"Failed to write news item: {record.get('symbol', '')} {record.get('url', '')}"
            )
            return False

    # ------------------------------------------------------------------
    # Flush & cleanup
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Manually flush pending rows to QuestDB."""
        if self._sender is not None:
            try:
                self._sender.flush()
            except Exception:
                logger.exception("Failed to flush QuestDB sender")

    def close(self) -> None:
        """Flush remaining data and close the sender."""
        if self._sender is not None:
            try:
                self._sender.flush()
                self._sender.close()
                logger.info("QuestDB sender closed")
            except Exception:
                logger.exception("Error closing QuestDB sender")
            finally:
                self._sender = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "QuestDBWriter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
        return None
