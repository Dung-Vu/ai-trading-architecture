"""QuestDB writer for persisting market data via InfluxDB line protocol."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

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
        for ddl in (self._DDL_TRADES, self._DDL_OHLCV, self._DDL_TICKER):
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
            self._sender.table("trades").symbol("symbol", symbol).symbol("side", side)
            self._sender.float64("price", price).float64("amount", amount)
            self._sender.symbol("trade_id", trade_id).symbol("exchange", exchange)
            self._sender.at(ts_ns)
        except Exception:
            logger.exception(f"Failed to write trade: {symbol} {side} {price}")

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
            self._sender.table("ohlcv").symbol("symbol", symbol).symbol("interval", interval)
            self._sender.float64("open", open_).float64("high", high)
            self._sender.float64("low", low).float64("close", close)
            self._sender.float64("volume", volume).int64("trades_count", trades_count)
            self._sender.at(ts_ns)
        except Exception:
            logger.exception(f"Failed to write OHLCV: {symbol} {interval}")

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
            self._sender.table("ticker_latest").symbol("symbol", symbol)
            self._sender.float64("bid", bid).float64("ask", ask)
            self._sender.symbol("exchange", exchange)
            self._sender.at(ts_ns)
        except Exception:
            logger.exception(f"Failed to write ticker: {symbol}")

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
