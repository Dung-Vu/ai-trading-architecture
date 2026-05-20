"""Binance market data connector using Cryptofeed async WebSocket."""

from __future__ import annotations

from loguru import logger

from cryptofeed import FeedHandler
from cryptofeed.callback import CandleCallback, TickerCallback, TradeCallback
from cryptofeed.defines import CANDLES, L2_BOOK, TICKER, TRADES
from cryptofeed.exchanges import Binance

from .config import DataConfig
from .quality_gates import QualityGates
from .questdb_writer import QuestDBWriter
from .redis_cache import RedisCache

# Map channel string names to Cryptofeed constants
_CHANNEL_MAP: dict[str, str] = {
    "TRADES": TRADES,
    "CANDLES": CANDLES,
    "TICKER": TICKER,
    "L2_BOOK": L2_BOOK,
}


class BinanceConnector:
    """Connects to Binance via Cryptofeed, validates data through QualityGates,
    writes to QuestDB, and caches in Redis.

    Parameters
    ----------
    config : DataConfig | None
        Pipeline configuration. If None, uses defaults.
    questdb_writer : QuestDBWriter
        Writer instance for persisting trades/OHLCV/tickers.
    redis_cache : RedisCache
        Cache instance for latest prices and pub/sub.
    quality_gates : QualityGates
        Validator for latency, spread, and price spikes.
    """

    # Sliding window of recent prices per symbol for spike detection
    _MAX_RECENT = 60  # Keep last 60 prices per symbol

    def __init__(
        self,
        config: DataConfig | None = None,
        questdb_writer: QuestDBWriter | None = None,
        redis_cache: RedisCache | None = None,
        quality_gates: QualityGates | None = None,
    ) -> None:
        self.config = config or DataConfig()
        self.questdb_writer = questdb_writer
        self.redis_cache = redis_cache
        self.quality_gates = quality_gates or QualityGates()

        self._feedhandler: FeedHandler | None = None
        self._recent_prices: dict[str, list[float]] = {
            s: [] for s in self.config.symbols
        }

    # ------------------------------------------------------------------
    # FeedHandler setup
    # ------------------------------------------------------------------

    def _create_feedhandler(self) -> FeedHandler:
        """Create and configure the Cryptofeed FeedHandler with Binance exchange.

        Sets up callbacks for each requested channel.

        Returns
        -------
        FeedHandler
            Configured FeedHandler instance (not yet started).
        """
        fh = FeedHandler()
        feed = Binance(
            config={
                # Map Cryptofeed symbol format if needed
            }
        )

        # Subscribe to requested channels
        if TRADES in self._resolved_channels():
            feed.add_feed(
                Binance(
                    channels={TRADES: self.config.symbols},
                    callbacks={TRADES: TradeCallback(self._trade_callback)},
                )
            )
            logger.info(f"Subscribed to TRADES: {self.config.symbols}")

        if CANDLES in self._resolved_channels():
            feed.add_feed(
                Binance(
                    channels={CANDLES: self.config.symbols},
                    callbacks={
                        CANDLES: CandleCallback(
                            self._candle_callback,
                            candle_interval=self.config.candle_interval,
                        )
                    },
                )
            )
            logger.info(
                f"Subscribed to CANDLES ({self.config.candle_interval}): "
                f"{self.config.symbols}"
            )

        if TICKER in self._resolved_channels():
            feed.add_feed(
                Binance(
                    channels={TICKER: self.config.symbols},
                    callbacks={TICKER: TickerCallback(self._ticker_callback)},
                )
            )
            logger.info(f"Subscribed to TICKER: {self.config.symbols}")

        self._feedhandler = fh
        return fh

    def _resolved_channels(self) -> set[str]:
        """Return the set of Cryptofeed channel constants to subscribe to."""
        return {_CHANNEL_MAP[c] for c in self.config.channels if c in _CHANNEL_MAP}

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _trade_callback(self, trade, receipt_ts: float) -> None:
        """Process a trade received from Binance WebSocket.

        Validates through quality gates, writes to QuestDB and Redis,
        and publishes via Redis pub/sub.

        Parameters
        ----------
        trade : Trade
            Cryptofeed Trade object.
        receipt_ts : float
            Epoch timestamp when the message was received.
        """
        try:
            # Convert to dict using Cryptofeed's built-in method
            data = trade.to_dict(numeric_type=float)

            symbol = data.get("symbol", "UNKNOWN")
            side = data.get("side", "unknown")
            price = data.get("price", 0.0)
            amount = data.get("amount", 0.0)
            trade_id = str(data.get("id", ""))
            exchange = "Binance"

            # Update recent prices for spike detection
            self._update_recent_price(symbol, price)

            # Quality gate validation
            passed, reason = self.quality_gates.validate_trade(
                trade_data={
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "amount": amount,
                    "bid": data.get("bid"),
                    "ask": data.get("ask"),
                },
                receipt_ts=receipt_ts,
                recent_prices=self._recent_prices.get(symbol, []),
            )

            if not passed:
                logger.warning(
                    f"Trade rejected by quality gate: {symbol} — {reason}"
                )
                return

            # Write to QuestDB
            if self.questdb_writer is not None:
                ts_ns = int(receipt_ts * 1_000_000_000)
                self.questdb_writer.write_trade(
                    symbol=symbol,
                    side=side,
                    price=price,
                    amount=amount,
                    trade_id=trade_id,
                    exchange=exchange,
                    ts_ns=ts_ns,
                )

            # Cache in Redis
            if self.redis_cache is not None:
                await self.redis_cache.set_latest_price(
                    symbol=symbol,
                    price=price,
                    side=side,
                    amount=amount,
                    exchange=exchange,
                    ts=receipt_ts,
                )
                await self.redis_cache.publish_price(
                    symbol=symbol,
                    data={
                        "symbol": symbol,
                        "price": price,
                        "side": side,
                        "amount": amount,
                        "exchange": exchange,
                        "ts": receipt_ts,
                    },
                )

        except Exception:
            logger.exception(f"Error processing trade callback for {trade}")

    async def _candle_callback(self, candle, receipt_ts: float) -> None:
        """Process a candle (OHLCV) received from Binance WebSocket.

        Only processes candles that are marked as closed by Cryptofeed.

        Parameters
        ----------
        candle : Candle
            Cryptofeed Candle object.
        receipt_ts : float
            Epoch timestamp when the message was received.
        """
        try:
            # Only process closed candles to avoid overwriting incomplete data
            if not candle.closed:
                logger.debug(f"Skipping open candle for {candle.symbol}")
                return

            data = candle.to_dict(numeric_type=float)

            symbol = data.get("symbol", "UNKNOWN")
            interval = self.config.candle_interval
            open_ = data.get("open", 0.0)
            high = data.get("high", 0.0)
            low = data.get("low", 0.0)
            close = data.get("close", 0.0)
            volume = data.get("volume", 0.0)
            trades_count = data.get("trades_count", 0)

            # Use candle close timestamp if available, else receipt
            candle_ts = getattr(candle, "timestamp", receipt_ts)
            ts_ns = int(candle_ts * 1_000_000_000)

            # Write to QuestDB
            if self.questdb_writer is not None:
                self.questdb_writer.write_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    trades_count=int(trades_count),
                    ts_ns=ts_ns,
                )
                logger.debug(
                    f"Wrote OHLCV: {symbol} {interval} "
                    f"O={open_} H={high} L={low} C={close} V={volume}"
                )

        except Exception:
            logger.exception(f"Error processing candle callback for {candle}")

    async def _ticker_callback(self, ticker, receipt_ts: float) -> None:
        """Process a ticker (bid/ask) update from Binance WebSocket.

        Writes to Redis cache only (no QuestDB persistence for tickers
        since they are high-frequency and handled via DEDUP UPSERT).

        Parameters
        ----------
        ticker : Ticker
            Cryptofeed Ticker object.
        receipt_ts : float
            Epoch timestamp when the message was received.
        """
        try:
            data = ticker.to_dict(numeric_type=float)

            symbol = data.get("symbol", "UNKNOWN")
            bid = data.get("bid", 0.0)
            ask = data.get("ask", 0.0)
            exchange = "Binance"

            # Cache in Redis
            if self.redis_cache is not None:
                await self.redis_cache.set_ticker(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    exchange=exchange,
                    ts=receipt_ts,
                )
                await self.redis_cache.publish_price(
                    symbol=symbol,
                    data={
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "exchange": exchange,
                        "ts": receipt_ts,
                    },
                )

                logger.trace(f"Ticker cached: {symbol} bid={bid} ask={ask}")

        except Exception:
            logger.exception(f"Error processing ticker callback for {ticker}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Cryptofeed FeedHandler. Blocks until stopped.

        Initializes the FeedHandler with Binance feeds and runs it.
        """
        if self._feedhandler is None:
            self._create_feedhandler()

        logger.info("Starting Binance FeedHandler...")
        try:
            if self._feedhandler is not None:
                self._feedhandler.run()
        except Exception:
            logger.exception("FeedHandler encountered an error")

    def stop(self) -> None:
        """Stop the Cryptofeed FeedHandler and flush pending data."""
        if self._feedhandler is not None:
            try:
                self._feedhandler.stop()
                logger.info("FeedHandler stopped")
            except Exception:
                logger.exception("Error stopping FeedHandler")

        # Flush QuestDB
        if self.questdb_writer is not None:
            try:
                self.questdb_writer.flush()
            except Exception:
                logger.exception("Error flushing QuestDB")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_recent_price(self, symbol: str, price: float) -> None:
        """Maintain a sliding window of recent prices for spike detection.

        Parameters
        ----------
        symbol : str
            Trading pair.
        price : float
            Latest price.
        """
        window = self._recent_prices.setdefault(symbol, [])
        window.append(price)
        if len(window) > self._MAX_RECENT:
            self._recent_prices[symbol] = window[-self._MAX_RECENT:]
