"""Exchange client for connecting to Binance via CCXT."""

from __future__ import annotations

from typing import Any

import ccxt
from loguru import logger

from src.config import get_default_exchange_name


class ExchangeClient:
    """Wrapper around a configurable CCXT exchange for spot trading."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        rate_limit_ms: int = 50,
        exchange_name: str | None = None,
    ) -> None:
        """Initialize exchange client.

        Args:
            api_key: Binance API key (empty for public endpoints).
            api_secret: Binance API secret (empty for public endpoints).
            testnet: Whether to connect to Binance testnet.
            rate_limit_ms: Minimum delay between requests in milliseconds.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._rate_limit_ms = rate_limit_ms
        self._exchange_name = exchange_name or get_default_exchange_name()
        self._exchange: Any | None = None

    @property
    def exchange(self) -> Any:
        """Return the underlying CCXT exchange instance."""
        if self._exchange is None:
            raise RuntimeError("Exchange not connected. Call connect() first.")
        return self._exchange

    @property
    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self._exchange is not None

    def connect(self) -> None:
        """Create and configure the underlying CCXT exchange instance."""
        config: dict[str, Any] = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
            "rateLimit": self._rate_limit_ms,
            "options": {
                "defaultType": "spot",
            },
        }

        exchange_factory = getattr(ccxt, self._exchange_name, None)
        if exchange_factory is None:
            raise ValueError(f"Unsupported CCXT exchange: {self._exchange_name}")

        self._exchange = exchange_factory(config)

        if self._testnet:
            sandbox_mode = getattr(self._exchange, "set_sandbox_mode", None)
            if callable(sandbox_mode):
                sandbox_mode(True)
            logger.info(f"Connected to {self._exchange_name} testnet (sandbox mode)")
        else:
            logger.info(f"Connected to {self._exchange_name} mainnet")

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current ticker for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT').

        Returns:
            Ticker dictionary with bid, ask, last, volume, etc.
        """
        logger.debug(f"Fetching ticker for {symbol}")
        return self.exchange.fetch_ticker(symbol)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> list[list]:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: Number of candles to fetch.

        Returns:
            List of [timestamp, open, high, low, close, volume] lists.
        """
        logger.debug(f"Fetching {limit} {timeframe} candles for {symbol}")
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_balance(self) -> dict:
        """Fetch account balance.

        Returns:
            Raw CCXT balance dictionary with free, used, and total balances.
        """
        logger.debug("Fetching account balance")
        raw = self.exchange.fetch_balance()
        return raw

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch open orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            List of open order dictionaries.
        """
        logger.debug(f"Fetching open orders for {symbol or 'all symbols'}")
        return self.exchange.fetch_open_orders(symbol)

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Fetch a specific order by ID.

        Args:
            order_id: Order ID.
            symbol: Trading pair symbol.

        Returns:
            Order dictionary.
        """
        logger.debug(f"Fetching order {order_id} for {symbol}")
        return self.exchange.fetch_order(order_id, symbol)

    def close(self) -> None:
        """Clean up exchange connection."""
        if self._exchange is not None:
            self._exchange = None
            logger.info("Exchange connection closed")
