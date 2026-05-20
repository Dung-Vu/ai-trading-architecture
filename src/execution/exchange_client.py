"""Exchange client for connecting to Binance via CCXT."""

from __future__ import annotations

from typing import Any

import ccxt
from loguru import logger


class ExchangeClient:
    """Wrapper around CCXT Binance exchange for spot trading."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        rate_limit_ms: int = 50,
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
        self._exchange: ccxt.binance | None = None

    @property
    def exchange(self) -> ccxt.binance:
        """Return the underlying CCXT exchange instance."""
        if self._exchange is None:
            raise RuntimeError("Exchange not connected. Call connect() first.")
        return self._exchange

    @property
    def is_connected(self) -> bool:
        """Check if exchange is connected."""
        return self._exchange is not None

    def connect(self) -> None:
        """Create and configure the CCXT Binance exchange instance."""
        config: dict[str, Any] = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
            "rateLimit": self._rate_limit_ms,
            "options": {
                "defaultType": "spot",
            },
        }

        self._exchange = ccxt.binance(config)

        if self._testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("Connected to Binance testnet (sandbox mode)")
        else:
            logger.info("Connected to Binance mainnet")

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
            Dictionary with free, used, and total balances per asset.
        """
        logger.debug("Fetching account balance")
        raw = self.exchange.fetch_balance()
        # Return only non-zero balances for readability
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
