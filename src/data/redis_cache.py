"""Redis cache for latest prices, tickers, and pub/sub price channels."""

from __future__ import annotations

import json
from typing import Any

from redis import asyncio as redis_asyncio
from loguru import logger

from src.config import get_default_redis_url


class RedisCache:
    """Async Redis client for caching market data and publishing updates.

    Key schema
    ----------
    - ``price:latest:{symbol}`` — Hash with latest trade fields
    - ``ticker:{symbol}`` — Hash with latest bid/ask
    - Pub/Sub channel: ``price:{symbol}``

    Parameters
    ----------
    url : str
        Redis connection URL (e.g. "redis://localhost:6379").
    """

    def __init__(self, url: str | None = None) -> None:
        self._url = url or get_default_redis_url()
        self._client: redis_asyncio.Redis | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the async Redis connection."""
        try:
            self._client = redis_asyncio.from_url(
                self._url,
                decode_responses=True,
                encoding="utf-8",
            )
            await self._client.ping()
            logger.info(f"Redis connected to {self._url}")
        except Exception:
            logger.exception("Failed to connect to Redis")
            raise

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("Redis connection closed")
            except Exception:
                logger.exception("Error closing Redis connection")
            finally:
                self._client = None

    def _require_client(self) -> redis_asyncio.Redis:
        """Return the client or raise if not connected."""
        if self._client is None:
            raise RuntimeError("RedisCache not connected — call connect() first")
        return self._client

    # ------------------------------------------------------------------
    # Latest price cache (HSET / HGETALL)
    # ------------------------------------------------------------------

    async def set_latest_price(
        self,
        symbol: str,
        price: float,
        side: str,
        amount: float,
        exchange: str,
        ts: float,
    ) -> None:
        """Cache the latest trade price for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTC-USDT").
        price : float
            Trade price.
        side : str
            Trade side ("buy" / "sell").
        amount : float
            Trade quantity.
        exchange : str
            Exchange name.
        ts : float
            Timestamp (epoch seconds).
        """
        client = self._require_client()
        key = f"price:latest:{symbol}"
        try:
            await client.hset(
                key,
                mapping={
                    "price": str(price),
                    "side": side,
                    "amount": str(amount),
                    "exchange": exchange,
                    "ts": str(ts),
                },
            )
        except Exception:
            logger.exception(f"Failed to set latest price for {symbol}")

    async def get_latest_price(self, symbol: str) -> dict[str, str] | None:
        """Retrieve the latest cached price for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair.

        Returns
        -------
        dict | None
            Hash fields as strings, or None if key doesn't exist.
        """
        client = self._require_client()
        key = f"price:latest:{symbol}"
        try:
            data = await client.hgetall(key)
            return data if data else None
        except Exception:
            logger.exception(f"Failed to get latest price for {symbol}")
            return None

    # ------------------------------------------------------------------
    # Ticker cache
    # ------------------------------------------------------------------

    async def set_ticker(
        self,
        symbol: str,
        bid: float,
        ask: float,
        exchange: str,
        ts: float,
    ) -> None:
        """Cache the latest ticker (bid/ask) for a symbol.

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
        ts : float
            Timestamp (epoch seconds).
        """
        client = self._require_client()
        key = f"ticker:{symbol}"
        try:
            await client.hset(
                key,
                mapping={
                    "bid": str(bid),
                    "ask": str(ask),
                    "exchange": exchange,
                    "ts": str(ts),
                },
            )
        except Exception:
            logger.exception(f"Failed to set ticker for {symbol}")

    async def get_ticker(self, symbol: str) -> dict[str, str] | None:
        """Retrieve the latest cached ticker for a symbol.

        Parameters
        ----------
        symbol : str
            Trading pair.

        Returns
        -------
        dict | None
            Ticker fields as strings, or None if key doesn't exist.
        """
        client = self._require_client()
        key = f"ticker:{symbol}"
        try:
            data = await client.hgetall(key)
            return data if data else None
        except Exception:
            logger.exception(f"Failed to get ticker for {symbol}")
            return None

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish_price(self, symbol: str, data: dict[str, Any]) -> int:
        """Publish a price update to the ``price:{symbol}`` channel.

        Parameters
        ----------
        symbol : str
            Trading pair.
        data : dict
            Price data to publish (will be JSON-encoded).

        Returns
        -------
        int
            Number of subscribers that received the message.
        """
        client = self._require_client()
        channel = f"price:{symbol}"
        try:
            payload = json.dumps(data)
            num_subscribers = await client.publish(channel, payload)
            logger.trace(f"Published to {channel}: {num_subscribers} subscribers")
            return num_subscribers
        except Exception:
            logger.exception(f"Failed to publish price for {symbol}")
            return 0

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RedisCache":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        await self.close()
        return None
