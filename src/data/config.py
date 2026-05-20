"""Data pipeline configuration using Pydantic for validation."""

from __future__ import annotations

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    """Configuration for the data pipeline module.

    Parameters
    ----------
    symbols : list[str]
        Trading symbol pairs to subscribe to (e.g. ["BTC-USDT"]).
    channels : list[str]
        Cryptofeed channel names to subscribe to (TRADES, CANDLES, TICKER).
    candle_interval : str
        Candlestick interval (e.g. "1m", "5m", "1h").
    redis_url : str
        Redis connection URL.
    questdb_addr : str
        QuestDB sender address (host:port for InfluxDB line protocol).
    max_latency_ms : int
        Maximum acceptable message latency in milliseconds.
    z_score_threshold : float
        Z-score threshold for price spike detection.
    max_spread_pct : float
        Maximum acceptable bid-ask spread as a percentage of mid price.
    """

    symbols: list[str] = Field(
        default=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        description="Trading symbol pairs to subscribe to",
    )
    channels: list[str] = Field(
        default=["TRADES", "CANDLES", "TICKER"],
        description="Cryptofeed channels to subscribe to",
    )
    candle_interval: str = Field(
        default="1m",
        description="Candlestick interval (e.g. '1m', '5m')",
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )
    questdb_addr: str = Field(
        default="localhost:9000",
        description="QuestDB sender address",
    )
    max_latency_ms: int = Field(
        default=5000,
        description="Maximum acceptable message latency (ms)",
    )
    z_score_threshold: float = Field(
        default=3.0,
        description="Z-score threshold for price spike detection",
    )
    max_spread_pct: float = Field(
        default=1.0,
        description="Maximum bid-ask spread (% of mid price)",
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols list cannot be empty")
        return v

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, v: list[str]) -> list[str]:
        valid = {"TRADES", "CANDLES", "TICKER", "L2_BOOK", "L3_BOOK"}
        for ch in v:
            if ch not in valid:
                raise ValueError(
                    f"Invalid channel '{ch}'. Must be one of {valid}"
                )
        return v

    @field_validator("candle_interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        valid_intervals = {
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M",
        }
        if v not in valid_intervals:
            raise ValueError(
                f"Invalid candle_interval '{v}'. Must be one of {valid_intervals}"
            )
        return v

    @field_validator("max_latency_ms")
    @classmethod
    def validate_latency(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_latency_ms must be positive")
        return v

    @field_validator("z_score_threshold")
    @classmethod
    def validate_z_score(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("z_score_threshold must be positive")
        return v

    @field_validator("max_spread_pct")
    @classmethod
    def validate_spread(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_spread_pct must be positive")
        return v

    @classmethod
    def load_from_env(cls) -> "DataConfig":
        """Load configuration from environment variables.

        Environment variables (all optional, defaults used when not set):
            DATA_SYMBOLS       - Comma-separated symbols (e.g. "BTC-USDT,ETH-USDT")
            DATA_CHANNELS      - Comma-separated channels (e.g. "TRADES,CANDLES,TICKER")
            DATA_CANDLE_INTERVAL
            DATA_REDIS_URL
            DATA_QUESTDB_ADDR
            DATA_MAX_LATENCY_MS
            DATA_Z_SCORE_THRESHOLD
            DATA_MAX_SPREAD_PCT

        Returns
        -------
        DataConfig
            Validated configuration instance.
        """
        env_map: dict[str, str] = {}

        if syms := os.getenv("DATA_SYMBOLS"):
            env_map["symbols"] = [s.strip() for s in syms.split(",") if s.strip()]
        if chs := os.getenv("DATA_CHANNELS"):
            env_map["channels"] = [c.strip() for c in chs.split(",") if c.strip()]
        if os.getenv("DATA_CANDLE_INTERVAL"):
            env_map["candle_interval"] = os.environ["DATA_CANDLE_INTERVAL"]
        if os.getenv("DATA_REDIS_URL"):
            env_map["redis_url"] = os.environ["DATA_REDIS_URL"]
        if os.getenv("DATA_QUESTDB_ADDR"):
            env_map["questdb_addr"] = os.environ["DATA_QUESTDB_ADDR"]
        if os.getenv("DATA_MAX_LATENCY_MS"):
            env_map["max_latency_ms"] = int(os.environ["DATA_MAX_LATENCY_MS"])
        if os.getenv("DATA_Z_SCORE_THRESHOLD"):
            env_map["z_score_threshold"] = float(os.environ["DATA_Z_SCORE_THRESHOLD"])
        if os.getenv("DATA_MAX_SPREAD_PCT"):
            env_map["max_spread_pct"] = float(os.environ["DATA_MAX_SPREAD_PCT"])

        return cls(**env_map)
