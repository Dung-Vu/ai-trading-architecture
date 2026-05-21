"""Data pipeline configuration with lightweight runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    env_float,
    env_int,
    env_str,
    get_data_channels,
    get_data_symbols,
    get_default_questdb_http_addr,
    get_default_redis_url,
)


_VALID_CHANNELS = {"TRADES", "CANDLES", "TICKER", "L2_BOOK", "L3_BOOK"}
_VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}


@dataclass(slots=True)
class DataConfig:
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

    symbols: list[str] = field(default_factory=get_data_symbols)
    channels: list[str] = field(default_factory=get_data_channels)
    candle_interval: str = "1m"
    redis_url: str = field(default_factory=get_default_redis_url)
    questdb_addr: str = field(default_factory=get_default_questdb_http_addr)
    max_latency_ms: int = 5000
    z_score_threshold: float = 3.0
    max_spread_pct: float = 1.0

    def __post_init__(self) -> None:
        self.symbols = self.validate_symbols(list(self.symbols))
        self.channels = self.validate_channels(list(self.channels))
        self.candle_interval = self.validate_interval(str(self.candle_interval))
        self.max_latency_ms = self.validate_latency(int(self.max_latency_ms))
        self.z_score_threshold = self.validate_z_score(float(self.z_score_threshold))
        self.max_spread_pct = self.validate_spread(float(self.max_spread_pct))

    @staticmethod
    def validate_symbols(value: list[str]) -> list[str]:
        if not value:
            raise ValueError("symbols list cannot be empty")
        return value

    @staticmethod
    def validate_channels(value: list[str]) -> list[str]:
        for channel in value:
            if channel not in _VALID_CHANNELS:
                raise ValueError(
                    f"Invalid channel '{channel}'. Must be one of {_VALID_CHANNELS}"
                )
        return value

    @staticmethod
    def validate_interval(value: str) -> str:
        if value not in _VALID_INTERVALS:
            raise ValueError(
                f"Invalid candle_interval '{value}'. Must be one of {_VALID_INTERVALS}"
            )
        return value

    @staticmethod
    def validate_latency(value: int) -> int:
        if value <= 0:
            raise ValueError("max_latency_ms must be positive")
        return value

    @staticmethod
    def validate_z_score(value: float) -> float:
        if value <= 0:
            raise ValueError("z_score_threshold must be positive")
        return value

    @staticmethod
    def validate_spread(value: float) -> float:
        if value <= 0:
            raise ValueError("max_spread_pct must be positive")
        return value

    @classmethod
    def load_from_env(cls) -> "DataConfig":
        """Load configuration from environment variables.

        Environment variables (all optional, defaults used when not set):
            DATA_SYMBOLS       - Comma-separated symbols (e.g. "BTC-USDT,ETH-USDT")
            DATA_CHANNELS      - Comma-separated channels (e.g. "TRADES,CANDLES,TICKER")
            DATA_CANDLE_INTERVAL
            REDIS_URL / DATA_REDIS_URL
            QUESTDB_HTTP_ADDR / DATA_QUESTDB_ADDR
            DATA_MAX_LATENCY_MS
            DATA_Z_SCORE_THRESHOLD
            DATA_MAX_SPREAD_PCT

        Returns
        -------
        DataConfig
            Validated configuration instance.
        """
        return cls(
            symbols=get_data_symbols(),
            channels=get_data_channels(),
            candle_interval=env_str("DATA_CANDLE_INTERVAL", "1m"),
            redis_url=get_default_redis_url(),
            questdb_addr=get_default_questdb_http_addr(),
            max_latency_ms=env_int("DATA_MAX_LATENCY_MS", 5000),
            z_score_threshold=env_float("DATA_Z_SCORE_THRESHOLD", 3.0),
            max_spread_pct=env_float("DATA_MAX_SPREAD_PCT", 1.0),
        )
