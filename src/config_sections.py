"""Typed config sections shared by the application config facade."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    mode: str = "dryrun"
    symbols: list[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    initial_capital: float = 10000.0
    interval: int = 60


@dataclass
class StrategyConfig:
    name: str = "sma_cross"
    sma_fast: int = 20
    sma_slow: int = 50
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_position_pct: float = 20.0
    max_leverage: int = 3


@dataclass
class DataConfig:
    exchange: str = "binance"
    testnet: bool = True
    candle_interval: str = "1m"
    symbols: list[str] = field(
        default_factory=lambda: ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    )


@dataclass
class MonitoringConfig:
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    log_level: str = "INFO"