"""
Configuration loader for the AI Trading Architecture project.
Loads settings from .env, settings.yaml, and command-line args.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


@dataclass
class TradingConfig:
    mode: str = "dryrun"  # dryrun, testnet, live
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    initial_capital: float = 10000.0


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
    symbols: list[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT", "SOL-USDT"])


@dataclass
class MonitoringConfig:
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    log_level: str = "INFO"


@dataclass
class AppConfig:
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Exchange API
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""

    # Database
    redis_url: str = "redis://localhost:6379"
    questdb_addr: str = "localhost:9000"
    database_url: str = "postgresql://trading_user:trading_pass@localhost:5432/trading_db"
    qdrant_url: str = "http://localhost:6333"

    # LLM (Phase 2+)
    litellm_model: str = "anthropic/claude-sonnet-4"


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    """Load configuration from settings.yaml and .env files."""
    base_dir = Path(__file__).parent.parent
    config = AppConfig()

    # Load .env
    env_file = Path(env_path) if env_path else base_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded .env from {env_file}")
    else:
        logger.warning(f"No .env file found at {env_file}")

    # Load settings.yaml
    yaml_path = Path(config_path) if config_path else base_dir / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            settings = yaml.safe_load(f)

        if "trading" in settings:
            config.trading = TradingConfig(**settings["trading"])
        if "strategy" in settings:
            config.strategy = StrategyConfig(**settings["strategy"])
        if "risk" in settings:
            config.risk = RiskConfig(**settings["risk"])
        if "data" in settings:
            config.data = DataConfig(**settings["data"])
        if "monitoring" in settings:
            config.monitoring = MonitoringConfig(**settings["monitoring"])

        logger.info(f"Loaded settings from {yaml_path}")
    else:
        logger.warning(f"No settings.yaml found at {yaml_path}")

    # Override with environment variables
    config.binance_api_key = os.getenv("BINANCE_API_KEY", "")
    config.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")
    config.binance_testnet_api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
    config.binance_testnet_api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    config.redis_url = os.getenv("REDIS_URL", config.redis_url)
    config.questdb_addr = os.getenv("QUESTDB_HTTP_ADDR", config.questdb_addr)
    config.database_url = os.getenv("DATABASE_URL", config.database_url)
    config.qdrant_url = os.getenv("QDRANT_URL", config.qdrant_url)

    if os.getenv("TRADING_MODE"):
        config.trading.mode = os.getenv("TRADING_MODE")
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        config.monitoring.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        config.monitoring.telegram_enabled = True
    if os.getenv("TELEGRAM_CHAT_ID"):
        config.monitoring.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if os.getenv("LITELLM_MODEL"):
        config.litellm_model = os.getenv("LITELLM_MODEL")
    if os.getenv("INITIAL_CAPITAL"):
        config.trading.initial_capital = float(os.getenv("INITIAL_CAPITAL"))

    return config
