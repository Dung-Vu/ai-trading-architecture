"""
Configuration loader for the AI Trading Architecture project.
Loads settings from .env, settings.yaml, and command-line args.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

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


def load_config(config_path: str | None = None, env_path: str | None = None) -> AppConfig:
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

    questdb_host = os.getenv("QUESTDB_HOST")
    questdb_port = os.getenv("QUESTDB_PORT")
    config.questdb_addr = os.getenv(
        "QUESTDB_HTTP_ADDR",
        os.getenv(
            "DATA_QUESTDB_ADDR",
            f"{questdb_host}:{questdb_port or '9000'}" if questdb_host else config.questdb_addr,
        ),
    )

    if os.getenv("DATABASE_URL"):
        config.database_url = os.environ["DATABASE_URL"]
    elif os.getenv("POSTGRES_HOST"):
        pg_user = os.getenv("POSTGRES_USER", "trading_user")
        pg_password = quote_plus(os.getenv("POSTGRES_PASSWORD", "trading_pass"))
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "trading_db")
        config.database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"

    config.qdrant_url = os.getenv("QDRANT_URL", config.qdrant_url)

    if trading_mode := os.getenv("TRADING_MODE"):
        config.trading.mode = trading_mode
    if telegram_token := os.getenv("TELEGRAM_BOT_TOKEN"):
        config.monitoring.telegram_bot_token = telegram_token
        config.monitoring.telegram_enabled = True
    if telegram_chat_id := os.getenv("TELEGRAM_CHAT_ID"):
        config.monitoring.telegram_chat_id = telegram_chat_id
    if litellm_model := os.getenv("LITELLM_MODEL"):
        config.litellm_model = litellm_model
    if initial_capital := os.getenv("INITIAL_CAPITAL"):
        config.trading.initial_capital = float(initial_capital)

    return config
