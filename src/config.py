"""
Configuration loader for the AI Trading Architecture project.
Loads settings from .env, settings.yaml, and command-line args.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.config_env import (
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json,
    env_str,
    env_str_alias,
)


BASE_DIR = Path(__file__).parent.parent
DEFAULT_ENV_FILE = BASE_DIR / ".env"

if DEFAULT_ENV_FILE.exists():
    load_dotenv(DEFAULT_ENV_FILE, override=False)


DEFAULT_TRADING_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_DATA_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
DEFAULT_DATA_CHANNELS = ["TRADES", "CANDLES", "TICKER"]
DEFAULT_DATABASE_URL = "postgresql://localhost:5432/trading_db"
DEFAULT_REDIS_URL = "redis://localhost:6379"
DEFAULT_QUESTDB_HTTP_ADDR = "localhost:9000"
DEFAULT_QUESTDB_ILP_ADDR = "localhost:9009"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_LITELLM_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_FALLBACK_LLM_MODEL = "openai/gpt-4o"
DEFAULT_DSPY_MODEL = DEFAULT_LITELLM_MODEL
DEFAULT_DEBATE_MAX_ROUNDS = 3
DEFAULT_DEBATE_TEMPERATURE = 0.7
DEFAULT_DEBATE_MAX_TOKENS = 4096
DEFAULT_DEBATE_TIMEOUT_SECONDS = 120.0
DEFAULT_DEBATE_CACHE_TTL_SECONDS = 30.0
DEFAULT_DEBATE_CACHE_MAX_ENTRIES = 128
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_CACHE_TTL_SECONDS = 30.0
DEFAULT_LLM_CACHE_MAX_ENTRIES = 256
DEFAULT_LLM_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_LLM_CIRCUIT_BREAKER_RESET_SECONDS = 60.0
DEFAULT_EXCHANGE_NAME = "binance"
DEFAULT_MEM0_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MEM0_LLM_MODEL = "gpt-4o-mini"
DEFAULT_NEWS_RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "CryptoNews": "https://cryptonews.com/news/feed/",
}
DEFAULT_CRYPTOPANIC_API_URL = "https://cryptopanic.com/api/v1/posts/"
DEFAULT_SIMULATED_BASE_PRICES = {
    "BTC/USDT": 67500.0,
    "ETH/USDT": 3450.0,
    "SOL/USDT": 145.0,
    "XRP/USDT": 0.52,
    "ADA/USDT": 0.45,
}
DEFAULT_QUANTITY_PCT = 0.10
DEFAULT_POSITION_SIZER_RISK_PCT = 0.02
DEFAULT_LOG_APP_NAME = "trading"
DEFAULT_LOG_ERROR_NAME = "error"
DEFAULT_LOG_ROTATION = "1 day"
DEFAULT_LOG_RETENTION = "30 days"
DEFAULT_LOG_ERROR_RETENTION = "90 days"
DEFAULT_PG_POOL_MIN_SIZE = 2
DEFAULT_PG_POOL_MAX_SIZE = 10
DEFAULT_PG_POOL_COMMAND_TIMEOUT = 30
DEFAULT_REDIS_CACHE_TTL_SECONDS = 86400
DEFAULT_KNOWLEDGE_GRAPH_TOP_N = 5
DEFAULT_KNOWLEDGE_GRAPH_MIN_OCCURRENCES = 3
DEFAULT_AUTOTUNE_MIN_TRADES = 20
DEFAULT_AUTOTUNE_SHARPE_DECAY_THRESHOLD = 0.5
DEFAULT_AUTOTUNE_WIN_RATE_DECAY_THRESHOLD = 0.40
DEFAULT_AUTOTUNE_DRAWDOWN_ALERT_THRESHOLD = 0.15
DEFAULT_AUTOTUNE_PARAM_RANGES = {
    "sma_fast": {"min": 5, "max": 50, "default": 20},
    "sma_slow": {"min": 20, "max": 200, "default": 50},
    "rsi_period": {"min": 7, "max": 21, "default": 14},
    "rsi_overbought": {"min": 65, "max": 80, "default": 70},
    "rsi_oversold": {"min": 20, "max": 35, "default": 30},
}
DEFAULT_AUTOTUNE_PARAM_COMBOS = [
    {"sma_fast": 10, "sma_slow": 30, "rsi_oversold": 35, "rsi_overbought": 65},
    {"sma_fast": 15, "sma_slow": 40, "rsi_oversold": 32, "rsi_overbought": 68},
    {"sma_fast": 20, "sma_slow": 50, "rsi_oversold": 30, "rsi_overbought": 70},
    {"sma_fast": 25, "sma_slow": 60, "rsi_oversold": 28, "rsi_overbought": 72},
    {"sma_fast": 10, "sma_slow": 40, "rsi_oversold": 30, "rsi_overbought": 75},
    {"sma_fast": 15, "sma_slow": 50, "rsi_oversold": 35, "rsi_overbought": 70},
]
DEFAULT_STRATEGY_OPTIMIZER_SMA_PARAM_SPACE = {
    "sma_fast": [10, 50],
    "sma_slow": [40, 200],
    "rsi_period": [7, 21],
    "rsi_overbought": [60, 80],
    "rsi_oversold": [20, 40],
}
DEFAULT_STRATEGY_OPTIMIZER_BBANDS_PARAM_SPACE = {
    "bb_period": [10, 40],
    "bb_std_dev": [1.5, 3.0],
    "volume_factor": [1.0, 2.5],
}
DEFAULT_LOOP_INTERVAL_SECONDS = 60
DEFAULT_BBANDS_STOP_LOSS_PCT = 0.02
DEFAULT_QUESTDB_PG_HOST = "localhost"
DEFAULT_QUESTDB_PG_PORT = 8812
DEFAULT_QUESTDB_PG_USER = "admin"
DEFAULT_QUESTDB_PG_PASSWORD = "quest"
DEFAULT_WANDB_PROJECT = "ai-trading-architecture"
DEFAULT_INITIAL_CAPITAL = 10000.0


def get_trading_symbols() -> list[str]:
    return env_csv("TRADING_SYMBOLS", DEFAULT_TRADING_SYMBOLS)


def get_data_symbols() -> list[str]:
    return env_csv("DATA_SYMBOLS", DEFAULT_DATA_SYMBOLS)


def get_data_channels() -> list[str]:
    return env_csv("DATA_CHANNELS", DEFAULT_DATA_CHANNELS)


def get_default_database_url() -> str:
    return env_str("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_default_redis_url() -> str:
    return env_str_alias(("REDIS_URL", "DATA_REDIS_URL"), DEFAULT_REDIS_URL)


def get_default_questdb_http_addr() -> str:
    return env_str_alias(
        ("QUESTDB_HTTP_ADDR", "DATA_QUESTDB_ADDR"),
        DEFAULT_QUESTDB_HTTP_ADDR,
    )


def get_default_questdb_ilp_addr() -> str:
    return env_str("QUESTDB_ILP_ADDR", DEFAULT_QUESTDB_ILP_ADDR)


def get_default_qdrant_url() -> str:
    return env_str("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_default_litellm_model() -> str:
    return env_str("LITELLM_MODEL", DEFAULT_LITELLM_MODEL)


def get_default_fallback_litellm_model() -> str:
    return env_str("LITELLM_FALLBACK_MODEL", DEFAULT_FALLBACK_LLM_MODEL)


def get_default_dspy_model() -> str:
    return env_str("DSPY_DEFAULT_MODEL", DEFAULT_DSPY_MODEL)


def get_default_debate_max_rounds() -> int:
    return env_int("DEBATE_MAX_ROUNDS", DEFAULT_DEBATE_MAX_ROUNDS)


def get_default_debate_temperature() -> float:
    return env_float("DEBATE_TEMPERATURE", DEFAULT_DEBATE_TEMPERATURE)


def get_default_debate_max_tokens() -> int:
    return env_int("DEBATE_MAX_TOKENS", DEFAULT_DEBATE_MAX_TOKENS)


def get_default_debate_timeout_seconds() -> float:
    return env_float("DEBATE_TIMEOUT_SECONDS", DEFAULT_DEBATE_TIMEOUT_SECONDS)


def get_default_debate_cache_ttl_seconds() -> float:
    return env_float(
        "DEBATE_RESULT_CACHE_TTL_SECONDS",
        DEFAULT_DEBATE_CACHE_TTL_SECONDS,
    )


def get_default_debate_cache_max_entries() -> int:
    return env_int(
        "DEBATE_RESULT_CACHE_MAX_ENTRIES",
        DEFAULT_DEBATE_CACHE_MAX_ENTRIES,
    )


def get_default_llm_max_retries() -> int:
    return env_int("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES)


def get_default_llm_cache_ttl_seconds() -> float:
    return env_float("LLM_CACHE_TTL_SECONDS", DEFAULT_LLM_CACHE_TTL_SECONDS)


def get_default_llm_cache_max_entries() -> int:
    return env_int("LLM_CACHE_MAX_ENTRIES", DEFAULT_LLM_CACHE_MAX_ENTRIES)


def get_default_llm_circuit_breaker_threshold() -> int:
    return env_int(
        "LLM_CIRCUIT_BREAKER_THRESHOLD",
        DEFAULT_LLM_CIRCUIT_BREAKER_THRESHOLD,
    )


def get_default_llm_circuit_breaker_reset_seconds() -> float:
    return env_float(
        "LLM_CIRCUIT_BREAKER_RESET_SECONDS",
        DEFAULT_LLM_CIRCUIT_BREAKER_RESET_SECONDS,
    )


def get_default_exchange_name() -> str:
    return env_str("EXCHANGE_NAME", DEFAULT_EXCHANGE_NAME)


def get_default_mem0_embedding_model() -> str:
    return env_str("MEM0_EMBEDDING_MODEL", DEFAULT_MEM0_EMBEDDING_MODEL)


def get_default_mem0_llm_model() -> str:
    return env_str("MEM0_LLM_MODEL", DEFAULT_MEM0_LLM_MODEL)


def get_default_news_rss_feeds() -> dict[str, str]:
    raw = env_json("NEWS_RSS_FEEDS", DEFAULT_NEWS_RSS_FEEDS)
    return dict(raw) if isinstance(raw, dict) else copy.deepcopy(DEFAULT_NEWS_RSS_FEEDS)


def get_default_cryptopanic_api_url() -> str:
    return env_str("CRYPTOPANIC_API_URL", DEFAULT_CRYPTOPANIC_API_URL)


def get_default_cryptopanic_api_key() -> str:
    return env_str("CRYPTOPANIC_API_KEY", "")


def get_default_simulated_base_prices() -> dict[str, float]:
    raw = env_json("SIMULATED_BASE_PRICES", DEFAULT_SIMULATED_BASE_PRICES)
    return dict(raw) if isinstance(raw, dict) else copy.deepcopy(DEFAULT_SIMULATED_BASE_PRICES)


def get_default_quantity_pct() -> float:
    return env_float("DEFAULT_QUANTITY_PCT", DEFAULT_QUANTITY_PCT)


def get_default_position_sizer_risk_pct() -> float:
    return env_float("POSITION_SIZER_RISK_PCT", DEFAULT_POSITION_SIZER_RISK_PCT)


def get_default_log_app_name() -> str:
    return env_str("LOG_APP_NAME", DEFAULT_LOG_APP_NAME)


def get_default_log_error_name() -> str:
    return env_str("LOG_ERROR_NAME", DEFAULT_LOG_ERROR_NAME)


def get_default_log_rotation() -> str:
    return env_str("LOG_ROTATION", DEFAULT_LOG_ROTATION)


def get_default_log_retention() -> str:
    return env_str("LOG_RETENTION", DEFAULT_LOG_RETENTION)


def get_default_log_error_retention() -> str:
    return env_str("LOG_ERROR_RETENTION", DEFAULT_LOG_ERROR_RETENTION)


def get_default_pg_pool_min_size() -> int:
    return env_int("PG_POOL_MIN_SIZE", DEFAULT_PG_POOL_MIN_SIZE)


def get_default_pg_pool_max_size() -> int:
    return env_int("PG_POOL_MAX_SIZE", DEFAULT_PG_POOL_MAX_SIZE)


def get_default_pg_pool_command_timeout() -> int:
    return env_int("PG_POOL_COMMAND_TIMEOUT", DEFAULT_PG_POOL_COMMAND_TIMEOUT)


def get_default_redis_cache_ttl_seconds() -> int:
    return env_int("REDIS_CACHE_TTL_SECONDS", DEFAULT_REDIS_CACHE_TTL_SECONDS)


def get_default_knowledge_graph_top_n() -> int:
    return env_int("KNOWLEDGE_GRAPH_TOP_N", DEFAULT_KNOWLEDGE_GRAPH_TOP_N)


def get_default_knowledge_graph_min_occurrences() -> int:
    return env_int(
        "KNOWLEDGE_GRAPH_MIN_OCCURRENCES",
        DEFAULT_KNOWLEDGE_GRAPH_MIN_OCCURRENCES,
    )


def get_default_autotune_min_trades() -> int:
    return env_int("AUTOTUNE_MIN_TRADES", DEFAULT_AUTOTUNE_MIN_TRADES)


def get_default_autotune_sharpe_decay_threshold() -> float:
    return env_float(
        "AUTOTUNE_SHARPE_DECAY_THRESHOLD",
        DEFAULT_AUTOTUNE_SHARPE_DECAY_THRESHOLD,
    )


def get_default_autotune_win_rate_decay_threshold() -> float:
    return env_float(
        "AUTOTUNE_WIN_RATE_DECAY_THRESHOLD",
        DEFAULT_AUTOTUNE_WIN_RATE_DECAY_THRESHOLD,
    )


def get_default_autotune_drawdown_alert_threshold() -> float:
    return env_float(
        "AUTOTUNE_DRAWDOWN_ALERT_THRESHOLD",
        DEFAULT_AUTOTUNE_DRAWDOWN_ALERT_THRESHOLD,
    )


def get_default_autotune_param_ranges() -> dict[str, dict[str, float | int]]:
    raw = env_json("AUTOTUNE_PARAM_RANGES", DEFAULT_AUTOTUNE_PARAM_RANGES)
    return dict(raw) if isinstance(raw, dict) else copy.deepcopy(DEFAULT_AUTOTUNE_PARAM_RANGES)


def get_default_autotune_param_combos() -> list[dict[str, int]]:
    raw = env_json("AUTOTUNE_PARAM_COMBOS", DEFAULT_AUTOTUNE_PARAM_COMBOS)
    if not isinstance(raw, list):
        return copy.deepcopy(DEFAULT_AUTOTUNE_PARAM_COMBOS)

    combos: list[dict[str, int]] = []
    for item in raw:
        if isinstance(item, dict):
            combos.append({str(key): int(value) for key, value in item.items()})

    return combos or copy.deepcopy(DEFAULT_AUTOTUNE_PARAM_COMBOS)


def get_default_strategy_optimizer_sma_param_space() -> dict[str, tuple[float, float]]:
    raw = env_json(
        "OPTIMIZER_SMA_PARAM_SPACE",
        DEFAULT_STRATEGY_OPTIMIZER_SMA_PARAM_SPACE,
    )
    if not isinstance(raw, dict):
        raw = DEFAULT_STRATEGY_OPTIMIZER_SMA_PARAM_SPACE
    return {key: tuple(value) for key, value in raw.items()}


def get_default_strategy_optimizer_bbands_param_space() -> dict[str, tuple[float, float]]:
    raw = env_json(
        "OPTIMIZER_BBANDS_PARAM_SPACE",
        DEFAULT_STRATEGY_OPTIMIZER_BBANDS_PARAM_SPACE,
    )
    if not isinstance(raw, dict):
        raw = DEFAULT_STRATEGY_OPTIMIZER_BBANDS_PARAM_SPACE
    return {key: tuple(value) for key, value in raw.items()}


def get_default_loop_interval_seconds() -> int:
    return env_int("LOOP_INTERVAL_SECONDS", DEFAULT_LOOP_INTERVAL_SECONDS)


def get_default_initial_capital() -> float:
    return env_float("INITIAL_CAPITAL", DEFAULT_INITIAL_CAPITAL)


def get_default_bbands_stop_loss_pct() -> float:
    return env_float("BBANDS_STOP_LOSS_PCT", DEFAULT_BBANDS_STOP_LOSS_PCT)


@dataclass
class TradingConfig:
    mode: str = field(default_factory=lambda: env_str("TRADING_MODE", "dryrun"))
    symbols: list[str] = field(default_factory=get_trading_symbols)
    initial_capital: float = field(default_factory=get_default_initial_capital)
    interval: int = field(default_factory=get_default_loop_interval_seconds)


@dataclass
class StrategyConfig:
    name: str = "sma_cross"
    sma_fast: int = field(default_factory=lambda: env_int("STRATEGY_SMA_FAST", 20))
    sma_slow: int = field(default_factory=lambda: env_int("STRATEGY_SMA_SLOW", 50))
    rsi_period: int = field(
        default_factory=lambda: env_int("STRATEGY_RSI_PERIOD", 14)
    )
    rsi_overbought: int = field(
        default_factory=lambda: env_int("STRATEGY_RSI_OVERBOUGHT", 70)
    )
    rsi_oversold: int = field(
        default_factory=lambda: env_int("STRATEGY_RSI_OVERSOLD", 30)
    )


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = field(
        default_factory=lambda: env_float("MAX_DAILY_LOSS_PCT", 3.0)
    )
    max_drawdown_pct: float = field(
        default_factory=lambda: env_float("MAX_DRAWDOWN_PCT", 10.0)
    )
    max_position_pct: float = field(
        default_factory=lambda: env_float("MAX_POSITION_PCT", 20.0)
    )
    max_leverage: int = field(default_factory=lambda: env_int("MAX_LEVERAGE", 3))


@dataclass
class DataConfig:
    exchange: str = field(default_factory=get_default_exchange_name)
    testnet: bool = field(default_factory=lambda: env_bool("DATA_TESTNET", True))
    candle_interval: str = field(
        default_factory=lambda: env_str("DATA_CANDLE_INTERVAL", "1m")
    )
    symbols: list[str] = field(default_factory=get_data_symbols)


@dataclass
class MonitoringConfig:
    telegram_enabled: bool = field(
        default_factory=lambda: env_bool("TELEGRAM_ENABLED", False)
    )
    telegram_bot_token: str = field(
        default_factory=lambda: env_str("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: env_str("TELEGRAM_CHAT_ID", "")
    )
    log_level: str = field(default_factory=lambda: env_str("LOG_LEVEL", "INFO"))


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
    openai_api_key: str = field(default_factory=lambda: env_str("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: env_str("ANTHROPIC_API_KEY", "")
    )

    # Database
    redis_url: str = field(default_factory=get_default_redis_url)
    questdb_addr: str = field(default_factory=get_default_questdb_http_addr)
    questdb_ilp_addr: str = field(default_factory=get_default_questdb_ilp_addr)
    questdb_pg_host: str = field(
        default_factory=lambda: env_str("QUESTDB_PG_HOST", DEFAULT_QUESTDB_PG_HOST)
    )
    questdb_pg_port: int = field(
        default_factory=lambda: env_int("QUESTDB_PG_PORT", DEFAULT_QUESTDB_PG_PORT)
    )
    questdb_pg_user: str = field(
        default_factory=lambda: env_str("QUESTDB_PG_USER", DEFAULT_QUESTDB_PG_USER)
    )
    questdb_pg_password: str = field(
        default_factory=lambda: env_str(
            "QUESTDB_PG_PASSWORD",
            DEFAULT_QUESTDB_PG_PASSWORD,
        )
    )
    database_url: str = field(default_factory=get_default_database_url)
    qdrant_url: str = field(default_factory=get_default_qdrant_url)
    pg_pool_min_size: int = field(default_factory=get_default_pg_pool_min_size)
    pg_pool_max_size: int = field(default_factory=get_default_pg_pool_max_size)
    pg_pool_command_timeout: int = field(
        default_factory=get_default_pg_pool_command_timeout
    )
    redis_cache_ttl_seconds: int = field(
        default_factory=get_default_redis_cache_ttl_seconds
    )

    # LLM (Phase 2+)
    litellm_model: str = field(default_factory=get_default_litellm_model)
    litellm_fallback_model: str = field(
        default_factory=get_default_fallback_litellm_model
    )
    debate_max_rounds: int = field(default_factory=get_default_debate_max_rounds)
    debate_temperature: float = field(default_factory=get_default_debate_temperature)
    debate_max_tokens: int = field(default_factory=get_default_debate_max_tokens)
    debate_timeout_seconds: float = field(
        default_factory=get_default_debate_timeout_seconds
    )
    debate_cache_ttl_seconds: float = field(
        default_factory=get_default_debate_cache_ttl_seconds
    )
    debate_cache_max_entries: int = field(
        default_factory=get_default_debate_cache_max_entries
    )
    llm_max_retries: int = field(default_factory=get_default_llm_max_retries)
    llm_cache_ttl_seconds: float = field(
        default_factory=get_default_llm_cache_ttl_seconds
    )
    llm_cache_max_entries: int = field(
        default_factory=get_default_llm_cache_max_entries
    )
    llm_circuit_breaker_threshold: int = field(
        default_factory=get_default_llm_circuit_breaker_threshold
    )
    llm_circuit_breaker_reset_seconds: float = field(
        default_factory=get_default_llm_circuit_breaker_reset_seconds
    )

    # Runtime defaults
    exchange_name: str = field(default_factory=get_default_exchange_name)
    mem0_embedding_model: str = field(default_factory=get_default_mem0_embedding_model)
    mem0_llm_model: str = field(default_factory=get_default_mem0_llm_model)
    cryptopanic_api_key: str = field(default_factory=get_default_cryptopanic_api_key)
    news_rss_feeds: dict[str, str] = field(default_factory=get_default_news_rss_feeds)
    cryptopanic_api_url: str = field(
        default_factory=get_default_cryptopanic_api_url
    )
    simulated_base_prices: dict[str, float] = field(
        default_factory=get_default_simulated_base_prices
    )
    default_quantity_pct: float = field(default_factory=get_default_quantity_pct)
    position_sizer_risk_pct: float = field(
        default_factory=get_default_position_sizer_risk_pct
    )
    knowledge_graph_top_n: int = field(
        default_factory=get_default_knowledge_graph_top_n
    )
    knowledge_graph_min_occurrences: int = field(
        default_factory=get_default_knowledge_graph_min_occurrences
    )
    autotune_min_trades: int = field(default_factory=get_default_autotune_min_trades)
    autotune_sharpe_decay_threshold: float = field(
        default_factory=get_default_autotune_sharpe_decay_threshold
    )
    autotune_win_rate_decay_threshold: float = field(
        default_factory=get_default_autotune_win_rate_decay_threshold
    )
    autotune_drawdown_alert_threshold: float = field(
        default_factory=get_default_autotune_drawdown_alert_threshold
    )
    autotune_param_ranges: dict[str, dict[str, float | int]] = field(
        default_factory=get_default_autotune_param_ranges
    )
    autotune_param_combos: list[dict[str, int]] = field(
        default_factory=get_default_autotune_param_combos
    )
    loop_interval_seconds: int = field(default_factory=get_default_loop_interval_seconds)
    bbands_stop_loss_pct: float = field(default_factory=get_default_bbands_stop_loss_pct)
    dspy_default_model: str = field(default_factory=get_default_dspy_model)

    # Logging / external tooling
    log_app_name: str = field(default_factory=get_default_log_app_name)
    log_error_name: str = field(default_factory=get_default_log_error_name)
    log_rotation: str = field(default_factory=get_default_log_rotation)
    log_retention: str = field(default_factory=get_default_log_retention)
    log_error_retention: str = field(
        default_factory=get_default_log_error_retention
    )
    wandb_project: str = field(
        default_factory=lambda: env_str("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    )
    wandb_api_key: str = field(default_factory=lambda: env_str("WANDB_API_KEY", ""))


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    """Load configuration from settings.yaml and .env files."""
    base_dir = BASE_DIR
    config = AppConfig()

    # Load .env
    env_file = Path(env_path) if env_path else base_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.info(f"Loaded .env from {env_file}")
    else:
        logger.warning(f"No .env file found at {env_file}")

    # Load settings.yaml
    yaml_path = Path(config_path) if config_path else base_dir / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            settings = yaml.safe_load(f) or {}

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
    config.binance_api_key = env_str("BINANCE_API_KEY", config.binance_api_key)
    config.binance_api_secret = env_str("BINANCE_API_SECRET", config.binance_api_secret)
    config.binance_testnet_api_key = env_str(
        "BINANCE_TESTNET_API_KEY",
        config.binance_testnet_api_key,
    )
    config.binance_testnet_api_secret = env_str(
        "BINANCE_TESTNET_API_SECRET",
        config.binance_testnet_api_secret,
    )
    config.openai_api_key = env_str("OPENAI_API_KEY", config.openai_api_key)
    config.anthropic_api_key = env_str(
        "ANTHROPIC_API_KEY",
        config.anthropic_api_key,
    )
    config.redis_url = get_default_redis_url()
    config.questdb_addr = get_default_questdb_http_addr()
    config.questdb_ilp_addr = get_default_questdb_ilp_addr()
    config.questdb_pg_host = env_str("QUESTDB_PG_HOST", config.questdb_pg_host)
    config.questdb_pg_port = env_int("QUESTDB_PG_PORT", config.questdb_pg_port)
    config.questdb_pg_user = env_str("QUESTDB_PG_USER", config.questdb_pg_user)
    config.questdb_pg_password = env_str(
        "QUESTDB_PG_PASSWORD",
        config.questdb_pg_password,
    )
    config.database_url = get_default_database_url()
    config.qdrant_url = get_default_qdrant_url()
    config.pg_pool_min_size = get_default_pg_pool_min_size()
    config.pg_pool_max_size = get_default_pg_pool_max_size()
    config.pg_pool_command_timeout = get_default_pg_pool_command_timeout()
    config.redis_cache_ttl_seconds = get_default_redis_cache_ttl_seconds()
    config.litellm_model = get_default_litellm_model()
    config.litellm_fallback_model = get_default_fallback_litellm_model()
    config.debate_max_rounds = get_default_debate_max_rounds()
    config.debate_temperature = get_default_debate_temperature()
    config.debate_max_tokens = get_default_debate_max_tokens()
    config.debate_timeout_seconds = get_default_debate_timeout_seconds()
    config.debate_cache_ttl_seconds = get_default_debate_cache_ttl_seconds()
    config.debate_cache_max_entries = get_default_debate_cache_max_entries()
    config.llm_max_retries = get_default_llm_max_retries()
    config.llm_cache_ttl_seconds = get_default_llm_cache_ttl_seconds()
    config.llm_cache_max_entries = get_default_llm_cache_max_entries()
    config.llm_circuit_breaker_threshold = get_default_llm_circuit_breaker_threshold()
    config.llm_circuit_breaker_reset_seconds = (
        get_default_llm_circuit_breaker_reset_seconds()
    )
    config.exchange_name = get_default_exchange_name()
    config.mem0_embedding_model = get_default_mem0_embedding_model()
    config.mem0_llm_model = get_default_mem0_llm_model()
    config.cryptopanic_api_key = get_default_cryptopanic_api_key()
    config.news_rss_feeds = get_default_news_rss_feeds()
    config.cryptopanic_api_url = get_default_cryptopanic_api_url()
    config.simulated_base_prices = get_default_simulated_base_prices()
    config.default_quantity_pct = get_default_quantity_pct()
    config.position_sizer_risk_pct = get_default_position_sizer_risk_pct()
    config.knowledge_graph_top_n = get_default_knowledge_graph_top_n()
    config.knowledge_graph_min_occurrences = (
        get_default_knowledge_graph_min_occurrences()
    )
    config.autotune_min_trades = get_default_autotune_min_trades()
    config.autotune_sharpe_decay_threshold = (
        get_default_autotune_sharpe_decay_threshold()
    )
    config.autotune_win_rate_decay_threshold = (
        get_default_autotune_win_rate_decay_threshold()
    )
    config.autotune_drawdown_alert_threshold = (
        get_default_autotune_drawdown_alert_threshold()
    )
    config.autotune_param_ranges = get_default_autotune_param_ranges()
    config.autotune_param_combos = get_default_autotune_param_combos()
    config.loop_interval_seconds = get_default_loop_interval_seconds()
    config.bbands_stop_loss_pct = get_default_bbands_stop_loss_pct()
    config.dspy_default_model = get_default_dspy_model()
    config.log_app_name = get_default_log_app_name()
    config.log_error_name = get_default_log_error_name()
    config.log_rotation = get_default_log_rotation()
    config.log_retention = get_default_log_retention()
    config.log_error_retention = get_default_log_error_retention()
    config.wandb_project = env_str("WANDB_PROJECT", config.wandb_project)
    config.wandb_api_key = env_str("WANDB_API_KEY", config.wandb_api_key)

    config.trading.mode = env_str("TRADING_MODE", config.trading.mode)
    config.trading.symbols = env_csv("TRADING_SYMBOLS", config.trading.symbols)
    config.trading.initial_capital = env_float(
        "INITIAL_CAPITAL",
        config.trading.initial_capital,
    )
    config.trading.interval = env_int(
        "LOOP_INTERVAL_SECONDS",
        config.trading.interval,
    )
    config.strategy.sma_fast = env_int("STRATEGY_SMA_FAST", config.strategy.sma_fast)
    config.strategy.sma_slow = env_int("STRATEGY_SMA_SLOW", config.strategy.sma_slow)
    config.strategy.rsi_period = env_int(
        "STRATEGY_RSI_PERIOD",
        config.strategy.rsi_period,
    )
    config.strategy.rsi_overbought = env_int(
        "STRATEGY_RSI_OVERBOUGHT",
        config.strategy.rsi_overbought,
    )
    config.strategy.rsi_oversold = env_int(
        "STRATEGY_RSI_OVERSOLD",
        config.strategy.rsi_oversold,
    )
    config.risk.max_daily_loss_pct = env_float(
        "MAX_DAILY_LOSS_PCT",
        config.risk.max_daily_loss_pct,
    )
    config.risk.max_drawdown_pct = env_float(
        "MAX_DRAWDOWN_PCT",
        config.risk.max_drawdown_pct,
    )
    config.risk.max_position_pct = env_float(
        "MAX_POSITION_PCT",
        config.risk.max_position_pct,
    )
    config.risk.max_leverage = env_int("MAX_LEVERAGE", config.risk.max_leverage)
    config.data.exchange = env_str("EXCHANGE_NAME", config.data.exchange)
    config.data.testnet = env_bool("DATA_TESTNET", config.data.testnet)
    config.data.candle_interval = env_str(
        "DATA_CANDLE_INTERVAL",
        config.data.candle_interval,
    )
    config.data.symbols = env_csv("DATA_SYMBOLS", config.data.symbols)
    config.monitoring.log_level = env_str("LOG_LEVEL", config.monitoring.log_level)
    config.monitoring.telegram_bot_token = env_str(
        "TELEGRAM_BOT_TOKEN",
        config.monitoring.telegram_bot_token,
    )
    config.monitoring.telegram_chat_id = env_str(
        "TELEGRAM_CHAT_ID",
        config.monitoring.telegram_chat_id,
    )
    config.monitoring.telegram_enabled = (
        env_bool("TELEGRAM_ENABLED", config.monitoring.telegram_enabled)
        or bool(config.monitoring.telegram_bot_token and config.monitoring.telegram_chat_id)
    )

    return config
