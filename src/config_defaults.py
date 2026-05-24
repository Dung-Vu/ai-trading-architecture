"""Env-backed defaults for the public src.config facade."""

from __future__ import annotations

import copy
from pathlib import Path

from dotenv import load_dotenv

from src.config_env import (
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
DEFAULT_BAILIAN_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"
DEFAULT_DASHSCOPE_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LITELLM_MODEL = "bailian/qwen3.6-plus"
DEFAULT_FALLBACK_LLM_MODEL = "opencode-go/deepseek-v4-pro"
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
DEFAULT_MEM0_LLM_PROVIDER = "dashscope"
DEFAULT_MEM0_LLM_MODEL = "qwen3.6-plus"
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
DEFAULT_QUESTDB_PG_PASSWORD = ""
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


def get_default_bailian_base_url() -> str:
    return env_str_alias(
        ("BAILIAN_BASE_URL", "ANTHROPIC_BASE_URL"),
        DEFAULT_BAILIAN_BASE_URL,
    )


def get_default_dashscope_api_base() -> str:
    return env_str("DASHSCOPE_API_BASE", DEFAULT_DASHSCOPE_API_BASE)


def get_default_opencode_go_base_url() -> str:
    return env_str_alias(
        ("OPENCODE_GO_BASE_URL", "OPENCODE_BASE_URL"),
        DEFAULT_OPENCODE_GO_BASE_URL,
    )


def get_default_deepseek_base_url() -> str:
    return env_str("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)


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


def get_default_mem0_llm_provider() -> str:
    return env_str("MEM0_LLM_PROVIDER", DEFAULT_MEM0_LLM_PROVIDER)


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


__all__ = [
    "BASE_DIR",
    *(name for name in globals() if name.startswith("DEFAULT_")),
    *(name for name in globals() if name.startswith("get_")),
]
