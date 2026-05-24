"""Public config facade for the trading architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config_env import (
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json,
    env_str,
    env_str_alias,
)
from src.config_sections import (
    DataConfig,
    MonitoringConfig,
    RiskConfig,
    StrategyConfig,
    TradingConfig,
)
from src.config_defaults import *  # noqa: F403


def build_trading_config() -> TradingConfig:
    return TradingConfig(
        mode=env_str("TRADING_MODE", "dryrun"),
        symbols=get_trading_symbols(),
        initial_capital=get_default_initial_capital(),
        interval=get_default_loop_interval_seconds(),
    )


def build_strategy_config() -> StrategyConfig:
    return StrategyConfig(
        name=env_str("STRATEGY_NAME", "sma_cross"),
        sma_fast=env_int("STRATEGY_SMA_FAST", 20),
        sma_slow=env_int("STRATEGY_SMA_SLOW", 50),
        rsi_period=env_int("STRATEGY_RSI_PERIOD", 14),
        rsi_overbought=env_int("STRATEGY_RSI_OVERBOUGHT", 70),
        rsi_oversold=env_int("STRATEGY_RSI_OVERSOLD", 30),
    )


def build_risk_config() -> RiskConfig:
    return RiskConfig(
        max_daily_loss_pct=env_float("MAX_DAILY_LOSS_PCT", 3.0),
        max_drawdown_pct=env_float("MAX_DRAWDOWN_PCT", 10.0),
        max_position_pct=env_float("MAX_POSITION_PCT", 20.0),
        max_leverage=env_int("MAX_LEVERAGE", 3),
    )


def build_data_config() -> DataConfig:
    return DataConfig(
        exchange=get_default_exchange_name(),
        testnet=env_bool("DATA_TESTNET", True),
        candle_interval=env_str("DATA_CANDLE_INTERVAL", "1m"),
        symbols=get_data_symbols(),
    )


def build_monitoring_config() -> MonitoringConfig:
    return MonitoringConfig(
        telegram_enabled=env_bool("TELEGRAM_ENABLED", False),
        telegram_bot_token=env_str("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=env_str("TELEGRAM_CHAT_ID", ""),
        log_level=env_str("LOG_LEVEL", "INFO"),
    )


@dataclass
class AppConfig:
    trading: TradingConfig = field(default_factory=build_trading_config)
    strategy: StrategyConfig = field(default_factory=build_strategy_config)
    risk: RiskConfig = field(default_factory=build_risk_config)
    data: DataConfig = field(default_factory=build_data_config)
    monitoring: MonitoringConfig = field(default_factory=build_monitoring_config)

    # Exchange API
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""
    openai_api_key: str = field(default_factory=lambda: env_str("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: env_str("ANTHROPIC_API_KEY", "")
    )
    anthropic_auth_token: str = field(
        default_factory=lambda: env_str("ANTHROPIC_AUTH_TOKEN", "")
    )
    bailian_api_key: str = field(
        default_factory=lambda: env_str_alias(
            ("BAILIAN_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DASHSCOPE_API_KEY"),
            "",
        )
    )
    dashscope_api_key: str = field(default_factory=lambda: env_str("DASHSCOPE_API_KEY", ""))
    opencode_api_key: str = field(default_factory=lambda: env_str("OPENCODE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: env_str("DEEPSEEK_API_KEY", ""))

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
    bailian_base_url: str = field(default_factory=get_default_bailian_base_url)
    dashscope_api_base: str = field(default_factory=get_default_dashscope_api_base)
    opencode_go_base_url: str = field(default_factory=get_default_opencode_go_base_url)
    deepseek_base_url: str = field(default_factory=get_default_deepseek_base_url)
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
    mem0_llm_provider: str = field(default_factory=get_default_mem0_llm_provider)
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


SECTION_TYPES: dict[str, type[Any]] = {
    "trading": TradingConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "data": DataConfig,
    "monitoring": MonitoringConfig,
}


APP_CONFIG_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("binance_api_key", lambda current: env_str("BINANCE_API_KEY", current)),
    ("binance_api_secret", lambda current: env_str("BINANCE_API_SECRET", current)),
    (
        "binance_testnet_api_key",
        lambda current: env_str("BINANCE_TESTNET_API_KEY", current),
    ),
    (
        "binance_testnet_api_secret",
        lambda current: env_str("BINANCE_TESTNET_API_SECRET", current),
    ),
    ("openai_api_key", lambda current: env_str("OPENAI_API_KEY", current)),
    ("anthropic_api_key", lambda current: env_str("ANTHROPIC_API_KEY", current)),
    ("anthropic_auth_token", lambda current: env_str("ANTHROPIC_AUTH_TOKEN", current)),
    (
        "bailian_api_key",
        lambda current: env_str_alias(
            ("BAILIAN_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DASHSCOPE_API_KEY"),
            current,
        ),
    ),
    ("dashscope_api_key", lambda current: env_str("DASHSCOPE_API_KEY", current)),
    ("opencode_api_key", lambda current: env_str("OPENCODE_API_KEY", current)),
    ("deepseek_api_key", lambda current: env_str("DEEPSEEK_API_KEY", current)),
    ("redis_url", lambda current: env_str_alias(("REDIS_URL", "DATA_REDIS_URL"), current)),
    (
        "questdb_addr",
        lambda current: env_str_alias(("QUESTDB_HTTP_ADDR", "DATA_QUESTDB_ADDR"), current),
    ),
    ("questdb_ilp_addr", lambda current: env_str("QUESTDB_ILP_ADDR", current)),
    ("questdb_pg_host", lambda current: env_str("QUESTDB_PG_HOST", current)),
    ("questdb_pg_port", lambda current: env_int("QUESTDB_PG_PORT", current)),
    ("questdb_pg_user", lambda current: env_str("QUESTDB_PG_USER", current)),
    ("questdb_pg_password", lambda current: env_str("QUESTDB_PG_PASSWORD", current)),
    ("database_url", lambda current: env_str("DATABASE_URL", current)),
    ("qdrant_url", lambda current: env_str("QDRANT_URL", current)),
    ("pg_pool_min_size", lambda current: env_int("PG_POOL_MIN_SIZE", current)),
    ("pg_pool_max_size", lambda current: env_int("PG_POOL_MAX_SIZE", current)),
    (
        "pg_pool_command_timeout",
        lambda current: env_int("PG_POOL_COMMAND_TIMEOUT", current),
    ),
    (
        "redis_cache_ttl_seconds",
        lambda current: env_int("REDIS_CACHE_TTL_SECONDS", current),
    ),
    ("litellm_model", lambda current: env_str("LITELLM_MODEL", current)),
    (
        "litellm_fallback_model",
        lambda current: env_str("LITELLM_FALLBACK_MODEL", current),
    ),
    (
        "bailian_base_url",
        lambda current: env_str_alias(("BAILIAN_BASE_URL", "ANTHROPIC_BASE_URL"), current),
    ),
    ("dashscope_api_base", lambda current: env_str("DASHSCOPE_API_BASE", current)),
    (
        "opencode_go_base_url",
        lambda current: env_str_alias(("OPENCODE_GO_BASE_URL", "OPENCODE_BASE_URL"), current),
    ),
    ("deepseek_base_url", lambda current: env_str("DEEPSEEK_BASE_URL", current)),
    ("debate_max_rounds", lambda current: env_int("DEBATE_MAX_ROUNDS", current)),
    ("debate_temperature", lambda current: env_float("DEBATE_TEMPERATURE", current)),
    ("debate_max_tokens", lambda current: env_int("DEBATE_MAX_TOKENS", current)),
    (
        "debate_timeout_seconds",
        lambda current: env_float("DEBATE_TIMEOUT_SECONDS", current),
    ),
    (
        "debate_cache_ttl_seconds",
        lambda current: env_float("DEBATE_RESULT_CACHE_TTL_SECONDS", current),
    ),
    (
        "debate_cache_max_entries",
        lambda current: env_int("DEBATE_RESULT_CACHE_MAX_ENTRIES", current),
    ),
    ("llm_max_retries", lambda current: env_int("LLM_MAX_RETRIES", current)),
    (
        "llm_cache_ttl_seconds",
        lambda current: env_float("LLM_CACHE_TTL_SECONDS", current),
    ),
    (
        "llm_cache_max_entries",
        lambda current: env_int("LLM_CACHE_MAX_ENTRIES", current),
    ),
    (
        "llm_circuit_breaker_threshold",
        lambda current: env_int("LLM_CIRCUIT_BREAKER_THRESHOLD", current),
    ),
    (
        "llm_circuit_breaker_reset_seconds",
        lambda current: env_float("LLM_CIRCUIT_BREAKER_RESET_SECONDS", current),
    ),
    ("exchange_name", lambda current: env_str("EXCHANGE_NAME", current)),
    ("mem0_embedding_model", lambda current: env_str("MEM0_EMBEDDING_MODEL", current)),
    ("mem0_llm_provider", lambda current: env_str("MEM0_LLM_PROVIDER", current)),
    ("mem0_llm_model", lambda current: env_str("MEM0_LLM_MODEL", current)),
    ("cryptopanic_api_key", lambda current: env_str("CRYPTOPANIC_API_KEY", current)),
    ("news_rss_feeds", lambda current: env_json("NEWS_RSS_FEEDS", current)),
    ("cryptopanic_api_url", lambda current: env_str("CRYPTOPANIC_API_URL", current)),
    ("simulated_base_prices", lambda current: env_json("SIMULATED_BASE_PRICES", current)),
    ("default_quantity_pct", lambda current: env_float("DEFAULT_QUANTITY_PCT", current)),
    (
        "position_sizer_risk_pct",
        lambda current: env_float("POSITION_SIZER_RISK_PCT", current),
    ),
    (
        "knowledge_graph_top_n",
        lambda current: env_int("KNOWLEDGE_GRAPH_TOP_N", current),
    ),
    (
        "knowledge_graph_min_occurrences",
        lambda current: env_int("KNOWLEDGE_GRAPH_MIN_OCCURRENCES", current),
    ),
    ("autotune_min_trades", lambda current: env_int("AUTOTUNE_MIN_TRADES", current)),
    (
        "autotune_sharpe_decay_threshold",
        lambda current: env_float("AUTOTUNE_SHARPE_DECAY_THRESHOLD", current),
    ),
    (
        "autotune_win_rate_decay_threshold",
        lambda current: env_float("AUTOTUNE_WIN_RATE_DECAY_THRESHOLD", current),
    ),
    (
        "autotune_drawdown_alert_threshold",
        lambda current: env_float("AUTOTUNE_DRAWDOWN_ALERT_THRESHOLD", current),
    ),
    ("autotune_param_ranges", lambda current: env_json("AUTOTUNE_PARAM_RANGES", current)),
    ("autotune_param_combos", lambda current: env_json("AUTOTUNE_PARAM_COMBOS", current)),
    (
        "loop_interval_seconds",
        lambda current: env_int("LOOP_INTERVAL_SECONDS", current),
    ),
    (
        "bbands_stop_loss_pct",
        lambda current: env_float("BBANDS_STOP_LOSS_PCT", current),
    ),
    ("dspy_default_model", lambda current: env_str("DSPY_DEFAULT_MODEL", current)),
    ("log_app_name", lambda current: env_str("LOG_APP_NAME", current)),
    ("log_error_name", lambda current: env_str("LOG_ERROR_NAME", current)),
    ("log_rotation", lambda current: env_str("LOG_ROTATION", current)),
    ("log_retention", lambda current: env_str("LOG_RETENTION", current)),
    (
        "log_error_retention",
        lambda current: env_str("LOG_ERROR_RETENTION", current),
    ),
    ("wandb_project", lambda current: env_str("WANDB_PROJECT", current)),
    ("wandb_api_key", lambda current: env_str("WANDB_API_KEY", current)),
)


from src.config_loader import load_config
