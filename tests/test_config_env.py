import json
from types import SimpleNamespace
from unittest.mock import patch

from src.config import load_config
from src.data.news_pipeline import NewsPipeline
from src.debate.llm_client import LLMClient
from src.debate.models import DebateConfig
from src.execution.position_sizer import PositionSizer
from src.memory.trade_memory import TradeMemory


def test_load_config_reads_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("QUESTDB_ILP_ADDR", "questdb:9009")
    monkeypatch.setenv("QUESTDB_PG_HOST", "questdb")
    monkeypatch.setenv("QUESTDB_PG_PORT", "8812")
    monkeypatch.setenv("QUESTDB_PG_USER", "admin")
    monkeypatch.setenv("QUESTDB_PG_PASSWORD", "quest")
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    monkeypatch.setenv("TRADING_SYMBOLS", "BTC/USDT,ETH/USDT")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "4.5")
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "12.5")
    monkeypatch.setenv("MAX_POSITION_PCT", "25")
    monkeypatch.setenv("MAX_LEVERAGE", "4")
    monkeypatch.setenv("STRATEGY_SMA_FAST", "12")
    monkeypatch.setenv("STRATEGY_SMA_SLOW", "48")
    monkeypatch.setenv("DEBATE_MAX_ROUNDS", "4")
    monkeypatch.setenv("LOOP_INTERVAL_SECONDS", "45")
    monkeypatch.setenv("MEM0_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("DSPY_DEFAULT_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("BBANDS_STOP_LOSS_PCT", "0.03")
    monkeypatch.setenv(
        "AUTOTUNE_PARAM_COMBOS",
        json.dumps([{"sma_fast": 9, "sma_slow": 31}]),
    )
    monkeypatch.setenv(
        "NEWS_RSS_FEEDS",
        json.dumps({"CoinDesk": "https://rss.example/coindesk.xml"}),
    )
    monkeypatch.setenv(
        "SIMULATED_BASE_PRICES",
        json.dumps({"BTC/USDT": 70000.0}),
    )

    config_path = tmp_path / "settings.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    config = load_config(
        config_path=str(config_path),
        env_path=str(tmp_path / ".env"),
    )

    assert config.openai_api_key == "openai-key"
    assert config.anthropic_api_key == "anthropic-key"
    assert config.questdb_ilp_addr == "questdb:9009"
    assert config.questdb_pg_host == "questdb"
    assert config.questdb_pg_port == 8812
    assert config.questdb_pg_user == "admin"
    assert config.questdb_pg_password == "quest"
    assert config.wandb_project == "env-project"
    assert config.wandb_api_key == "wandb-key"
    assert config.trading.symbols == ["BTC/USDT", "ETH/USDT"]
    assert config.risk.max_daily_loss_pct == 4.5
    assert config.risk.max_drawdown_pct == 12.5
    assert config.risk.max_position_pct == 25.0
    assert config.risk.max_leverage == 4
    assert config.strategy.sma_fast == 12
    assert config.strategy.sma_slow == 48
    assert config.debate_max_rounds == 4
    assert config.loop_interval_seconds == 45
    assert config.mem0_llm_model == "openai/gpt-4o-mini"
    assert config.dspy_default_model == "openai/gpt-4o-mini"
    assert config.bbands_stop_loss_pct == 0.03
    assert config.autotune_param_combos == [{"sma_fast": 9, "sma_slow": 31}]
    assert config.news_rss_feeds == {"CoinDesk": "https://rss.example/coindesk.xml"}
    assert config.simulated_base_prices == {"BTC/USDT": 70000.0}


def test_runtime_defaults_follow_env(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("DEBATE_MAX_ROUNDS", "2")
    monkeypatch.setenv("DEBATE_TEMPERATURE", "0.15")
    monkeypatch.setenv("DEBATE_MAX_TOKENS", "2048")
    monkeypatch.setenv("DEBATE_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("LLM_MAX_RETRIES", "7")
    monkeypatch.setenv("LLM_CACHE_TTL_SECONDS", "12")
    monkeypatch.setenv("LLM_CACHE_MAX_ENTRIES", "16")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "9")
    monkeypatch.setenv("LLM_CIRCUIT_BREAKER_RESET_SECONDS", "75")
    monkeypatch.setenv(
        "NEWS_RSS_FEEDS",
        json.dumps({"Example": "https://rss.example/feed.xml"}),
    )
    monkeypatch.setenv("CRYPTOPANIC_API_KEY", "panic-key")
    monkeypatch.setenv("CRYPTOPANIC_API_URL", "https://panic.example/api")
    monkeypatch.setenv("DATABASE_URL", "postgresql://env-db")
    monkeypatch.setenv("REDIS_URL", "redis://env-redis")
    monkeypatch.setenv("PG_POOL_MIN_SIZE", "4")
    monkeypatch.setenv("PG_POOL_MAX_SIZE", "12")
    monkeypatch.setenv("PG_POOL_COMMAND_TIMEOUT", "45")
    monkeypatch.setenv("REDIS_CACHE_TTL_SECONDS", "3600")
    monkeypatch.setenv("POSITION_SIZER_RISK_PCT", "0.05")

    debate = DebateConfig()
    news = NewsPipeline()
    memory = TradeMemory(enable_redis=False)
    with patch("src.debate.llm_client.litellm", SimpleNamespace()):
        client = LLMClient()

    assert debate.llm_model == "openai/gpt-4o-mini"
    assert debate.max_rounds == 2
    assert debate.temperature == 0.15
    assert debate.max_tokens == 2048
    assert debate.timeout_seconds == 45.0
    assert client.model == "openai/gpt-4o-mini"
    assert client.max_retries == 7
    assert client.cache_ttl_seconds == 12.0
    assert client.cache_max_entries == 16
    assert client.circuit_breaker_threshold == 9
    assert client.circuit_breaker_reset_seconds == 75.0
    assert news.cryptopanic_api_key == "panic-key"
    assert news.rss_feeds == {"Example": "https://rss.example/feed.xml"}
    assert news.cryptopanic_api_url == "https://panic.example/api"
    assert memory._db_url == "postgresql://env-db"
    assert memory._redis_url == "redis://env-redis"
    assert memory._pg_pool_min_size == 4
    assert memory._pg_pool_max_size == 12
    assert memory._pg_pool_command_timeout == 45
    assert memory._redis_cache_ttl_seconds == 3600
    assert PositionSizer.calc_van_tharp(100.0, 98.0, 10000.0) == 250.0


def test_load_config_exposes_cryptopanic_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYPTOPANIC_API_KEY", "panic-key")

    config_path = tmp_path / "settings.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    config = load_config(
        config_path=str(config_path),
        env_path=str(tmp_path / ".env"),
    )

    assert config.cryptopanic_api_key == "panic-key"


def test_load_config_merges_yaml_sections_and_alias_envs(tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ENABLED", "false")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat-id")
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("QUESTDB_HTTP_ADDR", raising=False)
        monkeypatch.setenv("DATA_REDIS_URL", "redis://alias-redis")
        monkeypatch.setenv("DATA_QUESTDB_ADDR", "questdb-alias:9000")

        config_path = tmp_path / "settings.yaml"
        config_path.write_text(
                """
trading:
    symbols:
        - SOL/USDT
strategy:
    name: bbands
    sma_fast: 11
monitoring:
    log_level: DEBUG
""".strip()
                + "\n",
                encoding="utf-8",
        )

        config = load_config(
                config_path=str(config_path),
                env_path=str(tmp_path / ".env"),
        )

        assert config.trading.symbols == ["SOL/USDT"]
        assert config.strategy.name == "bbands"
        assert config.strategy.sma_fast == 11
        assert config.monitoring.log_level == "DEBUG"
        assert config.redis_url == "redis://alias-redis"
        assert config.questdb_addr == "questdb-alias:9000"
        assert config.monitoring.telegram_enabled is True


def test_load_config_preserves_yaml_top_level_without_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATA_REDIS_URL", raising=False)
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    monkeypatch.delenv("NEWS_RSS_FEEDS", raising=False)
    monkeypatch.delenv("SIMULATED_BASE_PRICES", raising=False)

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        """
database_url: postgresql://yaml-db
redis_url: redis://yaml-redis
litellm_model: openai/yaml-model
news_rss_feeds:
    Example: https://rss.example/feed.xml
simulated_base_prices:
    BTC/USDT: 71000.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(
        config_path=str(config_path),
        env_path=str(tmp_path / ".env"),
    )

    assert config.database_url == "postgresql://yaml-db"
    assert config.redis_url == "redis://yaml-redis"
    assert config.litellm_model == "openai/yaml-model"
    assert config.news_rss_feeds == {"Example": "https://rss.example/feed.xml"}
    assert config.simulated_base_prices == {"BTC/USDT": 71000.0}
