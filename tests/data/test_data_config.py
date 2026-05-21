import pytest

from src.data.config import DataConfig


def test_load_from_env_applies_overrides(monkeypatch):
    monkeypatch.setenv("DATA_SYMBOLS", "BTC-USDT,ETH-USDT")
    monkeypatch.setenv("DATA_CHANNELS", "TRADES,CANDLES")
    monkeypatch.setenv("DATA_CANDLE_INTERVAL", "5m")
    monkeypatch.setenv("REDIS_URL", "redis://cache")
    monkeypatch.setenv("QUESTDB_HTTP_ADDR", "questdb:9009")
    monkeypatch.setenv("DATA_MAX_LATENCY_MS", "2500")
    monkeypatch.setenv("DATA_Z_SCORE_THRESHOLD", "4.5")
    monkeypatch.setenv("DATA_MAX_SPREAD_PCT", "0.8")

    config = DataConfig.load_from_env()

    assert config.symbols == ["BTC-USDT", "ETH-USDT"]
    assert config.channels == ["TRADES", "CANDLES"]
    assert config.candle_interval == "5m"
    assert config.redis_url == "redis://cache"
    assert config.questdb_addr == "questdb:9009"
    assert config.max_latency_ms == 2500
    assert config.z_score_threshold == 4.5
    assert config.max_spread_pct == 0.8


def test_load_from_env_keeps_data_aliases_for_compatibility(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("QUESTDB_HTTP_ADDR", raising=False)
    monkeypatch.setenv("DATA_REDIS_URL", "redis://compat-cache")
    monkeypatch.setenv("DATA_QUESTDB_ADDR", "compat-questdb:9009")

    config = DataConfig.load_from_env()

    assert config.redis_url == "redis://compat-cache"
    assert config.questdb_addr == "compat-questdb:9009"


def test_data_config_validates_channels():
    with pytest.raises(ValueError, match="Invalid channel"):
        DataConfig(channels=["TRADES", "BAD_CHANNEL"])


def test_data_config_coerces_numeric_inputs():
    config = DataConfig(max_latency_ms="1200", z_score_threshold="3.5", max_spread_pct="0.6")

    assert config.max_latency_ms == 1200
    assert config.z_score_threshold == 3.5
    assert config.max_spread_pct == 0.6