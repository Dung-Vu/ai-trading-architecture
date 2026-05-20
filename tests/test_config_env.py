"""Configuration environment variable mapping tests."""

from src.config import load_config
from src.data.config import DataConfig


def test_app_config_builds_service_urls_from_docker_env(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "trading_db")
    monkeypatch.setenv("POSTGRES_USER", "trading_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pa ss")
    monkeypatch.setenv("QUESTDB_HOST", "questdb")
    monkeypatch.setenv("QUESTDB_PORT", "9000")

    config = load_config(config_path=str(tmp_path / "missing.yaml"), env_path=str(tmp_path / "missing.env"))

    assert config.database_url == "postgresql://trading_user:pa+ss@postgres:5432/trading_db"
    assert config.questdb_addr == "questdb:9000"


def test_data_config_accepts_shared_service_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("QUESTDB_HOST", "questdb")
    monkeypatch.setenv("QUESTDB_PORT", "9000")

    config = DataConfig.load_from_env()

    assert config.redis_url == "redis://redis:6379/0"
    assert config.questdb_addr == "questdb:9000"
