"""Config loading and override orchestration for the public src.config facade."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.config import (
    APP_CONFIG_DEFAULTS,
    AppConfig,
    BASE_DIR,
    DEFAULT_AUTOTUNE_PARAM_COMBOS,
    DEFAULT_AUTOTUNE_PARAM_RANGES,
    DEFAULT_NEWS_RSS_FEEDS,
    DEFAULT_SIMULATED_BASE_PRICES,
    SECTION_TYPES,
)
from src.config_env import env_bool, env_csv, env_float, env_int, env_json, env_str, env_str_alias


@dataclass(frozen=True, slots=True)
class ConfigOverride:
    path: str
    resolver: Callable[[Any], Any]


def _get_config_value(config: Any, path: str) -> Any:
    current = config
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _set_config_value(config: Any, path: str, value: Any) -> None:
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _current_or_default_dict(current: Any, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(current, dict):
        return copy.deepcopy(current)
    return copy.deepcopy(default)


def _resolve_json_mapping(
    name: str,
    current: Any,
    default: dict[str, Any],
) -> dict[str, Any]:
    fallback = _current_or_default_dict(current, default)
    raw = env_json(name, fallback)
    return dict(raw) if isinstance(raw, dict) else fallback


def _coerce_param_combos(
    raw: Any,
    fallback: list[dict[str, int]],
) -> list[dict[str, int]]:
    if not isinstance(raw, list):
        return copy.deepcopy(fallback)

    combos: list[dict[str, int]] = []
    for item in raw:
        if isinstance(item, dict):
            combos.append({str(key): int(value) for key, value in item.items()})
    return combos or copy.deepcopy(fallback)


def _resolve_param_combos(current: Any) -> list[dict[str, int]]:
    fallback = (
        copy.deepcopy(current)
        if isinstance(current, list)
        else copy.deepcopy(DEFAULT_AUTOTUNE_PARAM_COMBOS)
    )
    raw = env_json("AUTOTUNE_PARAM_COMBOS", fallback)
    return _coerce_param_combos(raw, fallback)


def _merge_yaml_section(current: Any, section_type: type[Any], values: dict[str, Any]) -> Any:
    merged = {
        config_field.name: copy.deepcopy(getattr(current, config_field.name))
        for config_field in fields(section_type)
    }
    merged.update(values)
    return section_type(**merged)


def _apply_yaml_sections(config: AppConfig, settings: dict[str, Any]) -> None:
    for section_name, section_type in SECTION_TYPES.items():
        raw_section = settings.get(section_name)
        if isinstance(raw_section, dict):
            current = getattr(config, section_name)
            setattr(
                config,
                section_name,
                _merge_yaml_section(current, section_type, raw_section),
            )


def _apply_yaml_top_level_fields(config: AppConfig, settings: dict[str, Any]) -> None:
    section_names = set(SECTION_TYPES)
    field_names = {config_field.name for config_field in fields(AppConfig)}
    for key, value in settings.items():
        if key in section_names or key not in field_names:
            continue
        setattr(config, key, copy.deepcopy(value))


def _apply_overrides(config: AppConfig, overrides: tuple[ConfigOverride, ...]) -> None:
    for override in overrides:
        current = _get_config_value(config, override.path)
        _set_config_value(config, override.path, override.resolver(current))


APP_CONFIG_OVERRIDES: tuple[ConfigOverride, ...] = tuple(
    ConfigOverride(path, resolver) for path, resolver in APP_CONFIG_DEFAULTS
)


SECTION_OVERRIDES: tuple[ConfigOverride, ...] = (
    ConfigOverride("trading.mode", lambda current: env_str("TRADING_MODE", current)),
    ConfigOverride("trading.symbols", lambda current: env_csv("TRADING_SYMBOLS", current)),
    ConfigOverride(
        "trading.initial_capital",
        lambda current: env_float("INITIAL_CAPITAL", current),
    ),
    ConfigOverride(
        "trading.interval",
        lambda current: env_int("LOOP_INTERVAL_SECONDS", current),
    ),
    ConfigOverride("strategy.name", lambda current: env_str("STRATEGY_NAME", current)),
    ConfigOverride("strategy.sma_fast", lambda current: env_int("STRATEGY_SMA_FAST", current)),
    ConfigOverride("strategy.sma_slow", lambda current: env_int("STRATEGY_SMA_SLOW", current)),
    ConfigOverride(
        "strategy.rsi_period",
        lambda current: env_int("STRATEGY_RSI_PERIOD", current),
    ),
    ConfigOverride(
        "strategy.rsi_overbought",
        lambda current: env_int("STRATEGY_RSI_OVERBOUGHT", current),
    ),
    ConfigOverride(
        "strategy.rsi_oversold",
        lambda current: env_int("STRATEGY_RSI_OVERSOLD", current),
    ),
    ConfigOverride(
        "risk.max_daily_loss_pct",
        lambda current: env_float("MAX_DAILY_LOSS_PCT", current),
    ),
    ConfigOverride(
        "risk.max_drawdown_pct",
        lambda current: env_float("MAX_DRAWDOWN_PCT", current),
    ),
    ConfigOverride(
        "risk.max_position_pct",
        lambda current: env_float("MAX_POSITION_PCT", current),
    ),
    ConfigOverride("risk.max_leverage", lambda current: env_int("MAX_LEVERAGE", current)),
    ConfigOverride("data.exchange", lambda current: env_str("EXCHANGE_NAME", current)),
    ConfigOverride("data.testnet", lambda current: env_bool("DATA_TESTNET", current)),
    ConfigOverride(
        "data.candle_interval",
        lambda current: env_str("DATA_CANDLE_INTERVAL", current),
    ),
    ConfigOverride("data.symbols", lambda current: env_csv("DATA_SYMBOLS", current)),
    ConfigOverride(
        "monitoring.log_level",
        lambda current: env_str("LOG_LEVEL", current),
    ),
    ConfigOverride(
        "monitoring.telegram_bot_token",
        lambda current: env_str("TELEGRAM_BOT_TOKEN", current),
    ),
    ConfigOverride(
        "monitoring.telegram_chat_id",
        lambda current: env_str("TELEGRAM_CHAT_ID", current),
    ),
)


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    """Load configuration from settings.yaml and .env files."""
    base_dir = BASE_DIR
    config = AppConfig()

    env_file = Path(env_path) if env_path else base_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.info(f"Loaded .env from {env_file}")
    else:
        logger.warning(f"No .env file found at {env_file}")

    yaml_path = Path(config_path) if config_path else base_dir / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            settings = yaml.safe_load(f) or {}

        _apply_yaml_sections(config, settings)
        _apply_yaml_top_level_fields(config, settings)
        logger.info(f"Loaded settings from {yaml_path}")
    else:
        logger.warning(f"No settings.yaml found at {yaml_path}")

    _apply_overrides(config, APP_CONFIG_OVERRIDES)
    _apply_overrides(config, SECTION_OVERRIDES)
    config.monitoring.telegram_enabled = (
        env_bool("TELEGRAM_ENABLED", config.monitoring.telegram_enabled)
        or bool(config.monitoring.telegram_bot_token and config.monitoring.telegram_chat_id)
    )
    return config