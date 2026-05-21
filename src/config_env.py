"""Shared environment parsing helpers for configuration modules."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from loguru import logger


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_str_alias(names: tuple[str, ...], default: str) -> str:
    """Return the first non-empty environment value from a set of aliases."""
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {value!r}. Using default {default}.")
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}: {value!r}. Using default {default}.")
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning(f"Invalid boolean for {name}: {value!r}. Using default {default}.")
    return default


def env_csv(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value in (None, ""):
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def env_json(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value in (None, ""):
        return copy.deepcopy(default)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON for {name}. Using default value.")
        return copy.deepcopy(default)