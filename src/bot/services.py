"""Composition helpers for runtime services owned by trading bots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BOT_SERVICE_COMPAT_ATTRS = {
    "_trade_memory": "trade_memory",
    "_mem0_memory": "mem0_memory",
    "_knowledge_graph": "knowledge_graph",
    "_debate_engine": "debate_engine",
    "_risk_engine": "risk_engine",
    "_kill_switch": "kill_switch",
    "_executor": "executor",
    "_order_manager": "order_manager",
    "_news_pipeline": "news_pipeline",
    "_auto_tuner": "auto_tuner",
    "_weekly_reviewer": "weekly_reviewer",
    "_telegram_bot": "telegram_bot",
    "_redis_cache": "redis_cache",
    "_strategies": "strategies",
    "_dry_run_executor": "dry_run_executor",
}

FULL_BOT_RUNTIME_ATTRS = (
    "_trade_memory",
    "_mem0_memory",
    "_knowledge_graph",
    "_risk_engine",
    "_kill_switch",
    "_executor",
    "_order_manager",
    "_strategies",
    "_debate_engine",
    "_news_pipeline",
    "_auto_tuner",
    "_redis_cache",
    "_weekly_reviewer",
    "_telegram_bot",
)

LEAN_AI_BOT_RUNTIME_ATTRS = (
    "_telegram_bot",
    "_weekly_reviewer",
    "_debate_engine",
    "_strategies",
    "_redis_cache",
    "_order_manager",
    "_dry_run_executor",
    "_executor",
    "_kill_switch",
    "_risk_engine",
    "_trade_memory",
)


@dataclass(slots=True)
class BotServices:
    """Owned runtime services for trading bots.

    The bot classes keep their public/private attribute names for compatibility,
    but store the actual objects in this composition root.
    """

    trade_memory: Any = None
    mem0_memory: Any = None
    knowledge_graph: Any = None
    debate_engine: Any = None
    risk_engine: Any = None
    kill_switch: Any = None
    executor: Any = None
    order_manager: Any = None
    news_pipeline: Any = None
    auto_tuner: Any = None
    weekly_reviewer: Any = None
    telegram_bot: Any = None
    redis_cache: Any = None
    strategies: dict[str, dict[str, Any]] | None = field(default_factory=dict)
    dry_run_executor: Any = None

    def get_compat(self, attr_name: str) -> Any:
        return getattr(self, BOT_SERVICE_COMPAT_ATTRS[attr_name])

    def set_compat(self, attr_name: str, value: Any) -> None:
        setattr(self, BOT_SERVICE_COMPAT_ATTRS[attr_name], value)