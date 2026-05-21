"""Shared runtime helpers for constructing debate engine components."""

from __future__ import annotations

from typing import Any, Sequence

from src.config import (
    get_default_debate_cache_max_entries,
    get_default_debate_cache_ttl_seconds,
    get_default_debate_max_rounds,
    get_default_debate_max_tokens,
    get_default_debate_temperature,
    get_default_debate_timeout_seconds,
    get_default_fallback_litellm_model,
    get_default_litellm_model,
)

from .debate_engine import DebateEngine
from .llm_client import LLMClient
from .models import DebateConfig


def build_llm_api_keys(config: Any) -> dict[str, str] | None:
    """Collect configured provider API keys for LiteLLM routing."""
    api_keys = {
        provider: key
        for provider, key in {
            "openai": getattr(config, "openai_api_key", ""),
            "anthropic": getattr(config, "anthropic_api_key", ""),
        }.items()
        if key
    }
    return api_keys or None


def build_debate_engine(
    config: Any,
    symbols: Sequence[str],
) -> tuple[DebateEngine, str]:
    """Create a fully-configured debate engine for the supplied symbols."""
    llm_model = getattr(config, "litellm_model", get_default_litellm_model())
    debate_temperature = getattr(
        config,
        "debate_temperature",
        get_default_debate_temperature(),
    )

    llm_client = LLMClient(
        model=llm_model,
        temperature=debate_temperature,
        api_keys=build_llm_api_keys(config),
    )

    risk_config = getattr(config, "risk", None)
    risk_kwargs = {}
    if risk_config is not None:
        risk_kwargs = {
            "risk_max_daily_loss_pct": risk_config.max_daily_loss_pct,
            "risk_max_drawdown_pct": risk_config.max_drawdown_pct,
            "risk_max_position_pct": risk_config.max_position_pct,
            "risk_max_leverage": risk_config.max_leverage,
        }

    debate_config = DebateConfig(
        max_rounds=getattr(
            config,
            "debate_max_rounds",
            get_default_debate_max_rounds(),
        ),
        llm_model=llm_model,
        fallback_model=getattr(
            config,
            "litellm_fallback_model",
            get_default_fallback_litellm_model(),
        ),
        temperature=debate_temperature,
        symbols=list(symbols),
        max_tokens=getattr(
            config,
            "debate_max_tokens",
            get_default_debate_max_tokens(),
        ),
        timeout_seconds=getattr(
            config,
            "debate_timeout_seconds",
            get_default_debate_timeout_seconds(),
        ),
        result_cache_ttl_seconds=getattr(
            config,
            "debate_cache_ttl_seconds",
            get_default_debate_cache_ttl_seconds(),
        ),
        result_cache_max_entries=getattr(
            config,
            "debate_cache_max_entries",
            get_default_debate_cache_max_entries(),
        ),
        **risk_kwargs,
    )

    return DebateEngine(debate_config, llm_client), llm_model
