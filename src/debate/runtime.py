"""Shared runtime helpers for constructing debate engine components."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from loguru import logger

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


async def run_debate_round(
    debate_engine: Any,
    *,
    market_data: dict[str, Any],
    current_positions: dict[str, Any] | None = None,
    portfolio: dict[str, Any] | None = None,
    symbol: str = "BTC/USDT",
) -> Any:
    """Run a blocking debate engine call in the shared executor path."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: debate_engine.run_debate(
            market_data=market_data,
            current_positions=current_positions or {},
            portfolio=portfolio or {},
            symbol=symbol,
        ),
    )


def normalize_debate_result(
    result: Any,
    *,
    include_reason_alias: bool = False,
    include_round_count: bool = False,
) -> dict[str, Any]:
    """Normalize debate-engine outputs into the dict shape expected by bot layers."""
    if hasattr(result, "model_dump"):
        raw = result.model_dump()
    elif hasattr(result, "dict"):
        raw = result.dict()
    else:
        raw = {}

    reasoning = raw.get("reasoning", raw.get("reason", getattr(result, "reason", "")))
    rounds = raw.get("rounds", getattr(result, "rounds", []))

    normalized = {
        "action": raw.get("action", getattr(result, "action", "HOLD")),
        "confidence": raw.get("confidence", getattr(result, "confidence", 50.0)),
        "reasoning": reasoning,
        "stop_loss": raw.get("stop_loss", getattr(result, "stop_loss", 0.0)),
        "take_profit": raw.get("take_profit", getattr(result, "take_profit", 0.0)),
        "bull_argument": raw.get(
            "bull_argument",
            getattr(result, "bull_argument", ""),
        ),
        "bear_argument": raw.get(
            "bear_argument",
            getattr(result, "bear_argument", ""),
        ),
        "devil_argument": raw.get(
            "devil_argument",
            getattr(result, "devil_argument", ""),
        ),
        "risk_decision": raw.get(
            "risk_decision",
            getattr(result, "risk_decision", "APPROVE"),
        ),
        "risk_reasoning": raw.get(
            "risk_reasoning",
            getattr(result, "risk_reasoning", ""),
        ),
    }

    if include_reason_alias:
        normalized["reason"] = reasoning

    if include_round_count:
        if isinstance(rounds, list):
            normalized["rounds"] = len(rounds)
        elif isinstance(rounds, int):
            normalized["rounds"] = rounds
        else:
            logger.debug(f"Unexpected debate rounds payload type: {type(rounds).__name__}")
            normalized["rounds"] = 0

    return normalized


def build_llm_api_keys(config: Any) -> dict[str, str] | None:
    """Collect configured provider API keys for LiteLLM routing."""
    api_keys = {
        provider: key
        for provider, key in {
            "bailian": getattr(config, "bailian_api_key", ""),
            "dashscope": getattr(config, "dashscope_api_key", ""),
            "opencode-go": getattr(config, "opencode_api_key", ""),
            "deepseek": getattr(config, "deepseek_api_key", ""),
            "openai": getattr(config, "openai_api_key", ""),
            "anthropic": getattr(config, "anthropic_api_key", "")
            or getattr(config, "anthropic_auth_token", ""),
        }.items()
        if key
    }
    return api_keys or None


def build_llm_api_bases(config: Any) -> dict[str, str] | None:
    """Collect configured provider base URLs for LiteLLM routing."""
    api_bases = {
        provider: api_base
        for provider, api_base in {
            "bailian": getattr(config, "bailian_base_url", ""),
            "dashscope": getattr(config, "dashscope_api_base", ""),
            "opencode-go": getattr(config, "opencode_go_base_url", ""),
            "deepseek": getattr(config, "deepseek_base_url", ""),
        }.items()
        if api_base
    }
    return api_bases or None


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
        api_bases=build_llm_api_bases(config),
        fallback_model=getattr(
            config,
            "litellm_fallback_model",
            get_default_fallback_litellm_model(),
        ),
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
