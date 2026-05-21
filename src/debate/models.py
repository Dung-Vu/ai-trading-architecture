"""
Pydantic models for the AI Debate Engine.

All models use Pydantic v2 with strict typing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    get_default_debate_cache_max_entries,
    get_default_debate_cache_ttl_seconds,
    get_default_debate_max_rounds,
    get_default_debate_max_tokens,
    get_default_debate_temperature,
    get_default_debate_timeout_seconds,
    get_default_fallback_litellm_model,
    get_default_litellm_model,
    get_trading_symbols,
    env_float,
    env_int,
)


# ─── DebateRound ──────────────────────────────────────────────────────
class DebateRound(BaseModel):
    """Single round of debate from one agent."""

    round_number: int = Field(..., ge=1, description="Debate round number")
    agent_role: str = Field(
        ..., description="Role of the agent producing this round (bull, bear, devil)"
    )
    argument: str = Field(..., description="Full argument text from the agent")
    evidence: list[str] = Field(
        default_factory=list, description="List of specific evidence points cited"
    )
    stance: str = Field(
        ..., description="Agent's stance: BULLISH, BEARISH, SKEPTICAL"
    )
    raw_output: dict[str, Any] = Field(
        default_factory=dict, description="Raw JSON output from the LLM"
    )


# ─── DebateResult ─────────────────────────────────────────────────────
class DebateResult(BaseModel):
    """Final output of the complete debate process."""

    action: str = Field(
        ..., description="Final trading action: BUY, SELL, or HOLD"
    )
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence level (0-100)"
    )
    reason: str = Field(..., description="Synthesized reasoning for the decision")
    stop_loss: float = Field(..., ge=0.0, description="Stop-loss price level")
    take_profit: float = Field(..., ge=0.0, description="Take-profit price level")
    bull_argument: str = Field(
        default="", description="Summary of the strongest bull argument"
    )
    bear_argument: str = Field(
        default="", description="Summary of the strongest bear argument"
    )
    devil_argument: str = Field(
        default="", description="Summary of the devil's advocate critique"
    )
    risk_decision: str = Field(
        default="APPROVE",
        description="Risk Manager decision: APPROVE, REJECT, REDUCE, or FLATTEN",
    )
    risk_reasoning: str = Field(
        default="", description="Risk Manager's reasoning"
    )
    rounds: list[DebateRound] = Field(
        default_factory=list, description="All debate rounds"
    )
    symbol: str = Field(default="", description="Trading symbol")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (latency, tokens, etc.)"
    )


# ─── AgentOutput ──────────────────────────────────────────────────────
class AgentOutput(BaseModel):
    """Structured output from any debate agent."""

    action: str = Field(
        ..., description="Recommended action: BUY, SELL, or HOLD"
    )
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Confidence in the recommendation (0-100)"
    )
    reasoning: str = Field(..., description="Detailed reasoning with specific numbers")
    key_indicators: list[str] = Field(
        default_factory=list, description="Key technical/fundamental indicators cited"
    )
    risk_factors: list[str] = Field(
        default_factory=list, description="Risk factors identified"
    )
    suggested_stop_loss: float | None = Field(
        default=None, ge=0.0, description="Suggested stop-loss price"
    )
    suggested_take_profit: float | None = Field(
        default=None, ge=0.0, description="Suggested take-profit price"
    )
    # Devil's Advocate specific fields
    bull_rebuttal: str | None = Field(
        default=None, description="Rebuttal of bull's strongest point"
    )
    bear_rebuttal: str | None = Field(
        default=None, description="Rebuttal of bear's strongest point"
    )


# ─── DebateConfig ─────────────────────────────────────────────────────
class DebateConfig(BaseModel):
    """Configuration for a debate session."""

    max_rounds: int = Field(
        default_factory=get_default_debate_max_rounds,
        ge=1,
        le=5,
        description="Maximum number of debate rounds",
    )
    llm_model: str = Field(
        default_factory=get_default_litellm_model,
        description="LiteLLM model identifier (provider/model)",
    )
    fallback_model: str = Field(
        default_factory=get_default_fallback_litellm_model,
        description="Fallback model if primary fails",
    )
    temperature: float = Field(
        default_factory=get_default_debate_temperature,
        ge=0.0,
        le=1.5,
        description="LLM temperature for creativity",
    )
    symbols: list[str] = Field(
        default_factory=get_trading_symbols,
        description="Trading symbols to debate",
    )
    market_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Market data payload (price, indicators, volume, news, etc.)",
    )
    max_tokens: int = Field(
        default_factory=get_default_debate_max_tokens,
        ge=256,
        description="Max tokens for LLM response",
    )
    timeout_seconds: float = Field(
        default_factory=get_default_debate_timeout_seconds,
        gt=0.0,
        description="Timeout per LLM call in seconds",
    )
    result_cache_ttl_seconds: float = Field(
        default_factory=get_default_debate_cache_ttl_seconds,
        ge=0.0,
        description="TTL for exact debate-result caching",
    )
    result_cache_max_entries: int = Field(
        default_factory=get_default_debate_cache_max_entries,
        ge=1,
        description="Maximum number of cached debate results",
    )
    risk_max_daily_loss_pct: float = Field(
        default_factory=lambda: env_float("MAX_DAILY_LOSS_PCT", 3.0),
        ge=0.0,
        description="Risk manager max daily loss percentage",
    )
    risk_max_drawdown_pct: float = Field(
        default_factory=lambda: env_float("MAX_DRAWDOWN_PCT", 10.0),
        ge=0.0,
        description="Risk manager max drawdown percentage",
    )
    risk_max_position_pct: float = Field(
        default_factory=lambda: env_float("MAX_POSITION_PCT", 20.0),
        ge=0.0,
        description="Risk manager max single-position concentration percentage",
    )
    risk_max_leverage: int = Field(
        default_factory=lambda: env_int("MAX_LEVERAGE", 3),
        ge=1,
        description="Risk manager max leverage",
    )
