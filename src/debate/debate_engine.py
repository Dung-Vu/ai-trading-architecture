"""
LangGraph-based multi-agent debate engine.

Implements a triangular adversarial debate flow:
  Bull → Bear → Devil → (repeat N rounds) → Judge → Risk Manager

Uses LangGraph StateGraph for workflow orchestration with conditional
edges based on round count.
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from typing import Any

from loguru import logger
from src.shared_utils import trim_mapping_size

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None  # type: ignore[assignment]
from typing_extensions import TypedDict

from .agents import BearAgent, BullAgent, DevilsAdvocate, JudgeAgent, RiskManagerAgent
from .llm_client import LLMClient
from .models import DebateConfig, DebateResult, DebateRound
from .prompts import (
    BEAR_SYSTEM_PROMPT,
    BULL_SYSTEM_PROMPT,
    DEVIL_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    RISK_MANAGER_SYSTEM_PROMPT,
)


# ─── LangGraph State ─────────────────────────────────────────────────
class DebateState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    # Input
    market_data: dict[str, Any]
    current_positions: dict[str, Any]
    portfolio: dict[str, Any]
    config: dict[str, Any]

    # Accumulated debate data
    current_round: int
    max_rounds: int
    debate_rounds: list[dict[str, Any]]

    # Per-round agent outputs
    bull_output: dict[str, Any]
    bear_output: dict[str, Any]
    devil_output: dict[str, Any]

    # Final outputs
    judge_output: dict[str, Any]
    risk_output: dict[str, Any]

    # Metadata
    symbol: str
    start_time: float


# ─── DebateEngine ─────────────────────────────────────────────────────
class DebateEngine:
    """
    LangGraph-based multi-agent debate engine.

    Manages a triangular adversarial debate between Bull, Bear, and
    Devil's Advocate agents, followed by Judge synthesis and Risk Manager review.
    """

    def __init__(
        self,
        config: DebateConfig,
        llm_client: LLMClient,
    ) -> None:
        """
        Initialize the debate engine.

        Args:
            config: Debate configuration.
            llm_client: LLM client for making API calls.
        """
        self.config = config
        self.llm_client = llm_client

        # Instantiate agents
        self.bull = BullAgent(llm_client, temperature=config.temperature)
        self.bear = BearAgent(llm_client, temperature=config.temperature)
        self.devil = DevilsAdvocate(llm_client, temperature=config.temperature)
        self.judge = JudgeAgent(llm_client, temperature=config.temperature)
        risk_config = SimpleNamespace(
            max_daily_loss_pct=config.risk_max_daily_loss_pct,
            max_drawdown_pct=config.risk_max_drawdown_pct,
            max_position_pct=config.risk_max_position_pct,
            max_leverage=config.risk_max_leverage,
        )
        self.risk_manager = RiskManagerAgent(
            llm_client,
            temperature=config.temperature,
            risk_config=risk_config,
        )
        self._result_cache: dict[str, tuple[float, DebateResult]] = {}

        # Build the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph for the debate workflow."""
        if StateGraph is None or START is None or END is None:
            raise ImportError(
                "langgraph is required to use DebateEngine. Install with: pip install langgraph"
            )

        workflow = StateGraph(DebateState)

        # Add nodes
        workflow.add_node("bull_argue", self._bull_node)
        workflow.add_node("bear_argue", self._bear_node)
        workflow.add_node("devil_challenge", self._devil_node)
        workflow.add_node("judge_synthesize", self._judge_node)
        workflow.add_node("risk_review", self._risk_node)
        workflow.add_node("increment_round", self._increment_round_node)

        # Entry point: start with Bull
        workflow.add_edge(START, "bull_argue")

        # Bull → Bear
        workflow.add_edge("bull_argue", "bear_argue")

        # Bear → Devil
        workflow.add_edge("bear_argue", "devil_challenge")

        # Devil → conditional: increment_round OR judge
        workflow.add_conditional_edges(
            "devil_challenge",
            self._should_continue_debate,
            {
                "continue": "increment_round",
                "judge": "judge_synthesize",
            },
        )

        # increment_round → back to Bull for next round
        workflow.add_edge("increment_round", "bull_argue")

        # Judge → Risk Manager
        workflow.add_edge("judge_synthesize", "risk_review")

        # Risk Manager → END
        workflow.add_edge("risk_review", END)

        return workflow.compile()

    def run_debate(
        self,
        market_data: dict[str, Any],
        current_positions: dict[str, Any] | None = None,
        portfolio: dict[str, Any] | None = None,
        symbol: str = "BTC/USDT",
    ) -> DebateResult:
        """
        Run the complete debate workflow.

        Args:
            market_data: Current market data (price, indicators, volume, etc.).
            current_positions: Current open positions.
            portfolio: Portfolio state (equity, daily P&L, drawdown, etc.).
            symbol: Trading symbol being debated.

        Returns:
            DebateResult with final action, confidence, SL, TP, and reasoning.
        """
        cache_key = self._build_cache_key(
            market_data=market_data,
            current_positions=current_positions,
            portfolio=portfolio,
            symbol=symbol,
        )
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            cached_result = cached.model_copy(deep=True)
            cached_result.metadata = {
                **cached_result.metadata,
                "cache_hit": True,
                "total_time_seconds": 0.0,
            }
            logger.info(f"Debate cache hit for {symbol}")
            return cached_result

        start = time.monotonic()

        initial_state: DebateState = {
            "market_data": market_data,
            "current_positions": current_positions or {},
            "portfolio": portfolio or {},
            "config": self.config.model_dump(),
            "current_round": 1,
            "max_rounds": self.config.max_rounds,
            "debate_rounds": [],
            "bull_output": {},
            "bear_output": {},
            "devil_output": {},
            "judge_output": {},
            "risk_output": {},
            "symbol": symbol,
            "start_time": start,
        }

        logger.info(f"Starting debate for {symbol} — max {self.config.max_rounds} rounds")

        # Execute the graph
        final_state = self.graph.invoke(initial_state)

        # Build result from final state
        result = self._build_result(final_state, time.monotonic() - start)

        stats = self.llm_client.get_stats()
        result.metadata["llm_stats"] = stats
        result.metadata["total_time_seconds"] = round(time.monotonic() - start, 2)
        result.metadata["cache_hit"] = False

        self._cache_result(cache_key, result)

        logger.info(
            f"Debate complete for {symbol}: "
            f"action={result.action}, confidence={result.confidence}, "
            f"risk_decision={result.risk_decision}, "
            f"time={result.metadata['total_time_seconds']}s, "
            f"LLM calls={stats['total_calls']}"
        )

        return result

    def _build_cache_key(
        self,
        *,
        market_data: dict[str, Any],
        current_positions: dict[str, Any] | None,
        portfolio: dict[str, Any] | None,
        symbol: str,
    ) -> str:
        """Build a stable cache key for an exact debate input snapshot."""
        payload = {
            "symbol": symbol,
            "market_data": self._normalize_for_cache(market_data),
            "current_positions": self._normalize_for_cache(current_positions or {}),
            "portfolio": self._normalize_for_cache(portfolio or {}),
            "max_rounds": self.config.max_rounds,
            "temperature": self.config.temperature,
            "llm_model": self.config.llm_model,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def _normalize_for_cache(cls, value: Any) -> Any:
        """Strip volatile fields so identical market snapshots can be cached."""
        volatile_keys = {"timestamp", "created_at", "updated_at", "entry_time", "start_time"}
        if isinstance(value, dict):
            return {
                key: cls._normalize_for_cache(item)
                for key, item in value.items()
                if key not in volatile_keys
            }
        if isinstance(value, list):
            return [cls._normalize_for_cache(item) for item in value]
        return value

    def _get_cached_result(self, cache_key: str) -> DebateResult | None:
        """Return a cached debate result if its TTL has not expired."""
        cached = self._result_cache.get(cache_key)
        if cached is None:
            return None

        expires_at, result = cached
        now = time.monotonic()
        if expires_at <= now:
            self._result_cache.pop(cache_key, None)
            return None

        return result

    def _cache_result(self, cache_key: str, result: DebateResult) -> None:
        """Store a debate result in the short-lived exact-input cache."""
        ttl = self.config.result_cache_ttl_seconds
        if ttl <= 0:
            return

        expires_at = time.monotonic() + ttl
        self._result_cache[cache_key] = (expires_at, result.model_copy(deep=True))

        trim_mapping_size(self._result_cache, self.config.result_cache_max_entries)

    # ─── Node Functions ───────────────────────────────────────────────

    def _bull_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Bull agent presents bullish argument."""
        rnd = state["current_round"]
        logger.info(f"[Debate] Round {rnd}: Bull presenting argument")

        context: dict[str, Any] = {"round_number": rnd}

        # If this is round 2+, provide previous Devil's challenge
        if state.get("devil_output"):
            context["devil_rebuttal"] = state["devil_output"].get("reasoning", "")

        if state.get("bear_output"):
            context["bear_argument"] = state["bear_output"].get("reasoning", "")

        raw = self.bull.call_llm(state["market_data"], context)
        output = self.bull.parse_output(raw)

        # Create debate round record
        round_record = {
            "round_number": rnd,
            "agent_role": "bull",
            "argument": output.reasoning,
            "evidence": output.key_indicators,
            "stance": "BULLISH",
            "raw_output": raw,
        }

        return {
            "bull_output": raw,
            "debate_rounds": [*state["debate_rounds"], round_record],
        }

    def _bear_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Bear agent presents bearish argument."""
        rnd = state["current_round"]
        logger.info(f"[Debate] Round {rnd}: Bear presenting argument")

        context: dict[str, Any] = {"round_number": rnd}

        if state.get("devil_output"):
            context["devil_rebuttal"] = state["devil_output"].get("reasoning", "")

        if state.get("bull_output"):
            context["bull_argument"] = state["bull_output"].get("reasoning", "")

        raw = self.bear.call_llm(state["market_data"], context)
        output = self.bear.parse_output(raw)

        round_record = {
            "round_number": rnd,
            "agent_role": "bear",
            "argument": output.reasoning,
            "evidence": output.key_indicators,
            "stance": "BEARISH",
            "raw_output": raw,
        }

        return {
            "bear_output": raw,
            "debate_rounds": [*state["debate_rounds"], round_record],
        }

    def _devil_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Devil's Advocate challenges both sides."""
        rnd = state["current_round"]
        logger.info(f"[Debate] Round {rnd}: Devil challenging both sides")

        context: dict[str, Any] = {"round_number": rnd}

        if state.get("bull_output"):
            context["bull_argument"] = state["bull_output"].get("reasoning", "")

        if state.get("bear_output"):
            context["bear_argument"] = state["bear_output"].get("reasoning", "")

        # Previous devil rebuttal for anti-repetition
        previous_devil = None
        for r in reversed(state["debate_rounds"]):
            if r["agent_role"] == "devil":
                previous_devil = r.get("argument", "")
                break
        if previous_devil:
            context["previous_devil_rebuttal"] = previous_devil

        raw = self.devil.call_llm(state["market_data"], context)
        output = self.devil.parse_output(raw)

        round_record = {
            "round_number": rnd,
            "agent_role": "devil",
            "argument": output.reasoning,
            "evidence": output.key_indicators,
            "stance": "SKEPTICAL",
            "raw_output": raw,
        }

        return {
            "devil_output": raw,
            "debate_rounds": [*state["debate_rounds"], round_record],
        }

    def _judge_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Judge synthesizes all arguments."""
        logger.info("[Debate] Judge synthesizing all arguments")

        context: dict[str, Any] = {
            "debate_rounds": state["debate_rounds"],
        }

        raw = self.judge.call_llm(state["market_data"], context)

        return {"judge_output": raw}

    def _risk_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Risk Manager reviews the Judge's decision."""
        logger.info("[Debate] Risk Manager reviewing decision")

        context: dict[str, Any] = {
            "judge_decision": state.get("judge_output", {}),
            "portfolio": state.get("portfolio", {}),
            "current_positions": state.get("current_positions", {}),
        }

        raw = self.risk_manager.call_llm(state["market_data"], context)

        return {"risk_output": raw}

    def _increment_round_node(self, state: DebateState) -> dict[str, Any]:
        """LangGraph node: Increment the round counter."""
        new_round = state["current_round"] + 1
        logger.info(f"[Debate] Advancing to round {new_round}")
        return {"current_round": new_round}

    # ─── Conditional Edge ─────────────────────────────────────────────

    def _should_continue_debate(self, state: DebateState) -> str:
        """Determine whether to continue debate or move to Judge."""
        current = state["current_round"]
        max_r = state["max_rounds"]

        if current < max_r:
            return "continue"
        return "judge"

    # ─── Result Building ──────────────────────────────────────────────

    @staticmethod
    def _build_result(state: DebateState, elapsed: float) -> DebateResult:
        """Build the final DebateResult from the graph's final state."""
        judge = state.get("judge_output", {})
        risk = state.get("risk_output", {})

        # Extract risk decision
        risk_decision = risk.get("risk_decision", "APPROVE")
        risk_reasoning = risk.get("reasoning", "")

        # Use risk manager's recommended levels if available
        stop_loss = risk.get("recommended_stop_loss") or judge.get("stop_loss", 0.0)
        take_profit = risk.get("recommended_take_profit") or judge.get("take_profit", 0.0)

        # If risk manager says REJECT or FLATTEN, force HOLD
        action = judge.get("action", "HOLD")
        if risk_decision in ("REJECT", "FLATTEN"):
            action = "HOLD"

        # Preserve the latest argument from each role rather than pinning to round 1.
        bull_arg = ""
        bear_arg = ""
        devil_arg = ""
        rounds: list[DebateRound] = []

        for r in state.get("debate_rounds", []):
            rounds.append(DebateRound(**r))
            role = r.get("agent_role", "")
            if role == "bull":
                bull_arg = r.get("argument", "")
            elif role == "bear":
                bear_arg = r.get("argument", "")
            elif role == "devil":
                devil_arg = r.get("argument", "")

        return DebateResult(
            action=action,
            confidence=float(judge.get("confidence", 50.0)),
            reason=judge.get("reasoning", ""),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            bull_argument=bull_arg,
            bear_argument=bear_arg,
            devil_argument=devil_arg,
            risk_decision=risk_decision,
            risk_reasoning=risk_reasoning,
            rounds=rounds,
            symbol=state.get("symbol", ""),
            metadata={
                "total_time_seconds": round(elapsed, 2),
                "rounds_completed": len(rounds),
            },
        )
