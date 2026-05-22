"""
Mem0 Self-Learning Memory — Semantic trade memory with Qdrant vector store.

Provides semantic search over past trades, automatic memory updates with
outcome data, and human-readable summaries of trading patterns.

Fallback: If Mem0 or Qdrant are unavailable, uses a simple in-memory
list-based store with TF-IDF-like keyword matching.

Usage:
    >>> mem = Mem0Memory(qdrant_url="http://localhost:6333")
    >>> mem.add_trade_memory(trade_data, debate_data)
    >>> similar = mem.query_similar_trades("RSI oversold bounce BTC")
    >>> summary = mem.get_memory_summary("BTC/USDT")
    >>> mem.update_memory({"trade_id": "abc", "pnl": 150.0, "outcome": "win"})
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger
from src.config import (
    get_default_mem0_embedding_model,
    get_default_mem0_llm_model,
    get_default_qdrant_url,
)
from src.shared_utils import parse_iso_timestamp
from .fallback_store import _InMemoryStore

# ─── Optional Dependencies ─────────────────────────────────────────────

try:
    from mem0 import Memory as Mem0Client
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Mem0Client = None  # type: ignore

try:
    import qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    qdrant_client = None  # type: ignore


def yaml_safe_load(handle: Any) -> Any:
    """Safe YAML load helper kept near the config-loading call site."""
    try:
        import yaml

        return yaml.safe_load(handle)
    except ImportError:
        return {}


# ─── In-Memory Fallback Store ─────────────────────────────────────────



# ─── Mem0Memory ────────────────────────────────────────────────────────

class Mem0Memory:
    """
    Self-learning trade memory powered by Mem0 + Qdrant.

    Stores trade decisions with full context (market conditions, indicators,
    debate reasoning) and enables semantic retrieval of similar past trades.
    Automatically updates memories with outcome data to reinforce correct
    patterns and flag mistakes.

    Falls back to an in-memory keyword store if Mem0/Qdrant are unavailable.
    """

    def __init__(
        self,
        config_path: str | None = None,
        qdrant_url: str | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """
        Initialize Mem0Memory.

        Args:
            config_path: Path to Mem0 config YAML. If None, uses defaults.
            qdrant_url: Qdrant server URL for vector storage.
        """
        self._qdrant_url = qdrant_url or get_default_qdrant_url()
        self._embedding_model = embedding_model or get_default_mem0_embedding_model()
        self._llm_model = llm_model or get_default_mem0_llm_model()
        self._mem0_state_lock = RLock()
        self._use_mem0 = MEM0_AVAILABLE and QDRANT_AVAILABLE
        self._mem0_client: Any = None
        self._fallback: _InMemoryStore | None = None

        if self._use_mem0:
            try:
                config: dict[str, Any] = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "url": self._qdrant_url,
                            "collection_name": "trading_memories",
                            "embedding_model_dims": 384,
                        },
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": self._embedding_model,
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": self._llm_model,
                            "temperature": 0,
                        },
                    },
                }

                # Load from file if provided
                if config_path:
                    config_file = Path(config_path)
                    if config_file.exists():
                        with open(config_file) as f:
                            file_config = yaml_safe_load(f)
                        if isinstance(file_config, dict):
                            config.update(file_config)
                        logger.info(f"Loaded Mem0 config from {config_path}")

                self._mem0_client = Mem0Client(config=config)
                logger.info("✅ Mem0Memory initialized with Qdrant vector store")
            except Exception as exc:
                self._disable_mem0(
                    "⚠️ Mem0 init failed, using in-memory fallback",
                    exc,
                )

        if not self._is_mem0_enabled():
            self._ensure_fallback_store()
            logger.info("✅ Mem0Memory initialized with in-memory fallback store")

    def _is_mem0_enabled(self) -> bool:
        with self._mem0_state_lock:
            return self._use_mem0

    def _get_mem0_client(self) -> Any | None:
        with self._mem0_state_lock:
            if self._use_mem0 and self._mem0_client is not None:
                return self._mem0_client
            return None

    def _ensure_fallback_store(self) -> _InMemoryStore:
        with self._mem0_state_lock:
            if self._fallback is None:
                self._fallback = _InMemoryStore()
            return self._fallback

    def _disable_mem0(self, message: str, exc: Exception | None = None) -> None:
        with self._mem0_state_lock:
            self._use_mem0 = False
            self._mem0_client = None
            if self._fallback is None:
                self._fallback = _InMemoryStore()

        if exc is not None:
            logger.warning(f"{message} ({exc})")
        else:
            logger.warning(message)

    # ─── Core Methods ──────────────────────────────────────────────────

    def add_trade_memory(self, trade: dict, debate: dict) -> str:
        """
        Store a trade + debate context as memory with metadata.

        Args:
            trade: Trade data dict (symbol, side, price, quantity, indicators, etc.)
            debate: Debate engine result dict (action, confidence, reasoning, etc.)

        Returns:
            Memory ID for future reference/updates.
        """
        timestamp = trade.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Build rich memory text for embedding
        symbol = trade.get("symbol", "UNKNOWN")
        side = trade.get("side", "UNKNOWN")
        indicators = trade.get("indicators", {})
        conditions = trade.get("market_conditions", {})

        # Create a descriptive text for the memory
        memory_text = (
            f"Trade on {symbol}: {side} at ${trade.get('price', 0):,.2f}. "
            f"RSI={indicators.get('rsi', 'N/A')}, "
            f"SMA_fast={indicators.get('sma_fast', 'N/A')}, "
            f"SMA_slow={indicators.get('sma_slow', 'N/A')}. "
            f"Market condition: {conditions.get('trend', 'unknown')}. "
            f"Volume {'above' if conditions.get('volume_high', False) else 'below'} average. "
            f"Debate result: {debate.get('action', 'HOLD')} "
            f"(confidence={debate.get('confidence', 0):.0f}%). "
            f"Reasoning: {debate.get('reasoning', '')}."
        )

        metadata = {
            "symbol": symbol,
            "side": side,
            "price": trade.get("price", 0),
            "quantity": trade.get("quantity", 0),
            "strategy": trade.get("strategy", "unknown"),
            "timestamp": timestamp,
            "indicators": indicators,
            "market_conditions": conditions,
            "decision_reasoning": debate.get("reasoning", ""),
            "debate_action": debate.get("action", "HOLD"),
            "debate_confidence": debate.get("confidence", 0),
            "bull_argument": debate.get("bull_argument", ""),
            "bear_argument": debate.get("bear_argument", ""),
            "devil_argument": debate.get("devil_argument", ""),
            "risk_decision": debate.get("risk_decision", "APPROVE"),
            "stop_loss": trade.get("stop_loss"),
            "take_profit": trade.get("take_profit"),
            "outcome": "pending",
            "pnl": None,
            "_ts": parse_iso_timestamp(timestamp),
        }

        mem0_client = self._get_mem0_client()
        if mem0_client is not None:
            try:
                user_id = f"trader_{symbol}"
                result = mem0_client.add(
                    memory_text,
                    user_id=user_id,
                    metadata=metadata,
                )
                memory_id = result.get("memory_id", result.get("id", ""))
                logger.info(
                    f"[Mem0Memory] Stored memory for {symbol} {side} @ ${trade.get('price', 0):,.2f}"
                )
                return str(memory_id)
            except Exception as exc:
                self._disable_mem0("Mem0 add failed, falling back to in-memory", exc)

        # Fallback path
        fallback = self._fallback or self._ensure_fallback_store()
        if fallback is not None:
            memory_id = fallback.add(metadata)
            logger.info(
                f"[Mem0Memory/fallback] Stored memory for {symbol} {side} "
                f"@ ${trade.get('price', 0):,.2f}"
            )
            return memory_id

        logger.error("No memory store available (Mem0 + fallback both failed)")
        return ""

    def query_similar_trades(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for similar past trades using semantic similarity.

        Args:
            query: Natural language query describing current market setup.
                   e.g. "RSI oversold bounce with high volume on BTC"
            limit: Maximum number of results to return.

        Returns:
            List of dicts containing:
                - trade data (symbol, side, price, indicators)
                - what_worked: positive outcomes
                - what_failed: negative outcomes
                - lessons_learned: key takeaways
                - similarity_score: relevance score
        """
        results: list[dict[str, Any]] = []

        mem0_client = self._get_mem0_client()
        if mem0_client is not None:
            try:
                search_results = mem0_client.search(
                    query=query,
                    limit=limit * 2,  # Get more to filter
                )
                for item in search_results:
                    metadata = item.get("metadata", {})
                    score = item.get("score", 0.0)

                    outcome = metadata.get("outcome", "pending")
                    pnl = metadata.get("pnl", 0)

                    entry = {
                        "trade_data": {
                            "symbol": metadata.get("symbol", ""),
                            "side": metadata.get("side", ""),
                            "price": metadata.get("price", 0),
                            "indicators": metadata.get("indicators", {}),
                            "market_conditions": metadata.get("market_conditions", {}),
                            "timestamp": metadata.get("timestamp", ""),
                        },
                        "decision_reasoning": metadata.get("decision_reasoning", ""),
                        "debate_action": metadata.get("debate_action", ""),
                        "debate_confidence": metadata.get("debate_confidence", 0),
                        "outcome": outcome,
                        "pnl": pnl,
                        "what_worked": "",
                        "what_failed": "",
                        "lessons_learned": "",
                        "similarity_score": round(score, 4),
                        "memory_id": item.get("memory_id", item.get("id", "")),
                    }

                    # Auto-generate lessons from outcome
                    if outcome == "win" and pnl and pnl > 0:
                        entry["what_worked"] = (
                            f"Profitable trade: +${pnl:.2f}. "
                            f"Decision: {metadata.get('debate_action', '')} "
                            f"was correct."
                        )
                        entry["lessons_learned"] = (
                            f"This setup worked — {metadata.get('decision_reasoning', '')}"
                        )
                    elif outcome == "loss" and pnl and pnl < 0:
                        entry["what_failed"] = (
                            f"Lost ${abs(pnl):.2f}. "
                            f"Decision: {metadata.get('debate_action', '')} "
                            f"was incorrect."
                        )
                        entry["lessons_learned"] = (
                            f"Avoid this setup in similar conditions. "
                            f"Reconsider: {metadata.get('decision_reasoning', '')}"
                        )

                    results.append(entry)
                    if len(results) >= limit:
                        break

                logger.info(
                    f"[Mem0Memory] Found {len(results)} similar trades for query: '{query[:60]}'"
                )
                return results

            except Exception as exc:
                self._disable_mem0("Mem0 search failed, using fallback search", exc)

        # Fallback search
        if self._fallback is not None:
            raw_results = self._fallback.search(query, limit=limit)
            for mem in raw_results:
                entry = {
                    "trade_data": {
                        "symbol": mem.get("symbol", ""),
                        "side": mem.get("side", ""),
                        "price": mem.get("price", 0),
                        "indicators": mem.get("indicators", {}),
                        "market_conditions": mem.get("market_conditions", {}),
                        "timestamp": mem.get("timestamp", ""),
                    },
                    "decision_reasoning": mem.get("decision_reasoning", ""),
                    "debate_action": mem.get("debate_action", ""),
                    "debate_confidence": mem.get("debate_confidence", 0),
                    "outcome": mem.get("outcome", "pending"),
                    "pnl": mem.get("pnl"),
                    "what_worked": "",
                    "what_failed": "",
                    "lessons_learned": "",
                    "similarity_score": 0.5,  # Fallback has no real score
                    "memory_id": mem.get("_id", ""),
                }

                pnl = mem.get("pnl")
                outcome = mem.get("outcome", "pending")
                if outcome == "win" and pnl and pnl > 0:
                    entry["what_worked"] = f"Profitable: +${pnl:.2f}"
                elif outcome == "loss" and pnl and pnl < 0:
                    entry["what_failed"] = f"Lost: ${abs(pnl):.2f}"

                results.append(entry)

            logger.info(
                f"[Mem0Memory/fallback] Found {len(results)} similar trades"
            )

        return results

    def get_memory_summary(self, symbol: str | None = None) -> str:
        """
        Summarize all memories for a symbol or globally.

        Args:
            symbol: Optional symbol filter. If None, summarizes all memories.

        Returns:
            Human-readable summary string like:
            "I've seen this pattern 12 times before. 8 were profitable.
             Common mistakes: entering too early during low volume."
        """
        memories: list[dict[str, Any]] = []

        mem0_client = self._get_mem0_client()
        if mem0_client is not None:
            # Get all memories via a broad search
            try:
                all_results = mem0_client.search(
                    query="trade" if symbol is None else f"trade {symbol}",
                    limit=200,
                )
                for item in all_results:
                    mem = item.get("metadata", {})
                    if symbol is None or mem.get("symbol") == symbol:
                        memories.append(mem)
            except Exception as exc:
                self._disable_mem0("Mem0 summary search failed, using fallback", exc)

        if not memories and self._fallback is not None:
            memories = self._fallback.get_all(symbol=symbol)

        if not memories:
            return f"No trading memories found{' for ' + symbol if symbol else ''}."

        # Compute statistics
        total = len(memories)
        wins = sum(1 for m in memories if m.get("outcome") == "win")
        losses = sum(1 for m in memories if m.get("outcome") == "loss")
        pending = total - wins - losses

        win_rate = (wins / max(wins + losses, 1)) * 100

        pnl_values = [m.get("pnl", 0) for m in memories if m.get("pnl") is not None]
        total_pnl = sum(pnl_values)
        avg_pnl = total_pnl / max(len(pnl_values), 1)

        # Analyze patterns
        buys = sum(1 for m in memories if m.get("side") == "BUY")
        sells = sum(1 for m in memories if m.get("side") == "SELL")

        # Common conditions in wins vs losses
        win_conditions: dict[str, int] = {}
        loss_conditions: dict[str, int] = {}
        for m in memories:
            conditions = m.get("market_conditions", {})
            trend = conditions.get("trend", "unknown")
            if m.get("outcome") == "win":
                win_conditions[trend] = win_conditions.get(trend, 0) + 1
            elif m.get("outcome") == "loss":
                loss_conditions[trend] = loss_conditions.get(trend, 0) + 1

        # Build summary
        lines: list[str] = []
        scope = f" for {symbol}" if symbol else ""
        lines.append(f"📊 Trading Memory Summary{scope}")
        lines.append(f"   Total memories: {total}")
        lines.append(f"   Wins: {wins} | Losses: {losses} | Pending: {pending}")
        lines.append(f"   Win rate: {win_rate:.1f}%")
        lines.append(f"   Total P&L: ${total_pnl:+,.2f} | Avg P&L: ${avg_pnl:+,.2f}")
        lines.append(f"   Buys: {buys} | Sells: {sells}")

        if win_conditions:
            best_trend = max(win_conditions, key=win_conditions.get)  # type: ignore
            lines.append(
                f"   ✅ Best performing condition: {best_trend} "
                f"({win_conditions[best_trend]} wins)"
            )

        if loss_conditions:
            worst_trend = max(loss_conditions, key=loss_conditions.get)  # type: ignore
            lines.append(
                f"   ❌ Worst performing condition: {worst_trend} "
                f"({loss_conditions[worst_trend]} losses)"
            )

        # Confidence analysis
        high_conf_trades = [
            m for m in memories
            if (m.get("debate_confidence") or 0) >= 70 and m.get("pnl") is not None
        ]
        if high_conf_trades:
            high_conf_wins = sum(
                1 for m in high_conf_trades if m.get("outcome") == "win"
            )
            high_conf_rate = (high_conf_wins / len(high_conf_trades)) * 100
            lines.append(
                f"   High-confidence trades (≥70%): {len(high_conf_trades)}, "
                f"win rate: {high_conf_rate:.1f}%"
            )

        lines.append("")
        if total >= 12:
            lines.append(
                f"I've seen this pattern {total} times before. "
                f"{wins} were profitable. "
                f"Common mistakes: entering during {worst_trend if loss_conditions else 'unclear conditions'}."
            )
        elif total > 0:
            lines.append(
                f"I've seen {total} similar situation{'s' if total > 1 else ''} "
                f"{'so far' if pending > 0 else 'with resolved outcomes'}."
            )

        return "\n".join(lines)

    def update_memory(self, trade_outcome: dict) -> bool:
        """
        Update existing memories with outcome data.

        Reinforces correct patterns (win → boost confidence), flags
        incorrect ones (loss → lower confidence, add warning).

        Args:
            trade_outcome: Dict with keys:
                - trade_id or memory_id: ID of the memory to update
                - pnl: realized profit/loss
                - outcome: "win", "loss", or "break_even"
                - outcome_notes: optional human-readable notes
                - close_price: closing price
                - hold_duration: how long the trade was held

        Returns:
            True if update succeeded, False otherwise.
        """
        memory_id = trade_outcome.get("trade_id") or trade_outcome.get("memory_id")
        if not memory_id:
            logger.warning("[Mem0Memory] update_memory called without trade_id or memory_id")
            return False

        pnl = trade_outcome.get("pnl", 0)
        outcome = trade_outcome.get("outcome", "pending")

        # Determine reinforcement
        if outcome == "win":
            reinforcement = "positive"
            notes = trade_outcome.get(
                "outcome_notes",
                f"Trade was correct. PnL: +${pnl:.2f}",
            )
        elif outcome == "loss":
            reinforcement = "negative"
            notes = trade_outcome.get(
                "outcome_notes",
                f"Trade was incorrect. PnL: ${pnl:+.2f}",
            )
        else:
            reinforcement = "neutral"
            notes = trade_outcome.get("outcome_notes", "Break-even trade")

        updates = {
            "outcome": outcome,
            "pnl": pnl,
            "reinforcement": reinforcement,
            "outcome_notes": notes,
            "close_price": trade_outcome.get("close_price"),
            "hold_duration": trade_outcome.get("hold_duration"),
            "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        mem0_client = self._get_mem0_client()
        if mem0_client is not None:
            try:
                # Update via Mem0
                mem0_client.update(
                    memory_id=memory_id,
                    data=notes,
                )
                logger.info(
                    f"[Mem0Memory] Updated memory {memory_id}: "
                    f"outcome={outcome}, pnl=${pnl:+.2f}, "
                    f"reinforcement={reinforcement}"
                )
                return True
            except Exception as exc:
                self._disable_mem0("Mem0 update failed, using fallback", exc)

        # Fallback update
        if self._fallback is not None:
            success = self._fallback.update(memory_id, updates)
            if success:
                logger.info(
                    f"[Mem0Memory/fallback] Updated memory {memory_id}: "
                    f"outcome={outcome}, pnl=${pnl:+.2f}"
                )
            return success

        return False

    def clear_old_memories(self, days: int = 90) -> int:
        """
        Remove memories older than N days.

        Keeps recent context relevant and prevents stale patterns from
        influencing decisions.

        Args:
            days: Remove memories older than this many days (default: 90).

        Returns:
            Number of memories removed.
        """
        removed = 0

        mem0_client = self._get_mem0_client()
        if mem0_client is not None:
            try:
                # Mem0 doesn't have a direct bulk-delete-by-date API,
                # so we search for old memories and delete individually
                all_results = mem0_client.search(query="trade", limit=500)
                cutoff = datetime.now(timezone.utc).timestamp() - days * 86400

                for item in all_results:
                    mem = item.get("metadata", {})
                    ts = mem.get("_ts", 0) or parse_iso_timestamp(
                        mem.get("timestamp", "")
                    )
                    if ts < cutoff:
                        memory_id = item.get("memory_id", item.get("id", ""))
                        if memory_id:
                            try:
                                mem0_client.delete(memory_id=memory_id)
                                removed += 1
                            except Exception as exc:
                                logger.debug(
                                    f"[Mem0Memory] Failed to delete memory {memory_id}: {exc}"
                                )

                logger.info(
                    f"[Mem0Memory] Cleared {removed} memories older than {days} days"
                )
                return removed
            except Exception as exc:
                self._disable_mem0("Mem0 cleanup failed, using fallback", exc)

        if self._fallback is not None:
            removed = self._fallback.clear_old(days=days)
            logger.info(
                f"[Mem0Memory/fallback] Cleared {removed} old memories"
            )

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        if self._is_mem0_enabled():
            return {
                "store_type": "mem0_qdrant",
                "qdrant_url": self._qdrant_url,
                "status": "connected",
            }
        elif self._fallback is not None:
            return {
                "store_type": "in_memory_fallback",
                "memory_count": len(self._fallback),
                "status": "active",
            }
        return {"store_type": "none", "status": "unavailable"}

