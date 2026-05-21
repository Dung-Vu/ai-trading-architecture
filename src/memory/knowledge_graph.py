"""
Knowledge Graph — Trading pattern graph for storing and querying
condition→action→outcome relationships.

Stores patterns as nodes with edges representing transitions, enabling
the bot to recall "When RSI < 30 AND volume > avg → BUY → 70% win rate".

Persistence: serialize/deserialize to JSON for saving between sessions.

Usage:
    >>> kg = KnowledgeGraph()
    >>> kg.add_pattern("RSI < 30, volume > avg", "BUY", "win", 0.70)
    >>> matches = kg.query_pattern("RSI < 30")
    >>> top = kg.get_top_patterns(n=5)
    >>> kg.serialize()  # returns JSON string
    >>> kg.deserialize(json_string)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from loguru import logger

from src.config import (
    get_default_knowledge_graph_min_occurrences,
    get_default_knowledge_graph_top_n,
)


# ─── Pattern Data Class ────────────────────────────────────────────────

@dataclass
class PatternNode:
    """A single trading pattern node."""

    condition: str = ""
    action: str = "HOLD"
    outcome: str = "unknown"
    confidence: float = 0.5
    occurrences: int = 1
    wins: int = 0
    losses: int = 0
    avg_pnl: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def win_rate(self) -> float:
        """Calculate win rate from wins/losses."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternNode":
        """Create from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── KnowledgeGraph ────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Simple graph of trading patterns for condition→action→outcome lookup.

    Patterns are indexed by keywords extracted from the condition string,
    enabling fuzzy matching when querying with current market conditions.
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self._patterns: list[PatternNode] = []
        self._index: dict[str, list[int]] = {}  # keyword → [pattern indices]
        self._keyword_sets: list[set[str]] = []
        self._pattern_ids: list[str] = []
        self._pattern_id_to_index: dict[str, int] = {}
        self._version: int = 1
        logger.info("✅ KnowledgeGraph initialized")

    # ─── Pattern Management ────────────────────────────────────────────

    def add_pattern(
        self,
        condition: str,
        action: str,
        outcome: str,
        confidence: float,
        pnl: float = 0.0,
        tags: list[str] | None = None,
    ) -> str:
        """
        Add a pattern node to the graph.

        Args:
            condition: Market condition description.
                       e.g. "RSI < 30 AND volume > avg"
            action: Trading action taken. e.g. "BUY", "SELL", "HOLD"
            outcome: Result of the action. e.g. "win", "loss", "break_even"
            confidence: Confidence score (0.0 to 1.0).
            pnl: Realized P&L for this pattern instance.
            tags: Optional tags for categorization.

        Returns:
            Pattern ID string (condition hash + index).
        """
        pattern_id = self._make_pattern_id(condition, len(self._patterns))

        wins = 1 if outcome == "win" else 0
        losses = 1 if outcome == "loss" else 0

        node = PatternNode(
            condition=condition,
            action=action,
            outcome=outcome,
            confidence=confidence,
            occurrences=1,
            wins=wins,
            losses=losses,
            avg_pnl=pnl,
            tags=tags or [],
        )

        idx = len(self._patterns)
        self._patterns.append(node)

        # Build keyword index from condition
        keywords = self._extract_keywords(condition)
        keyword_set = set(keywords)
        self._keyword_sets.append(keyword_set)
        self._pattern_ids.append(pattern_id)
        self._pattern_id_to_index[pattern_id] = idx
        for kw in keywords:
            self._index.setdefault(kw, []).append(idx)

        logger.info(
            f"[KnowledgeGraph] Added pattern #{idx}: "
            f"{condition} → {action} ({outcome}, {confidence:.0%})"
        )
        return pattern_id

    def update_pattern(
        self,
        pattern_idx: int,
        outcome: str,
        pnl: float = 0.0,
    ) -> bool:
        """
        Update a pattern with new outcome data.

        Args:
            pattern_idx: Index of the pattern to update.
            outcome: "win", "loss", or "break_even".
            pnl: P&L of this instance.

        Returns:
            True if pattern was found and updated.
        """
        if pattern_idx < 0 or pattern_idx >= len(self._patterns):
            logger.warning(f"[KnowledgeGraph] Pattern index {pattern_idx} out of range")
            return False

        node = self._patterns[pattern_idx]
        node.occurrences += 1
        node.updated_at = time.time()
        node.outcome = outcome

        if outcome == "win":
            node.wins += 1
        elif outcome == "loss":
            node.losses += 1

        # Running average P&L
        total_pnl = node.avg_pnl * (node.occurrences - 1) + pnl
        node.avg_pnl = total_pnl / node.occurrences

        # Update confidence based on empirical win rate
        empirical_rate = node.win_rate
        node.confidence = 0.5 * node.confidence + 0.5 * empirical_rate

        logger.info(
            f"[KnowledgeGraph] Updated pattern #{pattern_idx}: "
            f"occurrences={node.occurrences}, "
            f"win_rate={node.win_rate:.0%}, "
            f"confidence={node.confidence:.0%}"
        )
        return True

    def update_pattern_by_id(
        self,
        pattern_id: str,
        outcome: str,
        pnl: float = 0.0,
    ) -> bool:
        """Update a pattern by its deterministic pattern ID."""
        pattern_idx = self._pattern_id_to_index.get(pattern_id)
        if pattern_idx is None:
            logger.warning(f"[KnowledgeGraph] Pattern ID {pattern_id} not found")
            return False

        return self.update_pattern(pattern_idx, outcome=outcome, pnl=pnl)

    # ─── Query Methods ─────────────────────────────────────────────────

    def query_pattern(self, condition: str) -> list[dict[str, Any]]:
        """
        Find patterns matching current market conditions.

        Uses keyword-based matching against the condition index.
        Results are sorted by confidence (highest first).

        Args:
            condition: Current market condition string.
                       e.g. "RSI below 30, volume above average, BTC uptrend"

        Returns:
            List of matching pattern dicts sorted by confidence.
        """
        query_keywords = set(self._extract_keywords(condition))
        if not query_keywords:
            return []

        # Score candidates by indexed keyword overlap without re-tokenizing every node.
        scored: list[tuple[float, int, PatternNode]] = []
        overlap_counts: dict[int, int] = {}

        for kw in query_keywords:
            for idx in self._index.get(kw, []):
                overlap_counts[idx] = overlap_counts.get(idx, 0) + 1

        for idx, overlap in overlap_counts.items():
            node = self._patterns[idx]
            node_keywords = self._keyword_sets[idx]
            keyword_score = overlap / max(len(query_keywords | node_keywords), 1)
            occurrence_weight = min(node.occurrences / 10, 1.0)
            score = keyword_score * node.confidence * (0.5 + 0.5 * occurrence_weight)
            scored.append((score, idx, node))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx, node in scored:
            results.append({
                "pattern_id": self._pattern_ids[idx],
                "condition": node.condition,
                "action": node.action,
                "outcome": node.outcome,
                "confidence": round(node.confidence, 4),
                "win_rate": round(node.win_rate, 4),
                "occurrences": node.occurrences,
                "wins": node.wins,
                "losses": node.losses,
                "avg_pnl": round(node.avg_pnl, 2),
                "match_score": round(score, 4),
                "tags": node.tags,
                "notes": node.notes,
            })

        logger.info(
            f"[KnowledgeGraph] Query '{condition[:60]}...' found {len(results)} patterns"
        )
        return results

    def get_top_patterns(self, n: int | None = None) -> list[dict[str, Any]]:
        """
        Return patterns sorted by historical success rate.

        Only includes patterns with at least 3 occurrences for statistical
        significance.

        Args:
            n: Number of top patterns to return.

        Returns:
            List of pattern dicts, best first.
        """
        top_n = n if n is not None else get_default_knowledge_graph_top_n()
        min_occurrences = get_default_knowledge_graph_min_occurrences()

        # Filter: need minimum occurrences
        qualified = [
            p for p in self._patterns
            if (p.wins + p.losses) >= min_occurrences
        ]

        # Sort by win rate, then by occurrences as tiebreaker
        qualified.sort(key=lambda p: (p.win_rate, p.occurrences), reverse=True)

        results = []
        for node in qualified[:top_n]:
            results.append({
                "condition": node.condition,
                "action": node.action,
                "win_rate": round(node.win_rate, 4),
                "confidence": round(node.confidence, 4),
                "occurrences": node.occurrences,
                "wins": node.wins,
                "losses": node.losses,
                "avg_pnl": round(node.avg_pnl, 2),
                "tags": node.tags,
            })

        logger.info(
            f"[KnowledgeGraph] Top {top_n} patterns returned "
            f"(from {len(qualified)} qualified)"
        )
        return results

    def get_pattern_stats(self) -> dict[str, Any]:
        """Get overall graph statistics."""
        if not self._patterns:
            return {"total_patterns": 0}

        total_wins = sum(p.wins for p in self._patterns)
        total_losses = sum(p.losses for p in self._patterns)
        total_occurrences = sum(p.occurrences for p in self._patterns)
        overall_win_rate = total_wins / max(total_wins + total_losses, 1)

        # Action distribution
        action_counts: dict[str, int] = {}
        for p in self._patterns:
            action_counts[p.action] = action_counts.get(p.action, 0) + 1

        return {
            "total_patterns": len(self._patterns),
            "total_occurrences": total_occurrences,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "overall_win_rate": round(overall_win_rate, 4),
            "action_distribution": action_counts,
            "avg_occurrences_per_pattern": round(
                total_occurrences / len(self._patterns), 1
            ),
        }

    # ─── Persistence ───────────────────────────────────────────────────

    def serialize(self) -> str:
        """Save graph to JSON string."""
        data = {
            "version": self._version,
            "pattern_count": len(self._patterns),
            "patterns": [p.to_dict() for p in self._patterns],
            "index": {k: v for k, v in self._index.items()},
        }
        return json.dumps(data, indent=2)

    def deserialize(self, data: str) -> None:
        """Load graph from JSON string."""
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            logger.error(f"[KnowledgeGraph] Failed to deserialize: {exc}")
            return

        self._patterns.clear()
        self._index.clear()
        self._keyword_sets.clear()
        self._pattern_ids.clear()
        self._pattern_id_to_index.clear()

        for p_data in parsed.get("patterns", []):
            node = PatternNode.from_dict(p_data)
            idx = len(self._patterns)
            self._patterns.append(node)

            # Rebuild index
            keywords = self._extract_keywords(node.condition)
            pattern_id = self._make_pattern_id(node.condition, idx)
            self._keyword_sets.append(set(keywords))
            self._pattern_ids.append(pattern_id)
            self._pattern_id_to_index[pattern_id] = idx
            for kw in keywords:
                self._index.setdefault(kw, []).append(idx)

        self._version = parsed.get("version", 1)

        logger.info(
            f"[KnowledgeGraph] Deserialized {len(self._patterns)} patterns "
            f"from JSON"
        )

    def save_to_file(self, filepath: str) -> None:
        """Save graph to a JSON file."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(self.serialize())

        logger.info(f"[KnowledgeGraph] Saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load graph from a JSON file."""
        try:
            with open(filepath) as f:
                self.deserialize(f.read())
        except FileNotFoundError:
            logger.warning(f"[KnowledgeGraph] File not found: {filepath}")
        except Exception as exc:
            logger.error(f"[KnowledgeGraph] Failed to load: {exc}")

    # ─── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from condition text."""
        import re

        # Lowercase and split
        tokens = re.findall(r'[a-z0-9]+', text.lower())

        # Filter out common stop words and very short tokens
        stop_words = {
            "and", "or", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "don", "now", "if", "but",
            "up", "down", "about", "that", "this", "these", "those", "it",
        }

        keywords = []
        for token in tokens:
            if len(token) > 1 and token not in stop_words:
                # Also keep numeric tokens (for RSI values, etc.)
                keywords.append(token)

        return keywords

    @staticmethod
    def _make_pattern_id(condition: str, idx: int) -> str:
        """Create a deterministic pattern identifier."""
        cond_hash = hashlib.md5(condition.encode("utf-8")).hexdigest()[:8]
        return f"pat_{cond_hash}_{idx}"

    def __len__(self) -> int:
        return len(self._patterns)
