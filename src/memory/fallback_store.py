"""In-memory fallback store for Mem0Memory."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

class _InMemoryStore:
    """Simple in-memory fallback when Mem0/Qdrant are unavailable.

    Uses keyword-based matching instead of vector embeddings.
    """

    def __init__(self) -> None:
        self._memories: list[dict[str, Any]] = []
        self._index: dict[str, list[int]] = {}  # keyword -> [indices]
        self._token_sets: list[set[str]] = []

    def add(self, memory: dict[str, Any]) -> str:
        """Add a memory and index its keywords."""
        idx = len(self._memories)
        self._memories.append(memory)
        self._token_sets.append(set())
        self._reindex_memory(idx)

        memory_id = hashlib.md5(f"{memory.get('timestamp', '')}-{idx}".encode()).hexdigest()[:12]
        memory["_id"] = memory_id
        memory["_idx"] = idx
        return memory_id

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Keyword-based similarity search."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        # Score only indexed candidate memories instead of scanning the full store.
        scores: list[tuple[float, dict[str, Any]]] = []
        overlap_counts: dict[int, int] = {}
        for token in query_tokens:
            for idx in self._index.get(token, []):
                overlap_counts[idx] = overlap_counts.get(idx, 0) + 1

        for idx, overlap in overlap_counts.items():
            mem_tokens = self._token_sets[idx]
            score = overlap / max(len(query_tokens | mem_tokens), 1)
            scores.append((score, self._memories[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scores[:limit]]

    def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Update a memory by its ID."""
        for idx, mem in enumerate(self._memories):
            if mem.get("_id") == memory_id:
                mem.update(updates)
                mem["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._reindex_memory(idx)
                return True
        return False

    def get_all(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get all memories, optionally filtered by symbol."""
        if symbol is None:
            return list(self._memories)
        return [m for m in self._memories if m.get("symbol") == symbol]

    def clear_old(self, days: int = 90) -> int:
        """Remove memories older than N days."""
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        original_len = len(self._memories)
        self._memories = [
            m for m in self._memories
            if m.get("_ts", 0) >= cutoff
        ]
        # Rebuild index
        self._index.clear()
        self._token_sets.clear()
        for idx, mem in enumerate(self._memories):
            mem["_idx"] = idx
            self._token_sets.append(set())
            self._reindex_memory(idx)
        return original_len - len(self._memories)

    def _reindex_memory(self, idx: int) -> None:
        """Refresh the inverted index for a single memory record."""
        if idx < 0 or idx >= len(self._memories):
            return

        for token_indices in self._index.values():
            while idx in token_indices:
                token_indices.remove(idx)

        text = " ".join(self._extract_text(self._memories[idx]))
        tokens = set(self._tokenize(text))
        self._token_sets[idx] = tokens
        for token in tokens:
            self._index.setdefault(token, []).append(idx)

    @staticmethod
    def _extract_text(memory: dict[str, Any]) -> list[str]:
        """Extract all text fields from a memory dict."""
        texts = []
        for key in ("symbol", "side", "decision_reasoning", "outcome_notes",
                     "market_conditions", "lessons", "action", "reasoning"):
            val = memory.get(key)
            if val:
                texts.append(str(val))
        # Also include indicator values as text
        indicators = memory.get("indicators", {})
        if isinstance(indicators, dict):
            for k, v in indicators.items():
                texts.append(f"{k}={v}")
        return texts

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        import re
        return re.findall(r'[a-z0-9]+', text.lower())

    def __len__(self) -> int:
        return len(self._memories)

