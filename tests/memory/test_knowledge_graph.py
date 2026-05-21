import pytest

from src.memory.knowledge_graph import KnowledgeGraph


def test_update_pattern_by_id_rolls_occurrences_and_average_pnl():
    graph = KnowledgeGraph()
    pattern_id = graph.add_pattern(
        condition="RSI below 30 and volume spike",
        action="BUY",
        outcome="win",
        confidence=0.8,
        pnl=100.0,
    )

    assert graph.update_pattern_by_id(pattern_id, outcome="loss", pnl=-50.0) is True

    match = graph.query_pattern("BTC RSI below 30 with volume spike")[0]
    assert match["pattern_id"] == pattern_id
    assert match["occurrences"] == 2
    assert match["wins"] == 1
    assert match["losses"] == 1
    assert match["avg_pnl"] == 25.0
    assert match["confidence"] == pytest.approx(0.65)


def test_get_top_patterns_requires_minimum_occurrences():
    graph = KnowledgeGraph()
    qualified_id = graph.add_pattern(
        condition="RSI below 30 and volume spike",
        action="BUY",
        outcome="win",
        confidence=0.7,
        pnl=120.0,
    )
    graph.update_pattern_by_id(qualified_id, outcome="win", pnl=80.0)
    graph.update_pattern_by_id(qualified_id, outcome="loss", pnl=-20.0)

    graph.add_pattern(
        condition="MACD bullish crossover",
        action="BUY",
        outcome="win",
        confidence=0.9,
        pnl=50.0,
    )

    top_patterns = graph.get_top_patterns(n=5)

    assert len(top_patterns) == 1
    assert top_patterns[0]["condition"] == "RSI below 30 and volume spike"
    assert top_patterns[0]["occurrences"] == 3


def test_serialize_round_trip_rebuilds_index_and_pattern_ids(tmp_path):
    graph = KnowledgeGraph()
    pattern_id = graph.add_pattern(
        condition="MACD bullish crossover with breakout",
        action="BUY",
        outcome="win",
        confidence=0.9,
        tags=["trend"],
    )
    payload = graph.serialize()

    restored = KnowledgeGraph()
    restored.deserialize(payload)

    assert len(restored) == 1
    restored_match = restored.query_pattern("bullish MACD breakout")[0]
    assert restored_match["pattern_id"] == pattern_id
    assert restored_match["tags"] == ["trend"]

    filepath = tmp_path / "knowledge_graph.json"
    restored.save_to_file(str(filepath))

    reloaded = KnowledgeGraph()
    reloaded.load_from_file(str(filepath))

    assert len(reloaded) == 1
    assert reloaded.get_pattern_stats()["total_patterns"] == 1