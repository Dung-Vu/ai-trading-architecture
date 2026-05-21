import builtins
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

if "langgraph.graph" not in sys.modules:
    fake_graph_module = types.ModuleType("langgraph.graph")

    class _FakeStateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_node(self, *_args, **_kwargs):
            return None

        def add_edge(self, *_args, **_kwargs):
            return None

        def add_conditional_edges(self, *_args, **_kwargs):
            return None

        def compile(self):
            return None

    fake_graph_module.END = "END"
    fake_graph_module.START = "START"
    fake_graph_module.StateGraph = _FakeStateGraph

    fake_langgraph_module = types.ModuleType("langgraph")
    fake_langgraph_module.graph = fake_graph_module
    sys.modules["langgraph"] = fake_langgraph_module
    sys.modules["langgraph.graph"] = fake_graph_module

from src.debate.debate_engine import DebateEngine
from src.debate.models import DebateConfig, DebateResult


class _DummyGraph:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return {
            **state,
            "debate_rounds": [
                {
                    "round_number": 1,
                    "agent_role": "bull",
                    "argument": "bull case",
                    "evidence": ["rsi"],
                    "stance": "BULLISH",
                    "raw_output": {"reasoning": "bull case"},
                }
            ],
            "judge_output": {
                "action": "BUY",
                "confidence": 77.0,
                "reasoning": "consensus",
                "stop_loss": 95.0,
                "take_profit": 110.0,
            },
            "risk_output": {
                "risk_decision": "APPROVE",
                "reasoning": "within limits",
                "recommended_stop_loss": 94.0,
                "recommended_take_profit": 111.0,
            },
        }


class _RejectingGraph:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return {
            **state,
            "debate_rounds": [
                {
                    "round_number": 1,
                    "agent_role": "bull",
                    "argument": "bull case",
                    "evidence": ["momentum"],
                    "stance": "BULLISH",
                    "raw_output": {"reasoning": "bull case"},
                },
                {
                    "round_number": 1,
                    "agent_role": "bear",
                    "argument": "bear case",
                    "evidence": ["resistance"],
                    "stance": "BEARISH",
                    "raw_output": {"reasoning": "bear case"},
                },
                {
                    "round_number": 1,
                    "agent_role": "devil",
                    "argument": "devil case",
                    "evidence": ["volatility"],
                    "stance": "SKEPTICAL",
                    "raw_output": {"reasoning": "devil case"},
                },
            ],
            "judge_output": {
                "action": "BUY",
                "confidence": 66.0,
                "reasoning": "judge prefers long",
                "stop_loss": 96.0,
                "take_profit": 112.0,
            },
            "risk_output": {
                "risk_decision": "REJECT",
                "reasoning": "drawdown too high",
                "recommended_stop_loss": 0.0,
                "recommended_take_profit": 0.0,
            },
        }


@patch("src.debate.debate_engine.DebateEngine._build_graph")
def test_debate_engine_caches_inputs_ignoring_timestamps(mock_build_graph):
    dummy_graph = _DummyGraph()
    mock_build_graph.return_value = dummy_graph

    llm_client = MagicMock()
    llm_client.get_stats.return_value = {"total_calls": 5}

    engine = DebateEngine(
        config=DebateConfig(
            llm_model="openai/gpt-4o-mini",
            max_rounds=1,
            result_cache_ttl_seconds=60.0,
        ),
        llm_client=llm_client,
    )

    first = engine.run_debate(
        market_data={
            "symbol": "BTC/USDT",
            "price": 100.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "indicators": {"rsi": 32.0},
        },
        current_positions={
            "BTC/USDT": {"quantity": 0.1, "entry_time": "2024-01-01T00:00:00Z"}
        },
        portfolio={"cash": 1000.0, "timestamp": "2024-01-01T00:00:00Z"},
        symbol="BTC/USDT",
    )

    second = engine.run_debate(
        market_data={
            "symbol": "BTC/USDT",
            "price": 100.0,
            "timestamp": "2024-01-01T00:01:00Z",
            "indicators": {"rsi": 32.0},
        },
        current_positions={
            "BTC/USDT": {"quantity": 0.1, "entry_time": "2024-01-01T00:01:00Z"}
        },
        portfolio={"cash": 1000.0, "timestamp": "2024-01-01T00:01:00Z"},
        symbol="BTC/USDT",
    )

    assert first.action == "BUY"
    assert second.action == "BUY"
    assert second.metadata["cache_hit"] is True
    assert dummy_graph.calls == 1


@patch("src.debate.debate_engine.DebateEngine._build_graph")
def test_debate_engine_builds_workflow_result_and_honors_risk_rejection(mock_build_graph):
    rejecting_graph = _RejectingGraph()
    mock_build_graph.return_value = rejecting_graph

    llm_client = MagicMock()
    llm_client.get_stats.return_value = {"total_calls": 4}

    engine = DebateEngine(
        config=DebateConfig(
            llm_model="openai/gpt-4o-mini",
            max_rounds=1,
            result_cache_ttl_seconds=0.0,
        ),
        llm_client=llm_client,
    )

    result = engine.run_debate(
        market_data={"symbol": "ETH/USDT", "price": 105.0, "indicators": {"rsi": 61.0}},
        current_positions={"ETH/USDT": {"quantity": 0.5}},
        portfolio={"cash": 5000.0},
        symbol="ETH/USDT",
    )

    assert result.action == "HOLD"
    assert result.risk_decision == "REJECT"
    assert result.risk_reasoning == "drawdown too high"
    assert result.bull_argument == "bull case"
    assert result.bear_argument == "bear case"
    assert result.devil_argument == "devil case"
    assert result.metadata["cache_hit"] is False
    assert rejecting_graph.calls == 1


def test_debate_engine_module_import_is_lazy_without_langgraph(monkeypatch):
    module_name = "src.debate._debate_engine_no_langgraph"
    module_path = Path(__file__).resolve().parents[2] / "src" / "debate" / "debate_engine.py"
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langgraph.graph":
            raise ImportError("langgraph missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    assert module.StateGraph is None
    assert module.START is None
    assert module.END is None


@patch("src.debate.debate_engine.DebateEngine._build_graph")
def test_debate_engine_cache_returns_detached_result_copy(mock_build_graph):
    dummy_graph = _DummyGraph()
    mock_build_graph.return_value = dummy_graph

    llm_client = MagicMock()
    llm_client.get_stats.return_value = {"total_calls": 3}

    engine = DebateEngine(
        config=DebateConfig(
            llm_model="openai/gpt-4o-mini",
            max_rounds=1,
            result_cache_ttl_seconds=60.0,
        ),
        llm_client=llm_client,
    )

    first = engine.run_debate(
        market_data={"symbol": "BTC/USDT", "price": 100.0, "indicators": {}},
        symbol="BTC/USDT",
    )
    first.metadata["mutated"] = True

    second = engine.run_debate(
        market_data={"symbol": "BTC/USDT", "price": 100.0, "indicators": {}},
        symbol="BTC/USDT",
    )

    assert second.metadata["cache_hit"] is True
    assert "mutated" not in second.metadata
    assert dummy_graph.calls == 1


def test_build_result_uses_latest_argument_from_each_role():
    result = DebateEngine._build_result(
        {
            "symbol": "BTC/USDT",
            "debate_rounds": [
                {
                    "round_number": 1,
                    "agent_role": "bull",
                    "argument": "bull round 1",
                    "evidence": [],
                    "stance": "BULLISH",
                    "raw_output": {},
                },
                {
                    "round_number": 2,
                    "agent_role": "bull",
                    "argument": "bull round 2",
                    "evidence": [],
                    "stance": "BULLISH",
                    "raw_output": {},
                },
                {
                    "round_number": 1,
                    "agent_role": "bear",
                    "argument": "bear round 1",
                    "evidence": [],
                    "stance": "BEARISH",
                    "raw_output": {},
                },
                {
                    "round_number": 3,
                    "agent_role": "bear",
                    "argument": "bear round 3",
                    "evidence": [],
                    "stance": "BEARISH",
                    "raw_output": {},
                },
                {
                    "round_number": 1,
                    "agent_role": "devil",
                    "argument": "devil round 1",
                    "evidence": [],
                    "stance": "SKEPTICAL",
                    "raw_output": {},
                },
                {
                    "round_number": 2,
                    "agent_role": "devil",
                    "argument": "devil round 2",
                    "evidence": [],
                    "stance": "SKEPTICAL",
                    "raw_output": {},
                },
            ],
            "judge_output": {
                "action": "BUY",
                "confidence": 70.0,
                "reasoning": "latest arguments win",
                "stop_loss": 0.0,
                "take_profit": 0.0,
            },
            "risk_output": {
                "risk_decision": "APPROVE",
                "reasoning": "ok",
            },
        },
        elapsed=1.23,
    )

    assert result.bull_argument == "bull round 2"
    assert result.bear_argument == "bear round 3"
    assert result.devil_argument == "devil round 2"


def test_debate_result_accepts_zero_stop_loss_and_take_profit():
    result = DebateResult(
        action="HOLD",
        confidence=55.0,
        reason="No trade",
        stop_loss=0.0,
        take_profit=0.0,
    )

    assert result.stop_loss == 0.0
    assert result.take_profit == 0.0
