import asyncio
from types import SimpleNamespace

from src.debate.runtime import normalize_debate_result, run_debate_round


def test_normalize_debate_result_adds_reason_alias_and_round_count():
    result = SimpleNamespace(
        action="BUY",
        confidence=77.0,
        reason="Bullish market structure",
        stop_loss=49000.0,
        take_profit=52000.0,
        bull_argument="Momentum is strong",
        bear_argument="Resistance nearby",
        devil_argument="Breakout could fail",
        risk_decision="APPROVE",
        risk_reasoning="Position size acceptable",
        rounds=[{"round": 1}, {"round": 2}],
    )

    normalized = normalize_debate_result(
        result,
        include_reason_alias=True,
        include_round_count=True,
    )

    assert normalized == {
        "action": "BUY",
        "confidence": 77.0,
        "reasoning": "Bullish market structure",
        "reason": "Bullish market structure",
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
        "bull_argument": "Momentum is strong",
        "bear_argument": "Resistance nearby",
        "devil_argument": "Breakout could fail",
        "risk_decision": "APPROVE",
        "risk_reasoning": "Position size acceptable",
        "rounds": 2,
    }


def test_run_debate_round_passes_context_to_engine():
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def run_debate(self, **kwargs):
            self.calls.append(kwargs)
            return {"action": "HOLD"}

    async def run_test():
        engine = FakeEngine()
        result = await run_debate_round(
            engine,
            market_data={"price": 50000.0},
            current_positions={"BTC/USDT": {"quantity": 0.1}},
            portfolio={"total_value": 10000.0},
            symbol="BTC/USDT",
        )

        assert result == {"action": "HOLD"}
        assert engine.calls == [
            {
                "market_data": {"price": 50000.0},
                "current_positions": {"BTC/USDT": {"quantity": 0.1}},
                "portfolio": {"total_value": 10000.0},
                "symbol": "BTC/USDT",
            }
        ]

    asyncio.run(run_test())