"""Audit logging tests for AI debate decisions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.main_ai import AITradingBot
from src.main_full import FullTradingBot


class _Config:
    class Trading:
        initial_capital = 10000.0

    trading = Trading()


@pytest.mark.asyncio
async def test_ai_bot_logs_debate_decision_without_trade():
    bot = AITradingBot(config=_Config(), symbols=["BTC/USDT"])
    bot._trade_memory = MagicMock()
    bot._trade_memory.log_debate = AsyncMock()

    await bot._log_debate_decision(
        "BTC/USDT",
        {
            "action": "HOLD",
            "confidence": 72.0,
            "risk_decision": "APPROVE",
            "rounds": [{"agent_role": "bull"}],
            "metadata": {"total_time_seconds": 1.2},
        },
    )

    bot._trade_memory.log_debate.assert_awaited_once()
    payload = bot._trade_memory.log_debate.await_args.args[0]
    assert payload["judge_action"] == "HOLD"
    assert payload["rounds"] == 1
    assert payload["latency_seconds"] == 1.2


@pytest.mark.asyncio
async def test_full_bot_logs_debate_decision_without_trade():
    bot = FullTradingBot(config=_Config(), symbols=["BTC/USDT"])
    bot._trade_memory = MagicMock()
    bot._trade_memory.log_debate = AsyncMock()

    await bot._log_debate_decision(
        "BTC/USDT",
        {
            "action": "SELL",
            "confidence": 65.0,
            "risk_decision": "REDUCE",
            "risk_reasoning": "lower size",
            "rounds": 2,
        },
    )

    bot._trade_memory.log_debate.assert_awaited_once()
    payload = bot._trade_memory.log_debate.await_args.args[0]
    assert payload["judge_action"] == "SELL"
    assert payload["risk_action"] == "REDUCE"
    assert payload["rounds"] == 2
