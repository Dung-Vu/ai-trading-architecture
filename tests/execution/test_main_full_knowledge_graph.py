from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.main_full import FullTradingBot
from src.memory.knowledge_graph import KnowledgeGraph


class MockConfig:
    def __init__(self):
        self.trading = SimpleNamespace(
            initial_capital=10000.0,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )
        self.risk = SimpleNamespace(
            max_daily_loss_pct=5.0,
            max_drawdown_pct=10.0,
            max_position_pct=20.0,
            max_leverage=3,
        )
        self.strategy = SimpleNamespace(
            name="ai_debate",
            sma_fast=10,
            sma_slow=30,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
        )
        self.database_url = "postgresql://postgres:postgres@localhost:5432/trading_db"
        self.redis_url = "redis://localhost:6379"
        self.binance_testnet_api_key = "mock_key"
        self.binance_testnet_api_secret = "mock_secret"


@pytest.mark.asyncio
async def test_knowledge_graph_pattern_is_updated_when_position_closes():
    bot = FullTradingBot(config=MockConfig(), mode="dryrun", symbols=["BTC/USDT"])
    bot._knowledge_graph = KnowledgeGraph()
    bot._executor = MagicMock()
    bot._executor.simulate_sell.return_value = {
        "trade_id": 11,
        "symbol": "BTC/USDT",
        "side": "sell",
        "quantity": 0.15,
        "price": 110.0,
        "revenue": 16.5,
        "pnl": 1.5,
        "pnl_pct": 10.0,
        "cash_total": 10016.5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    bot._positions["BTC/USDT"] = {
        "side": "LONG",
        "quantity": 0.15,
        "entry_price": 100.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }
    bot._get_portfolio_state = AsyncMock(return_value={
        "cash": 0.0,
        "positions": {
            "BTC/USDT": {
                "quantity": 0.15,
                "avg_price": 100.0,
            }
        },
        "total_value": 10016.5,
    })

    bot._update_knowledge_graph(
        {"rsi": 25.0, "volume_high": True},
        "BUY",
        {"symbol": "BTC/USDT"},
    )

    pending = bot._knowledge_graph.query_pattern("RSI < 30 AND volume > avg")
    assert pending[0]["outcome"] == "pending"

    result = await bot._execute_trade(
        symbol="BTC/USDT",
        action="SELL",
        price=110.0,
        debate_result={"confidence": 88.0},
    )

    updated = bot._knowledge_graph.query_pattern("RSI < 30 AND volume > avg")
    assert result is not None
    assert result["closed_position"]["knowledge_graph_pattern_id"]
    assert updated[0]["outcome"] == "win"
    assert updated[0]["wins"] == 1
    assert updated[0]["occurrences"] == 2