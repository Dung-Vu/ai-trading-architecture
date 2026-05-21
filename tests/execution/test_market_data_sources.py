from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.main_full import FullTradingBot


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
async def test_get_latest_price_falls_back_to_exchange_ticker_when_redis_misses():
    bot = FullTradingBot(config=MockConfig(), mode="testnet", symbols=["BTC/USDT"])
    bot._redis_cache = SimpleNamespace(get_latest_price=AsyncMock(return_value=None))
    bot._order_manager = SimpleNamespace(
        _client=SimpleNamespace(fetch_ticker=MagicMock(return_value={"last": 123.45}))
    )

    price = await bot._get_latest_price("BTC/USDT")

    assert price == 123.45
    bot._redis_cache.get_latest_price.assert_awaited_once_with("BTC-USDT")
    bot._order_manager._client.fetch_ticker.assert_called_once_with("BTC/USDT")


@pytest.mark.asyncio
async def test_build_market_data_skips_simulation_outside_dryrun_when_no_price():
    bot = FullTradingBot(config=MockConfig(), mode="testnet", symbols=["BTC/USDT"])
    bot._get_latest_price = AsyncMock(return_value=None)
    bot._simulate_price = MagicMock(return_value=50000.0)

    market_data = await bot._build_market_data("BTC/USDT")

    assert market_data is None
    bot._simulate_price.assert_not_called()


@pytest.mark.asyncio
async def test_build_market_data_keeps_indicators_stable_within_session():
    bot = FullTradingBot(config=MockConfig(), mode="dryrun", symbols=["BTC/USDT"])
    bot._get_latest_price = AsyncMock(return_value=None)
    bot._simulate_price = MagicMock(return_value=50000.0)

    first = await bot._build_market_data("BTC/USDT")
    second = await bot._build_market_data("BTC/USDT")

    assert first is not None
    assert second is not None
    assert first["price"] == second["price"] == 50000.0
    assert first["indicators"] == second["indicators"]
    assert first["market_conditions"] == second["market_conditions"]


def test_simulated_price_is_stable_within_session():
    bot = FullTradingBot(config=MockConfig(), mode="dryrun", symbols=["BTC/USDT"])

    first = bot._simulate_price("BTC/USDT")
    second = bot._simulate_price("BTC/USDT")

    assert first is not None
    assert second == first
