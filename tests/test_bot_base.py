from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.services import BotServices
from src.bot_base import BaseTradingBot
from src.runtime_status import RuntimeFailurePolicy


class _DummyBot(BaseTradingBot):
    def __init__(self) -> None:
        self.symbols = ["BTC/USDT"]
        self.strategy = "sma_cross"
        self.config = SimpleNamespace(
            strategy=SimpleNamespace(
                sma_fast=10,
                sma_slow=30,
                rsi_period=14,
                rsi_overbought=70,
                rsi_oversold=30,
            )
        )
        self._strategies = None


@pytest.mark.asyncio
async def test_cleanup_component_attr_awaits_async_close_and_clears_attribute():
    bot = _DummyBot()
    close = AsyncMock()
    bot._redis_cache = SimpleNamespace(close=close)

    await bot._cleanup_component_attr("_redis_cache")

    close.assert_awaited_once_with()
    assert bot._redis_cache is None


@pytest.mark.asyncio
async def test_cleanup_component_attrs_closes_components_in_reverse_order():
    bot = _DummyBot()
    closed: list[str] = []

    class _Component:
        def __init__(self, name: str) -> None:
            self._name = name

        def close(self) -> None:
            closed.append(self._name)

    bot._trade_memory = _Component("trade_memory")
    bot._telegram_bot = _Component("telegram_bot")

    await bot._cleanup_component_attrs("_trade_memory", "_telegram_bot")

    assert closed == ["telegram_bot", "trade_memory"]
    assert bot._trade_memory is None
    assert bot._telegram_bot is None


def test_generate_strategy_signal_short_circuits_ai_debate():
    bot = _DummyBot()
    bot.strategy = "ai_debate"

    with patch.object(bot, "_get_strategy_instance") as mock_get_strategy:
        result = bot._generate_strategy_signal("BTC/USDT", {"price": 100.0})

    assert result == "BUY"
    mock_get_strategy.assert_not_called()


def test_get_strategy_instance_creates_bundle_on_demand():
    bot = _DummyBot()
    strategy = MagicMock()
    bundle = {"sma_cross": strategy, "bbands": MagicMock()}

    with patch.object(bot, "_create_strategy_bundle", return_value=bundle) as mock_create:
        instance = bot._get_strategy_instance("sma_cross", "ETH/USDT")

    mock_create.assert_called_once_with("ETH/USDT")
    assert instance is strategy
    assert bot._strategies["ETH/USDT"] is bundle


def test_run_strategy_delegates_to_configured_strategy_instance():
    bot = _DummyBot()
    strategy = MagicMock()
    strategy.generate_signal.return_value = "SELL"
    bot._strategies = {
        "BTC/USDT": {"sma_cross": strategy, "bbands": MagicMock()},
    }

    market_data = {
        "symbol": "BTC/USDT",
        "price": 100.0,
        "indicators": {"rsi": 72.0},
    }
    result = bot._run_strategy("BTC/USDT", market_data)

    assert result == "SELL"
    strategy.generate_signal.assert_called_once_with(
        price=100.0,
        indicators={"rsi": 72.0},
        market_data=market_data,
    )


def test_generate_strategy_signal_returns_hold_for_missing_strategy():
    bot = _DummyBot()

    with patch.object(bot, "_get_strategy_instance", return_value=None):
        result = bot._generate_strategy_signal(
            "BTC/USDT",
            {"price": 100.0, "indicators": {}},
            strategy_name="unknown_strategy",
        )

    assert result == "HOLD"


def test_runtime_failure_records_typed_status():
    bot = _DummyBot()

    status = bot._runtime_failure(
        "redis_unavailable",
        "Redis unavailable during fetch",
        policy=RuntimeFailurePolicy.FALLBACK,
        log_level="debug",
    )

    assert not status
    assert status.code == "redis_unavailable"
    assert status.policy is RuntimeFailurePolicy.FALLBACK
    assert bot._last_runtime_status is status


def test_runtime_success_records_typed_status():
    bot = _DummyBot()

    status = bot._runtime_success("debate_executed", "Debate finished")

    assert status
    assert status.code == "debate_executed"
    assert bot._last_runtime_status is status


def test_service_container_routes_legacy_component_attrs():
    bot = _DummyBot()
    bot.services = BotServices()

    cache = SimpleNamespace(close=lambda: None)
    bot._redis_cache = cache
    bot._strategies = {"BTC/USDT": {"sma_cross": object(), "bbands": object()}}

    assert bot.services.redis_cache is cache
    assert bot._redis_cache is cache
    assert bot.services.strategies == bot._strategies