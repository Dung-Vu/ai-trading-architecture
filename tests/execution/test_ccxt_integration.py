"""
Integration tests for CCXT exchange order execution and portfolio state fetching.
Mocks ExchangeClient and OrderManager to verify that FullTradingBot and AITradingBot
correctly interact with the exchange in testnet mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from types import SimpleNamespace

from src.main_full import FullTradingBot
from src.main_ai import AITradingBot
from src.risk.kill_switch import KillSwitch
from src.risk.risk_engine import RiskEngine


class MockConfig:
    def __init__(self):
        class Trading:
            initial_capital = 10000.0
        class Risk:
            max_daily_loss_pct = 5.0
            max_drawdown_pct = 10.0
            max_position_pct = 20.0
            max_leverage = 3
        self.trading = Trading()
        self.risk = Risk()
        self.database_url = "postgresql://postgres:postgres@localhost:5432/trading_db"
        self.redis_url = "redis://localhost:6379"
        self.binance_testnet_api_key = "mock_key"
        self.binance_testnet_api_secret = "mock_secret"


class TestCCXTIntegration:
    @pytest.fixture
    def mock_config(self):
        return MockConfig()

    @pytest.fixture
    def mock_order_manager(self):
        manager = MagicMock()
        manager._client = MagicMock()
        manager._client.fetch_balance = MagicMock(return_value={
            "USDT": {"free": 10000.0, "total": 10000.0},
            "BTC": {"free": 0.5, "total": 0.5},
            "free": {"USDT": 10000.0, "BTC": 0.5},
            "used": {"USDT": 0.0, "BTC": 0.0},
            "total": {"USDT": 10000.0, "BTC": 0.5},
        })
        manager._precision_amount = MagicMock(side_effect=lambda symbol, amount: amount)
        manager._precision_price = MagicMock(side_effect=lambda symbol, price: price)
        manager.create_market_order = MagicMock(return_value={
            "id": "order-123",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "amount": 0.1,
            "price": 50000.0,
            "status": "closed",
        })
        return manager

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.save_trade = AsyncMock()
        db.save_active_positions = AsyncMock()
        db.get_active_positions = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    @patch("src.execution.exchange_client.ExchangeClient")
    @patch("src.execution.order_manager.OrderManager")
    async def test_full_bot_portfolio_and_execution_testnet(
        self, mock_om_class, mock_ec_class, mock_config, mock_order_manager, mock_db
    ):
        # Configure mocks
        mock_om_class.return_value = mock_order_manager
        
        # Initialize bot in testnet mode
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )
        
        # Inject mocks
        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._exchange_client = mock_order_manager._client
        
        # Mock additional helpers
        bot._get_latest_price = AsyncMock(return_value=50000.0)
        bot._run_risk_check = AsyncMock(return_value=True)

        # 1. Test portfolio state fetching
        portfolio = await bot._get_portfolio_state()
        assert portfolio["cash"] == 10000.0
        assert "BTC/USDT" in portfolio["positions"]
        assert portfolio["positions"]["BTC/USDT"]["quantity"] == 0.5

        # 2. Test trade execution (BUY)
        debate_result = {
            "confidence": 85.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "decision": "BUY",
        }
        exec_result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="BUY",
            price=50000.0,
            debate_result=debate_result
        )
        
        assert exec_result is not None
        assert exec_result["order_info"]["status"] == "closed"
        mock_order_manager.create_market_order.assert_called_once_with(
            symbol="BTC/USDT",
            side="buy",
            amount=0.02
        )

    @pytest.mark.asyncio
    @patch("src.execution.exchange_client.ExchangeClient")
    @patch("src.execution.order_manager.OrderManager")
    async def test_ai_bot_portfolio_and_execution_testnet(
        self, mock_om_class, mock_ec_class, mock_config, mock_order_manager, mock_db
    ):
        # Configure mocks
        mock_om_class.return_value = mock_order_manager
        
        # Initialize AI bot in testnet mode
        bot = AITradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )
        
        # Inject mocks
        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._exchange_client = mock_order_manager._client
        
        # Mock helpers
        bot._get_latest_price = AsyncMock(return_value=50000.0)
        bot._run_risk_check = AsyncMock(return_value=True)

        # 1. Test portfolio state fetching
        portfolio = await bot._get_portfolio_state()
        assert portfolio["cash"] == 10000.0
        assert "BTC/USDT" in portfolio["positions"]

        # 2. Test trade execution (SELL)
        # Reset mock call history first
        mock_order_manager.create_market_order.reset_mock()
        mock_order_manager.create_market_order.return_value["side"] = "sell"
        
        # Populate positions to control the quantity sold
        bot._positions["BTC/USDT"] = {
            "side": "LONG",
            "quantity": 0.2,
            "entry_price": 50000.0,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }

        debate_result = {
            "confidence": 90.0,
            "stop_loss": 51000.0,
            "take_profit": 48000.0,
            "decision": "SELL",
        }
        exec_result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="SELL",
            price=50000.0,
            debate_result=debate_result
        )
        
        assert exec_result is not None
        mock_order_manager.create_market_order.assert_called_once_with(
            symbol="BTC/USDT",
            side="sell",
            amount=0.2
        )

    @pytest.mark.asyncio
    async def test_ai_bot_sell_without_position_is_rejected(
        self, mock_config, mock_order_manager, mock_db
    ):
        bot = AITradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 10000.0,
            "positions": {},
            "total_value": 10000.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="SELL",
            price=50000.0,
            debate_result={"confidence": 90.0},
        )

        assert result is None
        mock_order_manager.create_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_bot_buy_handles_string_precision_from_exchange(
        self, mock_config, mock_order_manager, mock_db
    ):
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        mock_order_manager._precision_amount.return_value = "0.0200"
        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 10000.0,
            "positions": {},
            "total_value": 10000.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="BUY",
            price=50000.0,
            debate_result={"confidence": 80.0},
        )

        assert result is not None
        mock_order_manager.create_market_order.assert_called_once_with(
            symbol="BTC/USDT",
            side="buy",
            amount=0.02,
        )

    @pytest.mark.asyncio
    async def test_full_bot_buy_aborts_when_precision_application_fails(
        self, mock_config, mock_order_manager, mock_db
    ):
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        mock_order_manager._precision_amount.side_effect = RuntimeError("precision unavailable")
        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 10000.0,
            "positions": {},
            "total_value": 10000.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="BUY",
            price=50000.0,
            debate_result={"confidence": 80.0},
        )

        assert result is None
        mock_order_manager.create_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_bot_sell_without_position_is_rejected(
        self, mock_config, mock_order_manager, mock_db
    ):
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        bot._trade_memory = mock_db
        bot._order_manager = mock_order_manager
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 10000.0,
            "positions": {},
            "total_value": 10000.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="SELL",
            price=50000.0,
            debate_result={"confidence": 90.0},
        )

        assert result is None
        mock_order_manager.create_market_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_ai_bot_sell_uses_open_position_even_with_zero_cash(
        self, mock_config, mock_db
    ):
        bot = AITradingBot(
            config=mock_config,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )

        bot._trade_memory = mock_db
        bot._dry_run_executor = MagicMock()
        bot._dry_run_executor.simulate_sell.return_value = {
            "trade_id": 7,
            "symbol": "BTC/USDT",
            "side": "sell",
            "quantity": 0.2,
            "price": 50000.0,
            "revenue": 10000.0,
            "pnl": 250.0,
            "pnl_pct": 2.5,
            "cash_total": 10000.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 0.0,
            "positions": {
                "BTC/USDT": {
                    "quantity": 0.2,
                    "avg_price": 48750.0,
                }
            },
            "total_value": 10000.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="SELL",
            price=50000.0,
            debate_result={"confidence": 88.0},
        )

        assert result is not None
        bot._dry_run_executor.simulate_sell.assert_called_once_with(
            symbol="BTC/USDT",
            quantity=0.2,
            price=50000.0,
            timestamp=result["timestamp"],
        )

    @pytest.mark.asyncio
    async def test_full_bot_sell_uses_open_position_even_with_zero_cash(
        self, mock_config, mock_db
    ):
        bot = FullTradingBot(
            config=mock_config,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )

        bot._trade_memory = mock_db
        bot._executor = MagicMock()
        bot._executor.simulate_sell.return_value = {
            "trade_id": 11,
            "symbol": "BTC/USDT",
            "side": "sell",
            "quantity": 0.15,
            "price": 50000.0,
            "revenue": 7500.0,
            "pnl": 180.0,
            "pnl_pct": 2.4,
            "cash_total": 7500.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 0.0,
            "positions": {
                "BTC/USDT": {
                    "quantity": 0.15,
                    "avg_price": 48800.0,
                }
            },
            "total_value": 7500.0,
        })

        result = await bot._execute_trade(
            symbol="BTC/USDT",
            action="SELL",
            price=50000.0,
            debate_result={"confidence": 77.0},
        )

        assert result is not None
        bot._executor.simulate_sell.assert_called_once_with(
            symbol="BTC/USDT",
            quantity=0.15,
            price=50000.0,
            timestamp=result["timestamp"],
        )

    def test_ai_bot_bbands_signal_requires_volume_confirmation(self, mock_config):
        bot = AITradingBot(
            config=mock_config,
            mode="dryrun",
            strategy="bbands",
            symbols=["BTC/USDT"],
        )

        hold_signal = bot._bbands_signal({
            "price": 100.0,
            "indicators": {
                "bb_lower": 101.0,
                "bb_upper": 105.0,
                "volume_high": False,
            },
        })
        buy_signal = bot._bbands_signal({
            "price": 100.0,
            "indicators": {
                "bb_lower": 101.0,
                "bb_upper": 105.0,
                "volume_high": True,
            },
        })
        sell_signal = bot._bbands_signal({
            "price": 106.0,
            "indicators": {
                "bb_lower": 101.0,
                "bb_upper": 105.0,
                "volume_high": False,
            },
        })

        assert hold_signal == "HOLD"
        assert buy_signal == "BUY"
        assert sell_signal == "SELL"

    @pytest.mark.asyncio
    async def test_full_bot_drawdown_auto_triggers_kill_switch(self, mock_config):
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        bot._risk_engine = RiskEngine(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_position_pct=0.20,
            max_leverage=3,
        )
        bot._kill_switch = KillSwitch()
        bot._kill_switch.arm()
        bot._risk_engine.update_peak_equity(10000.0)
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 8900.0,
            "positions": {},
            "total_value": 8900.0,
        })

        await bot._check_drawdown_kill_switch()

        assert bot._kill_switch.is_active() is True
        assert bot._shutdown_event.is_set() is True

    @pytest.mark.asyncio
    async def test_ai_bot_run_debate_passes_portfolio_context(self, mock_config):
        bot = AITradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        bot._debate_engine = MagicMock()
        bot._debate_engine.run_debate = MagicMock(
            return_value=SimpleNamespace(
                action="BUY",
                confidence=72.0,
                reason="Bullish setup",
                stop_loss=49000.0,
                take_profit=52000.0,
            )
        )
        bot._get_portfolio_state = AsyncMock(return_value={
            "cash": 2500.0,
            "positions": {
                "BTC/USDT": {
                    "quantity": 0.15,
                    "entry_price": 49500.0,
                }
            },
            "total_value": 10050.0,
        })
        bot._positions["BTC/USDT"] = {
            "quantity": 0.2,
            "entry_price": 50000.0,
            "side": "LONG",
        }

        result = await bot._run_debate("BTC/USDT", {"price": 50000.0})

        assert result is not None
        assert result["action"] == "BUY"

        call_kwargs = bot._debate_engine.run_debate.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC/USDT"
        assert call_kwargs["portfolio"]["total_value"] == 10050.0
        assert call_kwargs["current_positions"]["BTC/USDT"]["quantity"] == 0.2

    @pytest.mark.asyncio
    async def test_full_bot_setup_delegates_phase_helpers(self, mock_config):
        bot = FullTradingBot(
            config=mock_config,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )

        bot._setup_memory_stack = AsyncMock()
        bot._setup_execution_stack = MagicMock()
        bot._setup_analysis_stack = AsyncMock()
        bot._setup_runtime_services = AsyncMock()
        bot._load_state = AsyncMock()
        bot._print_startup_summary = MagicMock()

        await bot.setup()

        bot._setup_memory_stack.assert_awaited_once_with()
        bot._setup_execution_stack.assert_called_once_with()
        bot._setup_analysis_stack.assert_awaited_once_with()
        bot._setup_runtime_services.assert_awaited_once_with()
        bot._load_state.assert_awaited_once_with()
        bot._print_startup_summary.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_full_bot_shutdown_cleans_owned_components(self, mock_config):
        bot = FullTradingBot(
            config=mock_config,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )

        bot._save_state = AsyncMock()
        bot._executor = MagicMock(
            get_portfolio=MagicMock(
                return_value={"total_value": 10000.0, "cash": 10000.0, "total_pnl": 0.0}
            ),
            get_trade_log=MagicMock(return_value=[]),
            close=MagicMock(),
        )
        bot._trade_memory = AsyncMock(close=AsyncMock())
        bot._mem0_memory = MagicMock(close=MagicMock())
        bot._knowledge_graph = MagicMock(save_to_file=MagicMock(), close=MagicMock())
        bot._risk_engine = MagicMock(close=MagicMock())
        bot._kill_switch = MagicMock(close=MagicMock())
        bot._order_manager = MagicMock(close=MagicMock())
        bot._strategies = {"BTC/USDT": {"strategy": object()}}
        bot._debate_engine = MagicMock(close=MagicMock())
        bot._news_pipeline = MagicMock(close=MagicMock())
        bot._auto_tuner = MagicMock(close=MagicMock())
        bot._redis_cache = AsyncMock(close=AsyncMock())
        bot._weekly_reviewer = MagicMock(close=MagicMock())
        bot._telegram_bot = MagicMock(close=MagicMock())

        executor = bot._executor
        trade_memory = bot._trade_memory
        mem0_memory = bot._mem0_memory
        knowledge_graph = bot._knowledge_graph
        risk_engine = bot._risk_engine
        kill_switch = bot._kill_switch
        order_manager = bot._order_manager
        debate_engine = bot._debate_engine
        news_pipeline = bot._news_pipeline
        auto_tuner = bot._auto_tuner
        redis_cache = bot._redis_cache
        weekly_reviewer = bot._weekly_reviewer
        telegram_bot = bot._telegram_bot

        await bot.shutdown()

        bot._save_state.assert_awaited_once_with()
        knowledge_graph.save_to_file.assert_called_once()
        executor.close.assert_called_once_with()
        trade_memory.close.assert_awaited_once_with()
        mem0_memory.close.assert_called_once_with()
        knowledge_graph.close.assert_called_once_with()
        risk_engine.close.assert_called_once_with()
        kill_switch.close.assert_called_once_with()
        order_manager.close.assert_called_once_with()
        debate_engine.close.assert_called_once_with()
        news_pipeline.close.assert_called_once_with()
        auto_tuner.close.assert_called_once_with()
        redis_cache.close.assert_awaited_once_with()
        weekly_reviewer.close.assert_called_once_with()
        telegram_bot.close.assert_called_once_with()
        assert bot._trade_memory is None
        assert bot._mem0_memory is None
        assert bot._knowledge_graph is None
        assert bot._risk_engine is None
        assert bot._kill_switch is None
        assert bot._executor is None
        assert bot._order_manager is None
        assert bot._strategies is None
        assert bot._debate_engine is None
        assert bot._news_pipeline is None
        assert bot._auto_tuner is None
        assert bot._redis_cache is None
        assert bot._weekly_reviewer is None
        assert bot._telegram_bot is None

    @pytest.mark.asyncio
    async def test_ai_bot_shutdown_cleans_debate_and_telegram_components(
        self, mock_config
    ):
        bot = AITradingBot(
            config=mock_config,
            mode="dryrun",
            symbols=["BTC/USDT"],
        )

        bot._dry_run_executor = MagicMock(
            get_portfolio=MagicMock(
                return_value={"total_value": 10000.0, "total_pnl": 0.0}
            )
        )
        bot._trade_memory = AsyncMock(close=AsyncMock())
        bot._redis_cache = AsyncMock(close=AsyncMock())
        bot._telegram_bot = MagicMock(close=MagicMock())
        bot._debate_engine = object()

        trade_memory = bot._trade_memory
        redis_cache = bot._redis_cache
        telegram_bot = bot._telegram_bot

        await bot.shutdown()

        trade_memory.close.assert_awaited_once_with()
        redis_cache.close.assert_awaited_once_with()
        telegram_bot.close.assert_called_once_with()
        assert bot._trade_memory is None
        assert bot._redis_cache is None
        assert bot._telegram_bot is None
        assert bot._debate_engine is None

    @patch("src.execution.exchange_client.ExchangeClient")
    @patch("src.execution.order_manager.OrderManager")
    def test_full_bot_setup_executor_fails_fast_on_connect_error(
        self, mock_om_class, mock_ec_class, mock_config
    ):
        bot = FullTradingBot(
            config=mock_config,
            mode="testnet",
            symbols=["BTC/USDT"],
        )

        mock_ec_class.return_value.connect.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="Failed to connect exchange client"):
            bot._setup_executor()

        mock_om_class.assert_not_called()
