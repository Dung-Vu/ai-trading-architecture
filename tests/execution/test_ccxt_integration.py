"""
Integration tests for CCXT exchange order execution and portfolio state fetching.
Mocks ExchangeClient and OrderManager to verify that FullTradingBot and AITradingBot
correctly interact with the exchange in testnet mode.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.main_full import FullTradingBot
from src.main_ai import AITradingBot


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
