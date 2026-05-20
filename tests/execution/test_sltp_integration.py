"""
Unit tests for the Stop-Loss / Take-Profit (SL/TP) simulation within the DryRunExecutor.
"""

import pytest

from src.execution.dry_run import DryRunExecutor


class TestSLTPIntegration:
    @pytest.fixture
    def executor(self):
        # Initialize executor with 10k USD balance
        return DryRunExecutor(initial_balance=10000.0)

    def test_stop_loss_trigger_long(self, executor):
        # 1. Open a LONG position: Buy 0.1 BTC at 50,000 USDT
        buy_result = executor.simulate_buy("BTC/USDT", 0.1, 50000.0)
        assert buy_result["symbol"] == "BTC/USDT"

        # Verify position was opened
        portfolio = executor.get_portfolio()
        assert "BTC/USDT" in portfolio["positions"]
        assert portfolio["positions"]["BTC/USDT"]["quantity"] == 0.1

        # 2. Define Stop Loss (SL) order: trigger when price drops below 49,000 USDT
        sl_order = {
            "id": "sl-001",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "stop_loss",
            "stop_price": 49000.0,
            "quantity": 0.1,
            "direction": "below",
        }

        # Test case A: Price drops to 49,500 USDT (not yet triggering SL)
        triggered_none = executor.simulate_sl_tp([sl_order], {"BTC/USDT": 49500.0})
        assert len(triggered_none) == 0
        assert "BTC/USDT" in executor.get_portfolio()["positions"]

        # Test case B: Price drops to 48,900 USDT (crosses below 49,000 SL threshold)
        triggered = executor.simulate_sl_tp([sl_order], {"BTC/USDT": 48900.0})
        assert len(triggered) == 1
        assert triggered[0]["triggered_by"] == "stop_loss"
        assert triggered[0]["trigger_order_id"] == "sl-001"
        assert triggered[0]["pnl"] == -110.0  # (48900 - 50000) * 0.1 = -110 USDT

        # Verify position is closed
        portfolio_after = executor.get_portfolio()
        assert "BTC/USDT" not in portfolio_after["positions"]
        assert portfolio_after["cash"] == 9890.0  # 10000 - 5000 + 4890 = 9890

    def test_take_profit_trigger_long(self, executor):
        # 1. Open a LONG position: Buy 0.1 BTC at 50,000 USDT
        buy_result = executor.simulate_buy("BTC/USDT", 0.1, 50000.0)
        assert buy_result["symbol"] == "BTC/USDT"

        # 2. Define Take Profit (TP) order: trigger when price goes above 55,000 USDT
        tp_order = {
            "id": "tp-001",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "take_profit",
            "stop_price": 55000.0,
            "quantity": 0.1,
            "direction": "above",
        }

        # Test case A: Price increases to 54,000 USDT (not yet triggering TP)
        triggered_none = executor.simulate_sl_tp([tp_order], {"BTC/USDT": 54000.0})
        assert len(triggered_none) == 0
        assert "BTC/USDT" in executor.get_portfolio()["positions"]

        # Test case B: Price increases to 55,100 USDT (crosses above 55,000 TP threshold)
        triggered = executor.simulate_sl_tp([tp_order], {"BTC/USDT": 55100.0})
        assert len(triggered) == 1
        assert triggered[0]["triggered_by"] == "take_profit"
        assert triggered[0]["trigger_order_id"] == "tp-001"
        assert triggered[0]["pnl"] == 510.0  # (55100 - 50000) * 0.1 = 510 USDT

        # Verify position is closed
        portfolio_after = executor.get_portfolio()
        assert "BTC/USDT" not in portfolio_after["positions"]
        assert portfolio_after["cash"] == 10510.0  # 10000 - 5000 + 5510 = 10510
