"""
Unit tests for DryRunExecutor module.
"""

import pytest
from src.execution.dry_run import DryRunExecutor


class TestDryRunExecutor:
    @pytest.fixture
    def executor(self):
        return DryRunExecutor(initial_balance=10000.0)

    def test_initial_state(self, executor):
        portfolio = executor.get_portfolio()
        assert portfolio["cash"] == 10000.0
        assert portfolio["total_value"] == 10000.0
        assert len(portfolio["positions"]) == 0

    def test_simulate_buy(self, executor):
        result = executor.simulate_buy("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")
        assert result["trade_id"] > 0
        assert result["cost"] == 5000.0

        portfolio = executor.get_portfolio()
        assert portfolio["cash"] == 5000.0
        assert "BTC/USDT" in portfolio["positions"]
        assert portfolio["positions"]["BTC/USDT"]["quantity"] == 0.1

    def test_simulate_sell(self, executor):
        executor.simulate_buy("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")
        result = executor.simulate_sell("BTC/USDT", 0.1, 52000.0, "2026-05-20T12:01:00")

        assert result["trade_id"] > 0
        assert result["revenue"] == 5200.0
        assert result["pnl"] == 200.0  # (52000 - 50000) * 0.1

        portfolio = executor.get_portfolio()
        assert portfolio["cash"] == 10200.0

    def test_insufficient_funds(self, executor):
        with pytest.raises(ValueError):
            executor.simulate_buy("BTC/USDT", 1.0, 50000.0, "2026-05-20T12:00:00")

    def test_sell_without_position(self, executor):
        with pytest.raises(ValueError):
            executor.simulate_sell("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")

    def test_trade_log(self, executor):
        executor.simulate_buy("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")
        executor.simulate_sell("BTC/USDT", 0.1, 52000.0, "2026-05-20T12:01:00")

        trades = executor.get_trade_log()
        assert len(trades) == 2
        assert trades[0]["side"] == "buy"
        assert trades[1]["side"] == "sell"
        assert trades[1]["pnl"] == 200.0

    def test_equity_curve(self, executor):
        executor.simulate_buy("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")
        curve = executor.get_equity_curve()
        assert len(curve) >= 1
        # First point should be initial balance
        assert curve[0][1] == 10000.0

    def test_sl_tp_trigger(self, executor):
        executor.simulate_buy("BTC/USDT", 0.1, 50000.0, "2026-05-20T12:00:00")

        sl_tp_orders = [
            {
                "id": "sl1",
                "symbol": "BTC/USDT",
                "side": "sell",
                "type": "stop_loss",
                "stop_price": 49000.0,
                "quantity": 0.1,
                "direction": "below",
            },
            {
                "id": "tp1",
                "symbol": "BTC/USDT",
                "side": "sell",
                "type": "take_profit",
                "stop_price": 55000.0,
                "quantity": 0.1,
                "direction": "above",
            },
        ]

        # Price hits stop loss
        triggered = executor.simulate_sl_tp(sl_tp_orders, {"BTC/USDT": 48500.0}, "2026-05-20T12:02:00")
        assert len(triggered) > 0
        assert any(t["triggered_by"] == "stop_loss" for t in triggered)
