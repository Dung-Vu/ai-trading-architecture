"""
Unit tests for AlertFormatter module.
"""

import pytest
from src.monitoring.alert_formatter import AlertFormatter


class TestAlertFormatter:
    def test_format_trade_alert_buy(self):
        msg = AlertFormatter.format_trade_alert("BUY", "BTC/USDT", 0.01, 50000.0)
        assert "BUY" in msg
        assert "BTC/USDT" in msg
        assert "50,000" in msg
        assert "🟢" in msg

    def test_format_trade_alert_sell_with_pnl(self):
        msg = AlertFormatter.format_trade_alert("SELL", "ETH/USDT", 1.0, 3500.0, pnl=200.0)
        assert "SELL" in msg
        assert "ETH/USDT" in msg
        assert "200" in msg
        assert "🟢" in msg  # Positive PnL

    def test_format_trade_alert_sell_with_loss(self):
        msg = AlertFormatter.format_trade_alert("SELL", "ETH/USDT", 1.0, 3500.0, pnl=-150.0)
        assert "🔴" in msg  # Negative PnL

    def test_format_status(self):
        msg = AlertFormatter.format_status(
            mode="dryrun",
            positions={"BTC/USDT": {"quantity": 0.01, "value": 500}},
            daily_pnl=125.30,
            total_value=10245.50,
            win_rate=62.5,
        )
        assert "dryrun" in msg.lower()
        assert "10,245" in msg
        assert "125.30" in msg
        assert "62.5" in msg

    def test_format_daily_report(self):
        msg = AlertFormatter.format_daily_report(
            date="2025-01-15",
            total_pnl=350.0,
            win_rate=65.0,
            total_trades=12,
            sharpe=1.85,
            max_dd=2.3,
            positions={"BTC/USDT": 5000},
        )
        assert "350" in msg
        assert "65" in msg
        assert "12" in msg
        assert "1.85" in msg
        assert "2.3" in msg
        assert "value=<code>$5,000.00</code>" in msg
        assert "@ $5,000.00" not in msg

    def test_format_error(self):
        msg = AlertFormatter.format_error("Connection timeout", "data_pipeline")
        assert "ERROR" in msg
        assert "Connection timeout" in msg
        assert "data_pipeline" in msg

    def test_format_kill_switch(self):
        from datetime import datetime
        msg = AlertFormatter.format_kill_switch("Max drawdown exceeded", datetime.now())
        assert "KILL SWITCH" in msg
        assert "drawdown" in msg
