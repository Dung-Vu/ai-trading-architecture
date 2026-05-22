"""
Unit tests for RiskEngine module.
"""

from datetime import datetime, timedelta, timezone

import pytest
from src.risk.risk_engine import RiskEngine


class TestRiskEngine:
    @pytest.fixture
    def engine(self):
        return RiskEngine(
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.10,
            max_position_pct=0.20,
            max_leverage=3,
        )

    def test_pre_trade_approved(self, engine):
        approved, reason = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0,
            current_equity=10000.0,
            start_equity=10000.0,
            positions={},
        )
        assert approved is True

    def test_pre_trade_daily_loss_exceeded(self, engine):
        engine.update_daily_pnl(-350.0)  # 3.5% loss on $10k
        engine.update_peak_equity(10000.0)

        approved, reason = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0,
            current_equity=9650.0,
            start_equity=10000.0,
            positions={},
        )
        assert approved is False
        assert "daily loss" in reason.lower()

    def test_pre_trade_drawdown_exceeded(self, engine):
        engine.update_peak_equity(10000.0)

        approved, reason = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0,
            current_equity=8900.0,  # 11% drawdown
            start_equity=10000.0,
            positions={},
        )
        assert approved is False
        assert "drawdown" in reason.lower()

    def test_pre_trade_concentration_exceeded(self, engine):
        positions = {
            "ETH/USDT": {"market_value": 2500.0},  # 25% of $10k
        }

        approved, reason = engine.pre_trade_checks(
            symbol="ETH/USDT",
            side="buy",
            quantity=0.1,
            price=3000.0,
            current_equity=10000.0,
            start_equity=10000.0,
            positions=positions,
        )
        assert approved is False
        assert "concentration" in reason.lower()

    def test_get_status(self, engine):
        engine.update_peak_equity(10000.0)
        engine.update_daily_pnl(-100.0)

        approved, _ = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0,
            current_equity=9800.0,
            start_equity=10000.0,
            positions={},
        )

        status = engine.get_status()
        assert approved is True
        assert hasattr(status, "daily_pnl")
        assert hasattr(status, "current_drawdown_pct")
        assert status.daily_pnl == -100.0
        assert status.daily_pnl_pct == pytest.approx(-0.01)
        assert status.current_drawdown_pct == pytest.approx(0.02)
        assert status.drawdown_limit_exceeded is False

    def test_get_status_drawdown_limit_exceeded(self, engine):
        engine.update_peak_equity(10000.0)

        approved, _ = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,
            price=50000.0,
            current_equity=8900.0,
            start_equity=10000.0,
            positions={},
        )

        status = engine.get_status()
        assert approved is False
        assert status.current_drawdown_pct == pytest.approx(0.11)
        assert status.drawdown_limit_exceeded is True

    def test_reset_daily(self, engine):
        engine.update_daily_pnl(-200.0)
        engine.reset_daily()

        status = engine.get_status()
        assert status.daily_pnl == 0.0

    def test_pre_trade_leverage_exceeded(self, engine):
        positions = {
            "ETH/USDT": {"market_value": 29500.0},
        }

        approved, reason = engine.pre_trade_checks(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=5000.0,
            current_equity=10000.0,
            start_equity=10000.0,
            positions=positions,
        )

        assert approved is False
        assert "leverage" in reason.lower()

    def test_get_status_resets_daily_pnl_after_day_boundary(self, engine):
        engine.update_daily_pnl(-125.0)
        engine._last_reset_date = datetime.now(timezone.utc).date() - timedelta(days=1)

        status = engine.get_status()

        assert status.daily_pnl == 0.0

    def test_update_daily_pnl_resets_before_appending_on_new_day(self, engine):
        engine.update_daily_pnl(-125.0)
        engine._last_reset_date = datetime.now(timezone.utc).date() - timedelta(days=1)

        engine.update_daily_pnl(-25.0)

        status = engine.get_status()
        assert status.daily_pnl == -25.0
