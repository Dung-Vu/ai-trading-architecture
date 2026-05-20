"""
Unit tests for PositionSizer module.
"""

import pytest
from src.execution.position_sizer import PositionSizer


class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        return PositionSizer(max_position_pct=0.20, max_leverage=3, daily_loss_limit_pct=0.03)

    def test_half_kelly_profitable(self, sizer):
        """Half-Kelly for a strategy with 60% win rate, 3% avg win, 2% avg loss."""
        size = sizer.calc_half_kelly(win_rate=0.60, avg_win=0.03, avg_loss=0.02, equity=10000)
        assert size > 0
        assert size <= 10000.0  # Should never exceed 100% of equity

    def test_half_kelly_unprofitable(self, sizer):
        """Half-Kelly for a losing strategy should return 0."""
        size = sizer.calc_half_kelly(win_rate=0.40, avg_win=0.02, avg_loss=0.03, equity=10000)
        assert size == 0.0

    def test_van_tharp_basic(self, sizer):
        """Van Tharp with 2% risk, $100 entry, $98 stop."""
        size = sizer.calc_van_tharp(entry_price=100.0, stop_loss_price=98.0, equity=10000, risk_pct=0.02)
        assert size > 0
        # size = (10000 * 0.02) / (100 - 98) = 200 / 2 = 100
        assert abs(size - 100.0) < 0.01

    def test_calc_position_size(self, sizer):
        result = sizer.calc_position_size(
            strategy="sma_cross",
            symbol="BTC/USDT",
            entry_price=50000.0,
            stop_loss_price=49000.0,
            equity=10000.0,
            win_rate=0.55,
            avg_win=0.03,
            avg_loss=0.02,
        )
        assert hasattr(result, "size_quote")
        assert hasattr(result, "size_base")
        assert hasattr(result, "method_used")
        assert result.size_quote > 0

    def test_check_daily_loss_pass(self, sizer):
        assert sizer.check_daily_loss(
            current_equity=9800, start_equity=10000, limit_pct=0.03
        ) is False  # 2% loss, within 3% limit

    def test_check_daily_loss_fail(self, sizer):
        assert sizer.check_daily_loss(
            current_equity=9600, start_equity=10000, limit_pct=0.03
        ) is True  # 4% loss, exceeds 3% limit

    def test_check_concentration_pass(self, sizer):
        assert sizer.check_concentration(
            position_value=1500, total_equity=10000, max_pct=0.20
        ) is False  # 15%, within 20%

    def test_check_concentration_fail(self, sizer):
        assert sizer.check_concentration(
            position_value=3000, total_equity=10000, max_pct=0.20
        ) is True  # 30%, exceeds 20%

    def test_check_leverage_pass(self, sizer):
        assert sizer.check_leverage(2, max_leverage=3) is False

    def test_check_leverage_fail(self, sizer):
        assert sizer.check_leverage(5, max_leverage=3) is True
