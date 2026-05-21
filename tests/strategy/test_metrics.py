"""
Unit tests for MetricsCalculator module.
"""

import pytest
import pandas as pd
from src.strategy.metrics import MetricsCalculator


class TestMetricsCalculator:
    def test_sharpe_ratio(self):
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002])
        sharpe = MetricsCalculator.calc_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_sharpe_ratio_negative(self):
        returns = pd.Series([-0.02, -0.01, -0.03, -0.005, -0.015])
        sharpe = MetricsCalculator.calc_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sharpe < 0  # Negative returns should give negative Sharpe

    def test_sortino_ratio(self):
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        sortino = MetricsCalculator.calc_sortino_ratio(returns, risk_free_rate=0.02)
        assert sortino > 0

    def test_max_drawdown(self):
        equity = pd.Series([10000, 10500, 11000, 10800, 9500, 9800, 10200])
        max_dd = MetricsCalculator.calc_max_drawdown(equity)
        assert max_dd > 0
        # Peak = 11000, trough = 9500, DD = (11000-9500)/11000 = 13.6%
        assert abs(max_dd - 0.13636) < 0.01

    def test_win_rate(self):
        trades = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}, {"pnl": 150},
        ]
        win_rate = MetricsCalculator.calc_win_rate(trades)
        assert win_rate == 0.6  # 3 wins out of 5

    def test_profit_factor(self):
        trades = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30},
        ]
        pf = MetricsCalculator.calc_profit_factor(trades)
        # Gross profit = 300, Gross loss = 80, PF = 300/80 = 3.75
        assert abs(pf - 3.75) < 0.01

    def test_expectancy(self):
        trades = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}, {"pnl": 150},
        ]
        expectancy = MetricsCalculator.calc_expectancy(trades)
        # Avg win = 150, Avg loss = 40, Win rate = 0.6, Loss rate = 0.4
        # Expectancy = 150 * 0.6 - 40 * 0.4 = 90 - 16 = 74
        assert abs(expectancy - 74.0) < 1.0

    def test_summarize(self):
        trades = [
            {"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}, {"pnl": 150},
        ]
        equity = pd.Series([10000, 10100, 10050, 10250, 10220, 10370])
        summary = MetricsCalculator.summarize(trades, equity)

        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown_pct" in summary
        assert "win_rate_pct" in summary
        assert "total_trades" in summary
        assert "profit_factor" in summary
        assert summary["total_trades"] == 5
        assert abs(summary["total_return_pct"] - 3.7) < 0.1  # (10370-10000)/10000 * 100

    def test_trade_pnl_sharpe_ratio(self):
        pnls = [100.0, -40.0, 80.0, -20.0, 60.0]
        sharpe = MetricsCalculator.calc_trade_pnl_sharpe_ratio(pnls, risk_free_rate=0.0)
        assert sharpe > 0

    def test_trade_pnl_max_drawdown(self):
        pnls = [100.0, 50.0, -75.0, -25.0, 20.0]
        max_dd = MetricsCalculator.calc_trade_pnl_max_drawdown(pnls)
        assert abs(max_dd - (100.0 / 150.0)) < 0.001
