import pytest

from src.memory.trade_memory import PerformanceSummary
from src.memory.weekly_review import WeeklyReviewer


class FakeTradeMemory:
    async def get_performance_summary(self, start_date=None, end_date=None):
        return PerformanceSummary(
            total_trades=4,
            winning_trades=3,
            losing_trades=1,
            win_rate=75.0,
            total_pnl=180.0,
            avg_pnl=45.0,
            best_trade=120.0,
            worst_trade=-30.0,
            sharpe_ratio=1.6,
            max_drawdown=6.5,
            profit_factor=2.4,
        )

    async def get_trade_history(self, start_date=None, end_date=None, limit=500):
        return [
            {
                "symbol": "BTC/USDT",
                "side": "SELL",
                "quantity": 0.1,
                "price": 51000.0,
                "pnl": 120.0,
                "pnl_pct": 6.0,
                "strategy": "sma_cross",
                "ai_confidence": 82.0,
            },
            {
                "symbol": "ETH/USDT",
                "side": "SELL",
                "quantity": 0.5,
                "price": 2900.0,
                "pnl": -30.0,
                "pnl_pct": -1.2,
                "strategy": "bbands",
                "ai_confidence": 61.0,
            },
        ]

    async def get_strategy_performance(self, start_date=None, end_date=None):
        return {
            "sma_cross": PerformanceSummary(
                total_trades=3,
                winning_trades=2,
                losing_trades=1,
                win_rate=66.7,
                total_pnl=140.0,
                avg_pnl=46.7,
                best_trade=120.0,
                worst_trade=-30.0,
                sharpe_ratio=1.4,
                max_drawdown=5.0,
                profit_factor=2.0,
            ),
            "bbands": PerformanceSummary(
                total_trades=1,
                winning_trades=1,
                losing_trades=0,
                win_rate=100.0,
                total_pnl=40.0,
                avg_pnl=40.0,
                best_trade=40.0,
                worst_trade=40.0,
                sharpe_ratio=1.1,
                max_drawdown=1.0,
                profit_factor=1.5,
            ),
        }

    async def get_trade_patterns(self):
        return {
            "symbol_side_patterns": [
                {
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "win_rate": 68,
                    "total_trades": 6,
                    "avg_pnl": 35.0,
                }
            ],
            "time_of_day_patterns": [
                {
                    "hour_utc": 12,
                    "win_rate": 66,
                    "avg_pnl": 22.0,
                }
            ],
            "confidence_correlation": {
                "high_confidence_avg_pnl": 30.0,
                "high_confidence_count": 3,
                "low_confidence_avg_pnl": -5.0,
                "low_confidence_count": 1,
            },
        }


@pytest.mark.asyncio
async def test_generate_report_is_async_and_returns_formatted_text(tmp_path):
    reviewer = WeeklyReviewer(FakeTradeMemory(), report_dir=str(tmp_path))

    report = await reviewer.generate_report()

    assert "## 📈 Performance Summary" in report
    assert "## 🏆 Best & Worst Trades" in report
    assert "## ⚔️ Strategy Comparison" in report
    assert "sma_cross" in report