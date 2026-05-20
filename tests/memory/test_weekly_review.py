from datetime import UTC, datetime

import pytest

from src.memory.trade_memory import PerformanceSummary
from src.memory.weekly_review import WeeklyReviewer


class _Memory:
    async def get_performance_summary(self, *args, **kwargs):
        return PerformanceSummary(
            total_trades=0,
            total_pnl=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
        )

    async def get_trade_history(self, *args, **kwargs):
        return []

    async def get_strategy_performance(self, *args, **kwargs):
        return {}

    async def get_trade_patterns(self, *args, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_generate_report_awaits_async_sections(tmp_path):
    reviewer = WeeklyReviewer(_Memory(), report_dir=str(tmp_path))

    report = await reviewer.generate_report(
        end_date=datetime(2026, 5, 20, tzinfo=UTC),
        lookback_days=7,
    )

    assert "# " in report
    assert "coroutine" not in report
    assert "Performance Summary" in report
