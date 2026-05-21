"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from ...schema import PerformanceSummary


async def build_performance_summary_section(
    memory: Any,
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Generate the performance summary section."""
    try:
        summary: PerformanceSummary = await memory.get_performance_summary(
            start_date=start_date, end_date=end_date
        )
    except Exception as exc:
        logger.warning(f"Failed to get performance summary: {exc}")
        return ["## 📈 Performance Summary\n\n*No data available for this period.*"]

    emoji = "🟢" if summary.total_pnl >= 0 else "🔴"

    lines = [
        "## 📈 Performance Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| {emoji} **Total P&L** | ${summary.total_pnl:+,.2f} |",
        f"| 📊 **Win Rate** | {summary.win_rate:.1f}% ({summary.winning_trades}W / {summary.losing_trades}L) |",
        f"| 📈 **Sharpe Ratio** | {summary.sharpe_ratio:.3f} |",
        f"| 📉 **Max Drawdown** | {summary.max_drawdown:.1f}% |",
        f"| 💰 **Avg P&L/Trade** | ${summary.avg_pnl:+,.2f} |",
        f"| 🏆 **Best Trade** | ${summary.best_trade:+,.2f} |",
        f"| 💔 **Worst Trade** | ${summary.worst_trade:+,.2f} |",
        f"| ⚖️ **Profit Factor** | {summary.profit_factor:.2f} |",
        f"| 📝 **Total Trades** | {summary.total_trades} |",
        "",
    ]

    # Commentary
    if summary.total_trades == 0:
        lines.append("*No closing trades in this period.*")
    else:
        if summary.win_rate >= 60:
            lines.append("✅ Win rate is strong (>60%).")
        elif summary.win_rate >= 45:
            lines.append("⚠️ Win rate is moderate. Consider refining entry signals.")
        else:
            lines.append("🔴 Win rate is below 45%. Significant strategy review needed.")

        if summary.sharpe_ratio >= 1.5:
            lines.append("✅ Risk-adjusted returns are excellent (Sharpe > 1.5).")
        elif summary.sharpe_ratio >= 1.0:
            lines.append("⚠️ Risk-adjusted returns are decent (Sharpe > 1.0).")
        else:
            lines.append("🔴 Risk-adjusted returns need improvement (Sharpe < 1.0).")

    return lines
