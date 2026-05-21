"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


async def build_strategy_comparison_section(
    memory: Any,
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Generate strategy comparison section."""
    try:
        strat_perf = await memory.get_strategy_performance(
            start_date=start_date, end_date=end_date
        )
    except Exception as exc:
        logger.warning(f"Failed to get strategy performance: {exc}")
        return ["## ⚔️ Strategy Comparison\n\n*No strategy data available.*"]

    if not strat_perf:
        return ["## ⚔️ Strategy Comparison\n\n*No strategies with closing trades in this period.*"]

    lines = [
        "## ⚔️ Strategy Comparison",
        "",
    ]

    # Sort by total P&L
    sorted_strats = sorted(
        strat_perf.items(),
        key=lambda x: x[1].total_pnl,
        reverse=True,
    )

    lines.append("| Strategy | Trades | Win Rate | Total P&L | Avg P&L | Sharpe |")
    lines.append("|----------|--------|----------|-----------|---------|--------|")

    for name, perf in sorted_strats:
        emoji = "🟢" if perf.total_pnl >= 0 else "🔴"
        lines.append(
            f"| {emoji} {name} "
            f"| {perf.total_trades} "
            f"| {perf.win_rate:.1f}% "
            f"| ${perf.total_pnl:+,.2f} "
            f"| ${perf.avg_pnl:+,.2f} "
            f"| {perf.sharpe_ratio:.2f} |"
        )

    lines.append("")

    # Winner commentary
    if sorted_strats:
        winner_name, winner_perf = sorted_strats[0]
        lines.append(
            f"🏆 **{winner_name}** was the top-performing strategy "
            f"this week with ${winner_perf.total_pnl:+,.2f} P&L."
        )
        lines.append("")

    return lines
