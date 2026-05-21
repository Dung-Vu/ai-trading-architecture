"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


async def build_action_items_section(
    memory: Any,
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Generate action items section."""
    try:
        summary = await memory.get_performance_summary(
            start_date=start_date, end_date=end_date
        )
    except Exception:
        return ["## 📋 Action Items\n\n*Insufficient data for action items.*"]

    lines = [
        "## 📋 Action Items",
        "",
    ]

    priority = 1

    if summary.total_trades == 0:
        lines.append(
            f"{priority}. **Priority:** Get more trade data. "
            "No meaningful analysis possible without completed trades."
        )
        return lines

    if summary.win_rate < 45:
        lines.append(
            f"{priority}. **Priority: HIGH** — Win rate below 45%. "
            f"Review entry signals and consider adding confirmation filters."
        )
        priority += 1

    if summary.max_drawdown > 10:
        lines.append(
            f"{priority}. **Priority: HIGH** — Max drawdown {summary.max_drawdown:.1f}%. "
            "Reduce position sizes or tighten stop-losses."
        )
        priority += 1

    if summary.sharpe_ratio < 1.0 and summary.total_trades > 10:
        lines.append(
            f"{priority}. **Priority: MEDIUM** — Sharpe ratio below 1.0. "
            "DSPy optimization should focus on improving risk-adjusted returns."
        )
        priority += 1

    if summary.profit_factor < 1.2:
        lines.append(
            f"{priority}. **Priority: MEDIUM** — Profit factor below 1.2. "
            "Either cut losers faster or let winners run longer."
        )
        priority += 1

    if priority == 1:
        lines.append(
            "✅ No critical issues detected. Continue current strategy and "
            "monitor for regression."
        )

    lines.append("")

    return lines
