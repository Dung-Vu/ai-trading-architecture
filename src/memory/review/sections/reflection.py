"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


async def build_reflection_section(
    memory: Any,
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Generate the reflective analysis section."""
    try:
        summary = await memory.get_performance_summary(
            start_date=start_date, end_date=end_date
        )
        patterns = await memory.get_trade_patterns()
    except Exception as exc:
        logger.warning(f"Failed to get reflection data: {exc}")
        return ["## 🤔 Reflection\n\n*Insufficient data for reflection.*"]

    lines = [
        "## 🤔 Reflection",
        "",
        "### ✅ What Went Well?",
        "",
    ]
    lines.extend(build_positive_reflection_lines(summary, patterns))
    lines.append("")
    lines.append("### ❌ What Went Poorly?")
    lines.append("")
    lines.extend(build_negative_reflection_lines(summary, patterns))
    lines.append("")
    lines.append("### 🔄 What to Change?")
    lines.append("")
    lines.extend(build_change_reflection_lines(summary, patterns))
    lines.append("")

    return lines


def build_positive_reflection_lines(
    summary: Any,
    patterns: dict[str, Any],
) -> list[str]:
    """Summarize the strongest outcomes and repeatable positive patterns."""
    if summary.total_trades <= 0:
        return [
            "- No completed trades to evaluate. Focus on getting more execution data."
        ]

    lines: list[str] = []
    if summary.win_rate >= 55:
        lines.append(
            f"- Overall win rate of {summary.win_rate:.1f}% is solid. "
            f"The trading approach is working."
        )

    if summary.profit_factor > 1.5:
        lines.append(
            f"- Profit factor of {summary.profit_factor:.2f} means "
            f"we're making more on winners than losing on losers."
        )

    if summary.best_trade > 0:
        lines.append(
            f"- Best trade earned ${summary.best_trade:+,.2f}. "
            f"The strategy can capture meaningful moves."
        )

    good_patterns = [
        pattern for pattern in patterns.get("symbol_side_patterns", [])
        if pattern["win_rate"] >= 60 and pattern["total_trades"] >= 5
    ]
    if good_patterns:
        symbols = ", ".join(
            f"{pattern['symbol']} {pattern['side']}" for pattern in good_patterns
        )
        lines.append(
            f"- High-win-rate patterns detected: {symbols}. "
            f"These setups should be prioritized."
        )

    return lines


def build_negative_reflection_lines(
    summary: Any,
    patterns: dict[str, Any],
) -> list[str]:
    """Summarize the main weak spots in recent trading performance."""
    if summary.total_trades <= 0:
        return ["- No trades yet to evaluate losses. This is the primary gap."]

    lines: list[str] = []
    if summary.win_rate < 45:
        lines.append(
            f"- Win rate of {summary.win_rate:.1f}% is concerning. "
            f"Entry signals need refinement."
        )

    if summary.max_drawdown > 10:
        lines.append(
            f"- Max drawdown of {summary.max_drawdown:.1f}% is significant. "
            f"Consider tighter stop-losses or smaller position sizes."
        )

    if summary.sharpe_ratio < 0.5:
        lines.append(
            f"- Sharpe ratio of {summary.sharpe_ratio:.2f} is poor. "
            f"Risk-adjusted returns need improvement."
        )

    bad_patterns = [
        pattern for pattern in patterns.get("symbol_side_patterns", [])
        if pattern["win_rate"] < 35 and pattern["total_trades"] >= 5
    ]
    if bad_patterns:
        symbols = ", ".join(
            f"{pattern['symbol']} {pattern['side']}" for pattern in bad_patterns
        )
        lines.append(
            f"- Low-win-rate patterns: {symbols}. "
            f"These setups should be avoided or reworked."
        )

    confidence_correlation = patterns.get("confidence_correlation", {})
    if confidence_correlation and confidence_correlation.get(
        "high_confidence_avg_pnl", 0
    ) < confidence_correlation.get("low_confidence_avg_pnl", 0):
        lines.append(
            "- AI confidence is INVERSELY correlated with outcomes. "
            "High confidence trades performed WORSE than low confidence ones. "
            "This suggests the AI is overconfident on bad setups."
        )

    return lines


def build_change_reflection_lines(
    summary: Any,
    patterns: dict[str, Any],
) -> list[str]:
    """Translate recent performance into concrete adjustments."""
    if summary.total_trades <= 0:
        return [
            "- No data-driven changes yet. Focus on executing more trades "
            "to build a statistical foundation."
        ]

    lines: list[str] = []
    if summary.win_rate < 50:
        lines.append(
            "- **Raise entry threshold:** Require higher AI confidence "
            "(≥70%) or stricter technical conditions before entering trades."
        )

    if summary.max_drawdown > 8:
        lines.append(
            "- **Tighten risk limits:** Reduce max position size or "
            "implement trailing stops to limit drawdowns."
        )

    confidence_correlation = patterns.get("confidence_correlation", {})
    if confidence_correlation and confidence_correlation.get(
        "high_confidence_avg_pnl", 0
    ) < 0:
        lines.append(
            "- **Calibrate AI confidence:** The AI is overconfident on "
            "losing trades. DSPy optimization should focus on better "
            "confidence calibration."
        )

    if not lines:
        lines.append(
            "- Performance is generally acceptable. Focus on incremental "
            "improvements rather than major changes."
        )

    return lines
