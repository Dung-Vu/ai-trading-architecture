"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


async def build_pattern_analysis_section(memory: Any) -> list[str]:
    """Generate pattern analysis section."""
    try:
        patterns = await memory.get_trade_patterns()
    except Exception as exc:
        logger.warning(f"Failed to get patterns: {exc}")
        return ["## 🔍 Pattern Analysis\n\n*Unable to analyze patterns.*"]

    lines = [
        "## 🔍 Pattern Analysis",
        "",
    ]

    # Symbol + Side patterns
    sp = patterns.get("symbol_side_patterns", [])
    if sp:
        lines.append("### Symbol + Side Performance")
        lines.append("")
        for p in sp:
            emoji = "🟢" if p["win_rate"] >= 55 else "🔴" if p["win_rate"] < 40 else "⚠️"
            lines.append(
                f"- {emoji} **{p['symbol']} {p['side']}**: "
                f"{p['win_rate']}% win rate ({p['total_trades']} trades), "
                f"avg P&L ${p['avg_pnl']:+,.2f}"
            )
        lines.append("")

    # Time of day patterns
    tp = patterns.get("time_of_day_patterns", [])
    if tp:
        lines.append("### Time-of-Day Patterns (UTC)")
        lines.append("")
        for p in sorted(tp, key=lambda x: x["avg_pnl"], reverse=True)[:5]:
            emoji = "🟢" if p["avg_pnl"] > 0 else "🔴"
            lines.append(
                f"- {emoji} **{p['hour_utc']:02d}:00 UTC**: "
                f"{p['win_rate']}% win rate, avg P&L ${p['avg_pnl']:+,.2f}"
            )
        lines.append("")

    # Confidence correlation
    cc = patterns.get("confidence_correlation", {})
    if cc:
        lines.append("### AI Confidence Correlation")
        lines.append("")
        lines.append(
            f"- High confidence (≥70%) avg P&L: ${cc.get('high_confidence_avg_pnl', 0):+.2f} "
            f"({cc.get('high_confidence_count', 0)} trades)"
        )
        lines.append(
            f"- Low confidence (<70%) avg P&L: ${cc.get('low_confidence_avg_pnl', 0):+.2f} "
            f"({cc.get('low_confidence_count', 0)} trades)"
        )
        lines.append("")

    return lines
