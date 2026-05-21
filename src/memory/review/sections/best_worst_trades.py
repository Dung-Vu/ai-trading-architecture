"""Weekly review section builder."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger


async def build_best_worst_trades_section(
    memory: Any,
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Generate best and worst trades section."""
    try:
        history = await memory.get_trade_history(
            start_date=start_date, end_date=end_date, limit=500
        )
    except Exception as exc:
        logger.warning(f"Failed to get trade history: {exc}")
        return ["## 🏆 Best & Worst Trades\n\n*No trade data available.*"]

    if not history:
        return ["## 🏆 Best & Worst Trades\n\n*No trades in this period.*"]

    # Find SELL trades with PnL
    pnl_trades = [t for t in history if t.get("side") == "SELL" and t.get("pnl", 0) != 0]

    if not pnl_trades:
        return ["## 🏆 Best & Worst Trades\n\n*No closing trades with realized P&L.*"]

    best = max(pnl_trades, key=lambda t: t.get("pnl", 0))
    worst = min(pnl_trades, key=lambda t: t.get("pnl", 0))

    lines = [
        "## 🏆 Best & Worst Trades",
        "",
        "### Best Trade",
        "",
        f"- **Symbol:** {best.get('symbol', 'N/A')}",
        f"- **Side:** {best.get('side', 'N/A')} {best.get('quantity', 0)} @ ${best.get('price', 0):,.2f}",
        f"- **P&L:** 🟢 ${best.get('pnl', 0):+.2f} ({best.get('pnl_pct', 0):+.1f}%)",
        f"- **Strategy:** {best.get('strategy', 'N/A')}",
        f"- **AI Confidence:** {best.get('ai_confidence', 'N/A')}",
        "",
        "### Worst Trade",
        "",
        f"- **Symbol:** {worst.get('symbol', 'N/A')}",
        f"- **Side:** {worst.get('side', 'N/A')} {worst.get('quantity', 0)} @ ${worst.get('price', 0):,.2f}",
        f"- **P&L:** 🔴 ${worst.get('pnl', 0):+.2f} ({worst.get('pnl_pct', 0):+.1f}%)",
        f"- **Strategy:** {worst.get('strategy', 'N/A')}",
        f"- **AI Confidence:** {worst.get('ai_confidence', 'N/A')}",
        "",
    ]

    # Add analysis
    best_conf = best.get("ai_confidence")
    worst_conf = worst.get("ai_confidence")

    if best_conf and worst_conf:
        if best_conf > worst_conf:
            lines.append("📌 AI confidence aligned with outcomes (high confidence on wins).")
        else:
            lines.append("⚠️ AI confidence was HIGHER on the losing trade than the winning trade. Review confidence calibration.")

    return lines
