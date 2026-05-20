"""
Utility functions for the AI Trading Streamlit Dashboard.

Provides mock data generation, formatting helpers, and data loading
logic that works standalone without database connections.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ─── Formatting helpers ────────────────────────────────────────────────

def format_currency(value: float) -> str:
    """Format a number as currency: '$12,345.67'."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def format_pct(value: float) -> str:
    """Format a number as percentage: '12.34%'."""
    return f"{value:.2f}%"


def color_pnl(value: float) -> str:
    """Return HTML color tag for P&L display — green for profit, red for loss."""
    if value >= 0:
        return f"<span style='color:#00E676'>{format_currency(value)}</span>"
    return f"<span style='color:#FF5252'>{format_currency(value)}</span>"


# ─── Mock data generation ─────────────────────────────────────────────

_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
_STRATEGIES = ["ai_debate", "sma_cross", "bbands", "momentum"]
_SIDES = ["BUY", "SELL"]
_JUDGE_ACTIONS = ["BUY", "SELL", "HOLD"]
_RISK_ACTIONS = ["APPROVE", "REJECT", "REDUCE"]

_BULL_ARGS = [
    "Strong support at key Fibonacci level with increasing volume confirmation",
    "Bullish divergence on RSI while price makes higher lows",
    "Breaking above 50-day MA with momentum from institutional accumulation",
    "Macro headwinds easing, liquidity conditions improving for risk assets",
    "On-chain metrics show long-term holders accumulating, supply shock incoming",
    "Golden cross forming on daily timeframe with rising OBV",
    "Funding rates neutral — room for long squeeze to fuel upward move",
]
_BEAR_ARGS = [
    "Distribution pattern visible on 4H with declining volume on rallies",
    "Overbought RSI with bearish divergence at major resistance zone",
    "Exchange inflows spiking — whales preparing to offload positions",
    "Macro uncertainty rising, DXY strengthening against risk assets",
    "Open interest at ATH while price stalls — potential long liquidation cascade",
    "Death cross looming on weekly, historical precedent shows further downside",
    "Funding rates excessively positive — overcrowded long positioning",
]
_DEVIL_ARGS = [
    "Both sides ignore the impact of upcoming regulatory decisions in major markets",
    "Correlation with traditional markets is being understated in both arguments",
    "Options expiry this week will create artificial volatility unrelated to fundamentals",
    "Liquidity conditions can change rapidly — current setup may be a bull/bear trap",
    "Historical patterns are breaking down due to structural market changes (ETF flows)",
    "Neither argument accounts for potential black swan events in the current regime",
]

_JUDGE_REASONS = [
    "Risk/reward favors long entry with tight stop below support",
    "Wait for pullback to retest broken resistance as new support",
    "Insufficient conviction — stay flat until clearer signal emerges",
    "Scale in gradually with 3 tranches, stop at recent swing low",
    "Short-term bullish but reduce size due to elevated macro risk",
]

_COMMON_PATTERNS = [
    {"name": "Morning Reversal", "description": "Strong reversals between 00:00-02:00 UTC", "frequency": 34},
    {"name": "Asian Range Breakout", "description": "Breakouts from Asian session consolidation", "frequency": 28},
    {"name": "US Open Momentum", "description": "Directional moves following NYSE open", "frequency": 42},
    {"name": "Friday Profit Taking", "description": "Systematic unwinding before weekend", "frequency": 19},
    {"name": "Low Volatility Squeeze", "description": "Volatility contraction followed by expansion", "frequency": 23},
]


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def load_mock_data(
    num_trades: int = 500,
    num_debates: int = 100,
    seed: int | None = 42,
) -> dict:
    """Generate realistic mock trading data for demo mode.

    Returns a dict with keys:
        - trades: list[dict] with trade records
        - debates: list[dict] with debate records
        - equity_curve: pd.Series of portfolio values over time
        - patterns: list[dict] of common patterns detected
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    now = datetime.utcnow()
    start_date = now - timedelta(days=90)

    # ── Generate equity curve ──
    dates = pd.date_range(start=start_date, end=now, freq="D")
    daily_returns = np.random.normal(0.001, 0.02, len(dates))
    # Add slight upward drift with occasional drawdowns
    daily_returns[::15] -= 0.04  # periodic drawdowns
    equity = 10000 * np.cumprod(1 + daily_returns)
    equity_curve = pd.Series(equity, index=dates)

    # ── Generate trades ──
    trades = []
    symbols_cycle = iter(_SYMBOLS)
    strategies_cycle = iter(_STRATEGIES)

    for i in range(num_trades):
        symbol = next(symbols_cycle, iter(_SYMBOLS)).__next__() if i % 4 == 0 else random.choice(_SYMBOLS)
        strategy = random.choice(_STRATEGIES)
        side = random.choice(_SIDES)

        # Realistic price ranges per symbol
        base_prices = {
            "BTC/USDT": 67500, "ETH/USDT": 3500, "SOL/USDT": 175,
            "BNB/USDT": 600, "XRP/USDT": 0.55,
        }
        base = base_prices.get(symbol, 100)
        price = base * (1 + random.uniform(-0.15, 0.15))

        # P&L: ~55% win rate, skewed distribution
        if random.random() < 0.55:
            pnl = abs(random.gauss(0, base * 0.015)) * random.uniform(0.5, 3.0)
        else:
            pnl = -abs(random.gauss(0, base * 0.01)) * random.uniform(0.3, 2.5)

        quantity = round(random.uniform(0.001, 0.5), 4)
        confidence = round(random.uniform(35, 95), 1)

        ts = _random_timestamp(start_date, now)
        trades.append({
            "id": i + 1,
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": round(price, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / (price * quantity) * 100, 2) if price * quantity > 0 else 0.0,
            "strategy": strategy,
            "mode": random.choice(["dryrun", "dryrun", "testnet"]),
            "ai_confidence": confidence,
            "stop_loss": round(price * (1 - random.uniform(0.01, 0.05)), 2) if side == "BUY" else round(price * (1 + random.uniform(0.01, 0.05)), 2),
            "take_profit": round(price * (1 + random.uniform(0.02, 0.10)), 2) if side == "BUY" else round(price * (1 - random.uniform(0.02, 0.10)), 2),
            "order_id": f"ORD-{ts.strftime('%Y%m%d')}-{i:04d}",
        })

    # Sort trades by timestamp
    trades.sort(key=lambda t: t["timestamp"])

    # ── Generate debates ──
    debates = []
    for i in range(num_debates):
        ts = _random_timestamp(start_date, now)
        symbol = random.choice(_SYMBOLS)
        judge_action = random.choice(_JUDGE_ACTIONS)
        risk_action = random.choice(_RISK_ACTIONS)
        confidence = round(random.uniform(30, 95), 1)

        debates.append({
            "id": i + 1,
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "bull_arg": random.choice(_BULL_ARGS),
            "bear_arg": random.choice(_BEAR_ARGS),
            "devil_arg": random.choice(_DEVIL_ARGS),
            "judge_action": judge_action,
            "judge_confidence": confidence,
            "risk_action": risk_action,
            "risk_reasoning": random.choice(_JUDGE_REASONS),
            "rounds": random.randint(1, 3),
            "latency_seconds": round(random.uniform(2.5, 18.0), 2),
            "actual_outcome": random.choice(["profitable", "loss", "breakeven"]),
        })

    debates.sort(key=lambda d: d["timestamp"])

    return {
        "trades": trades,
        "debates": debates,
        "equity_curve": equity_curve,
        "patterns": _COMMON_PATTERNS,
    }
