"""Shared dashboard data and layout helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Any

from loguru import logger
import pandas as pd
import streamlit as st

from src.config import (
    get_default_database_url,
    get_default_initial_capital,
    get_default_redis_url,
)
from src.dashboard_utils import load_mock_data


PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#E6EDF3", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    margin=dict(l=50, r=20, t=30, b=50),
)


@dataclass
class DashboardContext:
    trades_df: pd.DataFrame
    debates_df: pd.DataFrame
    equity_curve: pd.Series
    patterns: list[dict]
    total_trades: int
    winning: int
    total_pnl: float
    win_rate: float
    sharpe: float
    max_dd: float


def load_dashboard_data() -> dict:
    """Load dashboard data, preferring persisted history with mock fallback."""
    if _should_use_mock_dashboard_data():
        return load_mock_data()

    try:
        live_data = _load_live_dashboard_data()
    except Exception as exc:
        logger.warning(f"[Dashboard] Falling back to mock data: {exc}")
        return load_mock_data()

    if _has_dashboard_history(live_data):
        return live_data

    logger.info("[Dashboard] No persisted trade history found; using mock data")
    return load_mock_data()


def _should_use_mock_dashboard_data(use_mock: bool | None = None) -> bool:
    """Return whether mock dashboard data is explicitly requested."""
    if use_mock is not None:
        return use_mock

    raw_value = os.getenv("DASHBOARD_USE_MOCK_DATA", "")
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _load_live_dashboard_data() -> dict[str, Any]:
    """Synchronously load persisted dashboard records."""
    return asyncio.run(_fetch_live_dashboard_data())


async def _fetch_live_dashboard_data() -> dict[str, Any]:
    """Fetch dashboard datasets from persistent trade memory."""
    from src.memory import TradeMemory

    memory = TradeMemory(
        db_url=get_default_database_url(),
        redis_url=get_default_redis_url(),
    )
    await memory.connect()
    try:
        trades = await memory.get_trade_history(limit=250)
        debates = await memory.get_debate_history(limit=250)
        patterns = await memory.get_trade_patterns(min_samples=3)
    finally:
        await memory.close()

    ordered_trades = sorted(trades, key=lambda item: item.get("timestamp", ""))
    ordered_debates = sorted(debates, key=lambda item: item.get("timestamp", ""))

    return {
        "trades": ordered_trades,
        "debates": ordered_debates,
        "equity_curve": _build_live_equity_curve(ordered_trades),
        "patterns": _flatten_trade_patterns(patterns),
    }


def _has_dashboard_history(data: dict[str, Any]) -> bool:
    """Return whether a live dashboard payload has enough data to render."""
    return bool(data.get("trades")) and bool(data.get("debates"))


def _build_live_equity_curve(trades: list[dict[str, Any]]) -> pd.Series:
    """Build an equity curve from persisted trade records."""
    if not trades:
        return pd.Series(dtype=float)

    baseline = get_default_initial_capital()
    timestamps: list[pd.Timestamp] = []
    equity_values: list[float] = []
    running_equity = baseline

    for trade in trades:
        timestamp = pd.to_datetime(trade.get("timestamp"), utc=True)
        cash_total = trade.get("cash_total")
        cash_remaining = trade.get("cash_remaining")

        if cash_total is not None:
            running_equity = float(cash_total)
        elif cash_remaining is not None:
            running_equity = float(cash_remaining)
        else:
            running_equity += float(trade.get("pnl", 0.0))

        timestamps.append(timestamp)
        equity_values.append(running_equity)

    return pd.Series(equity_values, index=pd.DatetimeIndex(timestamps))


def _flatten_trade_patterns(patterns: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert analytics buckets into the dashboard's flat pattern card shape."""
    flattened: list[dict[str, Any]] = []

    for item in patterns.get("symbol_side_patterns", []):
        flattened.append({
            "name": f"{item['symbol']} {item['side']}",
            "frequency": item["total_trades"],
            "description": (
                f"Win rate {item['win_rate']:.1f}% | "
                f"Avg PnL ${item['avg_pnl']:+.2f}"
            ),
        })

    for item in patterns.get("strategy_symbol_patterns", []):
        flattened.append({
            "name": f"{item['strategy']} on {item['symbol']}",
            "frequency": item["total_trades"],
            "description": (
                f"Win rate {item['win_rate']:.1f}% | "
                f"Avg PnL ${item['avg_pnl']:+.2f}"
            ),
        })

    for item in patterns.get("time_of_day_patterns", []):
        flattened.append({
            "name": f"UTC hour {item['hour_utc']:02d}",
            "frequency": item["total_trades"],
            "description": (
                f"Win rate {item['win_rate']:.1f}% | "
                f"Avg PnL ${item['avg_pnl']:+.2f}"
            ),
        })

    for item in patterns.get("risk_action_outcomes", []):
        flattened.append({
            "name": f"Judge {item['judge_action']}",
            "frequency": item["total_trades"],
            "description": (
                f"Win rate {item['win_rate']:.1f}% | "
                f"Avg PnL ${item['avg_pnl']:+.2f}"
            ),
        })

    confidence = patterns.get("confidence_correlation") or {}
    high_conf_count = int(confidence.get("high_confidence_count", 0) or 0)
    low_conf_count = int(confidence.get("low_confidence_count", 0) or 0)
    if high_conf_count or low_conf_count:
        flattened.append({
            "name": "Confidence bands",
            "frequency": high_conf_count + low_conf_count,
            "description": (
                f"High-conf avg ${float(confidence.get('high_confidence_avg_pnl', 0.0)):+.2f} | "
                f"Low-conf avg ${float(confidence.get('low_confidence_avg_pnl', 0.0)):+.2f}"
            ),
        })

    return flattened


def build_context(data: dict) -> DashboardContext:
    """Normalize raw dashboard data into DataFrames and computed metrics."""
    trades_df = pd.DataFrame(data["trades"])
    debates_df = pd.DataFrame(data["debates"])
    equity_curve = data["equity_curve"]

    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
    debates_df["timestamp"] = pd.to_datetime(debates_df["timestamp"])
    trades_df["is_win"] = trades_df["pnl"] > 0

    total_trades = len(trades_df)
    winning = int(trades_df["is_win"].sum())
    total_pnl = float(trades_df["pnl"].sum())
    win_rate = winning / total_trades * 100 if total_trades > 0 else 0.0

    returns = equity_curve.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std() * (252**0.5)
        if len(returns) > 1 and returns.std() > 0
        else 0.0
    )
    running_max = equity_curve.cummax()
    max_dd = float(((running_max - equity_curve) / running_max).max() * 100)

    return DashboardContext(
        trades_df=trades_df,
        debates_df=debates_df,
        equity_curve=equity_curve,
        patterns=data["patterns"],
        total_trades=total_trades,
        winning=winning,
        total_pnl=total_pnl,
        win_rate=float(win_rate),
        sharpe=float(sharpe),
        max_dd=max_dd,
    )


def apply_theme() -> None:
    """Apply compact dark-theme CSS overrides."""
    st.markdown(
        """
<style>
    .stApp { background-color: #0E1117; }
    .stMetric label { color: #8B949E; }
    .stMetric div[data-testid="stMetricValue"] { color: #E6EDF3; }
    .stDataFrame { background-color: #161B22; }
    [data-testid="stSidebar"] { background-color: #0D1117; }
    section.main > div { padding-top: 2rem; }
    h1, h2, h3 { color: #E6EDF3 !important; }
    div[data-baseweb="select"] > div { background-color: #161B22; color: #E6EDF3; }
    .stSelectbox label, .stSlider label { color: #8B949E; }
    input, .stTextInput > div > div > input { background-color: #161B22; color: #E6EDF3; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_sidebar(ctx: DashboardContext) -> str:
    """Render sidebar navigation and return selected page key."""
    page_options = {
        "Overview": "overview",
        "Trade Analysis": "trade_analysis",
        "AI Debate": "ai_debate",
        "Settings": "settings",
    }

    st.sidebar.title("AI Trading")
    st.sidebar.markdown("---")
    selected_label = st.sidebar.radio(
        "Navigation",
        list(page_options.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    start = ctx.trades_df["timestamp"].min().strftime("%Y-%m-%d")
    end = ctx.trades_df["timestamp"].max().strftime("%Y-%m-%d")
    st.sidebar.caption(f"Data: {start} -> {end}")
    st.sidebar.caption(f"Total trades: {ctx.total_trades}")
    return page_options[selected_label]
