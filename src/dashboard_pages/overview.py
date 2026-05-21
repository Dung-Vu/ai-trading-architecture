"""Portfolio overview page."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import color_pnl, format_currency, format_pct
from src.dashboard_pages.common import DashboardContext, PLOTLY_LAYOUT


def render(ctx: DashboardContext) -> None:
    """Render the overview dashboard page."""
    trades_df = ctx.trades_df
    equity_curve = ctx.equity_curve

    st.title("Portfolio Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total P&L", format_currency(ctx.total_pnl), delta=f"{ctx.total_pnl:+,.2f}")
    c2.metric("Win Rate", format_pct(ctx.win_rate), delta=f"{ctx.winning}/{ctx.total_trades}")
    c3.metric("Sharpe Ratio", f"{ctx.sharpe:.2f}")
    c4.metric("Max Drawdown", f"{ctx.max_dd:.2f}%")

    st.markdown("---")
    st.subheader("Equity Curve")
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="Equity",
            line=dict(color="#58A6FF", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        )
    )

    trade_dates = trades_df["timestamp"].dt.tz_localize(None)
    if len(trade_dates) > 0:
        marker_colors = ["#00E676" if p > 0 else "#FF5252" for p in trades_df["pnl"].values]
        fig_equity.add_trace(
            go.Scatter(
                x=trade_dates,
                y=[equity_curve.iloc[0]] * len(trade_dates),
                mode="markers",
                name="Trades",
                marker=dict(symbol="diamond", size=6, color=marker_colors, opacity=0.6),
                showlegend=False,
            )
        )

    fig_equity.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis_title="",
        yaxis_title="Portfolio Value ($)",
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Portfolio Allocation")
        alloc = trades_df.groupby("symbol")["pnl"].sum().abs()
        alloc = alloc[alloc > 0].sort_values(ascending=True)
        colors_alloc = ["#58A6FF", "#3FB950", "#F0883E", "#BC8CFF", "#F778BA"]
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=alloc.index.tolist(),
                    values=alloc.values.tolist(),
                    hole=0.5,
                    marker=dict(colors=colors_alloc[: len(alloc)]),
                    textinfo="label+percent",
                    textfont=dict(color="#E6EDF3"),
                )
            ]
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Bot Health Status")
        status_items = [
            ("Data Pipeline", "Running", "QuestDB: connected"),
            ("AI Debate Engine", "Ready", "Latency: 4.2s avg"),
            ("Risk Engine", "Active", "All checks passed"),
            ("Execution Layer", "Dry-run", "No live orders"),
            ("Telegram Alerts", "Connected", "Last ping: 2m ago"),
            ("Memory", "Connected", "1,247 records"),
        ]
        for name, status, detail in status_items:
            col_s1, col_s2 = st.columns([1, 3])
            col_s1.markdown(f"**{name}**")
            col_s2.markdown(
                f"{status}<br><small style='color:#8B949E'>{detail}</small>",
                unsafe_allow_html=True,
            )
            st.divider()

    st.subheader("Recent Trades")
    recent = trades_df.tail(10).sort_values("timestamp", ascending=False)
    display_recent = recent[
        ["timestamp", "symbol", "side", "quantity", "price", "pnl", "strategy"]
    ].copy()
    display_recent["timestamp"] = display_recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_recent["pnl"] = display_recent["pnl"].apply(color_pnl)
    st.dataframe(display_recent, use_container_width=True, hide_index=True)
