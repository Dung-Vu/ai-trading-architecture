"""Trade analysis dashboard page."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import color_pnl, format_currency, format_pct
from src.dashboard_pages.common import DashboardContext, PLOTLY_LAYOUT


def render(ctx: DashboardContext) -> None:
    """Render trade filters, history, and trade analytics charts."""
    trades_df = ctx.trades_df

    st.title("Trade Analysis")
    st.sidebar.subheader("Filters")
    symbols = ["All"] + sorted(trades_df["symbol"].unique().tolist())
    strategies = ["All"] + sorted(trades_df["strategy"].unique().tolist())
    sides = ["All"] + sorted(trades_df["side"].unique().tolist())

    sel_symbol = st.sidebar.selectbox("Symbol", symbols)
    sel_strategy = st.sidebar.selectbox("Strategy", strategies)
    sel_side = st.sidebar.selectbox("Side", sides)
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[
            trades_df["timestamp"].min().date(),
            trades_df["timestamp"].max().date(),
        ],
    )

    filtered = trades_df.copy()
    if sel_symbol != "All":
        filtered = filtered[filtered["symbol"] == sel_symbol]
    if sel_strategy != "All":
        filtered = filtered[filtered["strategy"] == sel_strategy]
    if sel_side != "All":
        filtered = filtered[filtered["side"] == sel_side]
    if len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt = pd.Timestamp(date_range[1]) + timedelta(days=1)
        filtered = filtered[
            (filtered["timestamp"] >= start_dt) & (filtered["timestamp"] < end_dt)
        ]

    f_total = len(filtered)
    f_pnl = filtered["pnl"].sum()
    f_wr = filtered["is_win"].sum() / f_total * 100 if f_total > 0 else 0
    st.caption(
        f"Showing {f_total} trades | P&L: {format_currency(f_pnl)} | "
        f"Win Rate: {format_pct(f_wr)}"
    )

    st.markdown("---")
    st.subheader("Trade History")
    display = filtered[
        [
            "timestamp",
            "symbol",
            "side",
            "quantity",
            "price",
            "pnl",
            "pnl_pct",
            "strategy",
            "ai_confidence",
        ]
    ].copy()
    display = display.sort_values("timestamp", ascending=False)
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display["pnl"] = display["pnl"].apply(color_pnl)
    display["pnl_pct"] = display["pnl_pct"].apply(lambda v: f"{v:+.2f}%")
    st.dataframe(display, use_container_width=True, hide_index=True, height=350)

    st.markdown("---")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("P&L Distribution")
        pnl_vals = filtered["pnl"].values
        bin_size = (
            abs(pnl_vals.max() - pnl_vals.min()) / 30 if len(pnl_vals) > 0 else None
        )
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=pnl_vals[pnl_vals > 0],
                xbins=dict(size=bin_size) if bin_size else {},
                marker_color="#00E676",
                opacity=0.8,
                name="Profit",
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=pnl_vals[pnl_vals <= 0],
                xbins=dict(size=bin_size) if bin_size else {},
                marker_color="#FF5252",
                opacity=0.8,
                name="Loss",
            )
        )
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            barmode="overlay",
            xaxis_title="P&L ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_c2:
        st.subheader("Win / Loss Ratio")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=f_wr,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Win Rate", "font": {"size": 16, "color": "#E6EDF3"}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor="#E6EDF3"),
                    bar=dict(color="#58A6FF"),
                    bgcolor="#21262D",
                    bordercolor="#30363D",
                    borderwidth=2,
                    steps=[
                        {"range": [0, 40], "color": "#FF5252"},
                        {"range": [40, 55], "color": "#F0883E"},
                        {"range": [55, 100], "color": "#3FB950"},
                    ],
                    threshold=dict(
                        line=dict(color="#E6EDF3", width=4),
                        thickness=0.75,
                        value=f_wr,
                    ),
                ),
                number=dict(font=dict(size=36, color="#E6EDF3")),
            )
        )
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.subheader("Cumulative Returns")
        ordered = filtered.sort_values("timestamp")
        cum_pnl = ordered["pnl"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(
                x=ordered["timestamp"],
                y=cum_pnl.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(88,166,255,0.1)",
                line=dict(color="#58A6FF", width=2),
            )
        )
        fig_cum.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_c4:
        st.subheader("Avg Hold Time by Symbol")
        hold_times = []
        for sym in filtered["symbol"].unique():
            sym_trades = filtered[filtered["symbol"] == sym].sort_values("timestamp")
            if len(sym_trades) > 1:
                diffs = sym_trades["timestamp"].diff().dt.total_seconds().dropna()
                hold_times.append({"symbol": sym, "avg_hours": round(diffs.mean() / 3600, 1)})

        if hold_times:
            hold_df = pd.DataFrame(hold_times).sort_values("avg_hours", ascending=True)
            fig_hold = go.Figure(
                go.Bar(
                    x=hold_df["avg_hours"],
                    y=hold_df["symbol"],
                    orientation="h",
                    marker=dict(
                        color=["#58A6FF", "#3FB950", "#F0883E", "#BC8CFF", "#F778BA"][
                            : len(hold_df)
                        ]
                    ),
                    text=hold_df["avg_hours"].apply(lambda v: f"{v:.1f}h"),
                    textposition="outside",
                    textfont=dict(color="#E6EDF3"),
                )
            )
            fig_hold.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis_title="Average Hours Between Trades",
                yaxis_title="Symbol",
                margin=dict(l=80, r=50, t=30, b=50),
            )
            st.plotly_chart(fig_hold, use_container_width=True)
        else:
            st.info("Not enough data to compute hold times.")
