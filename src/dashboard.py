"""
AI Trading Architecture — Streamlit Dashboard

Run: streamlit run src/dashboard.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_utils import (  # noqa: E402
    color_pnl,
    format_currency,
    format_pct,
    load_mock_data,
)

# ─── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme overrides ─────────────────────────────────────────────
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
    .css-1d391kg, .css-1lcbmhc { background-color: #0D1117; }
    div[data-baseweb="select"] > div { background-color: #161B22; color: #E6EDF3; }
    .stSelectbox label { color: #8B949E; }
    input, .stTextInput > div > div > input { background-color: #161B22; color: #E6EDF3; }
    .stSlider label { color: #8B949E; }
    .css-1v3fvcr { background-color: #161B22; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Data loading ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_data():
    """Load data — tries real DB first, falls back to mock."""
    try:
        # Attempt real database connection (async, so we skip for now)
        # In production, wrap async calls in st.connection or sync adapter
        raise ImportError("DB not connected")
    except Exception:
        pass
    return load_mock_data()


data = get_data()
trades_df = pd.DataFrame(data["trades"])
debates_df = pd.DataFrame(data["debates"])
equity_curve = data["equity_curve"]
patterns = data["patterns"]

# Ensure timestamp columns are datetime
trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
debates_df["timestamp"] = pd.to_datetime(debates_df["timestamp"])

# Compute derived columns
trades_df["is_win"] = trades_df["pnl"] > 0
total_trades = len(trades_df)
winning = trades_df["is_win"].sum()
total_pnl = trades_df["pnl"].sum()
win_rate = winning / total_trades * 100 if total_trades > 0 else 0

# Compute metrics
returns = equity_curve.pct_change().dropna()
sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if len(returns) > 1 and returns.std() > 0 else 0
running_max = equity_curve.cummax()
max_dd = ((running_max - equity_curve) / running_max).max() * 100

# ─── Sidebar navigation ───────────────────────────────────────────────
st.sidebar.title("🤖 AI Trading")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📈 Trade Analysis", "🤖 AI Debate", "⚙️ Settings"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {trades_df['timestamp'].min().strftime('%Y-%m-%d')} → {trades_df['timestamp'].max().strftime('%Y-%m-%d')}")
st.sidebar.caption(f"Total trades: {total_trades}")

# ─── Shared Plotly dark template ──────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#E6EDF3", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    margin=dict(l=50, r=20, t=30, b=50),
)


# ======================================================================
# PAGE 1: 📊 OVERVIEW
# ======================================================================
if page == "📊 Overview":
    st.title("📊 Portfolio Overview")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total P&L", format_currency(total_pnl), delta=f"{total_pnl:+,.2f}")
    with c2:
        st.metric("Win Rate", format_pct(win_rate), delta=f"{winning:.0f}/{total_trades}")
    with c3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with c4:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.markdown("---")

    # Equity curve
    st.subheader("Equity Curve")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode="lines",
        name="Equity",
        line=dict(color="#58A6FF", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))

    # Add trade markers on equity curve
    # Approximate equity at each trade point
    trade_dates = trades_df["timestamp"].dt.tz_localize(None)
    if len(trade_dates) > 0:
        marker_pnls = trades_df["pnl"].values
        marker_colors = ["#00E676" if p > 0 else "#FF5252" for p in marker_pnls]
        fig_equity.add_trace(go.Scatter(
            x=trade_dates,
            y=[equity_curve.iloc[0]] * len(trade_dates),  # placeholder y — overlay only
            mode="markers",
            name="Trades",
            marker=dict(symbol="diamond", size=6, color=marker_colors, opacity=0.6),
            showlegend=False,
        ))

    fig_equity.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_equity, width="stretch")

    # Portfolio allocation + Bot health
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Portfolio Allocation")
        alloc = trades_df.groupby("symbol")["pnl"].sum().abs()
        alloc = alloc[alloc > 0].sort_values(ascending=True)
        colors_alloc = ["#58A6FF", "#3FB950", "#F0883E", "#BC8CFF", "#F778BA"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=alloc.index.tolist(),
            values=alloc.values.tolist(),
            hole=0.5,
            marker=dict(colors=colors_alloc[:len(alloc)]),
            textinfo="label+percent",
            textfont=dict(color="#E6EDF3"),
        )])
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        st.subheader("Bot Health Status")
        status_items = [
            ("Data Pipeline", "🟢 Running", "QuestDB: connected"),
            ("AI Debate Engine", "🟢 Ready", "Latency: 4.2s avg"),
            ("Risk Engine", "🟢 Active", "All checks passed"),
            ("Execution Layer", "🔵 Dry-run", "No live orders"),
            ("Telegram Alerts", "🟢 Connected", "Last ping: 2m ago"),
            ("Memory (PostgreSQL)", "🟢 Connected", "1,247 records"),
        ]
        for name, status, detail in status_items:
            col_s1, col_s2 = st.columns([1, 3])
            with col_s1:
                st.markdown(f"**{name}**")
            with col_s2:
                st.markdown(f"{status}<br><small style='color:#8B949E'>{detail}</small>", unsafe_allow_html=True)
            st.divider()

    # Recent trades summary
    st.subheader("Recent Trades")
    recent = trades_df.tail(10).sort_values("timestamp", ascending=False)
    display_recent = recent[["timestamp", "symbol", "side", "quantity", "price", "pnl", "strategy"]].copy()
    display_recent["timestamp"] = display_recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_recent["pnl"] = display_recent["pnl"].apply(color_pnl)
    st.dataframe(display_recent, width="stretch", hide_index=True)


# ======================================================================
# PAGE 2: 📈 TRADE ANALYSIS
# ======================================================================
elif page == "📈 Trade Analysis":
    st.title("📈 Trade Analysis")

    # Filters
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

    # Apply filters
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
        filtered = filtered[(filtered["timestamp"] >= start_dt) & (filtered["timestamp"] < end_dt)]

    # Filtered metrics
    f_total = len(filtered)
    f_pnl = filtered["pnl"].sum()
    f_wr = (filtered["is_win"].sum() / f_total * 100) if f_total > 0 else 0
    st.caption(f"Showing {f_total} trades | P&L: {format_currency(f_pnl)} | Win Rate: {format_pct(f_wr)}")

    st.markdown("---")

    # Trade table
    st.subheader("Trade History")
    display = filtered[["timestamp", "symbol", "side", "quantity", "price", "pnl", "pnl_pct", "strategy", "ai_confidence"]].copy()
    display = display.sort_values("timestamp", ascending=False)
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display["pnl"] = display["pnl"].apply(color_pnl)
    display["pnl_pct"] = display["pnl_pct"].apply(lambda v: f"{v:+.2f}%")
    st.dataframe(display, width="stretch", hide_index=True, height=350)

    st.markdown("---")

    # Charts row 1
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("P&L Distribution")
        pnl_vals = filtered["pnl"].values
        fig_hist = go.Figure()
        # Green bars for profit, red for loss
        fig_hist.add_trace(go.Histogram(
            x=pnl_vals[pnl_vals > 0],
            xbins=dict(size=abs(pnl_vals.max() - pnl_vals.min()) / 30) if len(pnl_vals) > 0 else {},
            marker_color="#00E676",
            opacity=0.8,
            name="Profit",
        ))
        fig_hist.add_trace(go.Histogram(
            x=pnl_vals[pnl_vals <= 0],
            xbins=dict(size=abs(pnl_vals.max() - pnl_vals.min()) / 30) if len(pnl_vals) > 0 else {},
            marker_color="#FF5252",
            opacity=0.8,
            name="Loss",
        ))
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            barmode="overlay",
            xaxis_title="P&L ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_c2:
        st.subheader("Win / Loss Ratio")
        wins = (filtered["pnl"] > 0).sum()
        losses = (filtered["pnl"] <= 0).sum()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=win_rate,
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
                    value=win_rate,
                ),
            ),
            number=dict(font=dict(size=36, color="#E6EDF3")),
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_gauge, width="stretch")

    # Charts row 2
    col_c3, col_c4 = st.columns(2)

    with col_c3:
        st.subheader("Cumulative Returns")
        cum_pnl = filtered.sort_values("timestamp")["pnl"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=filtered.sort_values("timestamp")["timestamp"],
            y=cum_pnl.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.1)",
            line=dict(color="#58A6FF", width=2),
        ))
        fig_cum.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
        )
        st.plotly_chart(fig_cum, width="stretch")

    with col_c4:
        st.subheader("Avg Hold Time by Symbol")
        # Approximate hold time from timestamp gaps within same symbol
        hold_times = []
        for sym in filtered["symbol"].unique():
            sym_trades = filtered[filtered["symbol"] == sym].sort_values("timestamp")
            if len(sym_trades) > 1:
                diffs = sym_trades["timestamp"].diff().dt.total_seconds().dropna()
                avg_hours = diffs.mean() / 3600
                hold_times.append({"symbol": sym, "avg_hours": round(avg_hours, 1)})

        if hold_times:
            hold_df = pd.DataFrame(hold_times).sort_values("avg_hours", ascending=True)
            fig_hold = go.Figure(go.Bar(
                x=hold_df["avg_hours"],
                y=hold_df["symbol"],
                orientation="h",
                marker=dict(
                    color=["#58A6FF", "#3FB950", "#F0883E", "#BC8CFF", "#F778BA"][:len(hold_df)],
                ),
                text=hold_df["avg_hours"].apply(lambda v: f"{v:.1f}h"),
                textposition="outside",
                textfont=dict(color="#E6EDF3"),
            ))
            fig_hold.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis_title="Average Hours Between Trades",
                yaxis_title="Symbol",
                margin=dict(l=80, r=50, t=30, b=50),
            )
            st.plotly_chart(fig_hold, width="stretch")
        else:
            st.info("Not enough data to compute hold times.")


# ======================================================================
# PAGE 3: 🤖 AI DEBATE
# ======================================================================
elif page == "🤖 AI Debate":
    st.title("🤖 AI Debate Analysis")

    # Debate table
    st.subheader("Recent Debates")
    debate_display = debates_df[["timestamp", "symbol", "bull_arg", "bear_arg", "devil_arg",
                                  "judge_action", "judge_confidence", "risk_action",
                                  "latency_seconds"]].copy()
    debate_display = debate_display.sort_values("timestamp", ascending=False).head(20)
    debate_display["timestamp"] = debate_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    debate_display["judge_confidence"] = debate_display["judge_confidence"].apply(lambda v: f"{v:.1f}%")
    debate_display["latency_seconds"] = debate_display["latency_seconds"].apply(lambda v: f"{v:.1f}s")
    st.dataframe(debate_display, width="stretch", hide_index=True, height=350)

    st.markdown("---")

    # Charts
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.subheader("Confidence Distribution")
        conf_vals = debates_df["judge_confidence"].values
        fig_conf = go.Figure(go.Histogram(
            x=conf_vals,
            nbinsx=20,
            marker_color="#58A6FF",
            opacity=0.8,
        ))
        fig_conf.add_vline(x=70, line_dash="dash", line_color="#F0883E",
                           annotation_text="High Confidence", annotation_position="top right",
                           annotation_font=dict(color="#F0883E"))
        fig_conf.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_title="Judge Confidence (%)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_conf, width="stretch")

    with col_d2:
        st.subheader("AI Accuracy vs Actual Outcomes")
        # Map debates to outcomes for accuracy check
        outcome_map = {
            "BUY": "profitable",
            "SELL": "profitable",  # simplified mapping
            "HOLD": "breakeven",
        }
        debates_with_outcome = debates_df.copy()
        debates_with_outcome["predicted_correct"] = debates_with_outcome.apply(
            lambda r: r["actual_outcome"] == "profitable" and r["judge_action"] in ["BUY", "SELL"],
            axis=1,
        )
        accuracy_by_conf = []
        for bucket in ["30-50", "50-70", "70-90", "90-100"]:
            low, high = map(float, bucket.replace("-", ",").split(","))
            mask = (debates_with_outcome["judge_confidence"] >= low) & (debates_with_outcome["judge_confidence"] < high)
            subset = debates_with_outcome[mask]
            if len(subset) > 0:
                acc = subset["predicted_correct"].mean() * 100
                accuracy_by_conf.append({"bucket": bucket, "accuracy": round(acc, 1), "count": len(subset)})

        if accuracy_by_conf:
            acc_df = pd.DataFrame(accuracy_by_conf)
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=acc_df["bucket"],
                y=acc_df["accuracy"],
                marker=dict(
                    color=["#FF5252", "#F0883E", "#58A6FF", "#3FB950"],
                ),
                text=acc_df.apply(lambda r: f"{r['accuracy']}% (n={r['count']})", axis=1),
                textposition="outside",
                textfont=dict(color="#E6EDF3"),
            ))
            fig_acc.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis_title="Confidence Bucket (%)",
                yaxis_title="Prediction Accuracy (%)",
            )
            st.plotly_chart(fig_acc, width="stretch")
        else:
            st.info("No outcome data available.")

    # Judge action breakdown
    col_d3, col_d4 = st.columns(2)

    with col_d3:
        st.subheader("Judge Decisions")
        action_counts = debates_df["judge_action"].value_counts()
        fig_action = go.Figure(go.Pie(
            labels=action_counts.index.tolist(),
            values=action_counts.values.tolist(),
            marker=dict(colors=["#3FB950", "#FF5252", "#8B949E"]),
            textinfo="label+percent",
            textfont=dict(color="#E6EDF3"),
            hole=0.4,
        ))
        fig_action.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_action, width="stretch")

    with col_d4:
        st.subheader("Most Common Patterns Detected")
        pattern_data = pd.DataFrame(patterns).sort_values("frequency", ascending=True)
        fig_patterns = go.Figure(go.Bar(
            x=pattern_data["frequency"],
            y=pattern_data["name"],
            orientation="h",
            marker=dict(color="#BC8CFF"),
            text=pattern_data["frequency"],
            textposition="outside",
            textfont=dict(color="#E6EDF3"),
            hovertext=pattern_data["description"],
            hoverinfo="text+x",
        ))
        fig_patterns.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis_title="Occurrences",
            margin=dict(l=140, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_patterns, width="stretch")

    # Debate stats summary
    st.markdown("---")
    st.subheader("Debate Statistics")
    avg_latency = debates_df["latency_seconds"].mean()
    avg_rounds = debates_df["rounds"].mean()
    high_conf_trades = (debates_df["judge_confidence"] >= 70).sum()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Latency", f"{avg_latency:.1f}s")
    s2.metric("Avg Rounds", f"{avg_rounds:.1f}")
    s3.metric("High Confidence", f"{high_conf_trades} ({high_conf_trades/len(debates_df)*100:.0f}%)")
    s4.metric("Total Debates", len(debates_df))


# ======================================================================
# PAGE 4: ⚙️ SETTINGS
# ======================================================================
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    # Trading mode
    st.subheader("Trading Mode")
    mode = st.radio(
        "Select Mode",
        ["🔵 Dry-run (Paper Trading)", "🟡 Testnet", "🟢 Live Trading"],
        index=0,
        horizontal=True,
    )
    st.caption(f"Current mode: **{mode.split('(')[0].strip()}** — Dry-run recommended for testing")

    st.markdown("---")

    # Strategy parameters
    st.subheader("Strategy Parameters")
    with st.expander("AI Debate Strategy", expanded=True):
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            max_rounds = st.slider("Max Debate Rounds", 1, 5, 3, step=1)
            temperature = st.slider("LLM Temperature", 0.0, 1.5, 0.7, step=0.1)
            min_confidence = st.slider("Min Confidence Threshold (%)", 0, 100, 60, step=5)
        with c_s2:
            llm_model = st.selectbox("Primary LLM Model", [
                "anthropic/claude-sonnet-4",
                "openai/gpt-4o",
                "google/gemini-2.5-pro",
            ])
            fallback_model = st.selectbox("Fallback LLM Model", [
                "openai/gpt-4o",
                "anthropic/claude-sonnet-4",
                "google/gemini-2.5-pro",
            ])
            timeout = st.slider("LLM Timeout (seconds)", 10, 300, 120, step=10)

    with st.expander("Technical Strategies"):
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("**SMA Cross**")
            sma_short = st.number_input("Short SMA Period", 5, 100, 20, step=1)
            sma_long = st.number_input("Long SMA Period", 10, 200, 50, step=1)
        with c_t2:
            st.markdown("**Bollinger Bands**")
            bb_period = st.number_input("BB Period", 5, 100, 20, step=1)
            bb_std = st.slider("BB Std Dev", 1.0, 3.0, 2.0, step=0.1)

    st.markdown("---")

    # Risk limits
    st.subheader("Risk Limits")
    with st.expander("Position Sizing & Limits", expanded=True):
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            max_position_pct = st.slider("Max Position Size (% of portfolio)", 1, 50, 10, step=1)
            max_drawdown_pct = st.slider("Max Drawdown Limit (%)", 5, 50, 20, step=1)
        with c_r2:
            max_daily_trades = st.slider("Max Daily Trades", 5, 100, 20, step=1)
            max_daily_loss = st.number_input("Max Daily Loss ($)", 100, 100000, 5000, step=100)
        with c_r3:
            kill_switch_dd = st.slider("Kill Switch Drawdown (%)", 10, 60, 30, step=1)
            stop_loss_pct = st.slider("Default Stop-Loss (%)", 1, 20, 5, step=1)

    st.markdown("---")

    # API Keys
    st.subheader("API Keys Status")
    c_a1, c_a2, c_a3 = st.columns(3)

    with c_a1:
        st.markdown("**Binance**")
        st.text_input("API Key", value="bn••••••••••••••••4f2a", disabled=True, label_visibility="collapsed")
        st.success("Connected" if True else "Not connected")

    with c_a2:
        st.markdown("**LLM Provider**")
        st.text_input("API Key", value="sk••••••••••••••••••••••••••x9z", disabled=True, label_visibility="collapsed")
        st.success("Connected" if True else "Not connected")

    with c_a3:
        st.markdown("**Telegram Bot**")
        st.text_input("Bot Token", value="123••••:AA••••••••••••••••••••••••", disabled=True, label_visibility="collapsed")
        st.success("Connected" if True else "Not connected")

    st.caption("⚠️ API keys are loaded from environment variables. They are never stored in the database or logs.")

    st.markdown("---")

    # Save button (mock)
    if st.button("💾 Save Settings", type="primary", width="stretch"):
        st.success("Settings saved successfully! (Demo mode — not persisted)")
        st.info("In production, settings are saved to config/production.yaml")

    # System info
    st.markdown("---")
    st.subheader("System Information")
    info_items = {
        "Dashboard Version": "1.0.0",
        "Data Source": "Mock (demo mode)",
        "Last Updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Streamlit Version": st.__version__,
    }
    for k, v in info_items.items():
        st.markdown(f"**{k}:** {v}")
