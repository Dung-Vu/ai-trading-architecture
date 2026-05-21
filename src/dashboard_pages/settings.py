"""Dashboard settings page."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import streamlit as st

from src.dashboard_pages.common import DashboardContext


def render(_ctx: DashboardContext) -> None:
    """Render editable demo settings."""
    st.title("Settings")

    st.subheader("Trading Mode")
    mode = st.radio(
        "Select Mode",
        ["Dry-run (Paper Trading)", "Testnet", "Live Trading"],
        index=0,
        horizontal=True,
    )
    st.caption(f"Current mode: **{mode.split('(')[0].strip()}** - dry-run recommended for testing")

    st.markdown("---")
    st.subheader("Strategy Parameters")
    with st.expander("AI Debate Strategy", expanded=True):
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.slider("Max Debate Rounds", 1, 5, 3, step=1)
            st.slider("LLM Temperature", 0.0, 1.5, 0.7, step=0.1)
            st.slider("Min Confidence Threshold (%)", 0, 100, 60, step=5)
        with c_s2:
            st.selectbox(
                "Primary LLM Model",
                [
                    "anthropic/claude-sonnet-4",
                    "openai/gpt-4o",
                    "google/gemini-2.5-pro",
                ],
            )
            st.selectbox(
                "Fallback LLM Model",
                [
                    "openai/gpt-4o",
                    "anthropic/claude-sonnet-4",
                    "google/gemini-2.5-pro",
                ],
            )
            st.slider("LLM Timeout (seconds)", 10, 300, 120, step=10)

    with st.expander("Technical Strategies"):
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("**SMA Cross**")
            st.number_input("Short SMA Period", 5, 100, 20, step=1)
            st.number_input("Long SMA Period", 10, 200, 50, step=1)
        with c_t2:
            st.markdown("**Bollinger Bands**")
            st.number_input("BB Period", 5, 100, 20, step=1)
            st.slider("BB Std Dev", 1.0, 3.0, 2.0, step=0.1)

    st.markdown("---")
    st.subheader("Risk Limits")
    with st.expander("Position Sizing & Limits", expanded=True):
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            st.slider("Max Position Size (% of portfolio)", 1, 50, 10, step=1)
            st.slider("Max Drawdown Limit (%)", 5, 50, 20, step=1)
        with c_r2:
            st.slider("Max Daily Trades", 5, 100, 20, step=1)
            st.number_input("Max Daily Loss ($)", 100, 100000, 5000, step=100)
        with c_r3:
            st.slider("Kill Switch Drawdown (%)", 10, 60, 30, step=1)
            st.slider("Default Stop-Loss (%)", 1, 20, 5, step=1)

    st.markdown("---")
    st.subheader("API Keys Status")
    c_a1, c_a2, c_a3 = st.columns(3)
    for col, name in [
        (c_a1, "Exchange"),
        (c_a2, "LLM Provider"),
        (c_a3, "Telegram Bot"),
    ]:
        with col:
            st.markdown(f"**{name}**")
            st.text_input(
                "Status",
                value="Loaded from environment",
                disabled=True,
                label_visibility="collapsed",
            )
            st.success("Connected")

    st.caption("API keys are loaded from environment variables and are not stored in the dashboard.")

    st.markdown("---")
    if st.button("Save Settings", type="primary", use_container_width=True):
        st.success("Settings saved successfully. Demo mode does not persist changes.")
        st.info("In production, settings are saved to config/production.yaml")

    st.markdown("---")
    st.subheader("System Information")
    info_items = {
        "Dashboard Version": "1.0.0",
        "Data Source": "Mock (demo mode)",
        "Last Updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Streamlit Version": st.__version__,
    }
    for key, value in info_items.items():
        st.markdown(f"**{key}:** {value}")
