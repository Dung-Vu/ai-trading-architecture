"""
AI Trading Architecture Streamlit dashboard.

Run:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_pages import ai_debate, overview, settings, trade_analysis
from src.dashboard_pages.common import (
    apply_theme,
    build_context,
    load_dashboard_data,
    render_sidebar,
)


st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def get_data() -> dict:
    """Load dashboard data with Streamlit caching."""
    return load_dashboard_data()


def main() -> None:
    """Render the selected dashboard page."""
    apply_theme()
    ctx = build_context(get_data())
    page = render_sidebar(ctx)

    pages = {
        "overview": overview.render,
        "trade_analysis": trade_analysis.render,
        "ai_debate": ai_debate.render,
        "settings": settings.render,
    }
    pages[page](ctx)


main()
