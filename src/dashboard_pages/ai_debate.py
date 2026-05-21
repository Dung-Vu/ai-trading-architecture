"""AI debate analytics dashboard page."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_pages.common import DashboardContext, PLOTLY_LAYOUT


def render(ctx: DashboardContext) -> None:
    """Render AI debate history, accuracy, and pattern charts."""
    debates_df = ctx.debates_df

    st.title("AI Debate Analysis")
    st.subheader("Recent Debates")
    debate_display = debates_df[
        [
            "timestamp",
            "symbol",
            "bull_arg",
            "bear_arg",
            "devil_arg",
            "judge_action",
            "judge_confidence",
            "risk_action",
            "latency_seconds",
        ]
    ].copy()
    debate_display = debate_display.sort_values("timestamp", ascending=False).head(20)
    debate_display["timestamp"] = debate_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    debate_display["judge_confidence"] = debate_display["judge_confidence"].apply(
        lambda v: f"{v:.1f}%"
    )
    debate_display["latency_seconds"] = debate_display["latency_seconds"].apply(
        lambda v: f"{v:.1f}s"
    )
    st.dataframe(debate_display, use_container_width=True, hide_index=True, height=350)

    st.markdown("---")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.subheader("Confidence Distribution")
        fig_conf = go.Figure(
            go.Histogram(
                x=debates_df["judge_confidence"].values,
                nbinsx=20,
                marker_color="#58A6FF",
                opacity=0.8,
            )
        )
        fig_conf.add_vline(
            x=70,
            line_dash="dash",
            line_color="#F0883E",
            annotation_text="High Confidence",
            annotation_position="top right",
            annotation_font=dict(color="#F0883E"),
        )
        fig_conf.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            xaxis_title="Judge Confidence (%)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_d2:
        st.subheader("AI Accuracy vs Actual Outcomes")
        debates_with_outcome = debates_df.copy()
        debates_with_outcome["predicted_correct"] = debates_with_outcome.apply(
            lambda r: r["actual_outcome"] == "profitable"
            and r["judge_action"] in ["BUY", "SELL"],
            axis=1,
        )
        accuracy_by_conf = []
        for bucket in ["30-50", "50-70", "70-90", "90-100"]:
            low, high = map(float, bucket.replace("-", ",").split(","))
            mask = (debates_with_outcome["judge_confidence"] >= low) & (
                debates_with_outcome["judge_confidence"] < high
            )
            subset = debates_with_outcome[mask]
            if len(subset) > 0:
                accuracy_by_conf.append(
                    {
                        "bucket": bucket,
                        "accuracy": round(subset["predicted_correct"].mean() * 100, 1),
                        "count": len(subset),
                    }
                )

        if accuracy_by_conf:
            acc_df = pd.DataFrame(accuracy_by_conf)
            fig_acc = go.Figure()
            fig_acc.add_trace(
                go.Bar(
                    x=acc_df["bucket"],
                    y=acc_df["accuracy"],
                    marker=dict(color=["#FF5252", "#F0883E", "#58A6FF", "#3FB950"]),
                    text=acc_df.apply(
                        lambda r: f"{r['accuracy']}% (n={r['count']})", axis=1
                    ),
                    textposition="outside",
                    textfont=dict(color="#E6EDF3"),
                )
            )
            fig_acc.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                xaxis_title="Confidence Bucket (%)",
                yaxis_title="Prediction Accuracy (%)",
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No outcome data available.")

    col_d3, col_d4 = st.columns(2)
    with col_d3:
        st.subheader("Judge Decisions")
        action_counts = debates_df["judge_action"].value_counts()
        fig_action = go.Figure(
            go.Pie(
                labels=action_counts.index.tolist(),
                values=action_counts.values.tolist(),
                marker=dict(colors=["#3FB950", "#FF5252", "#8B949E"]),
                textinfo="label+percent",
                textfont=dict(color="#E6EDF3"),
                hole=0.4,
            )
        )
        fig_action.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig_action, use_container_width=True)

    with col_d4:
        st.subheader("Most Common Patterns Detected")
        pattern_data = pd.DataFrame(ctx.patterns).sort_values("frequency", ascending=True)
        fig_patterns = go.Figure(
            go.Bar(
                x=pattern_data["frequency"],
                y=pattern_data["name"],
                orientation="h",
                marker=dict(color="#BC8CFF"),
                text=pattern_data["frequency"],
                textposition="outside",
                textfont=dict(color="#E6EDF3"),
                hovertext=pattern_data["description"],
                hoverinfo="text+x",
            )
        )
        fig_patterns.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            xaxis_title="Occurrences",
            margin=dict(l=140, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_patterns, use_container_width=True)

    st.markdown("---")
    st.subheader("Debate Statistics")
    avg_latency = debates_df["latency_seconds"].mean()
    avg_rounds = debates_df["rounds"].mean()
    high_conf_trades = (debates_df["judge_confidence"] >= 70).sum()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Latency", f"{avg_latency:.1f}s")
    s2.metric("Avg Rounds", f"{avg_rounds:.1f}")
    s3.metric("High Confidence", f"{high_conf_trades} ({high_conf_trades / len(debates_df) * 100:.0f}%)")
    s4.metric("Total Debates", len(debates_df))
