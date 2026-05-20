"""Backtest Report Generator — HTML, Markdown, and JSON output."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go


class BacktestReport:
    """Generate production-ready backtest reports in HTML, Markdown, and JSON.

    Parameters
    ----------
    backtest_results : dict
        Standardised backtest result dict (see ``BacktestRunner.get_results``).
        Expected keys: total_return, sharpe_ratio, max_drawdown, win_rate,
        profit_factor, total_trades, equity_curve, trades, monthly_returns, etc.
    strategy_name : str
        Human-readable strategy name shown in report headers.
    symbol : str
        Trading pair symbol (e.g. ``"BTC/USDT"``).
    """

    def __init__(
        self,
        backtest_results: dict[str, Any],
        strategy_name: str,
        symbol: str,
    ) -> None:
        self.results = backtest_results
        self.strategy_name = strategy_name
        self.symbol = symbol
        self._generated_at = datetime.utcnow().isoformat()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate_html(self, output_path: str) -> str:
        """Create a standalone HTML report with embedded Plotly charts.

        Parameters
        ----------
        output_path : str
            Destination file path (``.html``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_html()
        path.write_text(html, encoding="utf-8")
        return str(path)

    def generate_markdown(self, output_path: str) -> str:
        """Create a Markdown report (charts saved as separate HTML files).

        Parameters
        ----------
        output_path : str
            Destination file path (``.md``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        md = self._build_markdown()
        path.write_text(md, encoding="utf-8")
        return str(path)

    def generate_json(self, output_path: str) -> str:
        """Export all metrics and trade data as JSON.

        Parameters
        ----------
        output_path : str
            Destination file path (``.json``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_json_payload()
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)

    # ------------------------------------------------------------------ #
    #  HTML builder
    # ------------------------------------------------------------------ #

    def _build_html(self) -> str:
        """Assemble the full HTML report string."""
        stats_html = self._render_stats_table()
        equity_html = self._build_equity_chart_html()
        monthly_html = self._build_monthly_returns_html()
        trades_html = self._render_trade_table()
        dd_html = self._build_drawdown_chart_html()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report — {self.strategy_name} | {self.symbol}</title>
<style>
  :root {{ --bg: #0f1117; --card: #1a1d2e; --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1; --green: #22c55e; --red: #ef4444; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.6; }}
  h1 {{ font-size: 1.8rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: .95rem; margin-bottom: 2rem; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .stat-card {{ background: var(--card); border-radius: 12px; padding: 1.2rem; text-align: center; }}
  .stat-card .label {{ color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: .3rem; }}
  .value.positive {{ color: var(--green); }}
  .value.negative {{ color: var(--red); }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: .6rem .8rem; text-align: left; border-bottom: 1px solid #2d3148; font-size: .9rem; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .chart-container {{ background: var(--card); border-radius: 12px; padding: 1rem; margin: 1rem 0; }}
  .trade-buy {{ color: var(--green); }}
  .trade-sell {{ color: var(--red); }}
  footer {{ margin-top: 3rem; color: var(--muted); font-size: .8rem; text-align: center; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p class="subtitle">{self.strategy_name} — {self.symbol} · Generated {self._generated_at}</p>

<h2>Summary Statistics</h2>
{stats_html}

<h2>Equity Curve</h2>
<div class="chart-container">{equity_html}</div>

<h2>Monthly Returns Heatmap</h2>
<div class="chart-container">{monthly_html}</div>

<h2>Drawdown</h2>
<div class="chart-container">{dd_html}</div>

<h2>Trade Log</h2>
{trades_html}

<footer>Generated by AI Trading Architecture — BacktestReport</footer>
</body>
</html>"""

    # ------------------------------------------------------------------ #
    #  Stats table (HTML)
    # ------------------------------------------------------------------ #

    def _render_stats_table(self) -> str:
        r = self.results
        items = [
            ("Total Return", _fmt_pct(r.get("total_return")), _is_positive(r.get("total_return", 0))),
            ("Sharpe Ratio", _fmt_float(r.get("sharpe_ratio"), 2), _is_positive(r.get("sharpe_ratio", 0))),
            ("Max Drawdown", _fmt_pct(r.get("max_drawdown")), False),
            ("Win Rate", _fmt_pct(r.get("win_rate")), _is_positive(r.get("win_rate", 0) - 50)),
            ("Profit Factor", _fmt_float(r.get("profit_factor"), 2), _is_positive(r.get("profit_factor", 1) - 1)),
            ("Total Trades", str(r.get("total_trades", "—")), None),
            ("CAGR", _fmt_pct(r.get("cagr")), _is_positive(r.get("cagr", 0))),
            ("Sortino", _fmt_float(r.get("sortino_ratio"), 2), _is_positive(r.get("sortino_ratio", 0))),
        ]
        cards = []
        for label, value, positive in items:
            cls = ""
            if positive is True:
                cls = "positive"
            elif positive is False:
                cls = "negative"
            cards.append(
                f'<div class="stat-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value {cls}">{value}</div>'
                f"</div>"
            )
        return f'<div class="stats-grid">{"".join(cards)}</div>'

    # ------------------------------------------------------------------ #
    #  Equity curve (Plotly → HTML)
    # ------------------------------------------------------------------ #

    def _build_equity_chart_html(self) -> str:
        equity = _extract_equity(self.results)
        if equity is None or equity.empty:
            return "<p style='color:#94a3b8'>No equity curve data.</p>"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Equity",
                line=dict(color="#6366f1", width=2),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.15)",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        return str(fig.to_html(include_plotlyjs=True, full_html=False))

    # ------------------------------------------------------------------ #
    #  Monthly returns heatmap (Plotly table)
    # ------------------------------------------------------------------ #

    def _build_monthly_returns_html(self) -> str:
        monthly = _extract_monthly_returns(self.results)
        if monthly is None or monthly.empty:
            return "<p style='color:#94a3b8'>No monthly returns data.</p>"

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=monthly.values,
                x=monthly.columns,
                y=monthly.index,
                colorscale="RdYlGn",
                zmid=0,
                text=monthly.map(lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
                texttemplate="%{text}",
                textfont=dict(size=11, color="white"),
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=60, r=20, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        return str(fig.to_html(include_plotlyjs=True, full_html=False))

    # ------------------------------------------------------------------ #
    #  Trade table (HTML)
    # ------------------------------------------------------------------ #

    def _render_trade_table(self) -> str:
        trades = _extract_trades(self.results)
        if not trades:
            return "<p style='color:#94a3b8'>No trade data.</p>"

        rows = []
        for t in trades[:200]:  # cap at 200 for readability
            side_cls = "trade-buy" if str(t.get("side", "")).lower() in ("buy", "long") else "trade-sell"
            pnl = t.get("pnl", t.get("profit", 0))
            pnl_cls = "trade-buy" if pnl >= 0 else "trade-sell"
            rows.append(
                f"<tr>"
                f"<td>{t.get('date', t.get('datetime', t.get('time', '')))}</td>"
                f"<td class='{side_cls}'>{t.get('side', '').upper()}</td>"
                f"<td>{t.get('symbol', self.symbol)}</td>"
                f"<td>{_fmt_float(t.get('price', t.get('fill_price')), 2)}</td>"
                f"<td>{t.get('quantity', t.get('size', ''))}</td>"
                f"<td class='{pnl_cls}'>{_fmt_float(pnl, 2)}</td>"
                f"</tr>"
            )
        return (
            "<table>"
            "<thead><tr><th>Date</th><th>Side</th><th>Symbol</th><th>Price</th><th>Qty</th><th>PnL</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    # ------------------------------------------------------------------ #
    #  Drawdown chart (Plotly → HTML)
    # ------------------------------------------------------------------ #

    def _build_drawdown_chart_html(self) -> str:
        equity = _extract_equity(self.results)
        if equity is None or equity.empty:
            return "<p style='color:#94a3b8'>No data for drawdown.</p>"

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name="Drawdown %",
                line=dict(color="#ef4444", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.2)",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        return str(fig.to_html(include_plotlyjs=True, full_html=False))

    # ------------------------------------------------------------------ #
    #  Markdown builder
    # ------------------------------------------------------------------ #

    def _build_markdown(self) -> str:
        r = self.results
        lines = [
            "# Backtest Report",
            "",
            f"**Strategy:** {self.strategy_name}",
            f"**Symbol:** {self.symbol}",
            f"**Generated:** {self._generated_at}",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Return | {_fmt_pct(r.get('total_return'))} |",
            f"| Sharpe Ratio | {_fmt_float(r.get('sharpe_ratio'), 2)} |",
            f"| Max Drawdown | {_fmt_pct(r.get('max_drawdown'))} |",
            f"| Win Rate | {_fmt_pct(r.get('win_rate'))} |",
            f"| Profit Factor | {_fmt_float(r.get('profit_factor'), 2)} |",
            f"| Total Trades | {r.get('total_trades', '—')} |",
            f"| CAGR | {_fmt_pct(r.get('cagr'))} |",
            f"| Sortino Ratio | {_fmt_float(r.get('sortino_ratio'), 2)} |",
            "",
        ]

        lines.append("## Equity Curve\n")
        lines.append("> *Run `generate_html()` for embedded charts or open `equity_curve.html`.*\n")

        lines.append("## Trade Log\n")
        trades = _extract_trades(self.results)
        if trades:
            lines.append("| Date | Side | Symbol | Price | Qty | PnL |")
            lines.append("|------|------|--------|-------|-----|-----|")
            for t in trades[:100]:
                pnl = t.get("pnl", t.get("profit", 0))
                lines.append(
                    f"| {t.get('date', t.get('datetime', ''))} "
                    f"| {t.get('side', '').upper()} "
                    f"| {t.get('symbol', self.symbol)} "
                    f"| {_fmt_float(t.get('price', t.get('fill_price')), 2)} "
                    f"| {t.get('quantity', t.get('size', ''))} "
                    f"| {pnl:.2f} |"
                )
        else:
            lines.append("*No trade data.*")

        lines.append("")
        lines.append("---")
        lines.append("*Generated by AI Trading Architecture — BacktestReport*")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  JSON payload
    # ------------------------------------------------------------------ #

    def _build_json_payload(self) -> dict[str, Any]:
        r = self.results
        equity = _extract_equity(r)
        monthly = _extract_monthly_returns(r)
        trades = _extract_trades(r)

        return {
            "report_metadata": {
                "strategy": self.strategy_name,
                "symbol": self.symbol,
                "generated_at": self._generated_at,
            },
            "summary_metrics": {
                "total_return": r.get("total_return"),
                "cagr": r.get("cagr"),
                "sharpe_ratio": r.get("sharpe_ratio"),
                "sortino_ratio": r.get("sortino_ratio"),
                "max_drawdown": r.get("max_drawdown"),
                "win_rate": r.get("win_rate"),
                "profit_factor": r.get("profit_factor"),
                "total_trades": r.get("total_trades"),
                "initial_budget": r.get("initial_budget", r.get("initial_capital")),
                "final_value": r.get("final_value"),
                "benchmark_return": r.get("benchmark_return"),
            },
            "equity_curve": (
                equity.reset_index().to_dict(orient="records")
                if equity is not None and not equity.empty
                else []
            ),
            "monthly_returns": (
                monthly.to_dict() if monthly is not None and not monthly.empty else {}
            ),
            "trades": trades,
            "parameters": r.get("parameters", {}),
        }


# ========================================================================
#  Internal helpers
# ========================================================================


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
        return f"{v:+.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _fmt_float(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _is_positive(value: float) -> bool:
    return value > 0


def _extract_equity(results: dict[str, Any]) -> pd.Series | None:
    """Extract equity curve as a DatetimeIndex Series."""
    for key in ("equity_curve", "portfolio_value", "portfolio", "value"):
        val = results.get(key)
        if val is None:
            continue
        if isinstance(val, pd.Series):
            return val
        if isinstance(val, pd.DataFrame):
            for col in ("value", "equity", "portfolio_value", "total"):
                if col in val.columns:
                    return val[col]
            return val.iloc[:, 0]
        if isinstance(val, (list, tuple)):
            return pd.Series(val)
    return None


def _extract_monthly_returns(results: dict[str, Any]) -> pd.DataFrame | None:
    """Return a Year × Month DataFrame of percentage returns.

    Looks for a pre-computed ``monthly_returns`` key, or derives from
    the equity curve.
    """
    mr = results.get("monthly_returns")
    if mr is not None:
        if isinstance(mr, pd.DataFrame):
            return mr
        if isinstance(mr, dict):
            return pd.DataFrame(mr)

    # Derive from equity curve
    equity = _extract_equity(results)
    if equity is None or equity.empty:
        return None

    monthly = equity.resample("ME").last().pct_change() * 100
    if monthly.empty:
        return None

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.strftime("%b"),
        "return": monthly.values,
    })
    pivot["month_order"] = monthly.index.month
    pivot = pivot.sort_values(["year", "month_order"])
    table = pivot.pivot_table(index="year", columns="month_order", values="return")
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    table.columns = [month_labels.get(c, c) for c in table.columns]
    return table


def _extract_trades(results: dict[str, Any]) -> list[dict]:
    """Extract trade list from results dict."""
    for key in ("trades", "filled_trades", "trade_log", "orders"):
        val = results.get(key)
        if val is None:
            continue
        if isinstance(val, pd.DataFrame):
            return cast(list[dict], val.to_dict(orient="records"))
        if isinstance(val, (list, tuple)):
            return list(val)
    return []
