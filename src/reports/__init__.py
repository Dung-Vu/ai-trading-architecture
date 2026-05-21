"""Backtest report generator facade."""

from __future__ import annotations

import json as jsonlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.reports.html import (
    build_drawdown_chart_html,
    build_equity_chart_html,
    build_html,
    build_monthly_returns_html,
    render_stats_table,
    render_trade_table,
)
from src.reports.json import build_json_payload
from src.reports.markdown import build_markdown
from src.reports.utils import (
    extract_equity,
    extract_monthly_returns,
    extract_trades,
    fmt_float,
    fmt_pct,
    is_positive,
)


class BacktestReport:
    """Generate production-ready backtest reports in HTML, Markdown, and JSON."""

    def __init__(
        self,
        backtest_results: dict[str, Any],
        strategy_name: str,
        symbol: str,
    ) -> None:
        self.results = backtest_results
        self.strategy_name = strategy_name
        self.symbol = symbol
        self._generated_at = datetime.now(timezone.utc).isoformat()

    def generate_html(self, output_path: str) -> str:
        """Create a standalone HTML report with embedded Plotly charts."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._build_html(), encoding="utf-8")
        return str(path)

    def generate_markdown(self, output_path: str) -> str:
        """Create a Markdown report."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._build_markdown(), encoding="utf-8")
        return str(path)

    def generate_json(self, output_path: str) -> str:
        """Export all metrics and trade data as JSON."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            jsonlib.dumps(self._build_json_payload(), indent=2, default=str),
            encoding="utf-8",
        )
        return str(path)

    def _build_html(self) -> str:
        return build_html(
            self.results,
            self.strategy_name,
            self.symbol,
            self._generated_at,
        )

    def _render_stats_table(self) -> str:
        return render_stats_table(self.results)

    def _build_equity_chart_html(self) -> str:
        return build_equity_chart_html(self.results)

    def _build_monthly_returns_html(self) -> str:
        return build_monthly_returns_html(self.results)

    def _render_trade_table(self) -> str:
        return render_trade_table(self.results, self.symbol)

    def _build_drawdown_chart_html(self) -> str:
        return build_drawdown_chart_html(self.results)

    def _build_markdown(self) -> str:
        return build_markdown(
            self.results,
            self.strategy_name,
            self.symbol,
            self._generated_at,
        )

    def _build_json_payload(self) -> dict[str, Any]:
        return build_json_payload(
            self.results,
            self.strategy_name,
            self.symbol,
            self._generated_at,
        )


def _fmt_pct(value: Any) -> str:
    return fmt_pct(value)


def _fmt_float(value: Any, decimals: int = 2) -> str:
    return fmt_float(value, decimals)


def _is_positive(value: float) -> bool:
    return is_positive(value)


def _extract_equity(results: dict[str, Any]) -> Optional[pd.Series]:
    return extract_equity(results)


def _extract_monthly_returns(results: dict[str, Any]) -> Optional[pd.DataFrame]:
    return extract_monthly_returns(results)


def _extract_trades(results: dict[str, Any]) -> list[dict]:
    return extract_trades(results)


__all__ = [
    "BacktestReport",
    "_extract_equity",
    "_extract_monthly_returns",
    "_extract_trades",
    "_fmt_float",
    "_fmt_pct",
    "_is_positive",
]
