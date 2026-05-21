"""Shared helpers for backtest report builders."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def format_currency(value: Any) -> str:
    """Format a number as currency with a leading dollar sign."""
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    sign = "-" if numeric < 0 else ""
    return f"{sign}${abs(numeric):,.2f}"


def format_percent(value: Any, *, signed: bool = False) -> str:
    """Format a number as a percentage string."""
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if signed:
        return f"{numeric:+.2f}%"
    return f"{numeric:.2f}%"


def fmt_pct(value: Any) -> str:
    return format_percent(value, signed=True)


def fmt_float(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def is_positive(value: float) -> bool:
    return value > 0


def extract_equity(results: dict[str, Any]) -> Optional[pd.Series]:
    """Extract equity curve as a Series."""
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


def extract_monthly_returns(results: dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return a Year x Month DataFrame of percentage returns."""
    monthly_returns = results.get("monthly_returns")
    if monthly_returns is not None:
        if isinstance(monthly_returns, pd.DataFrame):
            return monthly_returns
        if isinstance(monthly_returns, dict):
            return pd.DataFrame(monthly_returns)

    equity = extract_equity(results)
    if equity is None or equity.empty:
        return None

    monthly = equity.resample("ME").last().pct_change() * 100
    if monthly.empty:
        return None

    pivot = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month_order": monthly.index.month,
            "return": monthly.values,
        }
    )
    table = pivot.pivot_table(index="year", columns="month_order", values="return")
    month_labels = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    table.columns = [month_labels.get(c, c) for c in table.columns]
    return table


def extract_trades(results: dict[str, Any]) -> list[dict]:
    """Extract trade list from results dict."""
    for key in ("trades", "filled_trades", "trade_log", "orders"):
        val = results.get(key)
        if val is None:
            continue
        if isinstance(val, pd.DataFrame):
            return val.to_dict(orient="records")
        if isinstance(val, (list, tuple)):
            return list(val)
    return []
