"""JSON report payload builder for backtest results."""

from __future__ import annotations

from typing import Any

from src.reports.utils import (
    extract_equity,
    extract_monthly_returns,
    extract_trades,
)


def build_json_payload(
    results: dict[str, Any],
    strategy_name: str,
    symbol: str,
    generated_at: str,
) -> dict[str, Any]:
    """Build a JSON-serializable report payload."""
    equity = extract_equity(results)
    monthly = extract_monthly_returns(results)
    trades = extract_trades(results)

    return {
        "report_metadata": {
            "strategy": strategy_name,
            "symbol": symbol,
            "generated_at": generated_at,
        },
        "summary_metrics": {
            "total_return": results.get("total_return"),
            "cagr": results.get("cagr"),
            "sharpe_ratio": results.get("sharpe_ratio"),
            "sortino_ratio": results.get("sortino_ratio"),
            "max_drawdown": results.get("max_drawdown"),
            "win_rate": results.get("win_rate"),
            "profit_factor": results.get("profit_factor"),
            "total_trades": results.get("total_trades"),
            "initial_budget": results.get(
                "initial_budget", results.get("initial_capital")
            ),
            "final_value": results.get("final_value"),
            "benchmark_return": results.get("benchmark_return"),
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
        "parameters": results.get("parameters", {}),
    }
