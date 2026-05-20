"""Backtest runner for Lumibot strategies with CCXT data."""

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger

from .base import BaseStrategy


class BacktestRunner:
    """Configure and execute backtests using Lumibot + CCXT (Binance).

    Parameters
    ----------
    strategy_class :
        A subclass of :class:`BaseStrategy` to backtest.
    symbol : str
        Base asset symbol (e.g. ``"BTC"``).
    start_date : str or datetime
        Backtest start date.  Strings are parsed as ``YYYY-MM-DD``.
    end_date : str or datetime
        Backtest end date.
    parameters : dict, optional
        Strategy parameters dict (merged with strategy defaults).
    ccxt_exchange : str
        CCXT exchange identifier (default ``"binance"``).
    quote_asset : str
        Quote asset symbol (default ``"USDT"``).
    initial_capital : float
        Starting portfolio value (default 100 000).
    benchmark_asset : str or None
        Asset to use as benchmark (e.g. ``"SPY"``).  ``None`` skips
        benchmark.
    """

    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        start_date: str | datetime,
        end_date: str | datetime,
        parameters: dict[str, Any] | None = None,
        ccxt_exchange: str = "binance",
        quote_asset: str = "USDT",
        initial_capital: float = 100_000,
        benchmark_asset: str | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.quote_asset = quote_asset
        self.initial_capital = initial_capital
        self.ccxt_exchange = ccxt_exchange
        self.benchmark_asset = benchmark_asset

        # Parse dates
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = start_date

        if isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            self.end_date = end_date

        # Merge parameters
        self.parameters = {
            "symbol": symbol,
            "quote_asset": quote_asset,
            "initial_capital": initial_capital,
        }
        if parameters:
            self.parameters.update(parameters)

        self._results: dict[str, Any] | None = None
        self._strategy_instance = None

    def run(self) -> dict[str, Any]:
        """Execute the backtest and return raw results.

        Uses Lumibot's ``run_backtest`` classmethod with
        ``CcxtBacktesting`` as the data source.

        Returns
        -------
        dict
            Raw backtest results from Lumibot.
        """
        try:
            from lumibot.backtesting import CcxtBacktesting
        except ImportError:
            logger.error(
                "CcxtBacktesting not available. "
                "Ensure lumibot>=3.6.0 is installed."
            )
            raise

        logger.info(
            "Starting backtest: {} {} from {} to {}",
            self.strategy_class.__name__,
            self.symbol,
            self.start_date.date(),
            self.end_date.date(),
        )

        benchmark = None
        if self.benchmark_asset:
            from lumibot.entities import Asset

            benchmark = Asset(self.benchmark_asset)

        results, strat = self.strategy_class.run_backtest(
            CcxtBacktesting,
            self.start_date,
            self.end_date,
            parameters=self.parameters,
            benchmark_asset=benchmark,
            ccxt_exchange=self.ccxt_exchange,
            show_plot=False,
            show_tearsheet=False,
            quiet_logs=False,
        )

        self._results = results
        self._strategy_instance = strat

        logger.info("Backtest completed.")
        return cast(dict[str, Any], results)

    def get_results(self) -> dict[str, Any]:
        """Extract and summarise key metrics from the backtest results.

        Must be called after :meth:`run`.

        Returns
        -------
        dict
            Dictionary with standardised keys: ``total_return``,
            ``sharpe_ratio``, ``max_drawdown``, ``total_trades``, etc.
        """
        if self._results is None:
            raise RuntimeError("No results available. Call run() first.")

        r = self._results

        # Lumibot results structure may vary by version;
        # we handle the common fields with safe access.
        extracted = {
            "total_return": _safe_float(r, "total_return"),
            "cagr": _safe_float(r, "cagr"),
            "sharpe_ratio": _safe_float(r, "sharpe"),
            "sortino_ratio": _safe_float(r, "sortino"),
            "max_drawdown": _safe_float(r, "max_drawdown"),
            "total_trades": _safe_int(r, "total_trades"),
            "win_rate": _safe_float(r, "win_rate"),
            "profit_factor": _safe_float(r, "profit_factor"),
            "initial_budget": _safe_float(r, "initial_budget"),
            "final_value": _safe_float(r, "final_value"),
            "benchmark_return": _safe_float(r, "benchmark_return"),
            "strategy_name": self.strategy_class.__name__,
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "parameters": self.parameters,
            "ccxt_exchange": self.ccxt_exchange,
        }

        # If Lumibot version uses different key names, try fallbacks
        if extracted["total_return"] is None:
            extracted["total_return"] = _safe_float(r, "return")
        if extracted["sharpe_ratio"] is None:
            extracted["sharpe_ratio"] = _safe_float(r, "sharpe_ratio")
        if extracted["max_drawdown"] is None:
            extracted["max_drawdown"] = _safe_float(r, "max_dd")

        return extracted

    def plot_results(
        self, output_path: str | None = None
    ) -> str | None:
        """Create a Plotly chart showing price, indicators, and trades.

        Parameters
        ----------
        output_path : str, optional
            File path to save the HTML chart.  If ``None``, the chart
            is not saved (just returned).

        Returns
        -------
        str or None
            The output file path if ``output_path`` was given, else
            ``None``.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self._results is None:
            raise RuntimeError("No results available. Call run() first.")

        # Try to extract equity curve from results
        equity = _get_equity_curve(self._results)
        if equity is None or equity.empty:
            logger.warning("No equity curve data available for plotting.")
            return None

        trades_list = _get_trades(self._results)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=[
                f"{self.strategy_class.__name__} — {self.symbol}",
                "Portfolio Value",
            ],
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                name="Equity",
                line=dict(color="blue", width=1.5),
            ),
            row=1,
            col=1,
        )

        # Trade markers
        for t in trades_list:
            dt = t.get("datetime") or t.get("date") or t.get("time")
            side = t.get("side", "unknown")
            price = t.get("price", t.get("fill_price", 0))
            color = "green" if side == "buy" else "red"
            fig.add_trace(
                go.Scatter(
                    x=[dt],
                    y=[price],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up" if side == "buy" else "triangle-down",
                        size=12,
                        color=color,
                    ),
                    name=f"{side.upper()} trade",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Returns distribution (second subplot)
        if len(equity) > 1:
            returns = equity.pct_change().dropna()
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values,
                    name="Returns",
                    line=dict(color="orange", width=0.8),
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            height=800,
            template="plotly_white",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns", row=2, col=1)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(path))
            logger.info("Chart saved to {}", str(path))
            return str(path)

        return None


# ------------------------------------------------------------------- #
#  Helpers
# ------------------------------------------------------------------- #


def _safe_float(data: dict, key: str) -> float | None:
    """Safely extract a float from a dict."""
    val = data.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(data: dict, key: str) -> int | None:
    """Safely extract an int from a dict."""
    val = data.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _get_equity_curve(results: dict[str, Any]) -> pd.Series | None:
    """Extract equity curve from Lumibot results."""
    # Try common keys
    for key in ("equity_curve", "portfolio_value", "portfolio", "value"):
        val = results.get(key)
        if val is not None:
            if isinstance(val, pd.Series):
                return val
            if isinstance(val, pd.DataFrame):
                # Guess the value column
                for col in ["value", "equity", "portfolio_value", "total"]:
                    if col in val.columns:
                        return val[col]
                return val.iloc[:, 0]  # first column as fallback
            if isinstance(val, (list, tuple)):
                return pd.Series(val)
    return None


def _get_trades(results: dict[str, Any]) -> list[dict]:
    """Extract trade list from Lumibot results."""
    for key in ("trades", "filled_trades", "trade_log", "orders"):
        val = results.get(key)
        if val is not None and isinstance(val, (list, pd.DataFrame)):
            if isinstance(val, pd.DataFrame):
                return cast(list[dict], val.to_dict(orient="records"))
            return list(val)
    return []
