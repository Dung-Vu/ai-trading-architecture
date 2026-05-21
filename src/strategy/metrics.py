"""Performance metrics calculator for trading strategies."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class MetricsCalculator:
    """Static utility class for computing trading performance metrics.

    All methods are ``@staticmethod`` and can be called without
    instantiating the class.
    """

    # ------------------------------------------------------------------ #
    #  Risk-adjusted returns
    # ------------------------------------------------------------------ #

    @staticmethod
    def calc_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Compute the annualised Sharpe ratio.

        Parameters
        ----------
        returns : pd.Series
            Period returns (e.g. daily or hourly).  NaN values are
            dropped automatically.
        risk_free_rate : float
            Annualised risk-free rate (default 2 %).

        Returns
        -------
        float
            Sharpe ratio.  Returns 0.0 if there is insufficient data
            or zero variance.
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0

        # Annualise based on number of observations per year
        periods_per_year = MetricsCalculator._infer_periods_per_year(returns)

        excess = returns - risk_free_rate / periods_per_year
        mean_excess = excess.mean()
        std = returns.std()

        if std == 0:
            return 0.0

        sharpe = (mean_excess / std) * np.sqrt(periods_per_year)
        return float(sharpe)

    @staticmethod
    def calc_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Compute the annualised Sortino ratio.

        The Sortino ratio uses **downside** deviation instead of total
        standard deviation, penalising only harmful volatility.

        Parameters
        ----------
        returns : pd.Series
            Period returns.
        risk_free_rate : float
            Annualised risk-free rate (default 2 %).

        Returns
        -------
        float
            Sortino ratio, or 0.0 if insufficient data.
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0

        periods_per_year = MetricsCalculator._infer_periods_per_year(returns)

        excess = returns - risk_free_rate / periods_per_year
        mean_excess = excess.mean()

        # Downside deviation (only negative returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0  # no downside = ratio is undefined / infinite

        downside_dev = np.sqrt((downside**2).mean())
        if downside_dev == 0:
            return 0.0

        sortino = (mean_excess / downside_dev) * np.sqrt(periods_per_year)
        return float(sortino)

    @staticmethod
    def calc_trade_pnl_sharpe_ratio(
        pnls: list[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 250,
    ) -> float:
        """Compute Sharpe ratio directly from a chronological trade-PnL series."""
        if len(pnls) < 2:
            return 0.0

        pnl_series = pd.Series(pnls, dtype=float).dropna()
        if len(pnl_series) < 2:
            return 0.0

        std = pnl_series.std()
        if std == 0:
            return 0.0

        mean_excess = pnl_series.mean() - risk_free_rate / periods_per_year
        return float((mean_excess / std) * np.sqrt(periods_per_year))

    @staticmethod
    def calc_trade_pnl_max_drawdown(pnls: list[float]) -> float:
        """Compute drawdown from cumulative realised PnL, returned as a fraction."""
        if not pnls:
            return 0.0

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for pnl in pnls:
            equity += float(pnl)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)

        return float(max_drawdown)

    # ------------------------------------------------------------------ #
    #  Drawdown
    # ------------------------------------------------------------------ #

    @staticmethod
    def calc_max_drawdown(equity_curve: pd.Series) -> float:
        """Compute the maximum drawdown from an equity curve.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio value over time.

        Returns
        -------
        float
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15 %).
            Returns 0.0 if the curve is empty.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return 0.0

        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        return float(drawdown.max())

    # ------------------------------------------------------------------ #
    #  Trade-level metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def calc_win_rate(trades: list[dict]) -> float:
        """Compute the fraction of profitable trades.

        Parameters
        ----------
        trades : list[dict]
            Each dict must contain a ``"pnl"`` key (profit and loss
            for that trade).

        Returns
        -------
        float
            Win rate in [0, 1].  Returns 0.0 for empty trade lists.
        """
        if not trades:
            return 0.0

        pnls = [t.get("pnl", 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        return wins / len(pnls)

    @staticmethod
    def calc_profit_factor(trades: list[dict]) -> float:
        """Compute gross profit divided by gross loss.

        Parameters
        ----------
        trades : list[dict]
            Each dict must contain a ``"pnl"`` key.

        Returns
        -------
        float
            Profit factor.  Returns 0.0 if gross loss is zero
            (cannot divide) or there are no trades.
        """
        if not trades:
            return 0.0

        pnls = [t.get("pnl", 0) for t in trades]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        if gross_loss == 0:
            # All trades profitable — return infinity as a large number
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calc_expectancy(trades: list[dict]) -> float:
        """Compute the expected value per trade.

        Formula:
            ``avg_win * win_rate - avg_loss * loss_rate``

        Parameters
        ----------
        trades : list[dict]
            Each dict must contain a ``"pnl"`` key.

        Returns
        -------
        float
            Expectancy per trade.  Returns 0.0 for empty lists.
        """
        if not trades:
            return 0.0

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        n = len(pnls)
        win_rate = len(wins) / n if n else 0
        loss_rate = len(losses) / n if n else 0

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0

        return avg_win * win_rate - avg_loss * loss_rate

    # ------------------------------------------------------------------ #
    #  Combined summary
    # ------------------------------------------------------------------ #

    @staticmethod
    def summarize(
        trades: list[dict],
        equity_curve: Optional[pd.Series],
        risk_free_rate: float = 0.02,
    ) -> dict:
        """Compute all metrics and return them in a single dictionary.

        Parameters
        ----------
        trades : list[dict]
            Trade list with ``"pnl"`` keys.
        equity_curve : pd.Series
            Portfolio value over time.
        risk_free_rate : float
            Annualised risk-free rate for Sharpe/Sortino.

        Returns
        -------
        dict
            Contains: ``sharpe_ratio``, ``sortino_ratio``,
            ``max_drawdown``, ``win_rate``, ``profit_factor``,
            ``expectancy``, ``total_trades``, ``avg_pnl``,
            ``total_pnl``, ``start_value``, ``end_value``.
        """
        if equity_curve is None:
            equity_curve = pd.Series(dtype=float)

        returns = equity_curve.pct_change().dropna() if not equity_curve.empty else pd.Series()

        pnls = [t.get("pnl", 0) for t in trades]
        start_val = float(equity_curve.iloc[0]) if not equity_curve.empty else 0.0
        end_val = float(equity_curve.iloc[-1]) if not equity_curve.empty else 0.0
        total_return_pct = ((end_val - start_val) / start_val * 100) if start_val > 0 else 0.0
        max_dd = MetricsCalculator.calc_max_drawdown(equity_curve)
        win_rate = MetricsCalculator.calc_win_rate(trades)

        return {
            "sharpe_ratio": MetricsCalculator.calc_sharpe_ratio(
                returns, risk_free_rate
            ),
            "sortino_ratio": MetricsCalculator.calc_sortino_ratio(
                returns, risk_free_rate
            ),
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "win_rate": win_rate,
            "win_rate_pct": win_rate * 100,
            "profit_factor": MetricsCalculator.calc_profit_factor(trades),
            "expectancy": MetricsCalculator.calc_expectancy(trades),
            "total_trades": len(trades),
            "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
            "total_pnl": float(sum(pnls)),
            "start_value": start_val,
            "end_value": end_val,
            "total_return_pct": total_return_pct,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_periods_per_year(returns: pd.Series) -> float:
        """Estimate how many periods fit in a year from the index freq.

        Uses common frequency mappings.  Falls back to 252 (daily) if
        the frequency cannot be determined.
        """
        if returns.index.empty:
            return 252.0

        # Try to infer from index frequency
        freq = None
        if hasattr(returns.index, "freq") and returns.index.freq is not None:
            freq = returns.index.freq.name

        # Map common frequencies
        freq_map = {
            "D": 252,       # daily
            "B": 252,       # business daily
            "W": 52,        # weekly
            "M": 12,        # monthly
            "H": 252 * 24,  # hourly (approximate — market hours)
            "min": 252 * 390,  # minute
            "T": 252 * 390,    # minute (alias)
            "S": 252 * 390 * 60,  # second
        }

        if freq:
            # Handle compound frequencies like "2D", "30min"
            base = freq.rstrip("0123456789")
            return float(freq_map.get(base, 252))

        # Fallback: estimate from data span
        span = returns.index[-1] - returns.index[0]
        days = span.total_seconds() / 86400 if hasattr(span, "total_seconds") else 365
        n = len(returns)
        if days > 0:
            return n * (365 / days)
        return 252.0
