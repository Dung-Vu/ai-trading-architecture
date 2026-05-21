"""Optuna-based parameter optimizer for trading strategies.

Usage
-----
    from strategy.optimizer import ParameterOptimizer
    from strategy.sma_cross import SMACrossStrategy

    optimizer = ParameterOptimizer(
        strategy_class=SMACrossStrategy,
        symbol="BTC",
        start_date="2024-01-01",
        end_date="2024-03-31",
    )
    best_params = optimizer.optimize(n_trials=50)
    print(best_params)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import optuna
from loguru import logger

from src.config import (
    env_float,
    get_default_exchange_name,
    get_default_strategy_optimizer_bbands_param_space,
    get_default_strategy_optimizer_sma_param_space,
)
from .backtest import BacktestRunner
from .base import BaseStrategy


class ParameterOptimizer:
    """Optimize strategy parameters using Optuna Bayesian optimisation.

    Wraps :class:`BacktestRunner` inside an Optuna study and searches the
    hyper-parameter space for the combination that maximises the Sharpe
    ratio (or another configurable metric).

    Parameters
    ----------
    strategy_class :
        A subclass of :class:`BaseStrategy` to optimise.
    symbol : str
        Base asset symbol (e.g. ``"BTC"``).
    start_date : str
        Backtest start date (``YYYY-MM-DD``).
    end_date : str
        Backtest end date (``YYYY-MM-DD``).
    ccxt_exchange : str
        CCXT exchange identifier (default ``"binance"``).
    initial_capital : float
        Starting portfolio value for each trial.
    storage_path : str or Path, optional
        Where to save the Optuna study (SQLite).  If ``None``, an
        in-memory study is used.
    results_dir : str or Path, optional
        Directory to persist best-params JSON files.
    """

    # Default search space for SMA cross strategies
    SMA_PARAM_SPACE = get_default_strategy_optimizer_sma_param_space()

    # Default search space for Bollinger Band strategies
    BBANDS_PARAM_SPACE = get_default_strategy_optimizer_bbands_param_space()

    def __init__(
        self,
        strategy_class: type[BaseStrategy],
        symbol: str,
        start_date: str,
        end_date: str,
        ccxt_exchange: Optional[str] = None,
        initial_capital: Optional[float] = None,
        storage_path: Optional[str] = None,
        results_dir: Optional[str] = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.ccxt_exchange = ccxt_exchange or get_default_exchange_name()
        self.initial_capital = (
            initial_capital if initial_capital is not None else env_float("INITIAL_CAPITAL", 100_000)
        )

        # Optuna storage — SQLite for persistence
        if storage_path:
            self.storage = f"sqlite:///{storage_path}"
        else:
            self.storage = None

        # Results directory
        self.results_dir = Path(results_dir) if results_dir else Path("results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._study: Optional[optuna.study.Study] = None
        self._best_params: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def optimize(
        self,
        n_trials: int = 50,
        metric: str = "sharpe_ratio",
        direction: str = "maximize",
        param_space: Optional[dict[str, tuple]] = None,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        prune: bool = True,
    ) -> dict[str, Any]:
        """Run the Optuna optimisation study.

        Parameters
        ----------
        n_trials : int
            Number of trials to run (default 50).
        metric : str
            Metric to optimise.  Supported: ``"sharpe_ratio"``,
            ``"sortino_ratio"``, ``"total_return"``, ``"win_rate"``,
            ``"profit_factor"``, ``"calmar_ratio"``.
        direction : str
            ``"maximize"`` or ``"minimize"``.
        param_space : dict, optional
            Custom parameter space.  Each key maps to ``(min, max)``.
            Keys must match the strategy's parameter names.  If
            ``None``, the space is auto-detected from the strategy
            class name.
        timeout : int, optional
            Maximum seconds for the entire study.
        study_name : str, optional
            Human-readable name for the Optuna study.
        prune : bool
            Use ``MedianPruner`` to stop unpromising trials early.

        Returns
        -------
        dict
            The best parameter set found.
        """
        # Determine parameter space
        if param_space is None:
            param_space = self._detect_param_space()

        # Create Optuna study
        pruner = optuna.pruners.MedianPruner() if prune else None
        if study_name is None:
            study_name = (
                f"{self.strategy_class.__name__}_{self.symbol}_"
                f"{self.start_date}_{self.end_date}"
            )

        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=self.storage,
            pruner=pruner,
            load_if_exists=True,
        )

        # Bind metric and param_space to the objective
        def _objective(trial: optuna.Trial) -> float:
            return self._objective(trial, param_space, metric)

        logger.info(
            "Starting Optuna study: {} | {} trials | metric={} | params={}",
            study_name,
            n_trials,
            metric,
            list(param_space.keys()),
        )

        start_time = time.time()
        self._study.optimize(_objective, n_trials=n_trials, timeout=timeout)
        elapsed = time.time() - start_time

        # Extract best trial
        best_trial = self._study.best_trial
        self._best_params = dict(best_trial.params)

        logger.info(
            "Optimization complete in {:.1f}s — best {} = {:.4f}",
            elapsed,
            metric,
            best_trial.value,
        )
        logger.info("Best params: {}", self._best_params)

        # Persist results
        self._save_results(best_trial, metric, elapsed, param_space)

        return self._best_params

    def get_study(self) -> Optional[optuna.study.Study]:
        """Return the Optuna study object (after :meth:`optimize` is called)."""
        return self._study

    def get_best_params(self) -> Optional[dict[str, Any]]:
        """Return the best parameters found."""
        return self._best_params

    def plot_results(self, output_path: Optional[str] = None) -> None:
        """Generate and optionally save Optuna optimisation plots.

        Creates:
        - Parameter importance (bar chart)
        - Optimisation history (line chart)
        - Parameter slice (scatter)

        Parameters
        ----------
        output_path : str, optional
            Base file path (without extension).  Plots are saved as
            ``{output_path}_importance.png``, ``{output_path}_history.png``,
            etc.
        """
        if self._study is None:
            raise RuntimeError("No study available. Call optimize() first.")

        try:
            optuna.visualization.plot_optimization_history(self._study)
            optuna.visualization.plot_param_importances(self._study)
            optuna.visualization.plot_slice(self._study)
        except Exception as exc:
            logger.warning("Could not generate Optuna plots: {}", exc)

        if output_path and len(self._study.trials) > 1:
            base = Path(output_path)
            base.parent.mkdir(parents=True, exist_ok=True)

            # Save individual plots
            try:
                fig_history = optuna.visualization.plot_optimization_history(self._study)
                fig_history.write_image(str(base.with_suffix(".history.png")))
            except Exception:
                pass

            try:
                fig_imp = optuna.visualization.plot_param_importances(self._study)
                fig_imp.write_image(str(base.with_suffix(".importance.png")))
            except Exception:
                pass

            try:
                fig_slice = optuna.visualization.plot_slice(self._study)
                fig_slice.write_image(str(base.with_suffix(".slice.png")))
            except Exception:
                pass

            logger.info("Optimization plots saved to {}", base.parent)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _objective(
        self,
        trial: optuna.Trial,
        param_space: dict[str, tuple],
        metric: str,
    ) -> float:
        """Objective function for Optuna — runs a single backtest trial.

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial.
        param_space : dict
            Parameter space definition.
        metric : str
            Which metric to return as the objective value.

        Returns
        -------
        float
            The objective value (e.g. Sharpe ratio).
        """
        # Sample parameters
        params = self._sample_params(trial, param_space)

        # Build backtest runner
        runner = BacktestRunner(
            strategy_class=self.strategy_class,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters=params,
            ccxt_exchange=self.ccxt_exchange,
            initial_capital=self.initial_capital,
        )

        try:
            runner.run()
            results = runner.get_results()
        except Exception as exc:
            logger.warning("Trial {} failed: {}", trial.number, exc)
            # Return a very bad score so Optuna prunes this direction
            return float("-inf")

        # Extract the target metric
        value = results.get(metric)
        if value is None:
            # Fallback: try alternate key names
            fallback_keys = {
                "sharpe_ratio": ["sharpe"],
                "sortino_ratio": ["sortino"],
                "total_return": ["return"],
                "max_drawdown": ["max_dd"],
            }
            for alt in fallback_keys.get(metric, []):
                value = results.get(alt)
                if value is not None:
                    break

        if value is None:
            logger.warning("Trial {} — metric '{}' not found in results", trial.number, metric)
            return float("-inf")

        # Handle negative infinity or NaN
        if not (float("-inf") < value < float("inf")):
            return float("-inf")

        # Report intermediate value for pruning
        trial.report(value, step=trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    def _sample_params(
        self,
        trial: optuna.Trial,
        param_space: dict[str, tuple],
    ) -> dict[str, Any]:
        """Sample a parameter set from the search space for this trial.

        Parameters
        ----------
        trial : optuna.Trial
        param_space : dict
            Mapping of param name → ``(min, max)``.

        Returns
        -------
        dict
            Parameter dict ready to be passed to the strategy.
        """
        params: dict[str, Any] = {}

        for name, (low, high) in param_space.items():
            # Integer parameters for period-like names
            if any(kw in name for kw in ("period", "fast", "slow", "factor")):
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                # Float parameters
                params[name] = trial.suggest_float(name, float(low), float(high))

        return params

    def _detect_param_space(self) -> dict[str, tuple]:
        """Auto-detect the parameter space from the strategy class name."""
        name = self.strategy_class.__name__.lower()

        if "sma" in name or "cross" in name:
            logger.info("Auto-detected SMA param space")
            return self.SMA_PARAM_SPACE.copy()

        if "bb" in name or "bollinger" in name or "bband" in name:
            logger.info("Auto-detected Bollinger Bands param space")
            return self.BBANDS_PARAM_SPACE.copy()

        # Fallback: use all parameters
        logger.warning(
            "Could not auto-detect param space for {}. "
            "Using combined SMA+BB space.",
            self.strategy_class.__name__,
        )
        combined = {**self.SMA_PARAM_SPACE, **self.BBANDS_PARAM_SPACE}
        return combined

    def _save_results(
        self,
        best_trial: optuna.trial.FrozenTrial,
        metric: str,
        elapsed: float,
        param_space: dict[str, tuple],
    ) -> Path:
        """Save best parameters and study summary to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.strategy_class.__name__}_{self.symbol}_"
            f"{timestamp}.json"
        )
        filepath = self.results_dir / filename

        summary = {
            "strategy": self.strategy_class.__name__,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "ccxt_exchange": self.ccxt_exchange,
            "metric": metric,
            "n_trials": len(self._study.trials) if self._study else 0,
            "elapsed_seconds": round(elapsed, 2),
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "param_space": {k: list(v) for k, v in param_space.items()},
            "study_name": self._study.study_name if self._study else None,
            "timestamp": datetime.now().isoformat(),
        }

        filepath.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Results saved to {}", filepath)

        # Also save a "latest" symlink-style copy for easy access
        latest = self.results_dir / f"latest_{self.strategy_class.__name__}_{self.symbol}.json"
        latest.write_text(json.dumps(summary, indent=2, default=str))

        return filepath
