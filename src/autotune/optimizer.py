"""Optimizer, prompt update, and config persistence helpers for AutoTuner."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_default_autotune_param_combos
from src.runtime_helpers import build_backtest_dates, build_backtest_runner


class AutoTuneOptimizerMixin:
    """Optimization helpers used by ``AutoTuner``."""

    def _run_optimizer(
        self, trades: list[dict[str, Any]], current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run parameter optimization.

        Uses Lumibot backtesting when a strategy class is available, otherwise
        falls back to a heuristic estimator.
        """
        best_params = dict(self.PARAM_RANGES)
        best_score = current_metrics.get("sharpe_ratio", 0)

        param_combos = getattr(
            self,
            "PARAM_COMBOS",
            get_default_autotune_param_combos(),
        )

        symbol = "BTC"
        if trades:
            symbols = [t.get("symbol", "BTC/USDT") for t in trades]
            most_common = max(set(symbols), key=symbols.count)
            symbol = most_common.split("/")[0] if "/" in most_common else most_common

        strategy_class = getattr(self, "strategy_class", None)
        if not strategy_class:
            if self.strategy_name.lower() in ["sma_cross", "smacross"]:
                try:
                    from src.strategy.sma_cross import SMACrossStrategy

                    strategy_class = SMACrossStrategy
                except ImportError:
                    pass
            elif self.strategy_name.lower() in ["bbands", "bbands_strategy"]:
                try:
                    from src.strategy.bbands import BBandsStrategy

                    strategy_class = BBandsStrategy
                except ImportError:
                    pass

        backtest_successful = False
        if strategy_class:
            try:
                start_date, end_date = build_backtest_dates(90, use_utc=True)
                logger.info(
                    "Attempting real Lumibot backtest optimization for "
                    f"{symbol} (90-day window)..."
                )

                for combo in param_combos:
                    runner = build_backtest_runner(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        parameters=combo,
                        initial_capital=self.initial_capital,
                    )
                    runner.run()
                    results = runner.get_results()
                    simulated_score = results.get("sharpe_ratio", 0)
                    if simulated_score is None:
                        simulated_score = 0.0

                    if simulated_score > best_score:
                        best_score = simulated_score
                        best_params = combo

                backtest_successful = True
                logger.info("Lumibot backtest optimization completed successfully.")
            except Exception as exc:
                logger.warning(
                    f"Real backtest failed due to error: {exc}. "
                    "Falling back to heuristic _simulate_params estimator."
                )

        if not backtest_successful:
            logger.info("Running heuristic optimization estimator...")
            for combo in param_combos:
                simulated_score = self._simulate_params(trades, combo)
                if simulated_score > best_score:
                    best_score = simulated_score
                    best_params = combo

        return {
            "new_params": best_params,
            "previous_score": current_metrics.get("sharpe_ratio", 0),
            "expected_score": best_score,
            "improvement": round(best_score - current_metrics.get("sharpe_ratio", 0), 3),
        }

    def _simulate_params(
        self, trades: list[dict[str, Any]], params: dict[str, Any]
    ) -> float:
        """
        Simulate a parameter set against historical trades.

        This is a simple heuristic score based on how well RSI thresholds would
        have filtered the historical trades.
        """
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        filtered_pnls = []
        for trade in trades:
            indicators = trade.get("indicators", {})
            rsi = indicators.get("rsi", 50)
            side = trade.get("side", "BUY")

            if side == "BUY" and rsi <= rsi_oversold:
                filtered_pnls.append(trade.get("pnl", 0))
            elif side == "SELL" and rsi >= rsi_overbought:
                filtered_pnls.append(trade.get("pnl", 0))
            elif side == "BUY" and rsi > rsi_oversold:
                pass
            elif side == "SELL" and rsi < rsi_overbought:
                pass
            else:
                filtered_pnls.append(trade.get("pnl", 0))

        if not filtered_pnls:
            return 0

        mean = sum(filtered_pnls) / len(filtered_pnls)
        var = sum((p - mean) ** 2 for p in filtered_pnls) / max(
            len(filtered_pnls) - 1, 1
        )
        std = math.sqrt(var) if var > 0 else 0

        return (mean / std * math.sqrt(252)) if std > 0 else 0

    def _update_debate_prompts(self, optimization: dict[str, Any]) -> None:
        """Update debate agent prompts with new insights from optimization."""
        new_params = optimization.get("new_params", {})
        insights = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "key_insights": [],
            "parameter_changes": new_params,
        }

        for param, value in new_params.items():
            if param in self.PARAM_RANGES:
                default = self.PARAM_RANGES[param]["default"]
                if value != default:
                    direction = "increased" if value > default else "decreased"
                    insights["key_insights"].append(
                        f"{param} {direction} from {default} to {value} "
                        f"based on recent performance analysis"
                    )

        insights_path = self.config_dir / f"{self.strategy_name}_insights.json"
        with open(insights_path, "w") as f:
            json.dump(insights, f, indent=2)

        logger.info(f"Saved debate insights to {insights_path}")

    def _save_optimized_config(
        self, optimization: dict[str, Any], metrics: dict[str, Any]
    ) -> Path:
        """Save optimized configuration to file."""
        config = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "parameters": optimization.get("new_params", {}),
            "performance": metrics,
            "optimization_score": optimization.get("expected_score", 0),
            "improvement": optimization.get("improvement", 0),
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_optimized_{timestamp}.json"
        config_path = self.config_dir / filename

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        latest_path = self.config_dir / f"{self.strategy_name}_latest.json"
        with open(latest_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved optimized config to {config_path}")
        return config_path

    def _generate_comparison(
        self, current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a before/after comparison report."""
        if len(self._optimization_history) < 1:
            return {
                "status": "no_previous_data",
                "current_metrics": current_metrics,
            }

        prev_metrics_list = [
            r.get("metrics", {})
            for r in self._optimization_history
            if r.get("metrics")
        ]

        if not prev_metrics_list:
            return {
                "status": "first_run",
                "current_metrics": current_metrics,
            }

        avg_prev = {}
        for key in ["win_rate", "sharpe_ratio", "total_pnl"]:
            values = [m.get(key, 0) for m in prev_metrics_list if key in m]
            avg_prev[key] = sum(values) / len(values) if values else 0

        return {
            "status": "compared",
            "previous_average": {
                "win_rate": round(avg_prev.get("win_rate", 0), 4),
                "sharpe_ratio": round(avg_prev.get("sharpe_ratio", 0), 3),
                "total_pnl": round(avg_prev.get("total_pnl", 0), 2),
            },
            "current": {
                "win_rate": round(current_metrics.get("win_rate", 0), 4),
                "sharpe_ratio": round(current_metrics.get("sharpe_ratio", 0), 3),
                "total_pnl": round(current_metrics.get("total_pnl", 0), 2),
            },
            "changes": {
                "win_rate_delta": round(
                    current_metrics.get("win_rate", 0) - avg_prev.get("win_rate", 0),
                    4,
                ),
                "sharpe_delta": round(
                    current_metrics.get("sharpe_ratio", 0)
                    - avg_prev.get("sharpe_ratio", 0),
                    3,
                ),
                "pnl_delta": round(
                    current_metrics.get("total_pnl", 0)
                    - avg_prev.get("total_pnl", 0),
                    2,
                ),
            },
        }
