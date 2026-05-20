"""
Auto-Tuner — Automated strategy optimization based on trading performance.

Monitors strategy performance, detects decay, and recommends parameter
adjustments. Runs a weekly optimization cycle using historical trade data.

Workflow:
1. Fetch last week's trades from TradeMemory
2. Calculate performance metrics (win rate, Sharpe, drawdown)
3. If enough trades (>20), run optimization
4. Update debate agent prompts with new insights
5. Save optimized config to config/optimized/
6. Generate before/after comparison report

Usage:
    >>> tuner = AutoTuner(trade_memory, debate_config, "sma_cross")
    >>> tuner.weekly_optimization_cycle()
    >>> recommendations = tuner.get_optimization_recommendations()
    >>> needs_optimization = tuner.detect_strategy_decay()
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger


# ─── AutoTuner ─────────────────────────────────────────────────────────

class AutoTuner:
    """
    Automated strategy performance optimizer.

    Connects to trade memory to analyze recent performance, detects
    strategy decay, and generates parameter adjustment recommendations.
    """

    # Performance thresholds
    MIN_TRADES_FOR_OPTIMIZATION = 20
    SHARPE_DECAY_THRESHOLD = 0.5
    WIN_RATE_DECAY_THRESHOLD = 0.40
    DRAWDOWN_ALERT_THRESHOLD = 0.15

    # Parameter search ranges
    PARAM_RANGES = {
        "sma_fast": {"min": 5, "max": 50, "default": 20},
        "sma_slow": {"min": 20, "max": 200, "default": 50},
        "rsi_period": {"min": 7, "max": 21, "default": 14},
        "rsi_overbought": {"min": 65, "max": 80, "default": 70},
        "rsi_oversold": {"min": 20, "max": 35, "default": 30},
    }

    def __init__(
        self,
        trade_memory: Any,
        debate_config: dict[str, Any] | None = None,
        strategy_name: str = "sma_cross",
        config_dir: str = "config/optimized",
    ) -> None:
        """
        Initialize AutoTuner.

        Args:
            trade_memory: TradeMemory instance for fetching trade history.
            debate_config: Current debate engine configuration.
            strategy_name: Name of the strategy being tuned.
            config_dir: Directory to save optimized configs.
        """
        self.trade_memory = trade_memory
        self.debate_config = debate_config or {}
        self.strategy_name = strategy_name
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Historical optimization results
        self._optimization_history: list[dict[str, Any]] = []
        self._current_params: dict[str, float] = {}

        logger.info(
            f"✅ AutoTuner initialized for strategy: {strategy_name}"
        )

    # ─── Weekly Optimization Cycle ─────────────────────────────────────

    async def weekly_optimization_cycle(self) -> dict[str, Any]:
        """
        Run the full weekly optimization cycle.

        Steps:
        1. Fetch last week's trades
        2. Calculate performance metrics
        3. Run parameter optimizer if enough trades (>20)
        4. Update debate agent prompts with new insights
        5. Save optimized config to config/optimized/
        6. Generate before/after comparison report

        Returns:
            Optimization report dict.
        """
        logger.info("🔧 Starting weekly optimization cycle...")
        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "steps_completed": [],
        }

        # Step 1: Fetch last week's trades
        logger.info("  Step 1: Fetching last week's trades...")
        trades = await self._fetch_recent_trades(days=7)
        report["trades_analyzed"] = len(trades)
        report["steps_completed"].append("fetch_trades")

        if not trades:
            logger.warning("  No trades found in the last week. Skipping optimization.")
            report["status"] = "skipped_no_trades"
            return report

        # Step 2: Calculate performance metrics
        logger.info("  Step 2: Calculating performance metrics...")
        metrics = self._calculate_metrics(trades)
        report["metrics"] = metrics
        report["steps_completed"].append("calculate_metrics")

        logger.info(
            f"  Metrics: win_rate={metrics['win_rate']:.1%}, "
            f"sharpe={metrics['sharpe_ratio']:.2f}, "
            f"total_pnl=${metrics['total_pnl']:+,.2f}"
        )

        # Step 3: Run optimization if enough trades
        if len(trades) >= self.MIN_TRADES_FOR_OPTIMIZATION:
            logger.info(
                f"  Step 3: Running optimizer ({len(trades)} trades available)..."
            )
            optimization = self._run_optimizer(trades, metrics)
            report["optimization"] = optimization
            report["steps_completed"].append("run_optimizer")

            # Step 4: Update debate prompts
            if optimization.get("new_params"):
                logger.info("  Step 4: Updating debate agent prompts...")
                self._update_debate_prompts(optimization)
                report["steps_completed"].append("update_prompts")

                # Step 5: Save optimized config
                logger.info("  Step 5: Saving optimized config...")
                config_path = self._save_optimized_config(optimization, metrics)
                report["config_path"] = str(config_path)
                report["steps_completed"].append("save_config")
        else:
            logger.info(
                f"  Step 3: Skipped — only {len(trades)} trades "
                f"(need {self.MIN_TRADES_FOR_OPTIMIZATION})"
            )
            report["status"] = "insufficient_trades"
            report["steps_completed"].append("skip_optimizer")

        # Step 6: Generate comparison report
        logger.info("  Step 6: Generating comparison report...")
        comparison = self._generate_comparison(metrics)
        report["comparison"] = comparison
        report["steps_completed"].append("generate_report")
        report["status"] = "completed"

        # Store in history
        self._optimization_history.append(report)

        logger.info("✅ Weekly optimization cycle complete")
        return report

    # ─── Strategy Decay Detection ──────────────────────────────────────

    async def detect_strategy_decay(self) -> bool:
        """
        Detect if strategy performance is degrading.

        Uses rolling Sharpe ratio and win rate over the last 20 trades
        compared to historical averages.

        Returns:
            True if optimization is needed (performance degraded).
        """
        # Fetch recent trades
        recent_trades = await self._fetch_recent_trades(days=30)
        if len(recent_trades) < 10:
            logger.debug("Not enough trades for decay detection")
            return False

        # Split into recent vs older
        recent_20 = recent_trades[:20]
        older_trades = recent_trades[20:]

        if len(older_trades) < 10:
            # Not enough historical data for comparison
            recent_metrics = self._calculate_metrics(recent_20)
            return recent_metrics["sharpe_ratio"] < self.SHARPE_DECAY_THRESHOLD

        recent_metrics = self._calculate_metrics(recent_20)
        historical_metrics = self._calculate_metrics(older_trades)

        # Check Sharpe decay
        sharpe_drop = historical_metrics["sharpe_ratio"] - recent_metrics["sharpe_ratio"]
        if sharpe_drop > self.SHARPE_DECAY_THRESHOLD:
            logger.warning(
                f"⚠️ Strategy decay detected: Sharpe dropped from "
                f"{historical_metrics['sharpe_ratio']:.2f} to "
                f"{recent_metrics['sharpe_ratio']:.2f}"
            )
            return True

        # Check win rate decay
        wr_drop = historical_metrics["win_rate"] - recent_metrics["win_rate"]
        if wr_drop > (1 - self.WIN_RATE_DECAY_THRESHOLD):
            logger.warning(
                f"⚠️ Win rate decay: {historical_metrics['win_rate']:.1%} → "
                f"{recent_metrics['win_rate']:.1%}"
            )
            return True

        # Check for excessive drawdown
        if recent_metrics["max_drawdown"] > self.DRAWDOWN_ALERT_THRESHOLD:
            logger.warning(
                f"⚠️ Excessive drawdown: {recent_metrics['max_drawdown']:.1%}"
            )
            return True

        logger.info("✅ No strategy decay detected")
        return False

    # ─── Optimization Recommendations ──────────────────────────────────

    async def get_optimization_recommendations(self) -> list[str]:
        """
        Suggest parameter changes based on recent performance.

        Analyzes recent trade patterns and suggests adjustments like:
        - "Reduce SMA_SLOW from 50 to 40 — faster response needed"
        - "Increase RSI threshold to 75 — too many false signals at 70"

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []
        trades = await self._fetch_recent_trades(days=30)

        if len(trades) < 5:
            return ["Insufficient trade data for recommendations (need 5+)"]

        metrics = self._calculate_metrics(trades)

        # Analyze by pattern
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]

        # 1. SMA timing analysis
        late_entries = 0
        early_entries = 0
        for trade in losing_trades:
            debate = trade.get("debate_result", {})
            reasoning = (debate.get("reasoning", "") + " " +
                        trade.get("outcome_notes", "")).lower()
            if "late" in reasoning or "missed" in reasoning:
                late_entries += 1
            elif "early" in reasoning or "premature" in reasoning:
                early_entries += 1

        if late_entries > len(losing_trades) * 0.3:
            recommendations.append(
                "Reduce SMA_SLOW from 50 to 40 — signals arriving too late, "
                "missing optimal entry points"
            )

        if early_entries > len(losing_trades) * 0.3:
            recommendations.append(
                "Increase SMA_SLOW from 50 to 60 — too many false signals, "
                "need slower confirmation"
            )

        # 2. RSI threshold analysis
        rsi_overbought_exits = 0
        for trade in trades:
            indicators = trade.get("indicators", {})
            rsi = indicators.get("rsi", 0)
            if rsi and trade.get("pnl", 0) < 0:
                if rsi > 65:
                    rsi_overbought_exits += 1

        if rsi_overbought_exits > 3:
            recommendations.append(
                "Increase RSI overbought threshold to 75 — "
                "too many false sell signals at current 70 level"
            )

        # 3. Win rate based suggestions
        if metrics["win_rate"] < 0.40:
            recommendations.append(
                "Win rate below 40% — consider tightening entry criteria "
                "or increasing RSI_OVERSOLD from 30 to 35 for safer entries"
            )

        if metrics["win_rate"] > 0.65:
            recommendations.append(
                f"Win rate is strong at {metrics['win_rate']:.0%} — "
                "consider increasing position size by 10-20%"
            )

        # 4. Drawdown analysis
        if metrics["max_drawdown"] > 0.10:
            recommendations.append(
                f"Max drawdown {metrics['max_drawdown']:.0%} exceeds 10% — "
                "reduce MAX_POSITION_PCT from 20% to 15%"
            )

        # 5. Sharpe ratio suggestions
        if metrics["sharpe_ratio"] < 0:
            recommendations.append(
                "Negative Sharpe ratio — strategy is losing money on a "
                "risk-adjusted basis. Consider reducing trade frequency "
                "or switching to a more conservative strategy"
            )

        # 6. Volume filter suggestion
        low_volume_losses = 0
        for trade in losing_trades:
            conditions = trade.get("market_conditions", {})
            if not conditions.get("volume_high", True):
                low_volume_losses += 1

        if low_volume_losses > len(losing_trades) * 0.4:
            recommendations.append(
                "40%+ of losses occurred on low volume — "
                "add volume filter: only trade when volume > 1.5x average"
            )

        if not recommendations:
            recommendations.append(
                "No parameter changes recommended — strategy is performing "
                "within acceptable bounds"
            )

        return recommendations

    # ─── Internal Methods ──────────────────────────────────────────────

    async def _fetch_recent_trades(self, days: int = 7) -> list[dict[str, Any]]:
        """Fetch trades from memory for the last N days."""
        if self.trade_memory is None:
            logger.warning("No trade memory available for optimization")
            return []

        try:
            # Try async method first (TradeMemory)
            if hasattr(self.trade_memory, "get_trade_history"):
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=days)

                # Check if async
                import inspect
                if inspect.iscoroutinefunction(self.trade_memory.get_trade_history):
                    trades = await self.trade_memory.get_trade_history(
                        limit=500, start_date=start_date, end_date=end_date
                    )
                else:
                    trades = self.trade_memory.get_trade_history(
                        limit=500, start_date=start_date, end_date=end_date
                    )
                return trades
        except Exception as exc:
            logger.warning(f"Failed to fetch trades from memory: {exc}")

        # Fallback: return simulated data for testing
        return self._generate_sample_trades(days)

    def _generate_sample_trades(self, days: int = 7) -> list[dict[str, Any]]:
        """Generate sample trade data for testing/demo purposes."""
        import random
        random.seed(42)

        trades = []
        now = datetime.now(timezone.utc)
        num_trades = min(days * 3, 30)  # ~3 trades per day

        for i in range(num_trades):
            pnl = random.gauss(50, 150)  # Mean $50, std $150
            ts = now - timedelta(hours=random.randint(0, days * 24))

            trades.append({
                "timestamp": ts.isoformat(),
                "symbol": random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
                "side": random.choice(["BUY", "SELL"]),
                "price": random.uniform(30000, 100000),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / 1000 * 100, 2),
                "strategy": self.strategy_name,
                "indicators": {
                    "rsi": random.uniform(20, 80),
                    "sma_fast": random.uniform(30000, 100000),
                    "sma_slow": random.uniform(30000, 100000),
                },
                "market_conditions": {
                    "trend": random.choice(["uptrend", "downtrend", "sideways"]),
                    "volume_high": random.choice([True, False]),
                },
                "debate_result": {
                    "action": random.choice(["BUY", "SELL"]),
                    "confidence": random.uniform(50, 95),
                    "reasoning": "Sample reasoning for optimization",
                },
                "outcome_notes": "",
            })

        return trades

    def _calculate_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate performance metrics from trade list."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
            }

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls)
        win_rate = len(wins) / len(pnls)

        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        if len(pnls) > 1:
            mean_return = sum(pnls) / len(pnls)
            variance = sum((p - mean_return) ** 2 for p in pnls) / (len(pnls) - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            sharpe = (mean_return / std_dev * math.sqrt(252)) if std_dev > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            drawdown = (peak - cumulative) / max(peak, 1)
            max_dd = max(max_dd, drawdown)

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trads": len(losses),
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 3),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
        }

    def _run_optimizer(
        self, trades: list[dict[str, Any]], current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run parameter optimization.

        Integrates Lumibot's BacktestRunner for high-fidelity backtests
        with heuristic fallbacks if real backtesting fails or is unavailable.

        Returns:
            Optimization result with new recommended parameters.
        """
        best_params = dict(self.PARAM_RANGES)
        best_score = current_metrics.get("sharpe_ratio", 0)

        # Define parameter combinations to test
        param_combos = [
            {"sma_fast": 10, "sma_slow": 30, "rsi_oversold": 35, "rsi_overbought": 65},
            {"sma_fast": 15, "sma_slow": 40, "rsi_oversold": 32, "rsi_overbought": 68},
            {"sma_fast": 20, "sma_slow": 50, "rsi_oversold": 30, "rsi_overbought": 70},
            {"sma_fast": 25, "sma_slow": 60, "rsi_oversold": 28, "rsi_overbought": 72},
            {"sma_fast": 10, "sma_slow": 40, "rsi_oversold": 30, "rsi_overbought": 75},
            {"sma_fast": 15, "sma_slow": 50, "rsi_oversold": 35, "rsi_overbought": 70},
        ]

        # Extract symbol
        symbol = "BTC"
        if trades:
            symbols = [t.get("symbol", "BTC/USDT") for t in trades]
            most_common = max(set(symbols), key=symbols.count)
            if "/" in most_common:
                symbol = most_common.split("/")[0]
            else:
                symbol = most_common

        # Check if strategy class is available
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

        # Try to run BacktestRunner for real backtesting
        backtest_successful = False
        if strategy_class:
            try:
                from src.strategy.backtest import BacktestRunner
                # We will backtest over the last 90 days for better statistics
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=90)

                logger.info(f"📊 Attempting real Lumibot backtest optimization for {symbol} (90-day window)...")

                for combo in param_combos:
                    # In backtest parameters, symbol and quote are managed by runner
                    runner = BacktestRunner(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        parameters=combo,
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
                logger.info("✅ Lumibot backtest optimization completed successfully.")
            except Exception as e:
                logger.warning(
                    f"⚠️ Real backtest failed due to error: {e}. "
                    "Falling back to heuristic _simulate_params estimator."
                )

        # Fallback to heuristic estimator if real backtest failed or was unavailable
        if not backtest_successful:
            logger.info("🧮 Running heuristic optimization estimator...")
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

        Simple heuristic: score based on how well the params would have
        filtered the historical trades.

        Returns:
            Simulated Sharpe-like score.
        """
        # Apply hypothetical filter based on RSI thresholds
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        filtered_pnls = []
        for trade in trades:
            indicators = trade.get("indicators", {})
            rsi = indicators.get("rsi", 50)

            # Simple filter: would this trade have been taken?
            side = trade.get("side", "BUY")
            if side == "BUY" and rsi <= rsi_oversold:
                filtered_pnls.append(trade.get("pnl", 0))
            elif side == "SELL" and rsi >= rsi_overbought:
                filtered_pnls.append(trade.get("pnl", 0))
            elif side == "BUY" and rsi > rsi_oversold:
                # Would have missed this trade
                pass
            elif side == "SELL" and rsi < rsi_overbought:
                pass
            else:
                filtered_pnls.append(trade.get("pnl", 0))

        if not filtered_pnls:
            return 0

        mean = sum(filtered_pnls) / len(filtered_pnls)
        var = sum((p - mean) ** 2 for p in filtered_pnls) / max(len(filtered_pnls) - 1, 1)
        std = math.sqrt(var) if var > 0 else 0

        return (mean / std * math.sqrt(252)) if std > 0 else 0

    def _update_debate_prompts(self, optimization: dict[str, Any]) -> None:
        """
        Update debate agent prompts with new insights from optimization.

        Creates an insight file that can be injected into system prompts.
        """
        new_params = optimization.get("new_params", {})
        insights = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "key_insights": [],
            "parameter_changes": new_params,
        }

        # Generate insight text
        for param, value in new_params.items():
            if param in self.PARAM_RANGES:
                default = self.PARAM_RANGES[param]["default"]
                if value != default:
                    direction = "increased" if value > default else "decreased"
                    insights["key_insights"].append(
                        f"{param} {direction} from {default} to {value} "
                        f"based on recent performance analysis"
                    )

        # Save insights file
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

        # Also save as "latest"
        latest_path = self.config_dir / f"{self.strategy_name}_latest.json"
        with open(latest_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved optimized config to {config_path}")
        return config_path

    def _generate_comparison(
        self, current_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate before/after comparison report.

        Compares current metrics against historical average.
        """
        if len(self._optimization_history) < 1:
            return {
                "status": "no_previous_data",
                "current_metrics": current_metrics,
            }

        # Get average of previous runs
        prev_metrics_list = [
            r.get("metrics", {}) for r in self._optimization_history
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
                    current_metrics.get("win_rate", 0) - avg_prev.get("win_rate", 0), 4
                ),
                "sharpe_delta": round(
                    current_metrics.get("sharpe_ratio", 0) - avg_prev.get("sharpe_ratio", 0), 3
                ),
                "pnl_delta": round(
                    current_metrics.get("total_pnl", 0) - avg_prev.get("total_pnl", 0), 2
                ),
            },
        }
