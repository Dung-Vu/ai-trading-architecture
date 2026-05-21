"""
Auto-tuner for automated strategy optimization based on trading performance.

The package keeps the public ``AutoTuner`` import stable while moving metrics
and optimizer internals into focused modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.autotune.metrics import AutoTuneMetricsMixin
from src.autotune.optimizer import AutoTuneOptimizerMixin
from src.config import (
    get_default_autotune_drawdown_alert_threshold,
    get_default_autotune_min_trades,
    get_default_autotune_param_combos,
    get_default_autotune_param_ranges,
    get_default_initial_capital,
    get_default_autotune_sharpe_decay_threshold,
    get_default_autotune_win_rate_decay_threshold,
)
from src.memory.interfaces import TradeMemoryInterface


class AutoTuner(AutoTuneMetricsMixin, AutoTuneOptimizerMixin):
    """
    Automated strategy performance optimizer.

    Connects to trade memory to analyze recent performance, detects strategy
    decay, and generates parameter adjustment recommendations.
    """

    MIN_TRADES_FOR_OPTIMIZATION = get_default_autotune_min_trades()
    SHARPE_DECAY_THRESHOLD = get_default_autotune_sharpe_decay_threshold()
    WIN_RATE_DECAY_THRESHOLD = get_default_autotune_win_rate_decay_threshold()
    DRAWDOWN_ALERT_THRESHOLD = get_default_autotune_drawdown_alert_threshold()
    PARAM_RANGES = get_default_autotune_param_ranges()
    PARAM_COMBOS = get_default_autotune_param_combos()

    def __init__(
        self,
        trade_memory: TradeMemoryInterface,
        debate_config: dict[str, Any] | None = None,
        strategy_name: str = "sma_cross",
        config_dir: str = "config/optimized",
        initial_capital: float | None = None,
    ) -> None:
        self.trade_memory = trade_memory
        self.debate_config = debate_config or {}
        self.strategy_name = strategy_name
        self.initial_capital = (
            initial_capital
            if initial_capital is not None
            else get_default_initial_capital()
        )
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._optimization_history: list[dict[str, Any]] = []
        self._current_params: dict[str, float] = {}

        logger.info(f"AutoTuner initialized for strategy: {strategy_name}")

    async def weekly_optimization_cycle(self) -> dict[str, Any]:
        """Run the full weekly optimization cycle."""
        logger.info("Starting weekly optimization cycle...")
        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": self.strategy_name,
            "steps_completed": [],
        }

        logger.info("  Step 1: Fetching last week's trades...")
        trades = await self._fetch_recent_trades(days=7)
        report["trades_analyzed"] = len(trades)
        report["steps_completed"].append("fetch_trades")

        if not trades:
            logger.warning("  No trades found in the last week. Skipping optimization.")
            report["status"] = "skipped_no_trades"
            return report

        logger.info("  Step 2: Calculating performance metrics...")
        metrics = self._calculate_metrics(trades)
        report["metrics"] = metrics
        report["steps_completed"].append("calculate_metrics")

        logger.info(
            f"  Metrics: win_rate={metrics['win_rate']:.1%}, "
            f"sharpe={metrics['sharpe_ratio']:.2f}, "
            f"total_pnl=${metrics['total_pnl']:+,.2f}"
        )

        if len(trades) >= self.MIN_TRADES_FOR_OPTIMIZATION:
            logger.info(
                f"  Step 3: Running optimizer ({len(trades)} trades available)..."
            )
            optimization = self._run_optimizer(trades, metrics)
            report["optimization"] = optimization
            report["steps_completed"].append("run_optimizer")

            if optimization.get("new_params"):
                logger.info("  Step 4: Updating debate agent prompts...")
                self._update_debate_prompts(optimization)
                report["steps_completed"].append("update_prompts")

                logger.info("  Step 5: Saving optimized config...")
                config_path = self._save_optimized_config(optimization, metrics)
                report["config_path"] = str(config_path)
                report["steps_completed"].append("save_config")
        else:
            logger.info(
                f"  Step 3: Skipped - only {len(trades)} trades "
                f"(need {self.MIN_TRADES_FOR_OPTIMIZATION})"
            )
            report["status"] = "insufficient_trades"
            report["steps_completed"].append("skip_optimizer")

        logger.info("  Step 6: Generating comparison report...")
        comparison = self._generate_comparison(metrics)
        report["comparison"] = comparison
        report["steps_completed"].append("generate_report")
        report["status"] = "completed"

        self._optimization_history.append(report)

        logger.info("Weekly optimization cycle complete")
        return report

    async def detect_strategy_decay(self) -> bool:
        """
        Detect if strategy performance is degrading.

        Uses rolling Sharpe ratio and win rate over the last 20 trades compared
        to historical averages.
        """
        recent_trades = await self._fetch_recent_trades(days=30)
        if len(recent_trades) < 10:
            logger.debug("Not enough trades for decay detection")
            return False

        recent_20 = recent_trades[:20]
        older_trades = recent_trades[20:]

        if len(older_trades) < 10:
            recent_metrics = self._calculate_metrics(recent_20)
            return recent_metrics["sharpe_ratio"] < self.SHARPE_DECAY_THRESHOLD

        recent_metrics = self._calculate_metrics(recent_20)
        historical_metrics = self._calculate_metrics(older_trades)

        sharpe_drop = historical_metrics["sharpe_ratio"] - recent_metrics["sharpe_ratio"]
        if sharpe_drop > self.SHARPE_DECAY_THRESHOLD:
            logger.warning(
                f"Strategy decay detected: Sharpe dropped from "
                f"{historical_metrics['sharpe_ratio']:.2f} to "
                f"{recent_metrics['sharpe_ratio']:.2f}"
            )
            return True

        wr_drop = historical_metrics["win_rate"] - recent_metrics["win_rate"]
        if wr_drop > (1 - self.WIN_RATE_DECAY_THRESHOLD):
            logger.warning(
                f"Win rate decay: {historical_metrics['win_rate']:.1%} -> "
                f"{recent_metrics['win_rate']:.1%}"
            )
            return True

        if recent_metrics["max_drawdown"] > self.DRAWDOWN_ALERT_THRESHOLD:
            logger.warning(
                f"Excessive drawdown: {recent_metrics['max_drawdown']:.1%}"
            )
            return True

        logger.info("No strategy decay detected")
        return False

    async def get_optimization_recommendations(self) -> list[str]:
        """Suggest parameter changes based on recent performance."""
        recommendations: list[str] = []
        trades = await self._fetch_recent_trades(days=30)

        if len(trades) < 5:
            return ["Insufficient trade data for recommendations (need 5+)"]

        metrics = self._calculate_metrics(trades)

        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

        late_entries = 0
        early_entries = 0
        for trade in losing_trades:
            debate = trade.get("debate_result", {})
            reasoning = (
                debate.get("reasoning", "") + " " + trade.get("outcome_notes", "")
            ).lower()
            if "late" in reasoning or "missed" in reasoning:
                late_entries += 1
            elif "early" in reasoning or "premature" in reasoning:
                early_entries += 1

        if late_entries > len(losing_trades) * 0.3:
            recommendations.append(
                "Reduce SMA_SLOW from 50 to 40 - signals arriving too late, "
                "missing optimal entry points"
            )

        if early_entries > len(losing_trades) * 0.3:
            recommendations.append(
                "Increase SMA_SLOW from 50 to 60 - too many false signals, "
                "need slower confirmation"
            )

        rsi_overbought_exits = 0
        for trade in trades:
            indicators = trade.get("indicators", {})
            rsi = indicators.get("rsi", 0)
            if rsi and trade.get("pnl", 0) < 0 and rsi > 65:
                rsi_overbought_exits += 1

        if rsi_overbought_exits > 3:
            recommendations.append(
                "Increase RSI overbought threshold to 75 - "
                "too many false sell signals at current 70 level"
            )

        if metrics["win_rate"] < 0.40:
            recommendations.append(
                "Win rate below 40% - consider tightening entry criteria "
                "or increasing RSI_OVERSOLD from 30 to 35 for safer entries"
            )

        if metrics["win_rate"] > 0.65:
            recommendations.append(
                f"Win rate is strong at {metrics['win_rate']:.0%} - "
                "consider increasing position size by 10-20%"
            )

        if metrics["max_drawdown"] > 0.10:
            recommendations.append(
                f"Max drawdown {metrics['max_drawdown']:.0%} exceeds 10% - "
                "reduce MAX_POSITION_PCT from 20% to 15%"
            )

        if metrics["sharpe_ratio"] < 0:
            recommendations.append(
                "Negative Sharpe ratio - strategy is losing money on a "
                "risk-adjusted basis. Consider reducing trade frequency "
                "or switching to a more conservative strategy"
            )

        low_volume_losses = 0
        for trade in losing_trades:
            conditions = trade.get("market_conditions", {})
            if not conditions.get("volume_high", True):
                low_volume_losses += 1

        if low_volume_losses > len(losing_trades) * 0.4:
            recommendations.append(
                "40%+ of losses occurred on low volume - "
                "add volume filter: only trade when volume > 1.5x average"
            )

        if not recommendations:
            recommendations.append(
                "No parameter changes recommended - strategy is performing "
                "within acceptable bounds"
            )

        return recommendations


__all__ = ["AutoTuner"]
